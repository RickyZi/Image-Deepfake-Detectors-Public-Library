"""
train.py — unified training entry point for CLIP-D.

Called by launcher.py via:
    python -u train.py --name <name> --arch <arch> --task train [args...]

Two modes, selected automatically by the --ft flag:

  Mode A  (no --ft):  TrainingModel  — frozen CLIP backbone, only fc trained.
                       Saves to: checkpoint/<name>/weights/

  Mode B  (--ft):     FTModel        — LoRA fine-tuning (or block-unfreeze).
                       Loads from: checkpoint/<name>/weights/best.pt
                       Saves to:   checkpoint/<name>/ft_weights/

Log file
────────
  logs/train_<name>.log        (appended on each run)

One line per epoch:
  2025-06-01 14:23:01  INFO     epoch=  1  loss=0.4231  val_acc=0.7812  val_auc=0.8340  lr=1.00e-04

  # Train + test together
python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv True --ft --dataset seasons/autumn_TM01
"""

import os
import time
import tqdm
from datetime import datetime
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from utils.logger import create_logger
from utils.processing import add_processing_arguments
from parser import get_parser
import json


if __name__ == "__main__":
    parser = get_parser()
    parser = add_processing_arguments(parser)
    opt    = parser.parse_args()
    # print(opt)
    # breakpoint()
    opt.task = "train"   # tell make_processing() to apply augmentation
    dataset_name = opt.dataset.replace(os.sep, '_')
    # print(f"dataset: {dataset}")

    # ── Log file ──────────────────────────────────────────────────────────
    # os.makedirs("logs", exist_ok=True)
    # log_path = os.path.join("logs",  f"train_{opt.name}_{dataset_name}.log")
    # print(f"log_path")
    # breakpoint()
    # logger   = build_logger("train", log_path)

    # ── Dataset loaders ───────────────────────────────────────────────────
    if opt.tf2k:
        from utils.tf2k_dataset import tf2k_create_dataloader as _mk
        train_loader = _mk(opt, split="train")
        val_loader   = _mk(opt, split="val")
    else:
        from utils.dataset import create_dataloader as _mk
        train_loader = _mk(opt, split="train")
        val_loader   = _mk(opt, split="val")

    #logger.info(f"training batches   = {len(train_loader)}")
    #logger.info(f"validation batches = {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────
    if opt.ft:
        from utils.finetuning import FTModel
        from utils import EarlyStopping
        save_dir = os.path.join("checkpoint", opt.name, "ft_weights", dataset_name)
        
        # os.makedirs(os.path.join("checkpoint", opt.name, "ft_weights", dataset_name), exist_ok=True)
        model = FTModel(opt)
    else:
        from utils.training import TrainingModel
        from utils import EarlyStopping
        save_dir = os.path.join("checkpoint", opt.name, "weights", dataset_name)
        
        model = TrainingModel(opt)
    
    # breakpoint()

    os.makedirs(save_dir,  exist_ok=True)

    save_log = os.path.join(save_dir, "logs")
    # logger = create_logger(os.path.join(ckpt_dir, 'train.log'))
    # os.makedirs(save_log, exist_ok=True)
    # log_path = os.path.join(save_log,  f"train_{opt.name}_{dataset_name}.log")
    
    logger   = create_logger(os.path.join(save_dir, 'train.log'))

    logger.info(f"Training the R50_tf model on the {dataset_name} dataset - FT: {opt.ft} - unfreezeL4: {opt.r50unfreezeL4}")
    logger.info(f"Training settings: {json.dumps(vars(opt), indent=2, default=str)}")
    
    logger.info(f"training batches   = {len(train_loader)}")
    logger.info(f"validation batches = {len(val_loader)}")
    
    # ── Resume or fresh start ─────────────────────────────────────────────
    early_stopping = None
    start_epoch    = 0
 
    if opt.resume:
        # Scans save_dir for the highest-numbered <N>.pt and loads it.
        # Raises FileNotFoundError with a clear message if nothing is found.
        resumed_state = model.load_for_resume()
 
        # epoch stored in checkpoint = last completed epoch;
        # the loop must start from the NEXT one.
        start_epoch = resumed_state.get("epoch", 0)
 
        # Restore EarlyStopping so patience counting continues from where
        # it left off and best_score is not reset to a worse value.
        es_ckpt = resumed_state.get("early_stopping")
        if es_ckpt is not None:
            early_stopping = EarlyStopping(
                init_score=es_ckpt["best_score"],
                patience=opt.earlystop_epoch,
                delta=0.001,
                verbose=False,
                logger = logger
            )
            early_stopping.best_score = es_ckpt["best_score"]
            early_stopping.count_down = es_ckpt["count_down"]
            early_stopping.early_stop = es_ckpt["early_stop"]
            logger.info(
                f"EarlyStopping restored — best_score={es_ckpt['best_score']:.4f}  "
                f"count_down={es_ckpt['count_down']}  "
                f"early_stop={es_ckpt['early_stop']}"
            )
        else:
            logger.warning(
                "Checkpoint has no early_stopping state "
                "(saved before resume support was added). "
                "EarlyStopping will re-initialise at the next validation."
            )
        logger.info(f"Resuming — last completed epoch={start_epoch}  "
                    f"next epoch={start_epoch + 1}")
 
    run_start = time.time()
 
    for epoch in range(start_epoch, opt.num_epoches + 1):
 
        # ── Training pass ─────────────────────────────────────────────────
        # Skipped when epoch == start_epoch so we always validate the
        # loaded weights before modifying them.
        epoch_loss = float("nan")
        if epoch > start_epoch:
            losses = []
            pbar   = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
            for batch in pbar:
                loss = model.train_on_batch(batch).item()
                losses.append(loss)
                pbar.set_postfix(
                    loss=f"{loss:.4f}",
                    lr=f"{model.get_learning_rate():.2e}",
                )
            epoch_loss = sum(losses) / len(losses)
            # Pass early_stopping so its state is baked into the checkpoint
            # and a subsequent --resume can restore it exactly.
            model.save_networks(epoch, early_stopping=early_stopping)
 
        # ── Validation ────────────────────────────────────────────────────
        y_true, y_pred, _ = model.predict(val_loader)
        val_acc = balanced_accuracy_score(y_true, y_pred > 0.0)
        val_auc = roc_auc_score(y_true, y_pred)
        lr      = model.get_learning_rate()
 
        # ── Early stopping + checkpoint decision ──────────────────────────
        saved = False
        note  = ""
 
        if early_stopping is None:
            early_stopping = EarlyStopping(
                init_score=val_acc,
                patience=opt.earlystop_epoch,
                delta=0.001,
                verbose=False,
                logger = logger
            )
            saved = True
            model.save_networks("best")
            note = "init_best"
 
        else:
            if early_stopping(val_acc):
                saved = True
                model.save_networks("best")
                note = "new_best"
 
            if early_stopping.early_stop:
                cont = model.adjust_learning_rate()
                if cont:
                    new_lr = model.get_learning_rate()
                    note   = f"lr_drop→{new_lr:.2e}"
                    early_stopping.reset_counter()
                    logger.info(
                        f"{epoch:>6}  {epoch_loss:>10.4f}  {val_acc:>8.4f}  "
                        f"{val_auc:>8.4f}  {lr:>10.2e}  {str(saved):>5}  {note}"
                    )
                    continue
                else:
                    note = "early_stop"
                    logger.info(
                        f"{epoch:>6}  {epoch_loss:>10.4f}  {val_acc:>8.4f}  "
                        f"{val_auc:>8.4f}  {lr:>10.2e}  {str(saved):>5}  {note}"
                    )
                    break
 
        # ── Per-epoch log line ────────────────────────────────────────────
        logger.info(
            f"{epoch:>6}  {epoch_loss:>10.4f}  {val_acc:>8.4f}  "
            f"{val_auc:>8.4f}  {lr:>10.2e}  {str(saved):>5}  {note}"
        )
 
    # ── Run summary ───────────────────────────────────────────────────────
    elapsed = time.time() - run_start
    best    = early_stopping.best_score if early_stopping else float("nan")
    logger.info("=" * 70)
    logger.info(
        f"RUN END  elapsed={elapsed:.0f}s  best_val_acc={best:.4f}"
    )
    logger.info(f"Checkpoint dir : {model.save_dir}")
    logger.info(f"Log file       : {os.path.abspath(os.path.join(model.save_dir, 'train.log'))}")
    logger.info("=" * 70)

    # training finished ->  clean them up now and keep just best.pt.
    removed = 0
    for fname in os.listdir(model.save_dir):
        name, ext = os.path.splitext(fname)
        if ext == '.pt' and name.isdigit():
            os.remove(os.path.join(model.save_dir, fname))
            removed += 1
    print(f'Removed {removed} intermediate epoch checkpoint(s), kept best.pt', flush=True)
    logger.info(f'Removed {removed} intermediate epoch checkpoint(s), kept best.pt')
    logger.info("=" * 70)


    # start_epoch = model.total_steps // max(len(train_loader), 1)

    # # ── Log run header ────────────────────────────────────────────────────
    # logger.info("=" * 70)
    # logger.info(f"RUN START  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # logger.info(f"name       = {opt.name}")
    # logger.info(f"arch       = {opt.arch}")
    # logger.info(f"fine-tune  = {opt.ft}")
    # logger.info(f"lr         = {opt.lr}")
    # logger.info(f"weight_dec = {opt.weight_decay}")
    # logger.info(f"batch_size = {opt.batch_size}")
    # logger.info(f"data_keys  = {opt.data_keys}")
    # logger.info(f"data_root  = {opt.data_root}")
    # logger.info(f"split_file = {opt.split_file}")
    # logger.info(f"start_epoch= {start_epoch}")
    # logger.info("=" * 70)
    # # Column header for easy reading / grep
    # logger.info(
    #     f"{'epoch':>6}  {'train_loss':>10}  {'val_acc':>8}  "
    #     f"{'val_auc':>8}  {'lr':>10}  {'saved':>5}  note"
    # )
    # logger.info("-" * 70)

    # early_stopping = None
    # run_start      = time.time()

    # for epoch in range(start_epoch, opt.num_epoches + 1):

    #     # ── Training pass ─────────────────────────────────────────────────
    #     epoch_loss = float("nan")
    #     if epoch > start_epoch:
    #         losses = []
    #         pbar   = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    #         for batch in pbar:
    #             loss = model.train_on_batch(batch).item()
    #             losses.append(loss)
    #             pbar.set_postfix(
    #                 loss=f"{loss:.4f}",
    #                 lr=f"{model.get_learning_rate():.2e}",
    #             )
    #         epoch_loss = sum(losses) / len(losses)
    #         model.save_networks(epoch)

    #     # ── Validation ────────────────────────────────────────────────────
    #     y_true, y_pred, _ = model.predict(val_loader)
    #     val_acc = balanced_accuracy_score(y_true, y_pred > 0.0)
    #     val_auc = roc_auc_score(y_true, y_pred)
    #     lr      = model.get_learning_rate()

    #     # ── Early stopping + checkpoint decision ──────────────────────────
    #     saved = False
    #     note  = ""

    #     if early_stopping is None:
    #         early_stopping = EarlyStopping(
    #             init_score=val_acc,
    #             patience=opt.earlystop_epoch,
    #             delta=0.001,
    #             verbose=False,      # we log ourselves
    #         )
    #         saved = True
    #         model.save_networks("best")
    #         note = "init_best"

    #     else:
    #         if early_stopping(val_acc):
    #             saved = True
    #             model.save_networks("best")
    #             note = "new_best"

    #         if early_stopping.early_stop:
    #             cont = model.adjust_learning_rate()
    #             if cont:
    #                 new_lr = model.get_learning_rate()
    #                 note   = f"lr_drop→{new_lr:.2e}"
    #                 early_stopping.reset_counter()
    #                 logger.info(
    #                     f"{epoch:>6}  {epoch_loss:>10.4f}  {val_acc:>8.4f}  "
    #                     f"{val_auc:>8.4f}  {lr:>10.2e}  {str(saved):>5}  {note}"
    #                 )
    #                 continue
    #             else:
    #                 note = "early_stop"
    #                 logger.info(
    #                     f"{epoch:>6}  {epoch_loss:>10.4f}  {val_acc:>8.4f}  "
    #                     f"{val_auc:>8.4f}  {lr:>10.2e}  {str(saved):>5}  {note}"
    #                 )
    #                 break

    #     # ── Per-epoch log line ────────────────────────────────────────────
    #     logger.info(
    #         f"{epoch:>6}  {epoch_loss:>10.4f}  {val_acc:>8.4f}  "
    #         f"{val_auc:>8.4f}  {lr:>10.2e}  {str(saved):>5}  {note}"
    #     )

    # # ── Run summary ───────────────────────────────────────────────────────
    # elapsed = time.time() - run_start
    # best    = early_stopping.best_score if early_stopping else float("nan")
    # logger.info("=" * 70)
    # logger.info(
    #     f"RUN END  elapsed={elapsed:.0f}s  best_val_acc={best:.4f}"
    # )
    # logger.info(f"Checkpoint dir : checkpoint/{opt.name}/")
    # logger.info(f"Log file       : {os.path.abspath(log_path)}")
    # logger.info("=" * 70)