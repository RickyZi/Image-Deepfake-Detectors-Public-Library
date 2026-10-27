"""
train.py — unified training entry point for CLIP-D

Called by launcher.py via:
- LoRA fine-tuning:
    python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset seasons/autumn_TM01
- Frozen backbone training:
    python3 launcher.py --detector CLIP-D --phases both --dataset seasons/autumn_TM01 --ft
    (pretrained backbone is already trained by TB, so this is just fine-tuning the head -> weights-name = pretrained [default])

-> NOTE: LoRA is called by the --weights-name argument, which is passed to the model constructor.
The LoRA rank and targets are parsed from the weights-name string, e.g. "lora_r4_qv" means rank=4, targets=('q','v').

Train + test together:
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
    opt.task = "train"   
    dataset_name = opt.dataset.replace(os.sep, '_')

    # Dataset loaders
    if opt.tf2k:
        from utils.tf2k_dataset import tf2k_create_dataloader as _mk
        train_loader = _mk(opt, split="train")
        val_loader   = _mk(opt, split="val")
    else:
        from utils.dataset import create_dataloader as _mk
        train_loader = _mk(opt, split="train")
        val_loader   = _mk(opt, split="val")

    # Model selection
    if opt.ft and opt.social:
        from utils.finetuning import FTModel
        from utils import EarlyStopping
        save_dir = os.path.join("checkpoint", opt.name, "social", opt.social , "ft_weights", dataset_name)
        
        model = FTModel(opt)
    
    elif opt.ft:
        from utils.finetuning import FTModel
        from utils import EarlyStopping
        save_dir = os.path.join("checkpoint", opt.name, "ft_weights", dataset_name)
        
        model = FTModel(opt)
    else:
        from utils.training import TrainingModel
        from utils import EarlyStopping
        save_dir = os.path.join("checkpoint", opt.name, "weights", dataset_name)
        
        model = TrainingModel(opt)
    
    os.makedirs(save_dir,  exist_ok=True)

    save_log = os.path.join(save_dir, "logs")
    
    logger   = create_logger(os.path.join(save_dir, 'train.log'))

    logger.info(f"Training the {opt.name} model on the {dataset_name} dataset - FT: {opt.ft} - unfreezeL4: {opt.r50unfreezeL4}")
    logger.info(f"Training settings: {json.dumps(vars(opt), indent=2, default=str)}")
    
    logger.info(f"training batches   = {len(train_loader)}")
    logger.info(f"validation batches = {len(val_loader)}")
    
    # Resume training or fresh start
    early_stopping = None
    start_epoch    = 0
 
    if opt.resume:
        resumed_state = model.load_for_resume()
        start_epoch = resumed_state.get("epoch", 0)
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
            model.save_networks(epoch, early_stopping=early_stopping)
 
        # Validation
        y_true, y_pred, _ = model.predict(val_loader)
        val_acc = balanced_accuracy_score(y_true, y_pred > 0.0)
        val_auc = roc_auc_score(y_true, y_pred)
        lr      = model.get_learning_rate()

        # Early stopping + checkpoint decision 
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
 
        # Per-epoch log line
        logger.info(
            f"{epoch:>6}  {epoch_loss:>10.4f}  {val_acc:>8.4f}  "
            f"{val_auc:>8.4f}  {lr:>10.2e}  {str(saved):>5}  {note}"
        )
 
    # Run summary
    elapsed = time.time() - run_start
    best    = early_stopping.best_score if early_stopping else float("nan")
    logger.info("=" * 70)
    logger.info(
        f"RUN END  elapsed={elapsed:.0f}s  best_val_acc={best:.4f}"
    )
    logger.info(f"Checkpoint dir : {model.save_dir}")
    logger.info(f"Log file       : {os.path.abspath(os.path.join(model.save_dir, 'train.log'))}")
    logger.info("=" * 70)

    
    removed = 0
    for fname in os.listdir(model.save_dir):
        name, ext = os.path.splitext(fname)
        if ext == '.pt' and name.isdigit():
            os.remove(os.path.join(model.save_dir, fname))
            removed += 1
    print(f'Removed {removed} intermediate epoch checkpoint(s), kept best.pt', flush=True)
    logger.info(f'Removed {removed} intermediate epoch checkpoint(s), kept best.pt')
    logger.info("=" * 70)
