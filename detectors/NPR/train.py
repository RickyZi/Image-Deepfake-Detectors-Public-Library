import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
import numpy as np
from validate import validate
from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger, EarlyStopping, create_logger
from tqdm import tqdm
import random
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True

    return val_opt


def find_last_checkpoint(save_dir):
    """Look for the highest-numbered '{epoch}.pt' checkpoint in save_dir.
    Returns (epoch, path) or None if no per-epoch checkpoint exists."""
    if not os.path.isdir(save_dir):
        return None
    candidates = []
    for fname in os.listdir(save_dir):
        name, ext = os.path.splitext(fname)
        if ext == '.pt' and name.isdigit():
            candidates.append(int(name))
    if not candidates:
        return None
    last_epoch = max(candidates)
    return last_epoch, os.path.join(save_dir, f'{last_epoch}.pt')


if __name__ == '__main__':
    opt_train = TrainOptions().parse()
    seed_torch(100)

    print('  '.join(list(sys.argv)) )
    opt_val = get_val_opt()

    train_loader = create_dataloader(opt_train, split='train')
    val_loader = create_dataloader(opt_val, split='val')

    model = Trainer(opt_train)

    model.train()
    print(f'cwd: {os.getcwd()}')

    logger = create_logger(os.path.join(model.save_dir, 'train.log'))
    logger.info("=== Train settings ===")
    logger.info(f"Model name: {opt_train.name}")
    logger.info(f"Arch: {opt_train.arch}")
    logger.info(f"Fine-tune (--ft): {opt_train.ft}")
    logger.info(f"Dataset: {opt_train.dataset}")
    logger.info(f"Data keys: {opt_train.data_keys}")
    logger.info(f"Data root: {opt_train.data_root}")
    logger.info(f"Split file: {opt_train.split_file}")
    logger.info(f"lr: {opt_train.lr}, optim: {opt_train.optim}, batch_size: {opt_train.batch_size}")
    logger.info(f"earlystop_epoch: {opt_train.earlystop_epoch}, min_delta: {opt_train.min_delta}")
    logger.info(f"Trainable parameters: {model.trainable_params:,} / {model.total_params:,} "
                f"({100 * model.trainable_params / model.total_params:.2f}%)")
    logger.info(f"Save dir: {model.save_dir}")
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")

    start_epoch = 0
    early_stopping = None

    if opt_train.resume:
        found = find_last_checkpoint(model.save_dir)
        if found is not None:
            last_epoch, ckpt_path = found
            checkpoint = model.load_checkpoint(ckpt_path)

            es_best_score = checkpoint.get('early_stopping_best_score')
            es_count_down = checkpoint.get('early_stopping_count_down')
            if es_best_score is not None:
                early_stopping = EarlyStopping(
                    init_score=es_best_score,
                    patience=opt_train.earlystop_epoch,
                    delta=opt_train.min_delta,
                    verbose=True,
                )
                early_stopping.count_down = es_count_down if es_count_down is not None else opt_train.earlystop_epoch

            start_epoch = last_epoch + 1
            print(f'Resuming training from epoch {start_epoch} (loaded {ckpt_path})', flush=True)
            logger.info(f"Resumed from checkpoint {ckpt_path} - starting at epoch {start_epoch}, "
                        f"early_stopping best_score={getattr(early_stopping, 'best_score', None)}, "
                        f"count_down={getattr(early_stopping, 'count_down', None)}")
        else:
            print(f'--resume set but no checkpoint found in {model.save_dir} - starting fresh', flush=True)
            logger.info(f"--resume set but no checkpoint found in {model.save_dir} - starting fresh")

    for epoch in range(start_epoch, opt_train.niter):
        if epoch > 0:
            epoch_start_time = time.time()
            iter_data_time = time.time()
            epoch_iter = 0

            #for i, data in enumerate(train_loader):
            with tqdm(train_loader, unit='batch', mininterval=0.5) as tepoch:
                tepoch.set_description(f'Epoch {epoch}', refresh=False)
                for i, data in enumerate(tepoch):
                    model.total_steps += 1
                    epoch_iter += opt_train.batch_size

                    model.set_input(data)
                    model.optimize_parameters()
                    tepoch.set_postfix(loss=model.loss.item())

        # Validation
        model.eval()
        acc, ap = validate(model.model, val_loader)[:2]
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))
        logger.info(f"Epoch {epoch} - val acc: {acc:.4f}, val ap: {ap:.4f}")
        model.train()

        # Early stopping + best-model checkpoint decision. Replaces the
        # previous fixed-schedule LR decay (every --delr_freq epochs
        # regardless of validation performance) with a validation-triggered
        # one - appropriate for large from-scratch runs, not for a small
        # fine-tuning set where overfitting can set in within a handful of
        # epochs and a schedule that doesn't look at validation won't react.
        if early_stopping is None:
            early_stopping = EarlyStopping(
                init_score=acc,
                patience=opt_train.earlystop_epoch,
                delta=opt_train.min_delta,
                verbose=True,
            )
            print('Save best model', flush=True)
            model.save_networks('best')
            logger.info(f"Epoch {epoch} - Save best model - early stopping initialized "
                        f"with patience={opt_train.earlystop_epoch}, min_delta={opt_train.min_delta}")
        else:
            if early_stopping(acc):
                print('Save best model', flush=True)
                model.save_networks('best')
                logger.info(f"Epoch {epoch} - New best model saved with val acc={acc:.4f} - "
                            f"EarlyStopping count_down: {early_stopping.count_down} on {early_stopping.patience}")

        # Per-epoch resume checkpoint - saved every epoch, now that
        # early-stopping's decision for this epoch is final. Separate from
        # 'best.pt', which stays a bare state_dict.
        model.save_networks(epoch, extra={
            'early_stopping_best_score': early_stopping.best_score,
            'early_stopping_count_down': early_stopping.count_down,
        })
        logger.info(f"Epoch {epoch} - Saved resume checkpoint to {epoch}.pt")

        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped, continue training ...", flush=True)
                logger.info(f"Epoch {epoch} - Learning rate decayed to {model.optimizer.param_groups[0]['lr']:.2e} "
                            f"- reset early stopping counter and continue training")
                early_stopping.reset_counter()
            else:
                print("Early stopping.", flush=True)
                logger.info(f"Epoch {epoch} - Early stopping - learning rate already at minimum, "
                            f"no improvement for {opt_train.earlystop_epoch} epochs")
                break

    # Training finished (all epochs run, or early-stopped for good) - the
    # per-epoch checkpoints only exist to support --resume, so clean them
    # up now and keep just best.pt.
    removed = 0
    for fname in os.listdir(model.save_dir):
        name, ext = os.path.splitext(fname)
        if ext == '.pt' and name.isdigit():
            os.remove(os.path.join(model.save_dir, fname))
            removed += 1
    print(f'Removed {removed} intermediate epoch checkpoint(s), kept best.pt', flush=True)
    logger.info(f"Removed {removed} intermediate epoch checkpoint(s) from {model.save_dir} - kept best.pt")
    logger.info(f"Training completed. Best val acc: {early_stopping.best_score:.4f} - number of epochs run: {epoch+1}")