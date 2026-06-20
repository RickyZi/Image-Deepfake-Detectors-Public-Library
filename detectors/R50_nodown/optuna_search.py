"""
optuna_search.py
----------------
Hyperparameter search for R50_nodown fine-tuning using Optuna.

Supports two FT modes:
    --mode standard      Single-stage BCE fine-tune (FTModel)
    --mode contrastive   Two-stage SupCon + BCE (ContrastiveFTModel)

The script imports your existing pipeline modules, so no changes to
train.py / finetuning.py are needed. Each Optuna trial = one full FT run
with sampled hyperparameters; the trial returns the best validation
balanced accuracy seen during training.

Usage:
    python optuna_search.py \
        --mode standard \
        --n_trials 40 \
        --study_name R50_cinematic_CN01 \
        --pretrained_ckpt ./checkpoint/pretrained/weights/best.pt \
        --data_root /path/to/dataset \
        --split_file /path/to/test2k_splits.json

Results:
    - SQLite study DB: optuna_studies/<study_name>.db
    - Best params JSON: optuna_studies/<study_name>_best.json
    - Per-trial log CSV: optuna_studies/<study_name>_trials.csv


R50_nd args
{
    "name": "pretrained",
    "arch": "res50nodown",
    "task": "train",
    "device": "cuda:0",
    "split_file": "/home/rz/TB_WP3/Image-Deepfake-Detectors-Public-Library/test2k_splits.json",
    "data_root": "/home/rz/TB_WP3/Image-Deepfake-Detectors-Public-Library/truefake_2k/tf2k_lr_org/seasons/autumn_TM01",
    "data_keys": "realFFHQ:pre&realFORLAB:pre&gan1:pre&gan2:pre&gan3:pre&sd15:pre&sd2:pre&sd3:pre&sdXL:pre&flux:pre",
    "batch_size": 64,
    "num_threads": 4,
    "lr": 0.0001,
    "weight_decay": 0.0,
    "beta1": 0.9,
    "num_epoches": 1000,
    "earlystop_epoch": 5,
    "ft": true,
    "tf2k": true,
    "dataset": "seasons/autumn_TM01",
    "r50unfreezeL4": false,
    "cropSize": 96,
    "resize_prob": 0.2,
    "jitter_prob": 0.8,
    "colordist_prob": 0.2,
    "cutout_prob": 0.2,
    "noise_prob": 0.2,
    "blur_prob": 0.5,
    "cmp_prob": 0.5,
    "blur_sig": "0.1,3.0",
    "cmp_qual": "30,100",
    "resize_size": 256,
    "resize_ratio": 0.75,
    "norm_type": "resnet"
}

"""

import os
import sys
import json
import copy
import argparse
import warnings
from types import SimpleNamespace

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

import torch
import tqdm
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

# Project modules (assumes script is run from the detector folder)
from utils.finetuning import FTModel
# from utils.contrastive_finetuning import ContrastiveFTModel
from utils.tf2k_dataset import TrueFake_dataset
from utils import EarlyStopping


# ------------------------------------------------------------------ #
# Argument defaults — match the shapes parser.py + processing.py     #
# expect, so FTModel / dataloaders work without modification.        #
# ------------------------------------------------------------------ #

DEFAULT_DATA_KEYS = ('realFFHQ:pre&realFORLAB:pre&gan1:pre&gan2:pre&gan3:pre&sd15:pre&sd2:pre&sd3:pre&sdXL:pre&flux:pre')

def base_opt(args, trial_name: str) -> SimpleNamespace:
    """
    Build a SimpleNamespace that mirrors what argparse would produce.
    The Optuna objective overrides specific fields per trial.
    """

    # NEW: derive dataset identifier from data_root
    norm_root = os.path.normpath(args.data_root)
    parts = norm_root.split(os.sep)
    dataset_id = ('_'.join(parts[-2:]) if len(parts) >= 2 else os.path.basename(norm_root))

    print(f"dataset_id: {dataset_id}")

    return SimpleNamespace(
        # identity / IO
        name        = trial_name,
        arch        = 'res50nodown',
        task        = 'train',
        split       = 'train',
        device      = args.device,
        split_file  = args.split_file,
        data_root   = args.data_root,
        data_keys   = args.data_keys,

        # dataloader
        batch_size  = args.batch_size,
        num_threads = args.num_threads,

        # optimizer (overridden per trial)
        lr             = 1e-4,
        weight_decay   = 0.0,
        beta1          = 0.9,
        num_epoches    = args.max_epochs,
        earlystop_epoch = args.patience,

        ft          = True,
        tf2k        = True,
        dataset     = dataset_id,
        r50unfreezeL4 = False, # check only FT version for now
        
        # processing / augmentation (overridden per trial)
        norm_type      = 'resnet',
        cropSize       = 96,
        resize_size    = 256,
        resize_ratio   = 0.75,
        resize_prob    = 0.2,
        jitter_prob    = 0.8,
        colordist_prob = 0.2,
        cutout_prob    = 0.2,
        noise_prob     = 0.2,
        blur_prob      = 0.5,
        blur_sig       = [0.1, 3.0],
        cmp_prob       = 0.5,
        cmp_qual       = [30, 100],
        
    )


# ------------------------------------------------------------------ #
# Per-trial training loops — return best val balanced accuracy.       #
# Calls trial.report() + trial.should_prune() each epoch for pruning. #
# ------------------------------------------------------------------ #

def train_standard(opt, trial, pretrained_ckpt: str, train_loader, valid_loader) -> float:
    model = FTModel(opt)
    model.load_networks(pretrained_ckpt)
    model.freeze_backbone(opt.r50unfreezeL4)
    # d

    best_acc       = 0.0
    early_stopping = None

    for epoch in range(1, opt.num_epoches + 1):
        # --- train one epoch ---
        pbar = tqdm.tqdm(train_loader, leave=False, desc=f"[T{trial.number}] ep{epoch}")
        for data in pbar:
            loss = model.train_on_batch(data).item()
            pbar.set_description(f"[T{trial.number}] ep{epoch} loss={loss:.4f}")

        # --- validate ---
        y_true, y_pred, _ = model.predict(valid_loader)
        acc = balanced_accuracy_score(y_true, y_pred > 0.0)
        best_acc = max(best_acc, acc)

        # --- pruning ---
        trial.report(acc, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # --- early stopping ---
        if early_stopping is None:
            early_stopping = EarlyStopping(init_score=acc, patience=opt.earlystop_epoch, delta=0.001, verbose=False)
        else:
            early_stopping(acc)
            if early_stopping.early_stop:
                if not model.adjust_learning_rate():
                    break
                early_stopping.reset_counter()

    return best_acc


def train_contrastive(opt, trial, pretrained_ckpt: str,
                      train_loader, valid_loader) -> float:
    """
    Two-stage: SupCon pretraining → BCE fine-tune.
    We sum the contrastive_epochs and num_epoches as the pruning step axis.
    """
    model = ContrastiveFTModel(opt)
    model.load_networks(pretrained_ckpt)

    # ---------------- Stage 1: contrastive ---------------- #
    model._setup_stage1()
    s1_best = 0.0
    s1_count_down = opt.earlystop_epoch
    for epoch in range(1, opt.contrastive_epochs + 1):
        pbar = tqdm.tqdm(train_loader, leave=False,
                        desc=f"[T{trial.number}] S1 ep{epoch}")
        for data in pbar:
            loss = model.train_contrastive_batch(data).item()
            pbar.set_description(
                f"[T{trial.number}] S1 ep{epoch} loss={loss:.4f}")

        acc, _ = model._validate_contrastive(valid_loader)
        s1_best = max(s1_best, acc)
        trial.report(acc, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if acc > s1_best - 0.001:
            s1_count_down = opt.earlystop_epoch
        else:
            s1_count_down -= 1
            if s1_count_down <= 0:
                if not model.adjust_learning_rate():
                    break
                s1_count_down = opt.earlystop_epoch

    # ---------------- Stage 2: BCE ---------------- #
    model._setup_stage2(unfreeze_layer4=opt.unfreeze_layer4_stage2)
    s2_best = 0.0
    s2_count_down = opt.earlystop_epoch
    for epoch in range(1, opt.num_epoches + 1):
        pbar = tqdm.tqdm(train_loader, leave=False,
                        desc=f"[T{trial.number}] S2 ep{epoch}")
        for data in pbar:
            loss = model.train_bce_batch(data).item()
            pbar.set_description(
                f"[T{trial.number}] S2 ep{epoch} loss={loss:.4f}")

        acc, _ = model._validate_bce(valid_loader)
        s2_best = max(s2_best, acc)
        # Continue the pruning step axis from Stage 1
        trial.report(acc, step=opt.contrastive_epochs + epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if acc > s2_best - 0.001:
            s2_count_down = opt.earlystop_epoch
        else:
            s2_count_down -= 1
            if s2_count_down <= 0:
                if not model.adjust_learning_rate():
                    break
                s2_count_down = opt.earlystop_epoch

    # We optimize the final BCE-stage accuracy (that's what test uses)
    return s2_best


# ------------------------------------------------------------------ #
# Crop-size → batch-size mapping (Tesla T4, 16 GB VRAM)              #
# Keeps total pixel throughput per step roughly constant so that      #
# larger crops don't OOM while smaller crops still get good           #
# utilisation. Values leave ~30% memory headroom in FP32.             #
# ------------------------------------------------------------------ #

CROP_TO_BATCH = {
    64:  128,
    96:   64,
    128:  48,
    160:  32,
}



def tf2k_create_dataloader(opt, split=None):
    if split == "train":
        opt.split = 'train'
        is_train=True

    elif split == "val":
        opt.split = 'val'
        is_train=False
    
    elif split == "test":
        opt.split = 'test'
        opt.batch_size = 2
        is_train=False
    
    else:
        raise ValueError(f"Unknown split {split}")

    dataset = TrueFake_dataset(opt)
    # print(f"dataset: {dataset}")
    # breakpoint()
    # data_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=opt.batch_size,
    #     shuffle=is_train,
    #     num_workers=int(opt.num_threads),
    # )

    # ------------------------------------ #
    # add persistent workers to dataloader #
    # ------------------------------------ #
    num_workers = int(opt.num_threads)
    # persistent_workers requires num_workers > 0; pin_memory speeds host->GPU
    # transfers and is a free win when a CUDA device is in use.
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=torch.cuda.is_available()
    )

    return data_loader


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    # study
    parser.add_argument('--study_name', required=True)
    parser.add_argument('--n_trials',   type=int, default=40)
    parser.add_argument('--mode',       choices=['standard', 'contrastive'], default='standard')

    # paths
    parser.add_argument('--data_root',       required=True)
    parser.add_argument('--split_file',      required=True)
    parser.add_argument('--pretrained_ckpt', required=True)
    parser.add_argument('--data_keys',       default=DEFAULT_DATA_KEYS)

    # training budget
    parser.add_argument('--max_epochs', type=int, default=30, help='Max BCE epochs per trial (early stopping cuts shorter)')
    parser.add_argument('--contrastive_epochs', type=int, default=30, help='Max Stage-1 epochs in contrastive mode')
    parser.add_argument('--patience', type=int, default=10)

    # hardware / loader
    parser.add_argument('--device',      default='cuda:0')
    parser.add_argument('--batch_size',  type=int, default=64)
    parser.add_argument('--num_threads', type=int, default=4)

    # pruner
    parser.add_argument('--pruner_warmup',  type=int, default=5, help='Epochs before pruner can kill a trial')
    parser.add_argument('--pruner_startup', type=int, default=5, help='Trials before pruner activates')

    args = parser.parse_args()

    # --- prepare output directory ---
    out_dir = 'optuna_studies'
    os.makedirs(out_dir, exist_ok=True)
    db_path        = os.path.join(out_dir, f'{args.study_name}.db')
    best_path      = os.path.join(out_dir, f'{args.study_name}_best.json')
    trials_csv     = os.path.join(out_dir, f'{args.study_name}_trials.csv')

    # --- build dataloaders ONCE (shared across all trials) ---
    # Use the most permissive opt for dataset construction; the per-trial opt
    # mutates augmentation params, which only affect the transform pipeline.
    # We need a fresh dataloader per trial because the transform depends on
    # opt.cropSize etc., so we'll actually rebuild it inside the objective.
    # To avoid repeated dataset walks (slow), wrap loader creation in a cache.
    loader_cache = {}

    def get_loaders(opt):
        # Cache key = augmentation knobs that affect transform pipeline.
        key = (opt.cropSize, opt.jitter_prob, opt.blur_prob, opt.cmp_prob,
               opt.resize_prob, opt.colordist_prob, opt.cutout_prob, opt.noise_prob)
        if key not in loader_cache:
            train_loader = tf2k_create_dataloader(opt, split='train')
            valid_loader = tf2k_create_dataloader(opt, split='val')
            loader_cache[key] = (train_loader, valid_loader)
        return loader_cache[key]

    # We can't share loaders directly because tf2k_create_dataloader mutates opt.
    # Instead: build them inside the objective using a fresh opt copy.
    def objective(trial):
        opt = base_opt(args, trial_name=f"trial_{trial.number:03d}")

        # Sampled hyperparameters
        opt.lr           = trial.suggest_float('lr',           1e-5, 1e-2, log=True)
        opt.weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        opt.cropSize     = trial.suggest_categorical('cropSize', [64, 96, 128, 160])
        # Scale batch size with crop size to keep VRAM usage roughly constant
        # on the T4 (16 GB). Smaller crops -> larger batches, larger crops ->
        # smaller batches. Prevents OOM at cropSize=160 while keeping
        # throughput high at cropSize=64.
        opt.batch_size = CROP_TO_BATCH[opt.cropSize]
        opt.jitter_prob  = trial.suggest_float('jitter_prob', 0.0, 1.0)
        opt.blur_prob    = trial.suggest_float('blur_prob',   0.0, 1.0)
        opt.cmp_prob     = trial.suggest_float('cmp_prob',    0.0, 1.0)

        if args.mode == 'contrastive':
            opt.contrastive            = True
            opt.supcon_temperature     = trial.suggest_float(
                'supcon_temperature', 0.03, 0.5, log=True)
            opt.stage2_lr              = trial.suggest_float(
                'stage2_lr', 1e-6, 1e-3, log=True)
            opt.contrastive_epochs     = args.contrastive_epochs
            opt.unfreeze_layer4_stage2 = trial.suggest_categorical(
                'unfreeze_layer4_stage2', [False, True])
            
        print(f"opt: {opt}")
        # breakpoint()

        trial.set_user_attr('config', {k: v for k, v in vars(opt).items() if not k.startswith('_')})

        # Build per-trial loaders (transform depends on opt knobs)
        train_loader, valid_loader = get_loaders(opt)

        # breakpoint()

        try:
            if args.mode == 'standard':
                score = train_standard(opt, trial, args.pretrained_ckpt, train_loader, valid_loader)
            else:
                score = train_contrastive(opt, trial, args.pretrained_ckpt, train_loader, valid_loader)
            return score
        except optuna.TrialPruned:
            raise
        # except Exception as e:
        #     print(f"[T{trial.number}] FAILED: {type(e).__name__}: {e}")
        #     return 0.0

    # --- create / resume study ---
    sampler = TPESampler(seed=42, multivariate=True)
    pruner  = MedianPruner(
        n_startup_trials=args.pruner_startup,
        n_warmup_steps=args.pruner_warmup,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=f'sqlite:///{db_path}',
        load_if_exists=True,
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
    )

    print(f"\nStudy: {args.study_name}")
    print(f"Storage: {db_path}")
    print(f"Mode: {args.mode}")
    print(f"Existing trials: {len(study.trials)}")
    print(f"Running {args.n_trials} new trials...\n")

    study.optimize(objective, n_trials=args.n_trials,
                   show_progress_bar=False)

    # --- save best params ---
    print("\n" + "="*60)
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best value (val balanced acc): {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("="*60)

    with open(best_path, 'w') as f:
        json.dump({
            'study_name':  args.study_name,
            'mode':        args.mode,
            'B-ACC best_value':  study.best_value,
            'best_params': study.best_params,
            'best_trial':  study.best_trial.number,
            'n_trials':    len(study.trials),
        }, f, indent=2)
    print(f"Best params saved to: {best_path}")

    # --- save trial summary CSV ---
    df = study.trials_dataframe(
        attrs=('number', 'value', 'state', 'params', 'duration')
    )
    df.to_csv(trials_csv, index=False)
    print(f"Trial log saved to: {trials_csv}")


if __name__ == '__main__':
    main()