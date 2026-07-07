import os
from pathlib import Path
import torch
from torch.utils.data import random_split
import json

# ------------------------------------------------------------------ #
# extract rand split (60-20-20) from each subfolder and save to json #
# ------------------------------------------------------------------ #

# dataset_path = 'truefake_2k/dataset/' # TF_preSocial
# dataset_path = 'truefake_2k/tf2k_social/Facebook' # TF_social#
dataset_path = '/dataset-disk/tb_dataset/tf2k_lr/social/facebook/seasons/autumn-TM01'

SEED = 42 # fix seed for reproducibility

datasets = []

generator = torch.Generator().manual_seed(SEED)

for dataset_root, dataset_dirs, dataset_files in os.walk(Path(dataset_path)):  

    # if dataset_files:
    #     print(f"Found directory: {dataset_root} with files: {dataset_files}")


    if not dataset_files:
        continue

    # ignore json files in the dataset path
    if any(f.lower().endswith('.json') for f in dataset_files):
        continue

    # Path relative to dataset root, e.g.:
    #   Facebook/Real/FFHQ                     (Real, no sub)
    #   Facebook/Fake/StyleGAN/images-psi-0.5  (Fake, with sub)
    rel = os.path.relpath(dataset_root, dataset_path)
    parts = rel.split(os.sep)

    # print(f"parts: {parts}")

    # print(f"Found directory: {rel} with files: {len(dataset_files)}")

    if len(parts) < 2:
        continue
    
    # ----------------------------------------- #
    # mod : {Presocial, Facebook, Telegram, X}
    # label: Real of Fake
    # gen: FFHQ, StyleGAN, etc.
    # sub: e.g. images-psi-0.5 (only for Fake)
    # ----------------------------------------- #

    # mod = 'pre' # fixed for tf2k
    # mod = 
    if 'Facebook' in dataset_path:
        mod = 'fb'
    elif 'Telegram' in dataset_path:
        mod = 'tl'
    elif 'Twitter' in dataset_path:
        mod = 'tw'
    else:
        mod = 'pre'
    label, gen = parts[:2]
    sub = parts[2] if len(parts) > 2 else None
    group_id = os.path.join(gen, sub) if sub else gen # for fake subset

    images = sorted([
        os.path.splitext(f)[0] for f in dataset_files
        if os.path.splitext(f)[1].lower() in {'.jpg', '.png', '.jpeg'}
    ])
    # images are sorted by name and img estension removed
    if not images:
        continue
        
    # print(f"{mod} - {label} - {group_id} ")

    # print(f" {label:5s} | {group_id:45s} | {len(images):3d} files")

    # random_split needs a Sized object — a list works fine
    train_sub, val_sub, test_sub = random_split(images, [0.6, 0.2, 0.2], generator=generator)

    # Subset.__getitem__ indexes into the original list, so this recovers filenames
    train = [images[i] for i in train_sub.indices]
    val   = [images[i] for i in val_sub.indices]
    test  = [images[i] for i in test_sub.indices]

    # print(f"  {label:4s} | {group_id:40s} | {len(train):2d} / {len(val):2d} / {len(test):2d} | {len(train) + len(val) + len(test):3d} files")
    print(f"{mod:<10} | {label:<10} | {group_id:<40} | {len(train):2d} / {len(val):2d} / {len(test):2d} | {len(train) + len(val) + len(test):3d} files")

    datasets.append({
        'id':    group_id, 
        'mod':   mod, # fixed on tf2k
        'label':  label,
        'sub':   sub,
        'root':  dataset_root,
        'train': train,
        'val':   val,
        'test':  test,
    })

print()
for split in ('train', 'val', 'test'):
    total = sum(len(d[split]) for d in datasets)
    print(f"Total {split}: {total}")

print(f"\nTotal dataset groups: {len(datasets)}")
print(f"total datasets: {sum(len(d['train']) + len(d['val']) + len(d['test']) for d in datasets)}")

# concatenate splits across all datasets and write to json file
train_set, val_set, test_set = [], [], []
for dataset in datasets:
    train_set += [os.path.join(dataset['id'], f) for f in dataset['train']]
    val_set   += [os.path.join(dataset['id'], f) for f in dataset['val']]
    test_set  += [os.path.join(dataset['id'], f) for f in dataset['test']]

train_set = sorted(train_set)
val_set   = sorted(val_set)
test_set  = sorted(test_set)

output_file = 'tf2k_SOCIAL_splits_NEW.json'
#'tf2k_dataset_splits.json'

with open(output_file, "w") as f:
    json.dump({'train': train_set, 'val': val_set, 'test': test_set}, f, indent=2)

print(f"\nWrote {output_file}")

# --------------------------------------------------------------------------------------------------- #
# ------------------------------------------ SPLIT RESULTS ------------------------------------------ #  
# pre        | Real       | FFHQ                                     | 300 / 100 / 100 | 500 files
# pre        | Real       | FORLAB                                   | 300 / 100 / 100 | 500 files
# pre        | Fake       | StableDiffusionXL/general                | 23 /  8 /  7 |  38 files
# pre        | Fake       | StableDiffusionXL/landscapes             | 23 /  8 /  7 |  38 files
# pre        | Fake       | StableDiffusionXL/animals                | 23 /  8 /  7 |  38 files
# pre        | Fake       | StableDiffusionXL/faces                  | 23 /  8 /  7 |  38 files
# pre        | Fake       | StyleGAN3/conf-t-psi-0.5                 | 23 /  8 /  7 |  38 files
# pre        | Fake       | StyleGAN3/conf-t-psi-0.7                 | 23 /  8 /  7 |  38 files
# pre        | Fake       | StableDiffusion3/general                 | 23 /  8 /  7 |  38 files
# pre        | Fake       | StableDiffusion3/landscapes              | 23 /  8 /  7 |  38 files
# pre        | Fake       | StableDiffusion3/animals                 | 23 /  8 /  7 |  38 files
# pre        | Fake       | StableDiffusion3/faces                   | 23 /  8 /  7 |  38 files
# pre        | Fake       | StyleGAN2/conf-f-psi-1                   | 23 /  8 /  7 |  38 files
# pre        | Fake       | StyleGAN2/conf-f-psi-0.5                 | 23 /  8 /  7 |  38 files
# pre        | Fake       | StableDiffusion2/general                 | 24 /  8 /  7 |  39 files
# pre        | Fake       | StableDiffusion2/landscapes              | 24 /  8 /  7 |  39 files
# pre        | Fake       | StableDiffusion2/animals                 | 24 /  8 /  7 |  39 files
# pre        | Fake       | StableDiffusion2/faces                   | 24 /  8 /  7 |  39 files
# pre        | Fake       | StableDiffusion1.5/general               | 24 /  8 /  7 |  39 files
# pre        | Fake       | StableDiffusion1.5/landscapes            | 24 /  8 /  7 |  39 files
# pre        | Fake       | StableDiffusion1.5/animals               | 24 /  8 /  7 |  39 files
# pre        | Fake       | StableDiffusion1.5/faces                 | 24 /  8 /  7 |  39 files
# pre        | Fake       | FLUX.1/general                           | 24 /  8 /  7 |  39 files
# pre        | Fake       | FLUX.1/landscapes                        | 24 /  8 /  7 |  39 files
# pre        | Fake       | FLUX.1/animals                           | 24 /  8 /  7 |  39 files
# pre        | Fake       | FLUX.1/faces                             | 24 /  8 /  7 |  39 files
# pre        | Fake       | StyleGAN/images-psi-0.7                  | 23 /  8 /  7 |  38 files
# pre        | Fake       | StyleGAN/images-psi-0.5                  | 23 /  8 /  7 |  38 files
# Total train: 1210
# Total val: 408
# Total test: 382

# Total dataset groups: 28
# total datasets: 2000

# Wrote test2k_splits.json
# --------------------------------------------------------------------------------------------------- #
# SOCIAL SPLIT #
# ------------ #
# - FACEBOOK - #
# fb         | Real       | FFHQ                                     | 300 / 100 / 100 | 500 files
# fb         | Real       | FORLAB                                   | 300 / 100 / 100 | 500 files
# fb         | Fake       | StableDiffusionXL/general                | 23 /  8 /  7 |  38 files
# fb         | Fake       | StableDiffusionXL/landscapes             | 23 /  8 /  7 |  38 files
# fb         | Fake       | StableDiffusionXL/animals                | 23 /  8 /  7 |  38 files
# fb         | Fake       | StableDiffusionXL/faces                  | 23 /  8 /  7 |  38 files
# fb         | Fake       | StyleGAN3/conf-t-psi-0.5                 | 23 /  8 /  7 |  38 files
# fb         | Fake       | StyleGAN3/conf-t-psi-0.7                 | 23 /  8 /  7 |  38 files
# fb         | Fake       | StableDiffusion3/general                 | 23 /  8 /  7 |  38 files
# fb         | Fake       | StableDiffusion3/landscapes              | 23 /  8 /  7 |  38 files
# fb         | Fake       | StableDiffusion3/animals                 | 23 /  8 /  7 |  38 files
# fb         | Fake       | StableDiffusion3/faces                   | 23 /  8 /  7 |  38 files
# fb         | Fake       | StyleGAN2/conf-f-psi-1                   | 23 /  8 /  7 |  38 files
# fb         | Fake       | StyleGAN2/conf-f-psi-0.5                 | 23 /  8 /  7 |  38 files
# fb         | Fake       | StableDiffusion2/general                 | 24 /  8 /  7 |  39 files
# fb         | Fake       | StableDiffusion2/landscapes              | 24 /  8 /  7 |  39 files
# fb         | Fake       | StableDiffusion2/animals                 | 24 /  8 /  7 |  39 files
# fb         | Fake       | StableDiffusion2/faces                   | 24 /  8 /  7 |  39 files
# fb         | Fake       | StableDiffusion1.5/general               | 24 /  8 /  7 |  39 files
# fb         | Fake       | StableDiffusion1.5/landscapes            | 24 /  8 /  7 |  39 files
# fb         | Fake       | StableDiffusion1.5/animals               | 24 /  8 /  7 |  39 files
# fb         | Fake       | StableDiffusion1.5/faces                 | 24 /  8 /  7 |  39 files
# fb         | Fake       | FLUX.1/general                           | 24 /  8 /  7 |  39 files
# fb         | Fake       | FLUX.1/landscapes                        | 24 /  8 /  7 |  39 files
# fb         | Fake       | FLUX.1/animals                           | 24 /  8 /  7 |  39 files
# fb         | Fake       | FLUX.1/faces                             | 24 /  8 /  7 |  39 files
# fb         | Fake       | StyleGAN/images-psi-0.7                  | 23 /  8 /  7 |  38 files
# fb         | Fake       | StyleGAN/images-psi-0.5                  | 23 /  8 /  7 |  38 files

# Total train: 1210
# Total val: 408
# Total test: 382

# Total dataset groups: 28
# total datasets: 2000

# Wrote tf2k_SOCIAL_splits.json