import os
import torch
from torch.utils.data import random_split
import json


def safe_split(files, ratios=(0.7, 0.15, 0.15), generator=None):
    # safe_split actually does a 60-20-20
    n = len(files)
    print(f"len(files): {n}")
    if n < len(ratios):
        raise ValueError(f"Dataset too small ({n} files) to split into {len(ratios)} parts")

    sizes = [int(n * r) for r in ratios]
    remainder = n - sum(sizes)
    fractions = [(n * r) % 1 for r in ratios]

    print(f"sizes: {sizes}, remainder: {remainder}, fractions: {fractions}")

    for i in sorted(range(len(ratios)), key=lambda i: fractions[i], reverse=True)[:remainder]:
        sizes[i] += 1

    for i in range(len(sizes)):
        if sizes[i] == 0:
            largest = max(range(len(sizes)), key=lambda j: sizes[j])
            sizes[largest] -= 1
            sizes[i] = 1

    assert sum(sizes) == n, "Split sizes don't sum to n"
    # breakpoint()
    return random_split(files, sizes, generator=generator)


# dataset_path = '../truefake_2k/dataset/' 
dataset_path = './demo_images/demo_images/'


# ------------------------------------------------------------------ #
# IMPORTANT: run fix_folder_names.py --apply BEFORE this script.     #
# The split keys are read directly from disk folder names, so the    #
# folder names must already be correct (e.g. images-psi-0.5 not 0p5)#
# ------------------------------------------------------------------ #

datasets = []
for dataset_root, dataset_dirs, dataset_files in os.walk(dataset_path, topdown=True, followlinks=True): 
    if not dataset_files:
        continue

    # # ignore json files in the dataset path
    # if any(f.lower().endswith('.json') for f in dataset_files):
    #     continue

    # Path relative to dataset root, e.g.:
    #   Facebook/Real/FFHQ                     (Real, no sub)
    #   Facebook/Fake/StyleGAN/images-psi-0.5  (Fake, with sub)
    rel = os.path.relpath(dataset_root, dataset_path)
    parts = rel.split(os.sep)

    if len(parts) < 3:
        continue
    

    # ----------------------------------------- #
    # mod : {Presocial, Faceboo, Telegram, X}
    # label: Real of Fake
    # gen: FFHQ, StyleGAN, etc.
    # sub: e.g. images-psi-0.5 (only for Fake)
    # ----------------------------------------- #

    mod, label, gen = parts[0], parts[1], parts[2]  
    sub = parts[3] if len(parts) > 3 else None

    # Key prefix written into the split file — must match what dataset.py constructs:
    #   gen/sub  for Fake  e.g. "StyleGAN/images-psi-0.5"
    #   gen      for Real  e.g. "FFHQ"
    id = os.path.join(gen, sub) if sub else gen

    files = sorted([
        os.path.splitext(f)[0] for f in dataset_files
        if os.path.splitext(f)[1].lower() in ['.jpg', '.png', '.jpeg']
    ])
    if not files:
        continue

    print(f"  {label:4s} | {mod:10s} | {id:40s} | {len(files)} files")
    datasets.append({'id': id, 'mod': mod, 'label': label, 'root': dataset_root, 'files': files})

print(f"\nTotal dataset groups: {len(datasets)}\n")

train_set, val_set, test_set = [], [], []
generator = torch.Generator().manual_seed(42)

for dataset in datasets:
    files = dataset['files']
    print(f"files: {files}")
    trn, val, tst = safe_split(files, generator=generator)
    # trn, val, tst = random_split(files, [0.7, 0.15, 0.15], generator=generator)
    train_set += [os.path.join(dataset['id'], f) for f in trn]
    val_set   += [os.path.join(dataset['id'], f) for f in val]
    test_set  += [os.path.join(dataset['id'], f) for f in tst]

# Deduplicate while preserving determinism -> removed set
# note: imgs have the same rel_path but are extacted from different subfoders
train_set = sorted(train_set)
val_set   = sorted(val_set)
test_set  = sorted(test_set)

print(f"train={len(train_set)}, val={len(val_set)}, test={len(test_set)}, "
      f"total={len(train_set) + len(val_set) + len(test_set)}")

breakpoint()

# Sanity check: flag any 0p5 remnants that slipped through
bad = [k for k in train_set + val_set + test_set if '0p' in k]
if bad:
    print(f"\n[WARNING] {len(bad)} entries still contain '0p' — "
          f"run fix_folder_names.py --apply first!\n  e.g. {bad[0]}")
else:
    print("[OK] No '0p' naming issues detected in split keys.")

# output_file = "split_tf2k_TM01.json"
# with open(output_file, "w") as f:
#     json.dump({'train': train_set, 'val': val_set, 'test': test_set}, f, indent=2)

# print(f"\nWrote {output_file}")