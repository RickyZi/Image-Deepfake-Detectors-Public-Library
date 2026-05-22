import os
import torch
from torch.utils.data import random_split
import json


def safe_split(files, ratios=(0.8, 0.2), generator=None):
    n = len(files)
    print(f"safe split - len(files): {n}")
    if n < len(ratios):
        raise ValueError(f"Dataset too small ({n} files) to split into {len(ratios)} parts")

    sizes = [int(n * r) for r in ratios]
    remainder = n - sum(sizes)
    fractions = [(n * r) % 1 for r in ratios]

    for i in sorted(range(len(ratios)), key=lambda i: fractions[i], reverse=True)[:remainder]:
        sizes[i] += 1

    for i in range(len(sizes)):
        if sizes[i] == 0:
            largest = max(range(len(sizes)), key=lambda j: sizes[j])
            sizes[largest] -= 1
            sizes[i] = 1

    assert sum(sizes) == n, "Split sizes don't sum to n"
    return random_split(files, sizes, generator=generator)


# ------------------------------------------------------------------ #

# Hardcoded: only these 4 ids from PreSocial go into train/val
# Everything else in the dataset goes into test
TRAIN_VAL_IDS = {'StyleGAN2/conf-f-psi-1', 'StableDiffusionXL/general', 'FORLAB', 'FFHQ'}
TRAIN_VAL_MOD = 'PreSocial'

dataset_path = './demo_images/season_TM01/'

train_val_data = []
test_data      = []

for dataset_root, dataset_dirs, dataset_files in os.walk(dataset_path, topdown=True, followlinks=True):
    if not dataset_files:
        continue

    rel   = os.path.relpath(dataset_root, dataset_path)
    parts = rel.split(os.sep)
    if len(parts) < 3:
        continue

    mod, label, gen = parts[0], parts[1], parts[2]
    sub = parts[3] if len(parts) > 3 else None
    id_ = os.path.join(gen, sub) if sub else gen

    files = sorted([
        os.path.splitext(f)[0] for f in dataset_files
        if os.path.splitext(f)[1].lower() in ('.jpg', '.png', '.jpeg')
    ])
    if not files:
        continue

    entry = {'id': id_, 'mod': mod, 'label': label, 'root': dataset_root, 'files': files}

    if mod == TRAIN_VAL_MOD and id_ in TRAIN_VAL_IDS:
        print(f"  [train/val] {mod:10s} | {id_:40s}")
        train_val_data.append(entry)
    else:
        test_data.append(entry)

print(f"\nTotal train_val groups : {len(train_val_data)}")   # expected 4
print(f"Total test groups      : {len(test_data)}\n")        # expected 36

print("train_val ids:")
for d in train_val_data:
    print(f" id_: {d['id']}:{d['mod']}")
print("test ids:")
for d in test_data:
    print(f" id_: {d['id']}:{d['mod']}")

# ------------------------------------------------------------------ #
# Build splits
#   train / val  →  80 / 20  random split of the 4 PreSocial groups
#   test         →  all remaining samples (no random split)
# ------------------------------------------------------------------ #

train_set, val_set = [], []
generator = torch.Generator().manual_seed(42)

for dataset in train_val_data:
    trn, val = safe_split(dataset['files'], ratios=(0.8, 0.2), generator=generator)
    train_set += [os.path.join(dataset['id'], f) for f in trn]
    val_set   += [os.path.join(dataset['id'], f) for f in val]

test_set = [os.path.join(d['id'], f) for d in test_data for f in d['files']]

train_set = sorted(train_set)
val_set   = sorted(val_set)
test_set  = sorted(test_set)

print(f"\ntrain={len(train_set)}, val={len(val_set)}, test={len(test_set)}, "
      f"total={len(train_set) + len(val_set) + len(test_set)}")

# Sanity check: no unrenamed '0p' folders
bad = [k for k in train_set + val_set + test_set if '0p' in k]
if bad:
    print(f"\n[WARNING] {len(bad)} entries still contain '0p' — "
          f"run fix_folder_names.py --apply first!\n  e.g. {bad[0]}")
else:
    print("[OK] No '0p' naming issues detected.")

# Save split in json file
output_file = "split_prova_hc.json"
with open(output_file, "w") as f:
    json.dump({'train': train_set, 'val': val_set, 'test': test_set}, f, indent=2)

print(f"\nWrote {output_file}")