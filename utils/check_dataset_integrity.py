"""
check_dataset_integrity.py
---------------------------
Mirrors the exact path-resolution logic of TrueFake_dataset / tf2k_dataset.py
and tests every image that would be loaded by the DataLoader for a given split.

Usage:
    python check_dataset_integrity.py \
        --data_root  /path/to/dataset \
        --split_file /path/to/test2k_splits.json \
        --data_keys  "realFFHQ:pre&realFORLAB:pre&gan1:pre&..." \
        --split      train          # train | val | test
        --num_workers 0             # 0 = single process (safer for diagnosis)

What it checks per image, in order:
    1. os.path.exists()         — file present on disk
    2. os.path.getsize()        — not zero bytes
    3. Image.open().verify()    — PIL header parse (no pixel decode)
    4. Image.open().convert()   — full pixel decode (what the DataLoader does)
    5. tensor shape after transform

Any failure is printed with a [FAIL-N] tag so you can grep the output.
A summary table is printed at the end.
"""

import os
import sys
import json
import bisect
import argparse
import traceback
from collections import defaultdict

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ------------------------------------------------------------------ #
# Mirrors parse_tf2k_dataset() from tf2k_dataset.py exactly          #
# ------------------------------------------------------------------ #

GEN_KEYS = {
    'gan1':      ['StyleGAN'],
    'gan2':      ['StyleGAN2'],
    'gan3':      ['StyleGAN3'],
    'sd15':      ['StableDiffusion1.5'],
    'sd2':       ['StableDiffusion2'],
    'sd3':       ['StableDiffusion3'],
    'sdXL':      ['StableDiffusionXL'],
    'flux':      ['FLUX.1'],
    'realFFHQ':  ['FFHQ'],
    'realFORLAB':['FORLAB'],
}
GEN_KEYS['all']  = [v[0] for v in GEN_KEYS.values()]
GEN_KEYS['real'] = [v[0] for k, v in GEN_KEYS.items() if 'real' in k]


def parse_tf2k_dataset(data_keys, split):
    need_real = split in ('train', 'val') and not any(
        'real' in d.split(':')[0] for d in data_keys.split('&')
    )
    assert not need_real, 'Train/val data_keys contain no real data'

    dataset_list = []
    for entry in data_keys.split('&'):
        gen, _mod = entry.split(':')
        dataset_list.append({'gen': GEN_KEYS[gen]})
    return dataset_list


def _in_list(sorted_list, elem):
    i = bisect.bisect_left(sorted_list, elem)
    return i != len(sorted_list) and sorted_list[i] == elem


def collect_samples(data_root, split_list, dataset_list):
    """Reproduce TrueFake_dataset.__init__ sample collection."""
    samples = []   # (path, label, gen, sub)

    for d in dataset_list:
        generators = d['gen']
        for dataset_root, dataset_dirs, dataset_files in os.walk(
                data_root, topdown=True, followlinks=True):
            if dataset_dirs:
                continue

            rel   = os.path.relpath(dataset_root, data_root)
            parts = rel.split(os.sep)
            if len(parts) < 2:
                continue

            label, gen = parts[0], parts[1]
            sub = parts[2] if len(parts) > 2 else None

            if gen not in generators:
                continue

            for filename in sorted(dataset_files):
                ext = os.path.splitext(filename)[1].lower()
                if ext not in ('.png', '.jpg', '.jpeg'):
                    continue
                stem = os.path.splitext(filename)[0]
                key  = os.path.join(gen, sub, stem) if sub else os.path.join(gen, stem)

                if _in_list(split_list, key):
                    full_path = os.path.join(dataset_root, filename)
                    samples.append((full_path, label, gen, sub))

    return samples


# ------------------------------------------------------------------ #
# Per-image checks                                                    #
# ------------------------------------------------------------------ #

def check_image(path, transform=None):
    """
    Returns (ok: bool, stage: str, error: str).
    stage is the name of the check that failed, or 'ok'.
    """
    # 1. Existence
    if not os.path.exists(path):
        return False, 'exists', 'File not found on disk'

    # 2. Size
    size = os.path.getsize(path)
    if size == 0:
        return False, 'size', 'File is 0 bytes'

    # 3. PIL verify (header only)
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as e:
        return False, 'verify', str(e)

    # 4. Full decode (what DataLoader workers actually do)
    try:
        img = Image.open(path).convert('RGB')
    except Exception as e:
        return False, 'decode', str(e)

    # 5. Transform
    if transform is not None:
        try:
            tensor = transform(img)
            expected_shape = (3,)   # at least 3 channels
            if tensor.shape[0] != 3:
                return False, 'transform', f'Unexpected tensor shape: {tensor.shape}'
        except Exception as e:
            return False, 'transform', str(e)

    return True, 'ok', ''


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description='Dataset integrity checker')
    parser.add_argument('--data_root', default = 'truefake_2k/tf2k_lr_org/style/cinematic_CN01')
    parser.add_argument('--split_file', default = 'tf2k_SOCIAL_splits.json')
    parser.add_argument('--data_keys', default = 'realFFHQ:pre&realFORLAB:pre&gan1:pre&gan2:pre&gan3:pre&sd15:pre&sd2:pre&sd3:pre&sdXL:pre&flux:pre', help='e.g. "realFFHQ:pre&gan1:pre&flux:pre"')
    parser.add_argument('--split',       default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--with_transform', action='store_true', help='Also run the torchvision transform pipeline (requires processing.py to be importable)')
    args = parser.parse_args()

    # --- load split file ---
    print(f"\n{'='*60}")
    print(f"Split file : {args.split_file}")
    print(f"Data root  : {args.data_root}")
    print(f"Data keys  : {args.data_keys}")
    print(f"Split      : {args.split}")
    print(f"{'='*60}\n")

    with open(args.split_file) as f:
        split_data = json.load(f)
    split_list = sorted(split_data[args.split])
    print(f"Keys in split file [{args.split}]: {len(split_list)}")

    # --- print a few split keys for sanity ---
    print("First 5 split keys:")
    for k in split_list[:5]:
        print(f"  '{k}'")

    # --- collect samples ---
    dataset_list = parse_tf2k_dataset(args.data_keys, args.split)
    print(f"\nGenerator families to scan: "
          f"{[g for d in dataset_list for g in d['gen']]}")

    samples = collect_samples(args.data_root, split_list, dataset_list)
    print(f"\nSamples matched by dataset walker: {len(samples)}")
    print(f"Expected (from split file)       : {len(split_list)}")

    if len(samples) != len(split_list):
        print("\n[WARN] Mismatch between walker count and split file count.")
        # Show which split keys were NOT matched
        matched_keys = set()
        for full_path, label, gen, sub in samples:
            filename = os.path.basename(full_path)
            stem = os.path.splitext(filename)[0]
            key = os.path.join(gen, sub, stem) if sub else os.path.join(gen, stem)
            matched_keys.add(key)
        unmatched = [k for k in split_list if k not in matched_keys]
        print(f"  Unmatched split keys ({len(unmatched)}):")
        for k in unmatched[:20]:
            print(f"    '{k}'")
        if len(unmatched) > 20:
            print(f"    ... and {len(unmatched)-20} more")

    # --- optionally build transform ---
    transform = None
    if args.with_transform:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from utils.processing import make_processing

            class FakeOpt:
                task        = 'train'
                data_keys   = args.data_keys
                cropSize    = 96
                resize_prob = 0.0   # deterministic for testing
                jitter_prob = 0.0
                colordist_prob = 0.0
                cutout_prob = 0.0
                noise_prob  = 0.0
                blur_prob   = 0.0
                cmp_prob    = 0.0
                blur_sig    = [0.5]
                cmp_qual    = [75]
                resize_size = 256
                resize_ratio = 1.0
                norm_type   = 'resnet'

            transform = make_processing(FakeOpt())
            print("\nTransform pipeline built successfully.")
        except Exception as e:
            print(f"\n[WARN] Could not build transform ({e}). Skipping transform check.")
            transform = None

    # --- check every image ---
    print(f"\nChecking {len(samples)} images...\n")
    failures = []   # (path, stage, error)
    stage_counts = defaultdict(int)

    for i, (path, label, gen, sub) in enumerate(samples):
        ok, stage, error = check_image(path, transform)
        stage_counts[stage] += 1
        if not ok:
            failures.append((path, label, gen, sub, stage, error))
            print(f"[FAIL-{stage.upper()}] {path}\n"
                  f"           label={label}, gen={gen}, sub={sub}\n"
                  f"           error: {error}\n")
        elif (i + 1) % 200 == 0:
            print(f"  ... checked {i+1}/{len(samples)}")

    # --- summary ---
    print(f"\n{'='*60}")
    print(f"SUMMARY — split={args.split}, total={len(samples)}")
    print(f"{'='*60}")
    print(f"  Passed : {stage_counts['ok']}")
    print(f"  Failed : {len(failures)}")
    if failures:
        by_stage = defaultdict(list)
        for path, label, gen, sub, stage, error in failures:
            by_stage[stage].append(path)
        for stage, paths in sorted(by_stage.items()):
            print(f"\n  Stage '{stage}' failures ({len(paths)}):")
            for p in paths:
                print(f"    {p}")
    print(f"{'='*60}\n")

    # # --- key-building spot check for the known bad file ---
    # known_bad = 'StableDiffusion3/general/00374.jpg'
    # print(f"Spot-check for known bad file: '{known_bad}'")
    # stem = os.path.splitext(os.path.basename(known_bad))[0]
    # # Reconstruct key as the walker would
    # gen  = 'StableDiffusion3'
    # sub  = 'general'
    # key  = os.path.join(gen, sub, stem)
    # print(f"  Key that would be built : '{key}'")
    # print(f"  Key in split_list       : {_in_list(split_list, key)}")
    # # Check both Real and Fake directory locations
    # for label in ('Real', 'Fake'):
    #     candidate = os.path.join(args.data_root, label, gen, sub,
    #                              os.path.basename(known_bad))
    #     exists = os.path.exists(candidate)
    #     size   = os.path.getsize(candidate) if exists else -1
    #     print(f"  {label} path exists={exists}, size={size}B : {candidate}")


if __name__ == '__main__':
    main()
