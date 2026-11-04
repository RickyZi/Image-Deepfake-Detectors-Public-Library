"""
Reconstructs the original folder structure from flat renamed images.

Flat name format: {split}_{algo}_{optional_dir}_{filename}.ext
                  e.g. Fake_StableDiffusionXL_animals_02190.png
                       Fake_FLUXp1_conf-t-psi-0p5_00001.png

The script reverses the LR rename rules (0p5 -> 0.5, FLUXp1 -> FLUX.1, etc.)
and reconstructs: output_root/split/algo/optional_dir/filename.ext

NOTE: adapted for truefake_2k dataset structure

CHANGES: 
- REORGANIZE IN PLACE (no more separate output folder)
- Files are now MOVED (shutil.move) instead of copied, since we want the
  flat renamed originals to disappear once reorganized.
- The reorganized output now lands next to the input folder instead of
  under the script's working directory: if input is
      <dataset_root>/tf2k_lr/<preset>
  the output goes to
      <dataset_root>/tf2k_org/<preset>
  i.e. same parent folder as the input, just a sibling directory instead
  of "./tf2k_lr".

Run with: python3 utils/tf2k_org_imgs.py <input_root> 

Note: <input_root> should be the root of the tf2k_lr dataset, e.g.
    ./truefake_2k/tf2k_lr

Also the json file is used to check the number of images expected in each subfolder after reconstruction.
-> using as reference the full_social_split.json file, just for the counts for each subfolder (Real/Fake, algo, optional subdir). 

-> could be updated to perform also a check on the actual split.
"""

import os
import sys
import shutil
from pathlib import Path
import json

def load_expected(json_path: str) -> dict:
    if not Path(json_path).exists():
        sys.exit(f"ERROR: JSON not found: {json_path}")
    with open(json_path, 'r') as f:
        return json.load(f)

# Depth mapping for the flat filename structure:
# 0: social   -> {PreSocial, Facebook, Telegram, X}
# 1: split    -> {Real, Fake}
# 2: algo     -> {FFHQ, FORLAB, StyleGAN, ...}
# 3: subdir   -> optional, anything (general, conf-t-psi-0p5, animals, ...)
# 4+:           filename

VALID_NAMES_FLAT = {
    'split': {'Real', 'Fake'},
    'social': {'PreSocial', 'Facebook', 'Telegram', 'X'},
    'algo': {'FFHQ', 'FORLAB', 'StyleGAN', 'StyleGAN2', 'StyleGAN3',
        'StableDiffusion1p5', 'StableDiffusion2', 'StableDiffusion3',
        'StableDiffusionXL', 'FLUXp1',
        'StableDiffusion1.5', 'FLUX.1'},
}

# Reverse rules: encoded -> original
LR_REVERSE_RULES = {
    'images-psi-0p5':       'images-psi-0.5',
    'images-psi-0p7':       'images-psi-0.7',
    'conf-t-psi-0p5':       'conf-t-psi-0.5',
    'conf-t-psi-0p7':       'conf-t-psi-0.7',
    'conf-f-psi-0p5':       'conf-f-psi-0.5',
    'conf-f-psi-0p7':       'conf-f-psi-0.7',
    'conf-t-psi-1p0':       'conf-t-psi-1.0',
    'FLUXp1':               'FLUX.1',
    'StableDiffusion1p5':   'StableDiffusion1.5',
}

def add_missing_split_prefix(filename: str) -> str:
    # Check if the filename already has a split prefix
    if filename.startswith('Real_') or filename.startswith('Fake_'):
        return filename  # No change needed

    # Determine if the file is likely Real or Fake based on its name
    # This is a heuristic; you may want to refine this logic based on your dataset
    if 'FORLAB' in filename or 'FFHQ' in filename:
        return 'Real_' + filename  # Assume it's Real
    else:
        return 'Fake_' + filename  # Assume it's Fake


def reverse_part(part: str) -> str:
    return LR_REVERSE_RULES.get(part, part)


def parse_flat_name(filename: str) -> tuple[list[str], str] | None:
    filename = add_missing_split_prefix(filename)  # Ensure split prefix is present
    stem, ext = os.path.splitext(filename)
    tokens = stem.split('_')

    if len(tokens) < 3:
        print(f"  [WARN] Too few tokens to parse: {filename}")
        return None

    folder_parts = []

    idx = 0  # current token index

    # --- Depth 0: optional social platform ---
    if tokens[idx] in VALID_NAMES_FLAT['social']:
        folder_parts.append(reverse_part(tokens[idx]))
        idx += 1

    # --- Next: split (Real/Fake) ---
    if idx >= len(tokens) or tokens[idx] not in VALID_NAMES_FLAT['split']:
        print(f"  [WARN] Expected Real/Fake at token {idx}: {filename}")
        return None
    folder_parts.append(reverse_part(tokens[idx]))
    idx += 1

    # --- Next: algo ---
    if idx >= len(tokens) or tokens[idx] not in VALID_NAMES_FLAT['algo']:
        print(f"  [WARN] Expected algo at token {idx}: {filename}")
        return None
    folder_parts.append(reverse_part(tokens[idx]))
    idx += 1

    # --- Next: optional subdir (non-numeric) or filename ---
    if idx >= len(tokens):
        print(f"  [WARN] No filename found in: {filename}")
        return None

    if not tokens[idx].isdigit():
        # It's a subdir (e.g. 'general', 'conf-t-psi-0p5', 'animals')
        folder_parts.append(reverse_part(tokens[idx]))
        idx += 1

    # --- Remaining tokens: filename ---
    remaining = '_'.join(tokens[idx:])
    if not remaining:
        print(f"  [WARN] No filename stem after subdir in: {filename}")
        return None

    return folder_parts, remaining + ext


def reconstruct(input_path: str | os.PathLike[str], output_path: str | os.PathLike[str], expected: dict, dry_run: bool = True):
    os.makedirs(output_path, exist_ok=True)

    files = [
        f for f in os.listdir(input_path)
        if os.path.isfile(os.path.join(input_path, f))
        and not f.startswith('.')
    ]

    ok, skipped = 0, 0

    total = len(files)
    print(f"Processing folder {Path(input_path).name} images:")
    parsed_files = [(filename, parse_flat_name(filename)) for filename in sorted(files)]
    _print_progress(0, total)

    for processed, (filename, result) in enumerate(parsed_files, start=1):
        if result is None:
            skipped += 1
            _print_progress(processed, total)
            continue

        folder_parts, original_filename = result

        # Reconstruct destination path
        dst_dir  = os.path.join(output_path, *folder_parts)
        dst_path = os.path.join(dst_dir, original_filename)
        src_path = os.path.join(input_path, filename)

        # print(f"{'[DRY]' if dry_run else ''} {filename}")
        # print(f"  -> {os.path.join(*folder_parts, original_filename)}")

        if not dry_run:
            os.makedirs(dst_dir, exist_ok=True)
            shutil.move(src_path, dst_path)  # move (not copy) — src is removed once done

        ok += 1
        _print_progress(processed, total)

    print(f"\nDone. Re-organized: {ok}  Skipped: {skipped}")

    # check if reconstructed images contain the same number of images as original split
    if not dry_run:
        sanity_check(output_path, expected)


def _print_progress(current: int, total: int, width: int = 40):
    percentage = int(current / total * 100) if total else 0
    filled = int(width * current / total) if total else 0
    bar = '#' * filled + '-' * (width - filled)
    sys.stdout.write(f"\r[{bar}] {current}/{total} ({percentage:3d}%)")
    if current == total:
        sys.stdout.write("\n")
    sys.stdout.flush()


def sanity_check(output_path: str | os.PathLike[str], expected: dict):
    """
    Verify reconstructed folder counts match expected JSON counts.
    Checks total Real/Fake counts and per-subfolder counts.
    """
    green = "\033[32m"
    red = "\033[31m"
    reset = "\033[0m"

    print("\n" + "="*60)
    print("SANITY CHECK")
    print("="*60)

    errors   = []
    warnings = []

    # --- Count actual images per leaf folder ---
    actual_counts = {'real': {}, 'fake': {}}

    for dirpath, dirnames, files in os.walk(output_path):
        if not files:
            continue
        images = [f for f in files if not f.startswith('.') and
                  f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        if not images:
            continue

        rel = os.path.relpath(dirpath, output_path)
        parts = rel.split(os.sep)

        # Handle both with-social (4 parts) and without-social (3 parts) structures
        # We need split + algo/subdir path
        if 'Real' in parts:
            split_idx = parts.index('Real')
            split_key = 'real'
        elif 'Fake' in parts:
            split_idx = parts.index('Fake')
            split_key = 'fake'
        else:
            warnings.append(f"  [WARN] Could not determine Real/Fake for: {rel}")
            continue

        # subfolder key = algo/subdir (everything after split)
        sub_key = '/'.join(parts[split_idx + 1:])

        if sub_key not in actual_counts[split_key]:
            actual_counts[split_key][sub_key] = 0
        actual_counts[split_key][sub_key] += len(images)

    # --- Total Real/Fake ---
    total_real = sum(actual_counts['real'].values())
    total_fake = sum(actual_counts['fake'].values())

    exp_real = expected['n_real']
    exp_fake = expected['n_fake']

    status_real = "OK" if total_real == exp_real else "FAIL"
    status_fake = "OK" if total_fake == exp_fake else "FAIL"

    color_real = green if status_real == "OK" else red
    color_fake = green if status_fake == "OK" else red
    print(f"\n{color_real}Total Real: {total_real:>5} / {exp_real:>5}  [{status_real}]{reset}")
    print(f"{color_fake}Total Fake: {total_fake:>5} / {exp_fake:>5}  [{status_fake}]{reset}")

    if status_real == "FAIL":
        errors.append(f"Real total mismatch: got {total_real}, expected {exp_real}")
    if status_fake == "FAIL":
        errors.append(f"Fake total mismatch: got {total_fake}, expected {exp_fake}")

    # --- Per-subfolder: Real ---
    print("\n  --- Real subfolders ---")
    for sub_key, exp_count in expected['subfolder_counts']['real'].items():
        actual = actual_counts['real'].get(sub_key, 0)
        status = "OK" if actual == exp_count else "FAIL"
        color = green if status == "OK" else red
        print(f"{color}  {'OK' if status == 'OK' else 'FAIL':4}  {sub_key:<40} {actual:>5} / {exp_count:>5}{reset}")
        if status == "FAIL":
            errors.append(f"Real/{sub_key}: got {actual}, expected {exp_count}")

    # Check for unexpected real subfolders
    for sub_key in actual_counts['real']:
        if sub_key not in expected['subfolder_counts']['real']:
            warnings.append(f"Unexpected Real subfolder: {sub_key} ({actual_counts['real'][sub_key]} images)")

    # --- Per-subfolder: Fake ---
    print("\n  --- Fake subfolders ---")
    for sub_key, exp_count in expected['subfolder_counts']['fake'].items():
        actual = actual_counts['fake'].get(sub_key, 0)
        status = "OK" if actual == exp_count else "FAIL"
        color = green if status == "OK" else red
        print(f"{color}  {'OK' if status == 'OK' else 'FAIL':4}  {sub_key:<40} {actual:>5} / {exp_count:>5}{reset}")
        if status == "FAIL":
            errors.append(f"Fake/{sub_key}: got {actual}, expected {exp_count}")

    # Check for unexpected fake subfolders
    for sub_key in actual_counts['fake']:
        if sub_key not in expected['subfolder_counts']['fake']:
            warnings.append(f"Unexpected Fake subfolder: {sub_key} ({actual_counts['fake'][sub_key]} images)")

    # --- Summary ---
    print("\n" + "-"*60)
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  [WARN] {w}")

    if errors:
        print(f"{red}\nSANITY CHECK FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  [FAIL] {e}")
        print(reset, end="")
        breakpoint() # to inspect errors
    else:
        print(f"{green}SANITY CHECK PASSED — all counts match.{reset}")
    print("="*60)


if __name__ == "__main__":
    # run with: python3 utils/tf2k_org_imgs.py <input_root> <lr_preset>
    # e.g.      python3 utils/tf2k_org_imgs.py ./truefake_2k/tf2k_lr seasons
    # -------------------------------------------------------------------- #
    # This reorganizes IN PLACE: each subfolder under input_path (e.g.
    # seasons/spring_SP01) has its own flat images moved into freshly
    # created Real/Fake/algo/subdir folders *inside that same subfolder*.
    # Nothing is written outside the input tree.
    # -------------------------------------------------------------------- #
    input_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./truefake_2k/tf2k_lr")
    # lr_preset  = str(sys.argv[2]) if len(sys.argv) > 2 else str("seasons")
    json_path  = './full_social_split.json'

    # if 'facebook' in input_root.name.lower() or 'telegram' in input_root.name.lower() or 'twitter' in input_root.name.lower():
    #     json_path = './full_social_split.json'
    # else:
    #     json_path = './test2k_split.json'

    print(f"Input root: {input_root}")
    print(f"Using JSON for expected counts: {json_path}")

    # look for all preset folders under input_root (e.g. seasons, style, etc.)
    # presets = os.listdir(input_root)
    # stay at the top level of input_root, only include directories and not .zip files
    presets = [child.name for child in input_root.iterdir() if child.is_dir()]
    print(f"Found presets under {input_root}: {presets}")
    breakpoint()

    for preset in presets:
        input_path = input_root / preset

    # input_path = input_root / lr_preset

        if not input_path.exists():
            sys.exit(f"ERROR: Input path not found: {input_path}")

        # get subfolders from input_path -> e.g. spring_SP01, summer_SU01, ...
        inpt_subfolders = sorted(child for child in input_path.iterdir() if child.is_dir())
        print(inpt_subfolders)
        # breakpoint()

        # load expected counts from dataset JSON (sanity check)
        expected = load_expected(json_path)

        # for each subfolder, reorganize its own flat images into subfolders
        # inside itself (Real/Fake -> algo -> optional subdir -> filename)
        for subfolder in inpt_subfolders:
            output_subfolder = subfolder  # reorganize in place
            print(f"\nProcessing subfolder in place: {subfolder}")
            reconstruct(subfolder, output_subfolder, expected, dry_run=False)  # dry_run = True -> do a dry run to verify paths
            print(f"Reconstruction complete for subfolder: {subfolder}")
            # breakpoint()  # to make it easier to inspect the sanity check