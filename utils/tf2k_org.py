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


def reconstruct(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    expected: dict,
    dry_run: bool = True,
    filenames: list[str] | None = None,
):
    os.makedirs(output_path, exist_ok=True)

    if filenames is None:
        files = [
            f for f in os.listdir(input_path)
            if os.path.isfile(os.path.join(input_path, f))
            and not f.startswith('.')
        ]
    else:
        files = filenames

    print(f"Found {len(files)} files in {input_path}")

    ok, skipped = 0, 0
    organized_paths = []

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
        organized_paths.append(dst_path)

        # print(f"{'[DRY]' if dry_run else ''} {filename}")
        # print(f"  -> {os.path.join(*folder_parts, original_filename)}")

        if not dry_run:
            os.makedirs(dst_dir, exist_ok=True)
            shutil.move(src_path, dst_path)  # move (not copy) — src is removed once done

        ok += 1
        _print_progress(processed, total)

    print(f"\nDone. Re-organized: {ok}  Skipped: {skipped}")

    # check if reconstructed images contain the same number of images as original split
    if not dry_run and organized_paths:
        sanity_check(
            output_path,
            expected,
            image_paths=organized_paths if filenames is not None else None,
        )


def _print_progress(current: int, total: int, width: int = 40):
    percentage = int(current / total * 100) if total else 0
    filled = int(width * current / total) if total else 0
    bar = '#' * filled + '-' * (width - filled)
    sys.stdout.write(f"\r[{bar}] {current}/{total} ({percentage:3d}%)")
    if current == total:
        sys.stdout.write("\n")
    sys.stdout.flush()


def sanity_check(
    output_path: str | os.PathLike[str],
    expected: dict,
    image_paths: list[str] | None = None,
):
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

    if image_paths is None:
        image_folders = (
            (dirpath, files)
            for dirpath, _dirnames, files in os.walk(output_path)
        )
    else:
        image_folders = (
            (os.path.dirname(path), [os.path.basename(path)])
            for path in image_paths
        )

    for dirpath, files in image_folders:
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
        print(f"{green}SANITY CHECK PASSED — all counts match {reset}")
    print("="*60)


def find_file_folders(root: Path) -> list[Path]:
    """Find directories containing files, deepest first, before any files are moved"""
    # folders = []
    # for dirpath, _dirnames, filenames in os.walk(root, topdown=False):
    #     files = [
    #         filename for filename in filenames
    #         if not filename.startswith('.')
    #         and os.path.isfile(os.path.join(dirpath, filename))
    #     ]

    #     print(f"Found {len(files)} files in {dirpath}")
    #     if files:
    #         folders.append((Path(dirpath), sorted(files)))
    # return folders

    all_folders = sorted([
            Path(dir_path) 
            for dir_path, dirs, files in os.walk(input_root) 
            if files and (not "Fake" in dir_path and not "Real" in dir_path)
        ])
    return all_folders


if __name__ == "__main__":
    # run with: python3 utils/tf2k_org.py <input_root>
    # Quote input_root when its path contains spaces.
    # -------------------------------------------------------------------- #
    # This reorganizes files in place in every directory under input_root,
    # including input_root itself. Each directory's files are moved into
    # Real/Fake/algo/subdir folders within that same directory.
    # -------------------------------------------------------------------- #
    input_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./truefake_2k/tf2k_lr")
    json_path  = './full_social_split.json'

    print(f"Input root: {input_root}")
    print(f"Using JSON for expected counts: {json_path}")

    if not input_root.is_dir():
        sys.exit(f"ERROR: Input path not found or not a directory: {input_root}")

    print("\nFinding folders containing files...")
    all_folders = find_file_folders(input_root)
    if len(all_folders) == 0:
        print(f"\nNo folders containing files found under {input_root}. Images alredy preprocessed. \nExiting.")
        sys.exit(0)
    print(f"Found {len(all_folders)} folders containing files under {input_root}")
    expected = load_expected(json_path)
    # # breakpoint()

    # collect all folders containing files, deepst first
    # all_folders = [
    #     Path(dir_path) 
    #     for dir_path, dirs, files in os.walk(input_root) 
    #     if files
    # ]

    # print(f"Found {len(all_folders)} folders containing files under {input_root}")
    print(f"folders: {[str(folder) for folder in all_folders]}")
    # breakpoint()
    for i, folder in enumerate(all_folders, start=1):
        print(f"\nProcessing folder {i}/{len(all_folders)}: {folder.name}")
        if len(os.listdir(folder)) == 0:
            print(f"  [WARN] Folder is empty, skipping: {folder}")
            continue
        reconstruct(
            folder,
            folder,
            expected,
            dry_run=False,
        )
        print(f"Reconstruction complete for folder: {folder.name}")

    print("\ndone. All folders processed")

    # for folder, filenames in folders:
    #     print(f"\nProcessing folder: {folder}")
    #     reconstruct(
    #         folder,
    #         folder,
    #         expected,
    #         dry_run=False,
    #         filenames=filenames,
    #     )
    #     print(f"Reconstruction complete for folder: {folder}")