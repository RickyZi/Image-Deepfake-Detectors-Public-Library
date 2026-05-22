"""
fix_folder_names.py

Checks that all subfolders under demo_images/<season>/ follow the expected
naming conventions and renames them if necessary

Expected structure:
  <dataset_root>/
    <mod>/            e.g. PreSocial, Facebook, Telegram, X
      <label>/        Real | Fake
        <gen>/        e.g. FFHQ, FORLAB, StyleGAN, StyleGAN2, StyleGAN3,
                           StableDiffusion1.5, StableDiffusion2, StableDiffusion3,
                           StableDiffusionXL, FLUX.1
          [<sub>/]    e.g. images-psi-0.5, conf-f-psi-1, conf-t-psi-0.5, general
            <files>   *.jpg / *.png / *.jpeg

Known rename rules (applied bottom-up so children are renamed before parents):
  Subfolder names:
    images-psi-0p5  -> images-psi-0.5
    conf-t-psi-0p5  -> conf-t-psi-0.5
    conf-f-psi-0p5  -> conf-f-psi-0.5   
    -> 0p5 was the renaming solution to avoid renaming error when exporting/importing
    images from LR

Usage:
  # Dry run (default) — only prints what would be renamed:
  python fix_folder_names.py --dataset_root ./demo_images/season_TM01

  # Actually rename:
  python fix_folder_names.py --dataset_root ./demo_images/season_TM01 --apply
"""

import os
import argparse

# ---------------------------------------------------------------------------
# Rename rules: map wrong name -> correct name
# Add more entries here as needed.
# ---------------------------------------------------------------------------
RENAME_RULES = {
    # sub-folder level (psi notation using 'p' instead of '.')
    'images-psi-0p5':   'images-psi-0.5',
    'conf-t-psi-0p5':   'conf-t-psi-0.5',
    'conf-f-psi-0p5':   'conf-f-psi-0.5',
    'conf-t-psi-1p0':   'conf-t-psi-1.0',
    'conf-f-psi-1p0':   'conf-f-psi-1.0',
    # mod-level aliases (social platform names)
    'Twitter':          'X',
    'twitter':          'X',
    'Whatsapp':         'WhatsApp',
    # label-level aliases
    'fake':             'Fake',
    'real':             'Real',
    'FAKE':             'Fake',
    'REAL':             'Real',
}

# ---------------------------------------------------------------------------
# Expected valid names at each depth level (0-indexed after dataset_root)
# depth 0 = mod, depth 1 = label, depth 2 = gen, depth 3 = sub
# None means "anything goes" (e.g. sub-folder names are dataset-specific)
# ---------------------------------------------------------------------------
VALID_NAMES = {
    0: {'PreSocial', 'Facebook', 'Telegram', 'X'},
    1: {'Real', 'Fake'},
    2: {'FFHQ', 'FORLAB', 'StyleGAN', 'StyleGAN2', 'StyleGAN3',
        'StableDiffusion1.5', 'StableDiffusion2', 'StableDiffusion3',
        'StableDiffusionXL', 'FLUX.1'},
    3: None,  # sub-folder names vary; only rename-rule corrections applied
}


def collect_renames(dataset_root, rules):
    """
    Walk the tree bottom-up and collect (old_path, new_path) pairs
    for every folder whose name matches a rename rule or is not in VALID_NAMES.
    Bottom-up ensures we rename children before parents (avoiding broken paths).
    """
    renames = []  # list of (old_abs_path, new_abs_path)
    warnings = []

    for dirpath, dirnames, _ in os.walk(dataset_root, topdown=False, followlinks=True):
        # Compute depth relative to dataset_root
        rel = os.path.relpath(dirpath, dataset_root)
        if rel == '.':
            continue
        parts = rel.split(os.sep)
        depth = len(parts) - 1          # depth of the *last* component
        name  = parts[-1]               # the folder name we're inspecting

        # --- Apply rename rules ---
        if name in rules:
            correct = rules[name]
            old_path = dirpath
            new_path = os.path.join(os.path.dirname(dirpath), correct)
            renames.append((old_path, new_path))
            continue  # no further validation needed for this entry

        # --- Validate against known valid sets ---
        valid_set = VALID_NAMES.get(depth)
        if valid_set is not None and name not in valid_set:
            warnings.append(
                f"  [WARN] Unknown name at depth {depth} "
                f"(expected one of {sorted(valid_set)}): {dirpath}"
            )

    return renames, warnings


def apply_renames(renames, dry_run=True):
    """Print and optionally execute the collected renames."""
    if not renames:
        print("✓  No renames needed — all folder names look correct.")
        return

    label = "[DRY RUN]" if dry_run else "[RENAMING]"
    print(f"\n{label} {len(renames)} folder(s) to rename:\n")

    errors = []
    for old, new in renames:
        print(f"  {os.path.relpath(old)}")
        print(f"    -> {os.path.relpath(new)}")
        if not dry_run:
            if os.path.exists(new):
                msg = f"  [ERROR] Target already exists, skipping: {new}"
                print(msg)
                errors.append(msg)
            else:
                try:
                    os.rename(old, new)
                    print(f"    [OK]")
                except OSError as e:
                    msg = f"  [ERROR] {e}"
                    print(msg)
                    errors.append(msg)
        print()

    if dry_run:
        print("Run with --apply to perform the renames.")
    elif errors:
        print(f"\n{len(errors)} error(s) occurred:")
        for e in errors:
            print(e)
    else:
        print(f"✓  All {len(renames)} renames completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Check and fix demo_images subfolder names.")
    parser.add_argument(
        '--dataset_root', type=str,
        default='./demo_images/season_TM01',
        help='Path to the season folder (default: ./demo_images/season_TM01)'
    )
    parser.add_argument(
        '--apply', action='store_true',
        help='Actually rename folders (default: dry run only)'
    )
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    if not os.path.isdir(dataset_root):
        print(f"[ERROR] Dataset root not found: {dataset_root}")
        return

    print(f"Scanning: {dataset_root}\n")
    renames, warnings = collect_renames(dataset_root, RENAME_RULES)

    # Print validation warnings first
    if warnings:
        print(f"[WARNINGS] {len(warnings)} unexpected folder name(s) found "
              f"(no rename rule — manual review needed):")
        for w in warnings:
            print(w)
        print()

    apply_renames(renames, dry_run=not args.apply)


if __name__ == '__main__':
    main()