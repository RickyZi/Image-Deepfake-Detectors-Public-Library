# """
# Reconstructs the original folder structure from flat renamed images.

# Flat name format: {split}_{algo}_{optional_dir}_{filename}.ext
#                   e.g. Fake_StableDiffusionXL_animals_02190.png
#                        Fake_FLUXp1_conf-t-psi-0p5_00001.png

# The script reverses the LR rename rules (0p5 -> 0.5, FLUXp1 -> FLUX.1, etc.)
# and reconstructs: output_root/split/algo/optional_dir/filename.ext

# NOTE: adapted for truefake_2k dataset structure

# """

# import os
# import sys
# import shutil
# from pathlib import Path
# import json

# def load_expected(json_path: str) -> dict:
#     if not Path(json_path).exists():
#         sys.exit(f"ERROR: JSON not found: {json_path}")
#     with open(json_path, 'r') as f:
#         return json.load(f)

# # Depth mapping for the flat filename structure:
# # 0: social   -> {PreSocial, Facebook, Telegram, X}
# # 1: split    -> {Real, Fake}
# # 2: algo     -> {FFHQ, FORLAB, StyleGAN, ...}
# # 3: subdir   -> optional, anything (general, conf-t-psi-0p5, animals, ...)
# # 4+:           filename

# # VALID_NAMES_FLAT = {
# #     0: {'Real', 'Fake'},
# #     1: {'PreSocial', 'Facebook', 'Telegram', 'X'},
# #     2: {'FFHQ', 'FORLAB', 'StyleGAN', 'StyleGAN2', 'StyleGAN3',
# #         'StableDiffusion1p5', 'StableDiffusion2', 'StableDiffusion3',
# #         'StableDiffusionXL', 'FLUXp1',
# #         # also accept already-decoded names
# #         'StableDiffusion1.5', 'FLUX.1'},
# # }

# VALID_NAMES_FLAT = {
#     'split': {'Real', 'Fake'},
#     'social': {'PreSocial', 'Facebook', 'Telegram', 'X'},
#     'algo': {'FFHQ', 'FORLAB', 'StyleGAN', 'StyleGAN2', 'StyleGAN3',
#         'StableDiffusion1p5', 'StableDiffusion2', 'StableDiffusion3',
#         'StableDiffusionXL', 'FLUXp1',
#         'StableDiffusion1.5', 'FLUX.1'},
# }

# # Reverse rules: encoded -> original
# LR_REVERSE_RULES = {
#     'images-psi-0p5':       'images-psi-0.5',
#     'images-psi-0p7':       'images-psi-0.7',
#     'conf-t-psi-0p5':       'conf-t-psi-0.5',
#     'conf-t-psi-0p7':       'conf-t-psi-0.7',
#     'conf-f-psi-0p5':       'conf-f-psi-0.5',
#     'conf-f-psi-0p7':       'conf-f-psi-0.7',
#     'conf-t-psi-1p0':       'conf-t-psi-1.0',
#     'FLUXp1':               'FLUX.1',
#     'StableDiffusion1p5':   'StableDiffusion1.5',
# }


# def reverse_part(part: str) -> str:
#     return LR_REVERSE_RULES.get(part, part)


# def parse_flat_name(filename: str) -> tuple[list[str], str] | None:
#     stem, ext = os.path.splitext(filename)
#     tokens = stem.split('_')

#     if len(tokens) < 3:
#         print(f"  [WARN] Too few tokens to parse: {filename}")
#         return None

#     folder_parts = []

#     idx = 0  # current token index

#     # --- Depth 0: optional social platform ---
#     if tokens[idx] in VALID_NAMES_FLAT['social']:
#         folder_parts.append(reverse_part(tokens[idx]))
#         idx += 1

#     # --- Next: split (Real/Fake) ---
#     if idx >= len(tokens) or tokens[idx] not in VALID_NAMES_FLAT['split']:
#         print(f"  [WARN] Expected Real/Fake at token {idx}: {filename}")
#         return None
#     folder_parts.append(reverse_part(tokens[idx]))
#     idx += 1

#     # --- Next: algo ---
#     if idx >= len(tokens) or tokens[idx] not in VALID_NAMES_FLAT['algo']:
#         print(f"  [WARN] Expected algo at token {idx}: {filename}")
#         return None
#     folder_parts.append(reverse_part(tokens[idx]))
#     idx += 1

#     # --- Next: optional subdir (non-numeric) or filename ---
#     if idx >= len(tokens):
#         print(f"  [WARN] No filename found in: {filename}")
#         return None

#     if not tokens[idx].isdigit():
#         # It's a subdir (e.g. 'general', 'conf-t-psi-0p5', 'animals')
#         folder_parts.append(reverse_part(tokens[idx]))
#         idx += 1

#     # --- Remaining tokens: filename ---
#     remaining = '_'.join(tokens[idx:])
#     if not remaining:
#         print(f"  [WARN] No filename stem after subdir in: {filename}")
#         return None

#     return folder_parts, remaining + ext




#     # --------------------------------------------------- #

#     # # --- Depths 0, 1, 2: strict validation ---
#     # for depth in range(3):
#     #     token = tokens[depth]
#     #     valid_set = VALID_NAMES_FLAT[depth]
#     #     if token not in valid_set:
#     #         print(f"  [WARN] Token '{token}' not valid at depth {depth} "
#     #               f"(expected one of {sorted(valid_set)}): {filename}")
#     #         return None
#     #     folder_parts.append(reverse_part(token))

#     # # --- Depth 3: subdir OR filename ---
#     # # Real images (FFHQ, FORLAB) have no subdir, so token[3] is the filename stem.
#     # # Fake images have a subdir (e.g. 'general', 'conf-t-psi-0p5', 'animals'),
#     # # followed by the filename stem at token[4+].
#     # # Heuristic: if token[3] is purely numeric it's the filename stem, not a subdir.
#     # token3 = tokens[3]
#     # if token3.isdigit():
#     #     # No subdir — token[3] onward is the filename
#     #     original_filename = '_'.join(tokens[3:]) + ext
#     # else:
#     #     # Has subdir — consume token[3] as folder part
#     #     folder_parts.append(reverse_part(token3))
#     #     remaining = '_'.join(tokens[4:])
#     #     if not remaining:
#     #         print(f"  [WARN] No filename stem after subdir in: {filename}")
#     #         return None
#     #     original_filename = remaining + ext

#     # return folder_parts, original_filename


# def reconstruct(input_path: str, output_path: str, expected: dict, dry_run: bool = True):
#     os.makedirs(output_path, exist_ok=True)

#     files = [
#         f for f in os.listdir(input_path)
#         if os.path.isfile(os.path.join(input_path, f))
#         and not f.startswith('.')
#     ]

#     ok, skipped = 0, 0

#     for filename in sorted(files):
#         result = parse_flat_name(filename)
#         if result is None:
#             skipped += 1
#             continue

#         folder_parts, original_filename = result

#         # Reconstruct destination path
#         dst_dir  = os.path.join(output_path, *folder_parts)
#         dst_path = os.path.join(dst_dir, original_filename)
#         src_path = os.path.join(input_path, filename)

#         print(f"{'[DRY]' if dry_run else ''} {filename}")
#         print(f"  -> {os.path.join(*folder_parts, original_filename)}")

#         if not dry_run:
#             os.makedirs(dst_dir, exist_ok=True)
#             # shutil.copy2(src_path, dst_path) # copy data and metadata
#             shutil.move(src_path, dst_path)

#         ok += 1

#     print(f"\nDone. Re-organized: {ok}  Skipped: {skipped}")

#     # check if reconstructed images contain the same number of images as original split
#     if not dry_run:
#         sanity_check(output_path, expected)


# def sanity_check(output_path: str, expected: dict):
#     """
#     Verify reconstructed folder counts match expected JSON counts.
#     Checks total Real/Fake counts and per-subfolder counts.
#     """
#     print("\n" + "="*60)
#     print("SANITY CHECK")
#     print("="*60)

#     errors   = []
#     warnings = []

#     # --- Count actual images per leaf folder ---
#     actual_counts = {'real': {}, 'fake': {}}

#     for dirpath, dirnames, files in os.walk(output_path):
#         if not files:
#             continue
#         images = [f for f in files if not f.startswith('.') and
#                   f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
#         if not images:
#             continue

#         rel = os.path.relpath(dirpath, output_path)
#         parts = rel.split(os.sep)

#         # Handle both with-social (4 parts) and without-social (3 parts) structures
#         # We need split + algo/subdir path
#         if 'Real' in parts:
#             split_idx = parts.index('Real')
#             split_key = 'real'
#         elif 'Fake' in parts:
#             split_idx = parts.index('Fake')
#             split_key = 'fake'
#         else:
#             warnings.append(f"  [WARN] Could not determine Real/Fake for: {rel}")
#             continue

#         # subfolder key = algo/subdir (everything after split)
#         sub_key = '/'.join(parts[split_idx + 1:])

#         if sub_key not in actual_counts[split_key]:
#             actual_counts[split_key][sub_key] = 0
#         actual_counts[split_key][sub_key] += len(images)

#     # --- Total Real/Fake ---
#     total_real = sum(actual_counts['real'].values())
#     total_fake = sum(actual_counts['fake'].values())

#     exp_real = expected['n_real']
#     exp_fake = expected['n_fake']

#     status_real = "OK" if total_real == exp_real else "FAIL"
#     status_fake = "OK" if total_fake == exp_fake else "FAIL"

#     print(f"\n  Total Real: {total_real:>5} / {exp_real:>5}  [{status_real}]")
#     print(f"  Total Fake: {total_fake:>5} / {exp_fake:>5}  [{status_fake}]")

#     if status_real == "FAIL":
#         errors.append(f"Real total mismatch: got {total_real}, expected {exp_real}")
#     if status_fake == "FAIL":
#         errors.append(f"Fake total mismatch: got {total_fake}, expected {exp_fake}")

#     # --- Per-subfolder: Real ---
#     print("\n  --- Real subfolders ---")
#     for sub_key, exp_count in expected['subfolder_counts']['real'].items():
#         actual = actual_counts['real'].get(sub_key, 0)
#         status = "OK" if actual == exp_count else "FAIL"
#         print(f"  {'OK' if status == 'OK' else 'FAIL':4}  {sub_key:<40} {actual:>5} / {exp_count:>5}")
#         if status == "FAIL":
#             errors.append(f"Real/{sub_key}: got {actual}, expected {exp_count}")

#     # Check for unexpected real subfolders
#     for sub_key in actual_counts['real']:
#         if sub_key not in expected['subfolder_counts']['real']:
#             warnings.append(f"Unexpected Real subfolder: {sub_key} ({actual_counts['real'][sub_key]} images)")

#     # --- Per-subfolder: Fake ---
#     print("\n  --- Fake subfolders ---")
#     for sub_key, exp_count in expected['subfolder_counts']['fake'].items():
#         actual = actual_counts['fake'].get(sub_key, 0)
#         status = "OK" if actual == exp_count else "FAIL"
#         print(f"  {'OK' if status == 'OK' else 'FAIL':4}  {sub_key:<40} {actual:>5} / {exp_count:>5}")
#         if status == "FAIL":
#             errors.append(f"Fake/{sub_key}: got {actual}, expected {exp_count}")

#     # Check for unexpected fake subfolders
#     for sub_key in actual_counts['fake']:
#         if sub_key not in expected['subfolder_counts']['fake']:
#             warnings.append(f"Unexpected Fake subfolder: {sub_key} ({actual_counts['fake'][sub_key]} images)")

#     # --- Summary ---
#     print("\n" + "-"*60)
#     if warnings:
#         print("WARNINGS:")
#         for w in warnings:
#             print(f"  [WARN] {w}")

#     if errors:
#         print(f"\nSANITY CHECK FAILED — {len(errors)} error(s):")
#         for e in errors:
#             print(f"  [FAIL] {e}")
#     else:
#         print("SANITY CHECK PASSED — all counts match.")
#     print("="*60)


# # # --- Expected counts from JSON ---
# # EXPECTED = {
# #     "n_real": 1000,
# #     "n_fake": 1000,
# #     "subfolder_counts": {
# #         "real": {
# #             "FORLAB": 500,
# #             "FFHQ":   500,
# #         },
# #         "fake": {
# #             "StyleGAN/images-psi-0.7":        38,
# #             "StableDiffusion3/animals":        38,
# #             "StableDiffusion1.5/general":      39,
# #             "StableDiffusion3/general":        38,
# #             "StableDiffusionXL/faces":         38,
# #             "StableDiffusion1.5/faces":        39,
# #             "StableDiffusion2/landscapes":     39,
# #             "StyleGAN3/conf-t-psi-0.5":        38,
# #             "StableDiffusion1.5/animals":      39,
# #             "StyleGAN/images-psi-0.5":         38,
# #             "FLUX.1/faces":                    39,
# #             "StableDiffusionXL/animals":       38,
# #             "FLUX.1/landscapes":               39,
# #             "StableDiffusion2/faces":          39,
# #             "FLUX.1/general":                  39,
# #             "StyleGAN2/conf-f-psi-0.5":        38,
# #             "StyleGAN3/conf-t-psi-0.7":        38,
# #             "FLUX.1/animals":                  39,
# #             "StableDiffusion2/animals":        39,
# #             "StableDiffusion3/faces":          38,
# #             "StableDiffusion2/general":        39,
# #             "StableDiffusionXL/landscapes":    38,
# #             "StyleGAN2/conf-f-psi-1":          38,
# #             "StableDiffusion3/landscapes":     38,
# #             "StableDiffusionXL/general":       38,
# #             "StableDiffusion1.5/landscapes":   39,
# #         }
# #     }
# # }


# if __name__ == "__main__":
#     # input_path  = './truefake_2k/tf2k_lr/seasons/spring_SP01'
#     # /truefake_2k/adaptive
#     input_path = Path(sys.argv[1]) if len(sys.argv) > 2 else Path("./truefake_2k/tf2k_lr/seasons")
#     # input_path = './tf2k_lr/'
#     lr_preset = str(sys.argv[2]) if len(sys.argv) > 2 else str("seasons")
#     # output_path = './truefake_2k/tf_renamed/seasons/spring_SP01'
#     json_path = './full_social_split.json'

#     # run with python3 utils/tf2k_org_imgs.py ./tf2k_lr/ adaptive

#     input_path = Path(input_path) / lr_preset

#     if not Path(input_path).exists():
#         sys.exit(f"ERROR: Input path not found: {input_path}")

#     output_path = Path('./tf2k_org') / lr_preset
#     # get subfolders from input_path -> two main subfolders: seasons and style, get all subfolders
#     inpt_subfolders = sorted(child for child in input_path.iterdir() if child.is_dir())
#     print(inpt_subfolders)
#     breakpoint()

#     # load expected counts from dataset JSON (sanity check)
#     expected = load_expected(json_path)


#     # for each subfolder in input_path, reconstruct the structure in output_path and start processing
#     for subfolder in inpt_subfolders:
#         out_subfolder = output_path / subfolder.name
#         os.makedirs(out_subfolder, exist_ok=True)
#         print(f"Created output subfolder: {out_subfolder}")
#         print(f"\nProcessing subfolder: {subfolder} -> {out_subfolder}")

#         reconstruct(subfolder, out_subfolder, expected, dry_run=False)  # dry_run = True -> do a dry run to verify paths
#         print(f"Reconstruction complete for subfolder: {out_subfolder}")
#         breakpoint() # to make easier to inspect the sanity check

#     # reconstruct(input_path, output_path, expected, dry_run=False)  # set dry_run=False to actually copy

"""
Reconstructs the original folder structure from flat renamed images.

Flat name format: {split}_{algo}_{optional_dir}_{filename}.ext
                  e.g. Fake_StableDiffusionXL_animals_02190.png
                       Fake_FLUXp1_conf-t-psi-0p5_00001.png

The script reverses the LR rename rules (0p5 -> 0.5, FLUXp1 -> FLUX.1, etc.)
and reconstructs: output_root/split/algo/optional_dir/filename.ext

NOTE: adapted for truefake_2k dataset structure

CHANGES:
- Files are now MOVED (shutil.move) instead of copied, since we want the
  flat renamed originals to disappear once reorganized.
- The reorganized output now lands next to the input folder instead of
  under the script's working directory: if input is
      <dataset_root>/tf2k_lr/<preset>
  the output goes to
      <dataset_root>/tf2k_org/<preset>
  i.e. same parent folder as the input, just a sibling directory instead
  of "./tf2k_lr".
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


def reverse_part(part: str) -> str:
    return LR_REVERSE_RULES.get(part, part)


def parse_flat_name(filename: str) -> tuple[list[str], str] | None:
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


def reconstruct(input_path: str, output_path: str, expected: dict, dry_run: bool = True):
    os.makedirs(output_path, exist_ok=True)

    files = [
        f for f in os.listdir(input_path)
        if os.path.isfile(os.path.join(input_path, f))
        and not f.startswith('.')
    ]

    ok, skipped = 0, 0

    for filename in sorted(files):
        result = parse_flat_name(filename)
        if result is None:
            skipped += 1
            continue

        folder_parts, original_filename = result

        # Reconstruct destination path
        dst_dir  = os.path.join(output_path, *folder_parts)
        dst_path = os.path.join(dst_dir, original_filename)
        src_path = os.path.join(input_path, filename)

        print(f"{'[DRY]' if dry_run else ''} {filename}")
        print(f"  -> {os.path.join(*folder_parts, original_filename)}")

        if not dry_run:
            os.makedirs(dst_dir, exist_ok=True)
            shutil.move(src_path, dst_path)  # move (not copy) — src is removed once done

        ok += 1

    print(f"\nDone. Re-organized: {ok}  Skipped: {skipped}")

    # check if reconstructed images contain the same number of images as original split
    if not dry_run:
        sanity_check(output_path, expected)


def sanity_check(output_path: str, expected: dict):
    """
    Verify reconstructed folder counts match expected JSON counts.
    Checks total Real/Fake counts and per-subfolder counts.
    """
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

    print(f"\n  Total Real: {total_real:>5} / {exp_real:>5}  [{status_real}]")
    print(f"  Total Fake: {total_fake:>5} / {exp_fake:>5}  [{status_fake}]")

    if status_real == "FAIL":
        errors.append(f"Real total mismatch: got {total_real}, expected {exp_real}")
    if status_fake == "FAIL":
        errors.append(f"Fake total mismatch: got {total_fake}, expected {exp_fake}")

    # --- Per-subfolder: Real ---
    print("\n  --- Real subfolders ---")
    for sub_key, exp_count in expected['subfolder_counts']['real'].items():
        actual = actual_counts['real'].get(sub_key, 0)
        status = "OK" if actual == exp_count else "FAIL"
        print(f"  {'OK' if status == 'OK' else 'FAIL':4}  {sub_key:<40} {actual:>5} / {exp_count:>5}")
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
        print(f"  {'OK' if status == 'OK' else 'FAIL':4}  {sub_key:<40} {actual:>5} / {exp_count:>5}")
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
        print(f"\nSANITY CHECK FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  [FAIL] {e}")
    else:
        print("SANITY CHECK PASSED — all counts match.")
    print("="*60)


if __name__ == "__main__":
    # run with: python3 utils/tf2k_org_imgs.py <input_root> <lr_preset>
    # e.g.      python3 utils/tf2k_org_imgs.py ./truefake_2k/tf2k_lr seasons
    #
    # This reorganizes IN PLACE: each subfolder under input_path (e.g.
    # seasons/spring_SP01) has its own flat images moved into freshly
    # created Real/Fake/algo/subdir folders *inside that same subfolder*.
    # Nothing is written outside the input tree.
    input_root = Path(sys.argv[1]) if len(sys.argv) > 2 else Path("./truefake_2k/tf2k_lr")
    lr_preset  = str(sys.argv[2]) if len(sys.argv) > 2 else str("seasons")
    json_path  = './full_social_split.json'

    input_path = input_root / lr_preset

    if not input_path.exists():
        sys.exit(f"ERROR: Input path not found: {input_path}")

    # get subfolders from input_path -> e.g. spring_SP01, summer_SU01, ...
    inpt_subfolders = sorted(child for child in input_path.iterdir() if child.is_dir())
    print(inpt_subfolders)
    breakpoint()

    # load expected counts from dataset JSON (sanity check)
    expected = load_expected(json_path)

    # for each subfolder, reorganize its own flat images into subfolders
    # inside itself (Real/Fake -> algo -> optional subdir -> filename)
    for subfolder in inpt_subfolders:
        print(f"\nProcessing subfolder in place: {subfolder}")

        reconstruct(subfolder, subfolder, expected, dry_run=False)  # dry_run = True -> do a dry run to verify paths
        print(f"Reconstruction complete for subfolder: {subfolder}")
        breakpoint()  # to make it easier to inspect the sanity check