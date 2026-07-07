# create a copy of the folder in the desired path 
# folder_name = Social_Real/Fake_Algo_general_img_name.jg
# then reverse it
# Social/Real or Fake/ Algo / general / img_name
# note: general is fixed

"""
Utils script that copies the images from the ./demo_images/ folder
of the TB dataset, and rename them following their folder structure

Ex.
    - folder_name = Social_Real/Fake_Algo_general_img_name.jg
    - img_name = social_fake_algo_general_img-name.jpg

This structure is then used to recreated the folder structure when
exporting the images with presets applied from LR

"""

import os 
import sys
from pathlib import Path
import shutil
import json

LR_RENAME_RULES = {
    # sub-folder level (psi notation using 'p' instead of '.')
    'images-psi-0.5':       'images-psi-0p5',
    'images-psi-0.7':       'images-psi-0p7',
    'conf-t-psi-0.5':       'conf-t-psi-0p5',
    'conf-t-psi-0.7':       'conf-t-psi-0p7',
    'conf-f-psi-0.5':       'conf-f-psi-0p5',
    'conf-f-psi-0.7':       'conf-f-psi-0p7',
    'conf-t-psi-1.0':       'conf-t-psi-1p0',
    'FLUX.1':               'FLUXp1',
    'StableDiffusion1.5':   'StableDiffusion1p5'
    
    # # mod-level aliases (social platform names)
    # 'Twitter':          'X',
    # 'twitter':          'X',
    # 'Whatsapp':         'WhatsApp',
    # # label-level aliases
    # 'fake':             'Fake',
    # 'real':             'Real',
    # 'FAKE':             'Fake',
    # 'REAL':             'Real',
}

# Reverse mapping for parsing the renamed flat filenames back into original names.
LR_REVERSE_RULES = {v: k for k, v in LR_RENAME_RULES.items()}

VALID_NAMES = {
    0: {'PreSocial', 'Facebook', 'Telegram', 'X'},
    1: {'Real', 'Fake'},
    2: {'FFHQ', 'FORLAB', 'StyleGAN', 'StyleGAN2', 'StyleGAN3',
        'StableDiffusion1.5', 'StableDiffusion2', 'StableDiffusion3',
        'StableDiffusionXL', 'FLUX.1'},
    3: None,  # sub-folder names vary; only rename-rule corrections applied
}



def load_expected(json_path: str) -> dict:
    if not Path(json_path).exists():
        sys.exit(f"ERROR: JSON not found: {json_path}")
    with open(json_path, 'r') as f:
        return json.load(f)

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



def sanity_check_flat(output_path: str, expected: dict):
    """
    Verify flat-renamed output counts match expected JSON counts.
    Counts files by parsing the reconstructed flat filename prefix directly
    from the output directory — no parse_flat_name needed.
    """
    print("\n" + "="*60)
    print("SANITY CHECK — flat renamed files")
    print("="*60)

    errors   = []
    warnings = []
    actual_counts = {'real': {}, 'fake': {}}

    files = [
        f for f in os.listdir(output_path)
        if os.path.isfile(os.path.join(output_path, f))
        and not f.startswith('.')
        and f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
    ]

    for filename in files:
        stem, _ = os.path.splitext(filename)
        tokens  = stem.split('_')

        # Walk tokens to find split (Real/Fake)
        split_key = None
        split_idx = None
        for i, t in enumerate(tokens):
            if t == 'Real':
                split_key, split_idx = 'real', i
                break
            elif t == 'Fake':
                split_key, split_idx = 'fake', i
                break

        if split_key is None:
            warnings.append(f"No Real/Fake token found: {filename}")
            continue

        # algo = token right after split
        # subdir = token after algo if non-numeric, else absent
        # filename = remaining numeric tokens
        algo_idx = split_idx + 1
        if algo_idx >= len(tokens):
            warnings.append(f"No algo token found: {filename}")
            continue

        algo = LR_REVERSE_RULES.get(tokens[algo_idx], tokens[algo_idx])

        subdir_idx = algo_idx + 1
        if subdir_idx < len(tokens) and not tokens[subdir_idx].isdigit():
            subdir  = LR_REVERSE_RULES.get(tokens[subdir_idx], tokens[subdir_idx])
            sub_key = f"{algo}/{subdir}"
        else:
            sub_key = algo

        actual_counts[split_key][sub_key] = actual_counts[split_key].get(sub_key, 0) + 1

    # --- Totals ---
    total_real = sum(actual_counts['real'].values())
    total_fake = sum(actual_counts['fake'].values())
    exp_real   = expected['n_real']
    exp_fake   = expected['n_fake']

    print(f"\n  Total Real: {total_real:>5} / {exp_real:>5}  [{'OK' if total_real == exp_real else 'FAIL'}]")
    print(f"  Total Fake: {total_fake:>5} / {exp_fake:>5}  [{'OK' if total_fake == exp_fake else 'FAIL'}]")

    if total_real != exp_real:
        errors.append(f"Real total mismatch: got {total_real}, expected {exp_real}")
    if total_fake != exp_fake:
        errors.append(f"Fake total mismatch: got {total_fake}, expected {exp_fake}")

    # --- Per-subfolder: Real ---
    print("\n  --- Real subfolders ---")
    for sub_key, exp_count in expected['subfolder_counts']['real'].items():
        # sub_key = sub_key.replace('.', 'p')
        actual = actual_counts['real'].get(sub_key, 0)
        status = "OK" if actual == exp_count else "FAIL"
        print(f"  {status:4}  {sub_key:<40} {actual:>5} / {exp_count:>5}")
        if status == "FAIL":
            errors.append(f"Real/{sub_key}: got {actual}, expected {exp_count}")

    for sub_key in actual_counts['real']:
        if sub_key not in expected['subfolder_counts']['real']:
            warnings.append(f"Unexpected Real subfolder: {sub_key} ({actual_counts['real'][sub_key]} files)")

    # --- Per-subfolder: Fake ---
    print("\n  --- Fake subfolders ---")
    for sub_key, exp_count in expected['subfolder_counts']['fake'].items():
        actual = actual_counts['fake'].get(sub_key, 0)
        status = "OK" if actual == exp_count else "FAIL"
        print(f"  {status:4}  {sub_key:<40} {actual:>5} / {exp_count:>5}")
        if status == "FAIL":
            errors.append(f"Fake/{sub_key}: got {actual}, expected {exp_count}")

    for sub_key in actual_counts['fake']:
        if sub_key not in expected['subfolder_counts']['fake']:
            warnings.append(f"Unexpected Fake subfolder: {sub_key} ({actual_counts['fake'][sub_key]} files)")

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

def copy_images(input_path: str, output_path: str, expected: dict):
    os.makedirs(output_path, exist_ok=True)

    for dirpath, dirnames, files in os.walk(input_path):
        if not files:
            continue

        rel = os.path.relpath(dirpath, input_path)
        if rel == '.':
            continue

        parts = rel.split(os.sep)
        renamed_parts = [LR_RENAME_RULES.get(p, p) for p in parts]

        for file in files:
            src_path = os.path.join(dirpath, file)
            dst_name = '_'.join(renamed_parts) + '_' + file
            dst_path = os.path.join(output_path, dst_name)
            print(f"Copying: {src_path} -> {dst_path}")
            shutil.copy(src_path, dst_path)

    sanity_check_flat(output_path, expected)

if __name__ == "__main__":

    # copy imgs into out folder original folder structure as name
    
    input_path = Path('/dataset-disk/tb_dataset/tf2k_lr/tf_dataset/adaptive/blurbg_subtle')
    #'../truefake_2k/'
    # # fixed
    output_path = Path('/dataset-disk/tb_dataset/tf2k_lr/tf_dataset/adaptive/blurbg_subtle_renamed')
    #'../truefake_2k/tf_renamed' # fixed
    # os.makedirs(out_path, exist_ok = True)

    json_path = './test_sample_2000.json'
    # json_path = './test2k_splits.json'

    expected = load_expected(json_path)

    if not Path(input_path).exists():
        sys.exit(f"ERROR: Path not found: {input_path}")

    

    subfolders = sorted(
        child for child in input_path.iterdir() if child.is_dir()
    )

    if subfolders:
        for sub in subfolders:
            img_path = input_path / sub.name
            print(f"Subfolder: {sub.name}")
            print(f"img_path: {img_path}")
            # breakpoint()
            out_path = Path(output_path / sub.name)
            print(out_path)
            if not Path(out_path).exists():
                os.makedirs(out_path, exist_ok = True)

            print(f"Renaming images from {img_path} to {out_path}...")
            breakpoint()
            copy_images(img_path, out_path, expected)
            sanity_check_flat(out_path, expected)
            print("done!")
            breakpoint()

    else:
        if not Path(output_path).exists():
            os.makedirs(output_path, exist_ok = True)

        print(f"Renaming images from {input_path} to {output_path}...")
        copy_images(input_path, output_path, expected)
        sanity_check_flat(output_path, expected)
        print("done!")

    # print(subfolders)
    # breakpoint()

    # -------------------------------------------- #
    # paths can be added also as command line args
    # input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    # out_path = Path(sys.argv[2]) if len(sys.argv) > 1 else None

    # if not input_path.exists():
    #     sys.exit(f"ERROR: Path not found: {input_path}")

    # if not out_path.exists():
    #     os.makedirs(out_path, exist_ok = True)
    # -------------------------------------------- #


    

    # dataset_root = os.path.abspath(input_path)
    # if not os.path.isdir(dataset_root):
    #     print(f"[ERROR] Dataset root not found: {dataset_root}")
    #     sys.exit(1)

    # print(f"Scanning: {dataset_root}\n")
    # renames, warnings = collect_renames(dataset_root, LR_RENAME_RULES)

    # # print(f"renames: {renames}")
    # for rename in renames:
    #     print(f"rename: {rename}")





    




