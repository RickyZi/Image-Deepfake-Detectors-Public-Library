"""
reference_consistency_check.py

For datasets with NO train/val/test split — just a collection of images
organized into subfolders (e.g. season-autumn-tm01, season-winter-tm01, ...),
each with the same real/fake/algorithm/subfolder structure.

It:
  1. Walks <dataset_path> and, for every image, builds an id of the form
     "<algorithm>[/<subfolder>]/<filename_without_ext>" (e.g. "FORLAB/00046"
     or "FLUX.1/animals/00025").
  2. Groups images by "subfolder" (everything above the real/fake folder,
     e.g. "seasons/spring-SP01" -> "seasons-spring-SP01"; or "(root)" if
     real/fake sits directly under dataset_path).
  3. Takes the FIRST subfolder (alphabetically) as the REFERENCE. Prints its
     total image count (expected to be ~2k) and a per-algorithm/subfolder
     numerosity breakdown.
  4. Saves the reference image id list to a JSON file.
  5. Compares every other subfolder against the reference:
       - per-algorithm numerosity mismatches
       - which specific images are missing / extra
  6. Prints a final MATCH / MISMATCH summary table, one line per subfolder.

Usage:
    python reference_consistency_check.py /dataset-disk/tb_dataset/tf2k_lr/social/facebook/
    python reference_consistency_check.py /path/to/social/telegram/ --output telegram_reference.json
"""

import argparse
import json
import os
from collections import defaultdict

REAL_FAKE_NAMES = {"real", "fake"}


def classify_path(root, base_path):
    """
    Given a directory `root` under `base_path`, return (group_key, real_fake, algo_key)
    or None if `root` is not inside a real/fake branch with an algorithm subfolder.

    group_key: everything above the real/fake folder, flattened with '-'
               (e.g. 'seasons/spring-SP01' -> 'seasons-spring-SP01'), or '(root)'
               if real/fake sits directly under base_path.
    real_fake: 'real' or 'fake'
    algo_key:  everything below real/fake, joined with '/' (e.g. 'FORLAB' or
               'FLUX.1/animals').
    """
    rel = os.path.relpath(root, base_path)
    if rel == ".":
        return None
    parts = rel.split(os.sep)

    idx = next((i for i, p in enumerate(parts) if p.lower() in REAL_FAKE_NAMES), None)
    if idx is None:
        return None

    algo_sub_parts = parts[idx + 1:]
    if not algo_sub_parts:
        return None

    group_parts = parts[:idx]
    group_key = "-".join(group_parts) if group_parts else "(root)"
    real_fake = parts[idx].lower()
    algo_key = "/".join(algo_sub_parts)
    return group_key, real_fake, algo_key


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_path", help="Path to the social_name folder, e.g. .../social/facebook/")
    parser.add_argument("--output", default=None,
                         help="Output path for the reference JSON. "
                              "Default: <social_name>_reference.json in the current directory.")
    args = parser.parse_args()

    dataset_path = args.dataset_path.rstrip("/")
    social_name = os.path.basename(dataset_path)

    # group -> set of ids ("algo_key/filename")
    images_by_group = defaultdict(set)
    # group -> algo_key -> count
    counts_by_group = defaultdict(lambda: defaultdict(int))

    total_files = 0
    for root, _dirs, files in os.walk(dataset_path):
        if not files:
            continue
        classified = classify_path(root, dataset_path)
        if classified is None:
            continue
        group_key, _real_fake, algo_key = classified

        for fname in files:
            stem, _ext = os.path.splitext(fname)
            full_id = f"{algo_key}/{stem}"
            total_files += 1
            images_by_group[group_key].add(full_id)
            counts_by_group[group_key][algo_key] += 1

    groups = sorted(images_by_group.keys())
    # print(groups)
    # breakpoint()

    if not groups:
        print(f"WARNING: no real/fake branches with algorithm subfolders found under {dataset_path}. "
              f"Check the path and folder structure.")
        return

    reference_group = groups[5]
    ref_ids = images_by_group[reference_group]
    ref_counts = counts_by_group[reference_group]

    # ---- Save reference list ----
    output = {
        "social_name": social_name,
        "reference_group": reference_group,
        "total_images": len(ref_ids),
        "images": sorted(ref_ids),
    }
    out_path = args.output or f"{social_name}_reference.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # ---- Console report ----
    print(f"Dataset path : {dataset_path}")
    print(f"Social name  : {social_name}")
    print(f"Files scanned: {total_files}")
    print(f"Subfolders found ({len(groups)}): {groups}")
    print(f"Reference subfolder (first one): {reference_group}")
    print(f"Reference total images: {len(ref_ids)}")

    print(f"\n=== Reference numerosity per algorithm/subfolder ('{reference_group}') ===")
    for algo_key in sorted(ref_counts.keys()):
        print(f"  * {algo_key} -> {ref_counts[algo_key]} images")

    # ---- Compare every other subfolder against the reference ----
    grand_missing = 0
    grand_extra = 0
    grand_numerosity_mismatches = 0
    match_status = {}  # group -> bool

    for g in groups:
        if g == reference_group:
            match_status[g] = True
            continue

        g_ids = images_by_group[g]
        g_counts = counts_by_group[g]

        missing_ids = ref_ids - g_ids
        extra_ids = g_ids - ref_ids
        grand_missing += len(missing_ids)
        grand_extra += len(extra_ids)

        is_match = (g_ids == ref_ids)
        match_status[g] = is_match

        print(f"\n=== Subfolder: {g} ===")
        print(f"  total images: {len(g_ids)} (reference: {len(ref_ids)})")

        all_algos = sorted(set(ref_counts.keys()) | set(g_counts.keys()))
        for algo_key in all_algos:
            ref_c = ref_counts.get(algo_key, 0)
            g_c = g_counts.get(algo_key, 0)
            if ref_c != g_c:
                grand_numerosity_mismatches += 1
                print(f"  NUMEROSITY MISMATCH [{algo_key}]: reference {ref_c}, found {g_c}")

        if missing_ids:
            sample = sorted(missing_ids)[:5]
            more = " ..." if len(missing_ids) > 5 else ""
            print(f"  missing ({len(missing_ids)}) e.g.: {sample}{more}")
        if extra_ids:
            sample = sorted(extra_ids)[:5]
            more = " ..." if len(extra_ids) > 5 else ""
            print(f"  extra ({len(extra_ids)}) e.g.: {sample}{more}")
        if is_match:
            print("  OK — identical to reference.")

    # ---- Final subfolder summary table ----
    print(f"\n=== SUBFOLDER SUMMARY (reference: '{reference_group}', {len(ref_ids)} images) ===")
    name_width = max(len(g) for g in groups) + 2
    for g in groups:
        g_ids = images_by_group[g]
        if g == reference_group:
            print(f"  {g:<{name_width}} MATCH   (reference, {len(g_ids)} images)")
        elif match_status[g]:
            print(f"  {g:<{name_width}} MATCH   ({len(g_ids)} images)")
        else:
            missing = len(ref_ids - g_ids)
            extra = len(g_ids - ref_ids)
            print(f"  {g:<{name_width}} MISMATCH  ({len(g_ids)} images; {missing} missing, {extra} extra)")

    print("\n=== FINAL SUMMARY ===")
    print(f"Total missing (present in reference, absent elsewhere): {grand_missing}")
    print(f"Total extra   (present elsewhere, absent from reference): {grand_extra}")
    print(f"Total numerosity mismatches (algorithm/subfolder level): {grand_numerosity_mismatches}")
    all_match = all(match_status.values())
    print(f"All subfolders match reference '{reference_group}': {'YES' if all_match else 'NO'}")
    print(f"Reference JSON written to: {out_path}")


if __name__ == "__main__":
    main()