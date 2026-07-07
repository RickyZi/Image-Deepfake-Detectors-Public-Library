# """
# sanity_check_split.py

# Cross-checks the images physically present on disk against a train/val/test
# split JSON, for a given social_name folder (facebook, telegram, twitter).

# For every image found under:
#     <dataset_path>/[optional group folders/].../<real|fake>/<algorithm>/[subfolder/]<id>.<ext>

# it builds the id as "<algorithm>[/<subfolder>]/<filename_without_ext>" (matching
# the format used in the split JSON, e.g. "FFHQ/00014" or "FLUX.1/animals/00025")
# and checks whether that id appears in the split JSON's train/val/test lists.

# The dataset can have multiple "groups" above the real/fake folder (e.g. a
# seasons/spring-SP01 style batch structure). Each group is expected to be an
# independent full copy of the dataset, so this script also cross-checks that
# every group has the exact same set of matched/unmatched ids as every other
# group, and reports any group whose composition diverges.

# Output:
#   - A SINGLE overall JSON file "<social_name>_sanity_check.json" (not one per
#     group/subfolder) containing the union, across all groups, of:
#       * "train" / "val" / "test": ids found on disk that match the original split
#       * "unmatched": ids found on disk that do NOT appear in any split of the
#         original JSON
#   - A console report with:
#       * missing images per split (listed in split JSON, not found on disk), per group
#       * extra/unmatched images per group (found on disk, not in split JSON)
#       * per-class (algorithm/subfolder) numerosity mismatches, comparing the
#         expected count (derived from the split JSON) against what's found on disk
#       * a cross-group consistency check: whether all groups share the exact
#         same matched/unmatched id sets

# Usage:
#     python sanity_check_split.py /dataset-disk/tb_dataset/tf2k_lr/social/facebook/ splits.json
#     python sanity_check_split.py /path/to/social/telegram/ splits.json --output telegram_sanity_check.json
# """

# import argparse
# import json
# import os
# from collections import defaultdict

# REAL_FAKE_NAMES = {"real", "fake"}


# def classify_path(root, base_path):
#     """
#     Given a directory `root` under `base_path`, return (group_key, real_fake, algo_key)
#     or None if `root` is not inside a real/fake branch with an algorithm subfolder.

#     group_key: everything above the real/fake folder, flattened with '-'
#                (e.g. 'seasons/spring-SP01' -> 'seasons-spring-SP01'), or '(root)'
#                if real/fake sits directly under base_path.
#     real_fake: 'real' or 'fake'
#     algo_key:  everything below real/fake, joined with '/' (e.g. 'FFHQ' or
#                'StyleGAN/images-psi-0.7'), matching the split JSON's id prefix.
#     """
#     rel = os.path.relpath(root, base_path)
#     if rel == ".":
#         return None
#     parts = rel.split(os.sep)

#     idx = next((i for i, p in enumerate(parts) if p.lower() in REAL_FAKE_NAMES), None)
#     if idx is None:
#         return None

#     algo_sub_parts = parts[idx + 1:]
#     if not algo_sub_parts:
#         return None

#     group_parts = parts[:idx]
#     group_key = "-".join(group_parts) if group_parts else "(root)"
#     real_fake = parts[idx].lower()
#     algo_key = "/".join(algo_sub_parts)
#     return group_key, real_fake, algo_key


# def build_split_lookup(splits):
#     """
#     Returns:
#       id_to_split: dict mapping "algo_key/filename" -> split_name ('train'/'val'/'test')
#       expected_counts: dict[split_name][algo_key] -> expected count of images
#     """
#     id_to_split = {}
#     expected_counts = defaultdict(lambda: defaultdict(int))

#     for split_name, ids in splits.items():
#         for full_id in ids:
#             if full_id in id_to_split:
#                 print(f"WARNING: id '{full_id}' appears in multiple splits "
#                       f"('{id_to_split[full_id]}' and '{split_name}')")
#             id_to_split[full_id] = split_name
#             algo_key = full_id.rsplit("/", 1)[0]
#             expected_counts[split_name][algo_key] += 1

#     return id_to_split, expected_counts


# def main():
#     parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
#     parser.add_argument("dataset_path", help="Path to the social_name folder, e.g. .../social/facebook/")
#     parser.add_argument("split_json", help="Path to the train/val/test split JSON")
#     parser.add_argument("--output", default=None,
#                          help="Output path for the sanity-check JSON. "
#                               "Default: <social_name>_sanity_check.json in the current directory.")
#     args = parser.parse_args()

#     dataset_path = args.dataset_path.rstrip("/")
#     social_name = os.path.basename(dataset_path)

#     with open(args.split_json, "r") as f:
#         splits = json.load(f)

#     id_to_split, expected_counts = build_split_lookup(splits)

#     matched = defaultdict(lambda: defaultdict(list))                       # group -> split_name -> [ids]
#     unmatched = defaultdict(list)                                          # group -> [ids]
#     found_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # group -> split_name -> algo_key -> count

#     total_files = 0
#     for root, _dirs, files in os.walk(dataset_path):
#         if not files:
#             continue
#         classified = classify_path(root, dataset_path)
#         if classified is None:
#             continue
#         group_key, _real_fake, algo_key = classified

#         for fname in files:
#             stem, _ext = os.path.splitext(fname)
#             full_id = f"{algo_key}/{stem}"
#             total_files += 1

#             if full_id in id_to_split:
#                 split_name = id_to_split[full_id]
#                 matched[group_key][split_name].append(full_id)
#                 found_counts[group_key][split_name][algo_key] += 1
#             else:
#                 unmatched[group_key].append(full_id)

#     groups = sorted(set(list(matched.keys()) + list(unmatched.keys())))

#     if not groups:
#         print(f"WARNING: no real/fake branches with algorithm subfolders found under {dataset_path}. "
#               f"Check the path and folder structure.")
#         return

#     # ---- Per-group sets, used both for the consistency check and to build the overall JSON ----
#     group_sets = {}
#     for g in groups:
#         group_sets[g] = {
#             "train": set(matched[g].get("train", [])),
#             "val": set(matched[g].get("val", [])),
#             "test": set(matched[g].get("test", [])),
#             "unmatched": set(unmatched[g]),
#         }

#     # ---- Cross-group consistency check ----
#     # Every group is expected to be an independent full copy of the dataset,
#     # so all groups should share the exact same matched/unmatched id sets.
#     groups_consistent = True
#     if len(groups) > 1:
#         reference_group = groups[0]
#         for category in ["train", "val", "test", "unmatched"]:
#             ref_set = group_sets[reference_group][category]
#             for g in groups[1:]:
#                 if group_sets[g][category] != ref_set:
#                     groups_consistent = False

#     # ---- Build a SINGLE overall JSON (union across all groups) ----
#     output = {
#         "train": sorted(set().union(*(group_sets[g]["train"] for g in groups))),
#         "val": sorted(set().union(*(group_sets[g]["val"] for g in groups))),
#         "test": sorted(set().union(*(group_sets[g]["test"] for g in groups))),
#         "unmatched": sorted(set().union(*(group_sets[g]["unmatched"] for g in groups))),
#     }

#     out_path = args.output or f"{social_name}_sanity_check.json"
#     with open(out_path, "w") as f:
#         json.dump(output, f, indent=2)

#     # ---- Console report ----
#     print(f"Dataset path : {dataset_path}")
#     print(f"Social name  : {social_name}")
#     print(f"Files scanned: {total_files}")
#     print(f"Groups found : {groups}")

#     grand_missing = 0
#     grand_extra = 0
#     grand_mismatched_classes = 0

#     for g in groups:
#         print(f"\n=== Group: {g} ===")
#         for split_name in ["train", "val", "test"]:
#             expected_ids = set(splits.get(split_name, []))
#             found_ids = set(matched[g].get(split_name, []))
#             missing_ids = expected_ids - found_ids
#             grand_missing += len(missing_ids)

#             print(f"  [{split_name}] expected {len(expected_ids)}, found {len(found_ids)}, "
#                   f"missing {len(missing_ids)}")
#             if missing_ids:
#                 sample = sorted(missing_ids)[:5]
#                 more = " ..." if len(missing_ids) > 5 else ""
#                 print(f"      missing e.g.: {sample}{more}")

#             exp_counts = expected_counts.get(split_name, {})
#             fnd_counts = found_counts[g].get(split_name, {})
#             for algo_key in sorted(set(exp_counts.keys()) | set(fnd_counts.keys())):
#                 exp_c = exp_counts.get(algo_key, 0)
#                 fnd_c = fnd_counts.get(algo_key, 0)
#                 if exp_c != fnd_c:
#                     grand_mismatched_classes += 1
#                     print(f"      NUMEROSITY MISMATCH [{split_name}/{algo_key}]: expected {exp_c}, found {fnd_c}")

#         extra_count = len(unmatched[g])
#         grand_extra += extra_count
#         print(f"  [unmatched] {extra_count} image(s) on disk not present in any split")
#         if extra_count:
#             sample = sorted(unmatched[g])[:5]
#             more = " ..." if extra_count > 5 else ""
#             print(f"      extra e.g.: {sample}{more}")

#     # ---- Cross-group consistency report ----
#     print("\n=== CROSS-GROUP CONSISTENCY CHECK ===")
#     if len(groups) <= 1:
#         print(f"Only one group found ({groups[0]}) — nothing to compare across groups.")
#     elif groups_consistent:
#         print(f"OK — all {len(groups)} groups have identical train/val/test/unmatched id sets.")
#     else:
#         print(f"MISMATCH — the {len(groups)} groups do NOT all share the same id sets.")
#         reference_group = groups[0]
#         for category in ["train", "val", "test", "unmatched"]:
#             ref_set = group_sets[reference_group][category]
#             for g in groups[1:]:
#                 g_set = group_sets[g][category]
#                 if g_set != ref_set:
#                     only_in_ref = ref_set - g_set
#                     only_in_g = g_set - ref_set
#                     print(f"  [{category}] '{g}' differs from reference '{reference_group}':")
#                     if only_in_ref:
#                         sample = sorted(only_in_ref)[:5]
#                         more = " ..." if len(only_in_ref) > 5 else ""
#                         print(f"      present in '{reference_group}' but not in '{g}' "
#                               f"({len(only_in_ref)}): {sample}{more}")
#                     if only_in_g:
#                         sample = sorted(only_in_g)[:5]
#                         more = " ..." if len(only_in_g) > 5 else ""
#                         print(f"      present in '{g}' but not in '{reference_group}' "
#                               f"({len(only_in_g)}): {sample}{more}")

#     print("\n=== FINAL SUMMARY ===")
#     print(f"Total missing (listed in split, not found on disk): {grand_missing}")
#     print(f"Total extra   (found on disk, not listed in split): {grand_extra}")
#     print(f"Total class-level numerosity mismatches:            {grand_mismatched_classes}")
#     print(f"Groups consistent with each other: {'YES' if (len(groups) <= 1 or groups_consistent) else 'NO'}")
#     print(f"Sanity-check JSON written to: {out_path}")


# if __name__ == "__main__":
#     main()

"""
sanity_check_split.py

Cross-checks the images physically present on disk against a train/val/test
split JSON, for a given social_name folder (facebook, telegram, twitter).

For every image found under:
    <dataset_path>/[optional group folders/].../<real|fake>/<algorithm>/[subfolder/]<id>.<ext>

it builds the id as "<algorithm>[/<subfolder>]/<filename_without_ext>" (matching
the format used in the split JSON, e.g. "FFHQ/00014" or "FLUX.1/animals/00025")
and checks whether that id appears in the split JSON's train/val/test lists.

The dataset can have multiple "groups" above the real/fake folder (e.g. a
seasons/spring-SP01 style batch structure — or none, if real/fake sits
directly under dataset_path). The FIRST group found (alphabetically) is used
as the REFERENCE: its images are what get matched against the split JSON and
written to the output file. Every other group is then checked against that
same reference — flagging any image present in the reference but missing
from the other group, and any image present in the other group but not in
the reference.

Output:
  - A SINGLE overall JSON file "<social_name>_sanity_check.json" (not one per
    group/subfolder), built from the reference group only:
      * "train" / "val" / "test": reference-group ids that match the original split
      * "unmatched": reference-group ids found on disk that do NOT appear in
        any split of the original JSON
  - A console report with:
      * missing images per split (listed in split JSON, not found on disk), per group
      * extra/unmatched images per group (found on disk, not in split JSON)
      * per-class (algorithm/subfolder) numerosity mismatches, comparing the
        expected count (derived from the split JSON) against what's found on disk
      * a consistency check comparing every other group against the reference
        group, image by image

Usage:
    python sanity_check_split.py /dataset-disk/tb_dataset/tf2k_lr/social/facebook/ splits.json
    python sanity_check_split.py /path/to/social/telegram/ splits.json --output telegram_sanity_check.json
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
    algo_key:  everything below real/fake, joined with '/' (e.g. 'FFHQ' or
               'StyleGAN/images-psi-0.7'), matching the split JSON's id prefix.
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


def build_split_lookup(splits):
    """
    Returns:
      id_to_split: dict mapping "algo_key/filename" -> split_name ('train'/'val'/'test')
      expected_counts: dict[split_name][algo_key] -> expected count of images
    """
    id_to_split = {}
    expected_counts = defaultdict(lambda: defaultdict(int))

    for split_name, ids in splits.items():
        for full_id in ids:
            if full_id in id_to_split:
                print(f"WARNING: id '{full_id}' appears in multiple splits "
                      f"('{id_to_split[full_id]}' and '{split_name}')")
            id_to_split[full_id] = split_name
            algo_key = full_id.rsplit("/", 1)[0]
            expected_counts[split_name][algo_key] += 1

    return id_to_split, expected_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_path", help="Path to the social_name folder, e.g. .../social/facebook/")
    parser.add_argument("split_json", help="Path to the train/val/test split JSON")
    parser.add_argument("--output", default=None,
                         help="Output path for the sanity-check JSON. "
                              "Default: <social_name>_sanity_check.json in the current directory.")
    args = parser.parse_args()

    dataset_path = args.dataset_path.rstrip("/")
    social_name = os.path.basename(dataset_path)

    with open(args.split_json, "r") as f:
        splits = json.load(f)

    id_to_split, expected_counts = build_split_lookup(splits)

    matched = defaultdict(lambda: defaultdict(list))                       # group -> split_name -> [ids]
    unmatched = defaultdict(list)                                          # group -> [ids]
    found_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # group -> split_name -> algo_key -> count

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

            if full_id in id_to_split:
                split_name = id_to_split[full_id]
                matched[group_key][split_name].append(full_id)
                found_counts[group_key][split_name][algo_key] += 1
            else:
                unmatched[group_key].append(full_id)

    groups = sorted(set(list(matched.keys()) + list(unmatched.keys())))

    if not groups:
        print(f"WARNING: no real/fake branches with algorithm subfolders found under {dataset_path}. "
              f"Check the path and folder structure.")
        return

    # ---- Per-group sets ----
    group_sets = {}
    for g in groups:
        group_sets[g] = {
            "train": set(matched[g].get("train", [])),
            "val": set(matched[g].get("val", [])),
            "test": set(matched[g].get("test", [])),
            "unmatched": set(unmatched[g]),
        }

    # ---- Reference group ----
    # The first group found on disk (alphabetically) is used as the reference:
    # its train/val/test/unmatched image sets define what every other group's
    # subfolder is expected to contain.
    reference_group = groups[0]
    ref_sets = group_sets[reference_group]

    # ---- Build the overall JSON from the reference group only ----
    output = {
        "train": sorted(ref_sets["train"]),
        "val": sorted(ref_sets["val"]),
        "test": sorted(ref_sets["test"]),
        "unmatched": sorted(ref_sets["unmatched"]),
    }

    out_path = args.output or f"{social_name}_sanity_check.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # ---- Console report ----
    print(f"Dataset path : {dataset_path}")
    print(f"Social name  : {social_name}")
    print(f"Files scanned: {total_files}")
    print(f"Groups found : {groups}")
    print(f"Reference group (first folder): {reference_group}")

    grand_missing = 0
    grand_extra = 0
    grand_mismatched_classes = 0

    for g in groups:
        print(f"\n=== Group: {g} ===")
        for split_name in ["train", "val", "test"]:
            expected_ids = set(splits.get(split_name, []))
            found_ids = set(matched[g].get(split_name, []))
            missing_ids = expected_ids - found_ids
            grand_missing += len(missing_ids)

            print(f"  [{split_name}] expected {len(expected_ids)}, found {len(found_ids)}, "
                  f"missing {len(missing_ids)}")
            if missing_ids:
                sample = sorted(missing_ids)[:5]
                more = " ..." if len(missing_ids) > 5 else ""
                print(f"      missing e.g.: {sample}{more}")

            exp_counts = expected_counts.get(split_name, {})
            fnd_counts = found_counts[g].get(split_name, {})
            for algo_key in sorted(set(exp_counts.keys()) | set(fnd_counts.keys())):
                exp_c = exp_counts.get(algo_key, 0)
                fnd_c = fnd_counts.get(algo_key, 0)
                if exp_c != fnd_c:
                    grand_mismatched_classes += 1
                    print(f"      NUMEROSITY MISMATCH [{split_name}/{algo_key}]: expected {exp_c}, found {fnd_c}")

        extra_count = len(unmatched[g])
        grand_extra += extra_count
        print(f"  [unmatched] {extra_count} image(s) on disk not present in any split")
        if extra_count:
            sample = sorted(unmatched[g])[:5]
            more = " ..." if extra_count > 5 else ""
            print(f"      extra e.g.: {sample}{more}")

    # ---- Reference-group consistency report ----
    # Every group is expected to be an independent full copy of the dataset,
    # so every other group should contain the exact same images (by id) as
    # the reference group, split-for-split.
    print(f"\n=== CONSISTENCY CHECK vs. REFERENCE GROUP '{reference_group}' ===")
    if len(groups) <= 1:
        print(f"Only one group found ({reference_group}) — nothing to compare it against.")
        groups_consistent = True
    else:
        groups_consistent = True
        for g in groups:
            if g == reference_group:
                continue
            g_sets = group_sets[g]
            group_ok = True
            for category in ["train", "val", "test", "unmatched"]:
                ref_set = ref_sets[category]
                g_set = g_sets[category]
                if g_set != ref_set:
                    group_ok = False
                    groups_consistent = False
                    only_in_ref = ref_set - g_set
                    only_in_g = g_set - ref_set
                    print(f"  [{category}] '{g}' differs from reference '{reference_group}':")
                    if only_in_ref:
                        sample = sorted(only_in_ref)[:5]
                        more = " ..." if len(only_in_ref) > 5 else ""
                        print(f"      missing in '{g}' (present in reference) "
                              f"({len(only_in_ref)}): {sample}{more}")
                    if only_in_g:
                        sample = sorted(only_in_g)[:5]
                        more = " ..." if len(only_in_g) > 5 else ""
                        print(f"      extra in '{g}' (not in reference) "
                              f"({len(only_in_g)}): {sample}{more}")
            if group_ok:
                print(f"  '{g}' matches reference '{reference_group}' exactly. OK")

    print("\n=== FINAL SUMMARY ===")
    print(f"Total missing (listed in split, not found on disk): {grand_missing}")
    print(f"Total extra   (found on disk, not listed in split): {grand_extra}")
    print(f"Total class-level numerosity mismatches:            {grand_mismatched_classes}")
    print(f"All groups consistent with reference '{reference_group}': {'YES' if groups_consistent else 'NO'}")
    print(f"Sanity-check JSON written to: {out_path}")


if __name__ == "__main__":
    main()