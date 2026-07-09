#!/usr/bin/env python3
"""
compare_splits_to_reference.py

Checks whether every image listed in a train/val/test split JSON (e.g.
tf2k_SOCIAL_splits.json) is present in a reference image list (e.g.
facebook_reference.json, produced by reference_consistency_check.py).

For each split (train/val/test) it reports:
  - total images, how many match the reference, how many are missing
  - a per-class (algorithm[/subfolder]) breakdown of matches/missing
plus a grand total across all splits.

Usage:
    python compare_splits_to_reference.py \\
        --splits tf2k_SOCIAL_splits.json \\
        --reference facebook_reference.json
"""

import argparse
import json
from collections import defaultdict


def get_class(image_id: str) -> str:
    """'FLUX.1/animals/00025' -> 'FLUX.1/animals'; 'FFHQ/00014' -> 'FFHQ'"""
    return image_id.rsplit("/", 1)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", required=True, help="Path to the train/val/test split JSON")
    parser.add_argument("--reference", required=True, help="Path to the reference JSON (with an 'images' list)")
    args = parser.parse_args()

    with open(args.splits, "r") as f:
        splits = json.load(f)

    with open(args.reference, "r") as f:
        reference = json.load(f)

    reference_ids = set(reference["images"])
    social_name = reference.get("social_name", "?")
    reference_group = reference.get("reference_group", "?")

    print(f"Reference file : {args.reference}")
    print(f"  social_name    : {social_name}")
    print(f"  reference_group: {reference_group}")
    print(f"  total images   : {len(reference_ids)}")
    print(f"Splits file    : {args.splits}")
    print(f"  splits found   : {list(splits.keys())}")

    grand_total = 0
    grand_matched = 0
    grand_missing = 0
    grand_class_totals = defaultdict(lambda: {"matched": 0, "missing": 0})

    for split_name, image_ids in splits.items():
        print(f"\n=== Split: {split_name} ({len(image_ids)} images) ===")

        per_class = defaultdict(lambda: {"matched": 0, "missing": 0})
        missing_ids = []

        for image_id in image_ids:
            cls = get_class(image_id)
            if image_id in reference_ids:
                per_class[cls]["matched"] += 1
                grand_class_totals[cls]["matched"] += 1
            else:
                per_class[cls]["missing"] += 1
                grand_class_totals[cls]["missing"] += 1
                missing_ids.append(image_id)

        split_total = len(image_ids)
        split_matched = split_total - len(missing_ids)
        split_missing = len(missing_ids)

        grand_total += split_total
        grand_matched += split_matched
        grand_missing += split_missing

        print(f"  matched: {split_matched} / {split_total}   missing: {split_missing}")

        print(f"  {'class':<38} {'matched':>8} {'missing':>8}")
        print(f"  {'-'*56}")
        for cls in sorted(per_class.keys()):
            counts = per_class[cls]
            flag = "" if counts["missing"] == 0 else "  <-- MISSING"
            print(f"  {cls:<38} {counts['matched']:>8} {counts['missing']:>8}{flag}")

        if missing_ids:
            sample = sorted(missing_ids)[:10]
            more = " ..." if len(missing_ids) > 10 else ""
            print(f"\n  missing ids e.g.: {sample}{more}")

    # ---- Grand summary across all splits ----
    print(f"\n=== GRAND SUMMARY (all splits combined) ===")
    print(f"  {'class':<38} {'matched':>8} {'missing':>8}")
    print(f"  {'-'*56}")
    for cls in sorted(grand_class_totals.keys()):
        counts = grand_class_totals[cls]
        flag = "" if counts["missing"] == 0 else "  <-- MISSING"
        print(f"  {cls:<38} {counts['matched']:>8} {counts['missing']:>8}{flag}")

    print(f"\nTotal images across all splits : {grand_total}")
    print(f"Total matched in reference      : {grand_matched}")
    print(f"Total missing from reference     : {grand_missing}")
    print(f"All split images found in reference: {'YES' if grand_missing == 0 else 'NO'}")


if __name__ == "__main__":
    main()