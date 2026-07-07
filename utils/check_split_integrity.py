#!/usr/bin/env python3
"""
check_split_integrity.py

Verifies that every image referenced in a train/val/test split JSON file
actually exists on disk, inside a dataset organized as:

    <root>/
        <social_name>/           e.g. facebook, telegram, twitter
            <Real_or_Fake>/      e.g. Real, Fake
                <algorithm>/     e.g. FFHQ, FLUX.1, StyleGAN2, ...
                    [<subfolder>/]   optional, e.g. animals, faces, images-psi-0.5, conf-f-psi-1
                        <image_id>.<ext>

Each entry in the split JSON looks like "FFHQ/00014" (algorithm/id) or
"FLUX.1/animals/00025" (algorithm/subfolder/id). The script matches this
against every (social_name, Real_or_Fake) combination it finds under root,
since the same set of source images is expected to be replicated across
each social platform / real-fake branch (e.g. re-uploaded/re-compressed
copies).

Usage:
    python check_split_integrity.py /path/to/dataset_root splits.json \
        --splits test \
        [--extensions png jpg jpeg] \
        [--report missing_report.json]

If --splits is omitted, all splits found in the JSON (train/val/test) are
checked. Multiple splits can be given, e.g. --splits train val test.
"""

import argparse
import json
import os
import sys
from collections import defaultdict


def build_present_index(root, valid_extensions=None):
    """
    Walk the dataset root and build a mapping:

        (social_name, real_or_fake) -> set of "algorithm[/subfolder]/id" keys

    Assumes the fixed depth: root/social/real_or_fake/algorithm/[subfolder/]file
    i.e. algorithm always sits at depth 2 below root (0-indexed: social=0,
    real_or_fake=1, algorithm=2, then 0+ subfolder levels, then the file).
    """
    index = defaultdict(set)
    all_files_seen = 0

    for dirpath, _dirnames, filenames in os.walk(root):
        if not filenames:
            continue
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == ".":
            continue
        parts = rel_dir.split(os.sep)

        # Need at least social/real_or_fake/algorithm
        if len(parts) < 3:
            continue

        social, real_fake = parts[0], parts[1]
        algo_and_sub = parts[2:]  # algorithm + optional subfolder(s)

        for fname in filenames:
            stem, ext = os.path.splitext(fname)
            ext = ext.lower().lstrip(".")
            if valid_extensions and ext not in valid_extensions:
                continue
            all_files_seen += 1
            key = "/".join(algo_and_sub + [stem])
            index[(social, real_fake)].add(key)

    return index, all_files_seen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="Path to the dataset root directory")
    parser.add_argument("split_json", default= './tf2k_SOCIAL_splits.json', help="Path to the split JSON file (train/val/test lists)")
    parser.add_argument("--splits", nargs="+", default=None, help="Which split(s) to check, e.g. test  or  train val test. Default: all present in JSON.")
    parser.add_argument("--extensions", nargs="+", default=None, help="Restrict scanned files to these extensions (without dot), e.g. png jpg. Default: all files.")
    parser.add_argument("--report", default=None, help="Optional path to write a JSON report of missing files.")
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        print(f"ERROR: root directory does not exist: {args.root}", file=sys.stderr)
        sys.exit(1)

    with open(args.split_json, "r") as f:
        splits = json.load(f)

    splits_to_check = args.splits if args.splits else list(splits.keys())
    for s in splits_to_check:
        if s not in splits:
            print(f"ERROR: split '{s}' not found in JSON (available: {list(splits.keys())})", file=sys.stderr)
            sys.exit(1)

    valid_ext = set(e.lower().lstrip(".") for e in args.extensions) if args.extensions else None

    print(f"Scanning dataset under: {args.root}")
    index, total_files = build_present_index(args.root, valid_ext)

    contexts = sorted(index.keys())
    if not contexts:
        print("WARNING: no (social_name/Real_or_Fake) contexts found — check the --root path and folder depth.")
    else:
        print(f"Found {total_files} candidate image files across {len(contexts)} (social_name, Real_or_Fake) contexts:")
        for c in contexts:
            print(f"  - {c[0]} / {c[1]}: {len(index[c])} unique image IDs")

    report = {}
    grand_total_missing = 0

    for split_name in splits_to_check:
        ids = splits[split_name]
        print(f"\n=== Checking split '{split_name}' ({len(ids)} entries) ===")
        split_report = {}

        for context in contexts:
            present_ids = index[context]
            missing = [img_id for img_id in ids if img_id not in present_ids]
            found_count = len(ids) - len(missing)
            print(f"  [{context[0]}/{context[1]}] found {found_count}/{len(ids)}"
                  + (f"  -> {len(missing)} MISSING" if missing else "  -> OK"))
            if missing:
                split_report["/".join(context)] = missing
                grand_total_missing += len(missing)

        if not contexts:
            # Fall back to a flat check directly against root (no social/realfake nesting)
            flat_present = set()
            for dirpath, _dirnames, filenames in os.walk(args.root):
                rel_dir = os.path.relpath(dirpath, args.root)
                if rel_dir == ".":
                    continue
                for fname in filenames:
                    stem, ext = os.path.splitext(fname)
                    ext = ext.lower().lstrip(".")
                    if valid_ext and ext not in valid_ext:
                        continue
                    flat_present.add(f"{rel_dir.replace(os.sep, '/')}/{stem}")
            missing = [img_id for img_id in ids if img_id not in flat_present]
            found_count = len(ids) - len(missing)
            print(f"  [flat root] found {found_count}/{len(ids)}"
                  + (f"  -> {len(missing)} MISSING" if missing else "  -> OK"))
            if missing:
                split_report["flat_root"] = missing
                grand_total_missing += len(missing)

        report[split_name] = split_report

    print(f"\nTotal missing entries across all checked splits/contexts: {grand_total_missing}")

    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Detailed missing-file report written to: {args.report}")


if __name__ == "__main__":
    main()