#!/usr/bin/env python3
"""Pick one random image per algorithm (real + fake) from a TF2K split JSON
and copy them into a new folder, renamed as
    [algo_name]_[algo_version]_[sub_category]_[image_name]

Usage:
    python sample_tf2k.py tf2k.json
    python sample_tf2k.py tf2k.json --out tf2k_rand_samples --seed 42
    python sample_tf2k.py tf2k.json --per-subfolder   # one per fake subfolder
"""
import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

# "StableDiffusion1.5" -> (StableDiffusion, 1.5), "StableDiffusionXL" -> (StableDiffusion, XL),
# "FLUX.1" -> (FLUX, 1), "StyleGAN2" -> (StyleGAN, 2), "StyleGAN" -> (StyleGAN, "")
ALGO_RE = re.compile(r"^([A-Za-z]+?)[._]?(\d+(?:\.\d+)?|XL)?$")


def parse_fake(split_entry):
    """'StableDiffusion1.5/faces/01234' -> (algo, version, subcat)."""
    parts = split_entry.split("/")
    algo_dir, subcat = parts[0], "/".join(parts[1:-1])
    m = ALGO_RE.match(algo_dir)
    name, version = (m.group(1), m.group(2) or "") if m else (algo_dir, "")
    return name, version, subcat.replace("/", "-")


def parse_real(split_entry):
    """'FORLAB/03487' -> (FORLAB, '', '')."""
    return split_entry.split("/")[0], "", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_file")
    ap.add_argument("--out", default="tf2k_rand_samples")
    ap.add_argument("--seed", type=int, default=42,
                    help="default: seed stored in the JSON, else 42")
    ap.add_argument("--per-subfolder", action="store_false",
                    help="one fake image per subfolder instead of per algo+version")
    args = ap.parse_args()

    data = json.loads(Path(args.json_file).read_text())
    rng = random.Random(args.seed if args.seed is not None else data.get("seed", 42))

    groups = defaultdict(list)  # group key -> [(sample, algo, version, subcat)]
    for s in data["samples"]:
        if s["label"] == "real":
            algo, ver, sub = parse_real(s["split_entry"])
            key = ("real", algo)
        else:
            algo, ver, sub = parse_fake(s["split_entry"])
            key = ("fake", algo, ver, sub) if args.per_subfolder else ("fake", algo, ver)
        groups[key].append((s, algo, ver, sub))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for key in sorted(groups):
        s, algo, ver, sub = rng.choice(groups[key])
        src = Path(s["path"])
        new_name = "_".join(p for p in (algo, ver, sub, src.name) if p)
        if not src.exists():
            print(f"[MISSING] {src}")
            continue
        shutil.copy2(src, out / new_name)
        print(f"{s['label']:4s} {src}  ->  {new_name}")

    print(f"\nDone: {len(list(out.iterdir()))} files in {out.resolve()}")


if __name__ == "__main__":
    main()