#!/usr/bin/env python3
"""
copy_reference_images.py
──────────────────────────
Reads reference JSON(s) (as produced by reference_consistency_check.py) —
each one listing the ~2k image ids that make up the canonical reference set
for one social platform — and copies just those images out of the FULL
source dataset into a new folder, preserving the structure:

    <output_dir>/<Social>/<Real|Fake>/<algorithm>/[<sub>/]<filename>

Source of each image is located by walking:
    <base_root>/<Social>/<real|fake>/<algorithm>/[<sub>/]<filename>.<ext>
(extension is auto-detected, since the reference JSON only stores ids
without extension, e.g. "FORLAB/00046")

After copying, a verification step checks:
  • every file exists at its destination
  • file size matches the source (fast) or md5 matches (default, exact)
  • per-algorithm[/subfolder] image counts on disk match a HARDCODED set of
    expected numerosities (see EXPECTED_COUNTS below)

Usage
─────
python copy_reference_images.py \\
    --references_dir ./references \\
    --base_root  /datasets-disk/tb_dataset/DeepShield_social \\
    --output_dir /path/to/output

# or point at individual files
python copy_reference_images.py \\
    --references facebook_reference.json telegram_reference.json \\
    --base_root  /datasets-disk/tb_dataset/DeepShield_social \\
    --output_dir /path/to/output
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
REAL_FAKE_NAMES = {"real", "fake"}

# ── hardcoded expected numerosity per algorithm[/subfolder] ──────────────────
# (as printed for the 'seasons-autumn_TM01' reference subfolder)
EXPECTED_COUNTS = {
    "FFHQ": 500,
    "FLUX.1/animals": 39,
    "FLUX.1/faces": 39,
    "FLUX.1/general": 39,
    "FLUX.1/landscapes": 39,
    "FORLAB": 500,
    "StableDiffusion1.5/animals": 39,
    "StableDiffusion1.5/faces": 39,
    "StableDiffusion1.5/general": 39,
    "StableDiffusion1.5/landscapes": 39,
    "StableDiffusion2/animals": 39,
    "StableDiffusion2/faces": 39,
    "StableDiffusion2/general": 39,
    "StableDiffusion2/landscapes": 39,
    "StableDiffusion3/animals": 38,
    "StableDiffusion3/faces": 38,
    "StableDiffusion3/general": 38,
    "StableDiffusion3/landscapes": 38,
    "StableDiffusionXL/animals": 38,
    "StableDiffusionXL/faces": 38,
    "StableDiffusionXL/general": 38,
    "StableDiffusionXL/landscapes": 38,
    "StyleGAN/images-psi-0.5": 38,
    "StyleGAN/images-psi-0.7": 38,
    "StyleGAN2/conf-f-psi-0.5": 38,
    "StyleGAN2/conf-f-psi-1": 38,
    "StyleGAN3/conf-t-psi-0.5": 38,
    "StyleGAN3/conf-t-psi-0.7": 38,
}


# ── helpers ──────────────────────────────────────────────────────────────────

def get_label_dir(branch: str) -> str:
    return "Real" if branch == "real" else "Fake"


def get_subfolder(image_id: str) -> str:
    """'FLUX.1/animals/00025' -> 'FLUX.1/animals'; 'FFHQ/00014' -> 'FFHQ'"""
    return image_id.rsplit("/", 1)[0]


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def mark(ok: bool) -> str:
    return "✅" if ok else "❌"


# ── index the full source dataset for one social ─────────────────────────────

def index_social_images(social_root: Path) -> dict:
    """
    Walk <base_root>/<social> once and build:
        id -> (filepath, branch)   where id = "<algo>[/<sub>]/<filename_no_ext>"
    """
    index = {}
    for root, _dirs, files in os.walk(social_root):
        if not files:
            continue
        rel = os.path.relpath(root, social_root)
        if rel == ".":
            continue
        parts = rel.split(os.sep)

        idx = next((i for i, p in enumerate(parts) if p.lower() in REAL_FAKE_NAMES), None)
        if idx is None:
            continue

        algo_sub_parts = parts[idx + 1:]
        if not algo_sub_parts:
            continue

        branch = parts[idx].lower()
        algo_key = "/".join(algo_sub_parts)

        for fname in files:
            stem, _ext = os.path.splitext(fname)
            full_id = f"{algo_key}/{stem}"
            index[full_id] = (Path(root) / fname, branch)

    return index


def resolve_dest(image_id: str, branch: str, src_path: Path, social: str, output_dir: Path) -> Path:
    label_dir = get_label_dir(branch)
    subfolder = get_subfolder(image_id)
    filename = src_path.name
    return output_dir / social / label_dir / subfolder / filename


# ── copy ─────────────────────────────────────────────────────────────────────

def copy_reference(social: str, image_ids: list,
                    base_root: Path, output_dir: Path,
                    verify_mode: str, verify_only: bool) -> dict:
    """Copy (or verify) all reference images for one social. Returns result dict."""

    n_ok = 0
    n_miss_src = 0
    n_errors = 0
    pairs = []  # (src, dest, subfolder) — src/dest are None if not found on disk

    social_root = base_root / social
    print(f"\n  Indexing source dataset …", end=" ", flush=True)
    index = index_social_images(social_root)
    print(f"{len(index):,} images indexed.")

    print(f"  Resolving {len(image_ids):,} reference images …", end=" ", flush=True)
    for image_id in image_ids:
        subfolder = get_subfolder(image_id)
        entry = index.get(image_id)
        if entry is None:
            n_miss_src += 1
            pairs.append((None, None, subfolder))
            continue
        src_path, branch = entry
        dest = resolve_dest(image_id, branch, src_path, social, output_dir)
        pairs.append((src_path, dest, subfolder))
    print("done.")

    if verify_only:
        print(f"  --verify_only: skipping copy.")
    else:
        print(f"  Copying {len(pairs):,} files …", end=" ", flush=True)
        for src, dest, _sub in pairs:
            if src is None:
                continue  # already counted as missing source
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                n_ok += 1
            except Exception as e:
                print(f"\n  ERROR {src} → {dest}: {e}")
                n_errors += 1
        print(f"done.  copied={n_ok:,}  missing_src={n_miss_src:,}  errors={n_errors:,}")

    # ── verification ─────────────────────────────────────────────────────────
    n_ver_ok = 0
    n_ver_fail = 0
    failed = []

    if verify_mode != "none":
        print(f"  Verifying ({verify_mode}) …", end=" ", flush=True)
        for src, dest, _sub in pairs:
            if src is None:
                continue  # already counted as missing source
            if not dest.is_file():
                n_ver_fail += 1
                failed.append((src, dest, "dest missing"))
                continue
            if verify_mode == "size":
                ok = src.stat().st_size == dest.stat().st_size
                reason = "size mismatch"
            else:
                ok = md5sum(src) == md5sum(dest)
                reason = "checksum mismatch"
            if ok:
                n_ver_ok += 1
            else:
                n_ver_fail += 1
                failed.append((src, dest, reason))
        print(f"done.  ok={n_ver_ok:,}  failed={n_ver_fail:,}")
        if failed:
            print(f"\n  Verification failures:")
            for src, dest, reason in failed[:20]:
                print(f"    [{reason}]  {src.name}")
            if len(failed) > 20:
                print(f"    … and {len(failed)-20} more")

    # ── numerosity check (hardcoded expected counts) ─────────────────────────
    act_counts: dict = defaultdict(int)
    for sub in EXPECTED_COUNTS:
        for label_dir in ("Real", "Fake"):
            folder = output_dir / social / label_dir / sub
            if folder.is_dir():
                act_counts[sub] += sum(
                    1 for f in folder.iterdir()
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTS
                )

    mismatches = [
        (sub, act_counts[sub], exp)
        for sub, exp in EXPECTED_COUNTS.items()
        if act_counts[sub] != exp
    ]
    num_ok = len(mismatches) == 0

    act_total = sum(act_counts.values())
    exp_total = sum(EXPECTED_COUNTS.values())
    tot_ok = act_total == exp_total

    print(f"\n  Numerosity  {mark(num_ok and tot_ok)}")
    print(f"    Total  : {act_total:>6,} / {exp_total:>6,}  {mark(tot_ok)}")

    if mismatches:
        print(f"    {'subfolder':<38}  {'actual':>7}  {'expect':>7}")
        print(f"    {'─'*55}")
        for sub, act, exp in sorted(mismatches):
            print(f"    {sub:<38}  {act:>7,}  {exp:>7,}  ❌")

    all_ok = (
        n_miss_src == 0 and
        n_errors == 0 and
        n_ver_fail == 0 and
        num_ok and tot_ok
    )

    return {
        "social": social,
        "n_copied": n_ok,
        "n_miss_src": n_miss_src,
        "n_errors": n_errors,
        "n_ver_fail": n_ver_fail,
        "num_ok": num_ok and tot_ok,
        "all_ok": all_ok,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--references_dir", type=Path, default='',
        help="Directory containing <social>_reference.json files",
    )
    parser.add_argument(
        "--references", nargs="+", type=Path,
        help="Explicit list of reference JSON files",
    )
    parser.add_argument("--base_root", default='/datasets-disk/tb_dataset/DeepShield_social/', type=Path,
                        help="Full dataset root (contains Facebook/, Telegram/, Twitter/)")
    parser.add_argument("--output_dir", type=Path, default='./reference_socials',
                        help="Destination root for copied images")
    parser.add_argument(
        "--verify", choices=["checksum", "size", "none"], default="checksum",
        help="Verification mode after copying  (default: checksum / md5)",
    )
    parser.add_argument(
        "--verify_only", action="store_true",
        help="Skip copying; only verify existing files and numerosity",
    )
    args = parser.parse_args()

    if not args.base_root.is_dir():
        sys.exit(f"[error] base_root not found: {args.base_root}")

    # Collect reference files
    if args.references_dir:
        if not args.references_dir.is_dir():
            sys.exit(f"[error] references_dir not found: {args.references_dir}")
        reference_files = sorted(args.references_dir.glob("*_reference.json"))
        if not reference_files:
            sys.exit(f"[error] No *_reference.json files in {args.references_dir}")
    else:
        reference_files = args.references
        if not reference_files:
            sys.exit("[error] Provide --references_dir or --references")
        for p in reference_files:
            if not p.is_file():
                sys.exit(f"[error] Reference file not found: {p}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*65}")
    print(f"  base_root  : {args.base_root}")
    print(f"  output_dir : {args.output_dir}")
    print(f"  verify     : {args.verify}")
    print(f"  reference files: {[p.name for p in reference_files]}")
    print(f"{'═'*65}")

    results = []

    for reference_file in reference_files:
        with open(reference_file) as f:
            d = json.load(f)

        social = d["social_name"]
        image_ids = d["images"]

        print(f"\n{'═'*65}")
        print(f"  SOCIAL: {social.upper()}  ({len(image_ids):,} reference images)")
        print(f"{'═'*65}")

        social_src = args.base_root / social
        if not social_src.is_dir():
            print(f"  ⚠️  Source directory not found: {social_src}  — skipping.")
            results.append({"social": social, "all_ok": False,
                            "error": "source dir missing"})
            continue

        res = copy_reference(
            social, image_ids, args.base_root, args.output_dir,
            args.verify, args.verify_only,
        )
        results.append(res)

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  SUMMARY")
    print(f"{'═'*65}")
    print(f"  {'Social':<12}  {'copied':>7}  {'miss_src':>8}  "
          f"{'err':>5}  {'ver_fail':>8}  {'numerosity':>10}  status")
    print(f"  {'─'*63}")

    overall_ok = True
    for r in results:
        if "error" in r:
            print(f"  {r['social']:<12}  {'—':>7}  {'—':>8}  {'—':>5}  "
                  f"{'—':>8}  {'—':>10}  ❌  {r['error']}")
            overall_ok = False
            continue
        ok = r["all_ok"]
        overall_ok &= ok
        print(
            f"  {r['social']:<12}  {r['n_copied']:>7,}  {r['n_miss_src']:>8,}  "
            f"{r['n_errors']:>5,}  {r['n_ver_fail']:>8,}  "
            f"{mark(r['num_ok']):>10}  {mark(ok)}"
        )

    print(f"\n  {'✅  All done.' if overall_ok else '❌  Issues found — see details above.'}")
    print(f"{'═'*65}\n")

    if not overall_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()