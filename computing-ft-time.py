"""
Collect fine-tuning elapsed times from train.log files and save them as JSON.

Expected layout (one train.log per preset folder):
    <root>/detectors/<detector>/checkpoint/<config>/<weights_dir>/<preset>/train.log
e.g.
    detectors/CLIP-D/checkpoint/lora_r4_qv/ft_weights/style_vintage_VN01/train.log
    detectors/R50_nodown/checkpoint/pretrained/ft_unfreezeL4_weights/style_film_inspired_coolbw_unfreezeL4/train.log

  preset      = folder that holds the log (a trailing "_unfreezeL4" is stripped so
                names match across detectors; the raw name is kept as preset_folder)
  config      = first folder under checkpoint/   (e.g. lora_r4_qv, pretrained)
  weights_dir = folder between config and preset (e.g. ft_weights)

Log choice per preset folder: the train_v<N>.log with the highest N if any exist,
otherwise train.log.

Elapsed time per log:
    1. "RUN END  elapsed=<N>s" if present (summed if it appears several times).
    2. Otherwise: last timestamp - first timestamp in the log.

Detector folders without checkpoint/ are skipped. Preset folders without a log are
kept in the summary with elapsed_s = null.

Output folder (default: detectors_training_elapsed_times/):
    runs.json                    flat list, one entry per preset folder / log
    summary_per_detector.json    detector -> stats, per-config stats, and preset list
    summary_per_preset.json      preset -> pooled stats across detectors
    overall_summary.json         stats over every log found
    summary_common_presets.json  detectors compared ONLY on the presets that have a
                                 valid time for every detector (same-preset comparison)

Usage:
    cd /second-disk/Image-Deepfake-Detectors-Public-Library && python collect_training_times.py
    (or from the detectors/ folder; --root PATH is optional)
"""
import argparse
import json
import re
import statistics as st
from datetime import datetime
from pathlib import Path
import shutil

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d{3})?)")
# RUN_END_RE = re.compile(r"RUN END\s+elapsed=(\d+(?:\.\d+)?)s")
LOG_RE = re.compile(r"^train(?:_v(\d+))?\.log$")  # train.log or train_v<N>.log
DETECTORS = ["CLIP-D", "R50_nodown", "R50_TF"]  # empty list -> scan all detectors
# longest first: R50_nodown uses "_unfreezeL4", R50_TF uses "_ft_unfreezeL4"
STRIP_SUFFIXES = ("_ft_unfreezeL4", "_unfreezeL4")
# folder names that are not presets (matched case-insensitively on the raw folder name)
IGNORE_DIRS = ["TB", "social"]
# fix misspelled preset folders so they merge with the correct name
PRESET_ALIASES = {"style_cinemantic2_CN11": "style_cinematic2_CN11"}

# Target folder requirements per detector: detector -> (config_name | None, weights_dir_name | None)
TARGET_PATHS = {
    "CLIP-D": ("lora_r4_qv", "ft_weights"),
    "R50_nodown": (None, "ft_unfreezeL4_weights"),
    "R50_TF": (None, "ft_unfreezeL4_weights"),
}


def parse_ts(s):
    fmt = "%Y-%m-%d %H:%M:%S,%f" if "," in s else "%Y-%m-%d %H:%M:%S"
    return datetime.strptime(s, fmt)


def elapsed_from_log(path):
    """Return (elapsed_seconds | None, source)."""
    first_ts = last_ts = None
    total, n_end = 0.0, 0
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                m = TS_RE.match(line)
                if m:
                    last_ts = m.group(1)
                    if first_ts is None:
                        first_ts = last_ts
                # m = RUN_END_RE.search(line)
                # if m:
                #     total += float(m.group(1))
                #     n_end += 1
    except OSError as e:
        return None, f"unreadable ({e})"

    if n_end:
        return total, "run_end"
    if first_ts and last_ts and first_ts != last_ts:
        return (parse_ts(last_ts) - parse_ts(first_ts)).total_seconds(), "timestamps"
    return None, "no_usable_timestamps"


def pick_log(pdir):
    """Latest log in a preset folder: highest train_v<N>.log, else train.log, else None."""
    versioned, plain = [], None
    for f in pdir.iterdir():
        m = LOG_RE.match(f.name)
        if m and f.is_file():
            if m.group(1):
                versioned.append((int(m.group(1)), f))
            else:
                plain = f
    return max(versioned)[1] if versioned else plain


def clean_preset(name):
    for suf in STRIP_SUFFIXES:
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    return PRESET_ALIASES.get(name, name)


def stats(values):
    """mean / median / sample variance / std, in seconds and hours."""
    values = [v for v in values if v is not None]
    if not values:
        return dict(n=0, mean_s=None, median_s=None, variance_s2=None, std_s=None,
                    mean_h=None, median_h=None)
    var = st.variance(values) if len(values) > 1 else None
    return dict(
        n=len(values),
        mean_s=round(st.mean(values), 2),
        median_s=round(st.median(values), 2),
        variance_s2=round(var, 2) if var is not None else None,
        std_s=round(var ** 0.5, 2) if var is not None else None,
        mean_h=round(st.mean(values) / 3600, 4),
        median_h=round(st.median(values) / 3600, 4),
    )


def collect(root, detectors, ignore):
    det_root = root / "detectors"
    if not det_root.is_dir():
        det_root = root  # allow pointing straight at .../detectors

    det_dirs = ([det_root / d for d in detectors] if detectors
                else sorted(p for p in det_root.iterdir() if p.is_dir() and p.name.lower() not in ignore))

    runs = []
    for det_dir in det_dirs:
        ckpt = det_dir / "checkpoint"
        if not ckpt.is_dir():
            print(f"[skip] no checkpoint folder: {det_dir}")
            continue

        target_cfg, target_wdir = TARGET_PATHS.get(det_dir.name, (None, None))

        # every folder holding a log, plus config/weights_dir/preset folders without one
        preset_dirs = {p.parent for p in ckpt.rglob("train*.log") if LOG_RE.match(p.name)}
        for cfg in (p for p in ckpt.iterdir() if p.is_dir() and p.name.lower() not in ignore):
            for wdir in (p for p in cfg.iterdir() if p.is_dir() and p.name.lower() not in ignore):
                preset_dirs.update(p for p in wdir.iterdir() if p.is_dir() and p.name.lower() not in ignore)

        for pdir in sorted(preset_dirs):
            # Check if any folder in the path relative to checkpoint matches IGNORE_DIRS
            rel_parts = pdir.relative_to(ckpt).parts
            if any(part.lower() in ignore for part in rel_parts):
                print(f"[ignore] {pdir}")
                continue

            # Ensure depth is at least <config>/<weights_dir>/<preset>
            if len(rel_parts) < 3:
                continue

            cfg_name, wdir_name = rel_parts[0], rel_parts[1]

            # Filter folders matching required config and weights_dir rules for detector
            if target_cfg and cfg_name != target_cfg:
                continue
            if target_wdir and wdir_name != target_wdir:
                continue

            log = pick_log(pdir)
            if log is not None:
                secs, src = elapsed_from_log(log)
                if secs is None:
                    print(f"[warn] {log}: {src}")
            else:
                secs, src = None, "log_missing"
                print(f"[skip] no train.log / train_v*.log in {pdir}")
            runs.append(dict(
                detector=det_dir.name,
                config=cfg_name,
                weights_dir=wdir_name,
                preset=clean_preset(pdir.name),
                preset_folder=pdir.name,
                log_found=log is not None,
                elapsed_s=secs,
                elapsed_h=round(secs / 3600, 4) if secs is not None else None,
                source=src,
                log=str(log) if log is not None else None,
            ))

    # presets seen for other detectors but with no folder at all for this one -> null
    all_presets = {r["preset"] for r in runs}
    for det in dict.fromkeys(r["detector"] for r in runs):
        have = {r["preset"] for r in runs if r["detector"] == det}
        for preset in sorted(all_presets - have):
            print(f"[missing] {det}: no folder for preset {preset}")
            runs.append(dict(
                detector=det, config=None, weights_dir=None, preset=preset,
                preset_folder=None, log_found=False, elapsed_s=None, elapsed_h=None,
                source="folder_missing", log=None,
            ))
    return runs


def build_summaries(runs):
    per_detector = {}
    for det in dict.fromkeys(r["detector"] for r in runs):
        d_runs = [r for r in runs if r["detector"] == det]
        per_config = {}
        for cfg in dict.fromkeys(r["config"] for r in d_runs if r["config"]):
            per_config[cfg] = stats([r["elapsed_s"] for r in d_runs if r["config"] == cfg])
        per_detector[det] = dict(
            stats=stats([r["elapsed_s"] for r in d_runs]),
            stats_per_config=per_config,
            presets=[dict(preset=r["preset"], config=r["config"],
                          weights_dir=r["weights_dir"], elapsed_s=r["elapsed_s"],
                          elapsed_h=r["elapsed_h"], log_found=r["log_found"],
                          source=r["source"])
                     for r in sorted(d_runs, key=lambda r: (r["config"] or "", r["preset"]))],
        )

    by_preset = {}
    for r in runs:
        by_preset.setdefault(r["preset"], []).append(r["elapsed_s"])
    per_preset = {p: stats(v) for p, v in sorted(by_preset.items())}
    overall = stats([r["elapsed_s"] for r in runs])
    return per_detector, per_preset, overall


def build_common(runs):
    """Compare detectors only on presets with a valid time for EVERY detector."""
    dets = list(dict.fromkeys(r["detector"] for r in runs))
    times = {d: {} for d in dets}  # detector -> preset -> [elapsed_s, ...]
    for r in runs:
        if r["elapsed_s"] is not None:
            times[r["detector"]].setdefault(r["preset"], []).append(r["elapsed_s"])

    common = sorted(set.intersection(*(set(t) for t in times.values()))) if dets else []
    all_presets = sorted({r["preset"] for r in runs})
    excluded = {p: [d for d in dets if p not in times[d]] for p in all_presets if p not in common}

    per_detector = {d: stats([st.mean(times[d][p]) for p in common]) for d in dets}
    per_preset = {p: {d: round(st.mean(times[d][p]), 2) for d in dets} for p in common}
    return dict(
        detectors=dets,
        n_common_presets=len(common),
        common_presets=common,
        stats_per_detector=per_detector,
        elapsed_s_per_preset=per_preset,
        excluded_presets=excluded,  # preset -> detectors with no valid time
    )


def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."),
                    help="repo root containing detectors/ (default: current folder); "
                         "the detectors/ folder itself also works")
    ap.add_argument("--out", type=Path, default=Path("detectors_training_elapsed_times"))
    ap.add_argument("--detectors", nargs="*", default=DETECTORS,
                    help="detector folder names; pass none to scan all")
    ap.add_argument("--ignore", nargs="*", default=IGNORE_DIRS,
                    help="folder names to ignore (not presets)")
    args = ap.parse_args()

    # Delete existing output directory if it exists
    if args.out.exists():
        shutil.rmtree(args.out)

    runs = collect(args.root, args.detectors, {n.lower() for n in args.ignore})
    per_detector, per_preset, overall = build_summaries(runs)
    common = build_common(runs)

    args.out.mkdir(parents=True, exist_ok=True)
    dump(runs, args.out / "runs.json")
    dump(per_detector, args.out / "summary_per_detector.json")
    dump(per_preset, args.out / "summary_per_preset.json")
    dump(overall, args.out / "overall_summary.json")
    dump(common, args.out / "summary_common_presets.json")

    found = [r for r in runs if r["log_found"]]
    print(f"\n{len(found)} logs found, {len(runs) - len(found)} preset folders without a log "
          f"({sum(r['source'] == 'timestamps' for r in runs)} via timestamp fallback)")
    print("\nAll available presets per detector:")
    for det, d in per_detector.items():
        s = d["stats"]
        print(f"  {det:12s} n={s['n']:3d}  mean={s['mean_h']}h  median={s['median_h']}h")

    print(f"\nSame-preset comparison ({common['n_common_presets']} presets with a valid time "
          f"for all detectors):")
    for det, s in common["stats_per_detector"].items():
        if s["n"]:
            print(f"  {det:12s} n={s['n']:3d}  mean={s['mean_s'] / 60:.1f} min  "
                  f" std ={s['std_s'] / 60:.1f}  median={s['median_s'] / 60:.1f} min  min var={s['variance_s2']} s2")
    print(f"  ({len(common['excluded_presets'])} presets excluded: missing for at least one detector)")
    print(f"Saved JSON files to {args.out.resolve()}")


if __name__ == "__main__":
    main()

## ----------------------------- ##
# 30 logs found, 0 preset folders without a log (30 via timestamp fallback)

# All available presets per detector:
#   CLIP-D       n= 10  mean=0.4551h  median=0.4115h
#   R50_nodown   n= 10  mean=0.2404h  median=0.2408h
#   R50_TF       n= 10  mean=0.1546h  median=0.1414h

# Same-preset comparison (10 presets with a valid time for all detectors):
#   CLIP-D       n= 10  mean=27.3 min  median=24.7 min  std =6.2 min var=137841.62 s2
#   R50_nodown   n= 10  mean=14.4 min  median=14.4 min  std =3.5 min var=43829.71 s2
#   R50_TF       n= 10  mean=9.3 min  median=8.5 min  std =4.9 min var=87545.65 s2
#   (0 presets excluded: missing for at least one detector)
# Saved JSON files to /second-disk/Image-Deepfake-Detectors-Public-Library/detectors_training_elapsed_times