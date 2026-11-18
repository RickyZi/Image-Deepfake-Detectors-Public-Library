#!/usr/bin/env python3
"""
List the runs that are missing (or incomplete) for a detector and write a retry script.

A run (trn, tst) counts as DONE only if
  <root>/<top>/<sub>/<trn>[_vs_<tst>]/<detector_dir>/*/metrics.json   exists and is valid JSON.
Anything else is MISSING (no folder) or INCOMPLETE (folder but no valid metrics.json).
The *.complete marker files are ignored on purpose.

Usage (from the folder containing plot_gen_matrix.py, e.g. utils/):
  python3 find_missing_runs.py --detectors CLIP-D --preset-set extended --extra-flags "<flags of the CLIP-D FT call>"
  python3 find_missing_runs.py --detectors R50_TF R50_nodown --preset-set wifs

Outputs (in --out, default ./missing_runs):
  <det>_missing.csv   trn, test, status
  <det>_retry.sh      one launcher.py command per missing run (aligned case first for each trn)
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.plot_gen_confusion_matrix import DETECTOR_LAYOUT, PRESET_SETS, find_metrics_file  # noqa: E402

# flags appended to every launcher.py call; CLIP-D flags are unknown here -> pass --extra-flags
DEFAULT_FLAGS = {"R50_TF": "--ft --r50unfreezeL4", "R50_nodown": "--ft --r50unfreezeL4", "CLIP-D": ""}
LAUNCHER_NAME = {"R50_TF": "R50_TF", "R50_nodown": "R50_nodown", "CLIP-D": "CLIP-D"}


def to_path(slug):  # "style_bw_BW01" -> "style/bw_BW01" (family = text before the first underscore)
    fam, rest = slug.split("_", 1)
    return f"{fam}/{rest}"


def is_valid(f):
    try:
        json.load(open(f))
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="generalization_results")
    ap.add_argument("--out", default="missing_runs")
    ap.add_argument("--detectors", nargs="+", default=list(DETECTOR_LAYOUT), choices=list(DETECTOR_LAYOUT))
    ap.add_argument("--preset-set", default="extended", choices=list(PRESET_SETS))
    ap.add_argument("--extra-flags", default=None, help="flags added to every launcher.py call")
    a = ap.parse_args()

    presets = PRESET_SETS[a.preset_set]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    for det in a.detectors:
        top, sub, det_folder = DETECTOR_LAYOUT[det]
        run_root = Path(a.root) / top / sub
        flags = a.extra_flags if a.extra_flags is not None else DEFAULT_FLAGS[det]
        missing = []
        for trn in presets:
            for tst in [trn] + [p for p in presets if p != trn]:   # aligned case first
                d = run_root / (trn if trn == tst else f"{trn}_vs_{tst}")
                f = find_metrics_file(d / det_folder) if d.is_dir() else None
                if f and is_valid(f):
                    continue
                missing.append((trn, tst, "incomplete" if d.is_dir() else "missing"))

        total = len(presets) ** 2
        print(f"{det}: {len(missing)} missing/incomplete out of {total} "
              f"({sum(s == 'incomplete' for *_, s in missing)} incomplete)")
        with open(out / f"{det}_missing.csv", "w") as fh:
            fh.write("trn,test,status\n")
            fh.writelines(f"{t},{s},{st}\n" for t, s, st in missing)
        with open(out / f"{det}_retry.sh", "w") as fh:
            fh.write("#!/usr/bin/env bash\n# only the missing runs for " + det + "\n")
            if det == "CLIP-D" and not flags:
                fh.write("# WARNING: add the CLIP-D flags (re-run with --extra-flags ...)\n")
            for t, s, _ in missing:
                fh.write(f"python3 launcher.py --detector {LAUNCHER_NAME[det]} --phases test "
                         f"--trn-dataset {to_path(t)} --dataset {to_path(s)} {flags}\n")
        (out / f"{det}_retry.sh").chmod(0o755)
        # per-training-preset summary of what is missing
        by = {}
        for t, s, _ in missing:
            by.setdefault(t, 0); by[t] += 1
        for t, n in sorted(by.items(), key=lambda x: -x[1])[:10]:
            print(f"   {t}: {n} missing")


if __name__ == "__main__":
    main()