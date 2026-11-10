#!/usr/bin/env python3
"""
Cross-preset generalization matrices from the generalization_results/pretrained tree:

  <root>/<trn>/<detector_dir>/<datasets-dir>/metrics.json              (aligned case)
  <root>/<trn>_vs_<tst>/<detector_dir>/<datasets-dir>/metrics.json     (non-aligned case)

Presets are discovered automatically from the folder names (split on "_vs_").

First step (to find the right metric key):
  python3 plot_generalization_from_results.py --root generalization_results/pretrained --list-keys

  found 275 runs in generalization_results/pretrained:
    generalization_results/pretrained/adaptive_blurbg_strong/R50_TF_ft_unfreezeL4/realFFHQ&realFORLAB&gan1&gan2&gan3&sd15&sd2&sd3&sdXL&flux/metrics.json
    TPR = 0.9395604395604396
    TNR = 0.86
    Acc = 0.8979057591623036
    Balanced Acc = 0.8997802197802198
    F1 = 0.8976377952755905
    AUC = 0.9446153846153846
    num_images = 382.0
    execution time = 87.63098788261414
Then:
  python3 plot_generalization_from_results.py --root generalization_results/pretrained --metric-key accuracy

  run with Balanced Acc (or Acc, or AUC) to get the same results as in the paper.
  python3 plot_gen_matrix.py --metric-key balanced_accuracy --detector-dirs R50_TF_ft_unfreezeL4 R50_nodown_ft_unfreezeL4

Outputs in --out: <det>_raw.png, <det>_delta.png (+ csv), compare_delta.png, all_results.csv
delta(trn, tst) = metric(trn, tst) - metric(trn, trn)   (row-wise, signed)
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILY_ORDER = ["seasons", "style", "adaptive", "subject"]

METRIC_KEY_ALIASES = {
    "accuracy": "Acc",
    "acc": "Acc",
    "balanced_acc": "Balanced Acc",
    "balanced_accuracy": "Balanced Acc",
    "auc": "AUC",
    "f1": "F1",
    "tpr": "TPR",
    "tnr": "TNR",
}

def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out[prefix[:-1]] = float(d)
    return out


def find_metrics_file(det_dir: Path):
    files = sorted(det_dir.glob("*/metrics.json"))
    return files[0] if files else None


def pick_key(flat, key):
    keys = list(flat)
    if key:
        key = METRIC_KEY_ALIASES.get(key.lower().replace(" ", "_").replace("-", "_"), key)
        if key in flat:
            return key
        cand = [k for k in keys if k.lower().endswith(key.lower())]
        if len(cand) == 1:
            return cand[0]
        sys.exit(f"metric key '{key}' is {'ambiguous' if cand else 'not found'}: {cand or keys}")
    for c in ("Balanced Acc", "Acc", "AUC", "F1", "TPR", "TNR"):
        cand = [k for k in keys if k.lower().endswith(c)]
        if len(cand) == 1:
            return cand[0]
    sys.exit(f"cannot guess the metric, pass --metric-key. Available keys:\n  " + "\n  ".join(keys))


def preset_sort_key(slug):
    fam = slug.split("_", 1)[0]
    return (FAMILY_ORDER.index(fam) if fam in FAMILY_ORDER else 99, slug)


label = lambda s: s.split("_", 1)[1] if "_" in s else s
family = lambda s: s.split("_", 1)[0]


def scan(root: Path):
    runs = []  # (trn, tst, dirpath)
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        parts = d.name.split("_vs_")
        if len(parts) == 1:
            runs.append((parts[0], parts[0], d))
        elif len(parts) == 2:
            runs.append((parts[0], parts[1], d))
    return runs


def draw(mat, presets, title, path, cmap, vmin=None, vmax=None, fmt="{:.1f}", cbar_label=""):
    n = len(presets)
    fig, ax = plt.subplots(figsize=(0.75 * n + 3, 0.7 * n + 2))
    im = ax.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([label(p) for p in presets], rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels([label(p) for p in presets], fontsize=8)
    ax.set_xlabel("Test preset"); ax.set_ylabel("Training preset"); ax.set_title(title)
    for i in range(n):
        for j in range(n):
            v = mat.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6)
        ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False, ec="black", lw=1.5))
    for i in range(1, n):  # family separators
        if family(presets[i]) != family(presets[i - 1]):
            ax.axhline(i - .5, color="white", lw=2.5); ax.axvline(i - .5, color="white", lw=2.5)
    fig.colorbar(im, ax=ax, fraction=0.04, label=cbar_label)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="generalization_results/pretrained")
    ap.add_argument("--out", default="generalization_results/matrices")
    ap.add_argument("--detector-dirs", nargs="+", default=["R50_TF_ft_unfreezeL4", "R50_nodown_ft_unfreezeL4"],
                    help="sub-folder name of each detector inside every run folder")
    ap.add_argument("--metric-key", default="balanced_accuracy", help="key in metrics.json (dotted path or unique suffix)")
    ap.add_argument("--list-keys", action="store_true")
    a = ap.parse_args()

    root = Path(a.root)
    runs = scan(root)
    print(f"found {len(runs)} runs in {root}:")
    if not runs:
        sys.exit(f"no run folders in {root}")

    if a.list_keys:
        for trn, tst, d in runs:
            for det in a.detector_dirs:
                f = find_metrics_file(d / det)
                if f:
                    print(f)
                    for k, v in flatten(json.load(open(f))).items():
                        print(f"  {k} = {v}")
                    return
        sys.exit("no metrics.json found: check --detector-dirs")

    rows, key = [], None
    for trn, tst, d in runs:
        for det in a.detector_dirs:
            f = find_metrics_file(d / det)
            if not f:
                continue
            flat = flatten(json.load(open(f)))
            key = key or pick_key(flat, a.metric_key)
            rows.append(dict(detector=det, trn=trn, test=tst, metric=flat.get(key, np.nan)))
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("no metrics.json found: check --detector-dirs and --root")
    print(f"metric key: {key}")

    presets = sorted(set(df.trn) | set(df.test), key=preset_sort_key)
    pct = df.metric.max() <= 1.0
    if pct:
        df["metric"] *= 100
    unit = "%" if pct else ""
    out = Path(a.out)
    for det in a.detector_dirs:
        (out / det).mkdir(parents=True, exist_ok=True)
    # out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / det / "all_results.csv", index=False)

    deltas = {}
    for det in a.detector_dirs:
        d = df[df.detector == det]
        if d.empty:
            print(f"[warn] nothing for {det}"); continue
        M = d.pivot(index="trn", columns="test", values="metric").reindex(index=presets, columns=presets)
        missing = int(M.isna().sum().sum())
        if missing:
            print(f"[warn] {det}: {missing} missing cells out of {len(presets) ** 2}")
        D = M - np.diag(M.values)[:, None]
        deltas[det] = D
        M.to_csv(out / det / f"{det}_raw.csv"); D.to_csv(out / det / f"{det}_delta.csv")
        draw(M, presets, f"{det}: {key} (diagonal = aligned)", out / det / f"{det}_raw.png", "viridis",
             cbar_label=f"{key} {unit}")
        lim = np.nanmax(np.abs(D.values))
        draw(D, presets, f"{det}: change in {key} vs aligned case", out / det / f"{det}_delta.png", "RdYlGn",
             -lim, lim, fmt="{:+.1f}", cbar_label=f"delta {unit}")
        off = D.values.copy(); np.fill_diagonal(off, np.nan)
        summ = pd.DataFrame({"aligned": np.diag(M.values), "mean_delta_offdiag": np.nanmean(off, axis=1),
                             "worst_delta": np.nanmin(off, axis=1)}, index=presets)
        summ.to_csv(out / det / f"{det}_summary.csv"); print(f"\n{det}\n{summ.round(2)}")

    if len(deltas) == 2:
        for det in a.detector_dirs:
            if det not in deltas:
                sys.exit(f"detector {det} missing, cannot compare")
            (n1, D1), (n2, D2) = deltas.items()
            C = D1 - D2
            lim = np.nanmax(np.abs(C.values))
            draw(C, presets, f"Delta({n1}) - Delta({n2}): positive = {n1} generalizes better",
                out / det / "compare_delta.png", "PuOr", -lim, lim, fmt="{:+.1f}", cbar_label="difference of deltas")


if __name__ == "__main__":
    main()