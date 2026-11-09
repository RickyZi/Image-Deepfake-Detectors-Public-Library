#!/usr/bin/env python3
"""
Build cross-preset generalization matrices from the logs written by run_generalization.sh.

For each detector it produces:
  1. <det>_raw.png    : metric matrix (rows = training preset, cols = test preset)
  2. <det>_delta.png  : signed drop vs aligned case, delta = metric(trn,tst) - metric(trn,trn)
                        (diverging colormap centred on 0: green = better than aligned, red = worse)
and, when both detectors are present, compare_delta.png (difference between the two models'
delta matrices) plus a CSV with everything.

Metric extraction: the LAST regex match in each log is used. Adapt --metric-regex to the
launcher's output (one capture group with the number). Values in [0,1] or in % both work.
"""
import argparse, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASETS = [
    "seasons/autumn_TM01", "seasons/spring_SP01", "seasons/summer_SM01", "seasons/winter_WN01",
    "style/bw_BW01", "style/cinematic_CN01", "style/cinematic2_CN11", "style/filminspired_warmgold",
    "style/futuristic_FT01", "style/vintage_VN01",
    "adaptive/blurbg_strong", "adaptive/blurbg_subtle", "adaptive/enhance_portrait",
    "adaptive/sky_bluedrama", "adaptive/subject_pop",
    "subject/landscape_LN01", "subject/travel2_TR11", "subject/travel_TR01",
]
slug = lambda s: s.replace("/", "_")
short = lambda s: s.split("/", 1)[1]


def collect(log_root: Path, regex: str, detectors):
    rx = re.compile(regex)
    rows = []
    for det in detectors:
        for trn in DATASETS:
            for tst in DATASETS:
                f = log_root / det / f"{slug(trn)}__{slug(tst)}.log"
                val = np.nan
                if f.exists():
                    m = rx.findall(f.read_text(errors="ignore"))
                    if m:
                        val = float(m[-1])
                rows.append(dict(detector=det, trn=trn, test=tst, metric=val))
    return pd.DataFrame(rows)


def matrices(df_det):
    M = df_det.pivot(index="trn", columns="test", values="metric").reindex(index=DATASETS, columns=DATASETS)
    diag = np.diag(M.values)[:, None]          # aligned score of each training preset
    D = M - diag                               # signed drop w.r.t. the aligned case (row-wise)
    return M, D


def draw(mat, title, path, cmap, vmin=None, vmax=None, fmt="{:.1f}", cbar_label=""):
    n = len(DATASETS)
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([short(d) for d in DATASETS], rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels([short(d) for d in DATASETS], fontsize=8)
    ax.set_xlabel("Test preset"); ax.set_ylabel("Training preset"); ax.set_title(title)
    for i in range(n):
        for j in range(n):
            v = mat.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6)
        ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False, ec="black", lw=1.5))  # aligned diagonal
    # separate preset families
    for b in (4, 10, 15):
        ax.axhline(b - .5, color="white", lw=2); ax.axvline(b - .5, color="white", lw=2)
    fig.colorbar(im, ax=ax, fraction=0.04, label=cbar_label)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-root", default="logs")
    ap.add_argument("--out", default="matrices")
    ap.add_argument("--detectors", nargs="+", default=["R50_TF", "R50_nodown"])
    ap.add_argument("--metric-regex", default=r"(?i)\bacc(?:uracy)?\b\s*[:=]\s*([0-9]*\.?[0-9]+)",
                    help="regex with ONE capture group for the metric; last match in the log is used")
    ap.add_argument("--metric-name", default="Accuracy")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(exist_ok=True, parents=True)
    df = collect(Path(a.log_root), a.metric_regex, a.detectors)
    df.to_csv(out / "all_results.csv", index=False)
    print(f"parsed {df.metric.notna().sum()}/{len(df)} runs")
    if df.metric.notna().sum() == 0:
        sys.exit("no metric found: check --metric-regex against one of the logs")

    deltas = {}
    for det in a.detectors:
        d = df[df.detector == det]
        if d.metric.notna().sum() == 0:
            continue
        M, D = matrices(d)
        deltas[det] = D
        M.to_csv(out / f"{det}_raw.csv"); D.to_csv(out / f"{det}_delta.csv")
        draw(M, f"{det}: {a.metric_name} (diagonal = aligned)", out / f"{det}_raw.png",
             "viridis", cbar_label=a.metric_name)
        lim = np.nanmax(np.abs(D.values))
        draw(D, f"{det}: {a.metric_name} change vs aligned case", out / f"{det}_delta.png",
             "RdYlGn", -lim, lim, fmt="{:+.1f}", cbar_label="delta vs aligned")

        # per-training-preset summary: mean off-diagonal change
        off = D.values.copy(); np.fill_diagonal(off, np.nan)
        summ = pd.DataFrame({"aligned": np.diag(M.values), "mean_delta_offdiag": np.nanmean(off, axis=1),
                             "worst_delta": np.nanmin(off, axis=1)}, index=DATASETS)
        summ.to_csv(out / f"{det}_summary.csv"); print(f"\n{det}\n{summ.round(2)}")

    if len(deltas) == 2:
        (n1, D1), (n2, D2) = deltas.items()
        C = D1 - D2
        lim = np.nanmax(np.abs(C.values))
        draw(C, f"Delta({n1}) - Delta({n2}): positive = {n1} generalizes better", out / "compare_delta.png",
             "PuOr", -lim, lim, fmt="{:+.1f}", cbar_label="difference of deltas")


if __name__ == "__main__":
    main()