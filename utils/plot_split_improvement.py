#!/usr/bin/env python3
"""
plot_split_improvement.py  (companion of plot_metric_tables.py, which must be importable)

For every detector and every split (PRE = aligned, plus the social splits) it computes the
improvement of the fine-tuned model over the baseline on the SAME dataset:

    improvement[detector][split][dataset][metric] = FT[metric] - baseline[metric]

and then
  1. prints, on the terminal, the mean Balanced-Acc improvement of each detector in each split,
     the mean over the social splits, the mean over all splits and the gap PRE - social;
  2. writes improvements.csv (one row per detector / split / dataset);
  3. for every pair of detectors and every split writes a "compare delta" table
        rows = datasets (+ MEAN), cols = metrics,
        cell = improvement(A) - improvement(B)        (green = A improves more than B)
        small text under the value = improvement(A) / improvement(B)
  4. for every pair of detectors writes a summary table across the splits
        rows = splits (+ mean social, mean all), same cells, averaged over the datasets.

Folder conventions (same as plot_metric_tables.py):
  baseline : results/<model>/pretrained[_<social>]/<dataset>/<model>_pretrained/**/metrics.json
  FT       : results/<model>/pretrained[_<social>]/<dataset>/<model>_ft[_unfreezeL4]/**/metrics.json
             (CLIP-D FT lives in results/CLIP-D/lora_r4_qv[_<social>]/)
  The reference folder 'dataset' is skipped, as in your FT tables.

Usage:
  python plot_split_improvement.py
  python plot_split_improvement.py --detectors R50_nodown CLIP-D --socials facebook telegram twitter
  python plot_split_improvement.py --detectors R50_nodown R50_nodown:unfreezeL4 CLIP-D   # 3 modes -> all pairs
"""
import argparse, contextlib, csv, io, itertools, re
from pathlib import Path

import numpy as np
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch

import plot_tables_singlerun as pmt
from plot_tables_singlerun import METRICS, CMAP_DIFF, CELL_W, CELL_H, LEFT_PAD, TOP_PAD

BA = "Balanced Acc"

# Optional manual overrides of the folders, {(model, split): (baseline_root, ft_root)}, split = "pre"
# or the social name. Use it when your folders do not follow the default convention, e.g.
#   PATH_OVERRIDES = {("CLIP-D", "pre"): ("results/CLIP-D/ft_unfreezeL4", "results/CLIP-D/lora_r4_qv")}
PATH_OVERRIDES = {}

# ------------------------------------------------------------------ dataset-name matching
# Baseline and FT folders spell the same dataset differently (autumn_TM01 / autumn-TM01,
# bw_BW01 / bw01, urban-architecture-UA08 / urbanarchitecture-UA08, enhance-portait ...),
# so datasets are matched on a canonical key instead of on the folder name.
LABELS = {}   # canonical key -> readable label (first folder name seen)
RAW = {}      # (detector label, split) -> {canonical key: (baseline BA, FT BA)}, filled by improvements()


def canon(name):
    s = re.sub(r"[^a-z0-9+]", "", name.lower())          # drop '_' and '-' and any other separator
    s = s.replace("portait", "portrait")                  # typo in the folder name
    return {"stylebw01": "stylebwbw01"}.get(s, s)         # style_bw01 == style_bw_BW01


# ------------------------------------------------------------------ paths / loading
def parse_detector(spec):
    """'R50_TF', 'R50_nodown', 'CLIP-D' (optionally 'MODEL:unfreezeL4' / 'MODEL:ft') -> (model, ft_subdir, label)

    Folder of the FT model inside every dataset folder:
      R50_TF / R50_nodown : <model>_ft_unfreezeL4   (default; ':ft' selects <model>_ft)
      CLIP-D              : CLIP-D_ft
    The baseline is always <model>_pretrained."""
    model, _, variant = spec.partition(":")
    if variant not in ("", "unfreezeL4", "ft"):
        raise SystemExit(f"unknown variant '{variant}' in '{spec}' (use ':unfreezeL4' or ':ft')")
    if model.startswith("R50"):
        unfreeze = variant != "ft"
        return model, f"{model}_ft_unfreezeL4" if unfreeze else f"{model}_ft", model if unfreeze or variant == "" else f"{model}:ft"
    return model, f"{model}_ft", model


def _first_existing(*cands):
    cands = [Path(c) for c in cands]
    return next((c for c in cands if c.is_dir()), cands[0])


def roots(model, social):
    """(baseline root, FT root) -- identical to the paths built in plot_metric_tables.main()."""
    ov = PATH_OVERRIDES.get((model, social or "pre"))
    if ov:
        return Path(ov[0]), Path(ov[1])
    # R50_*: baseline and FT live side by side in results/<model>/pretrained[_<social>]/<preset>/
    # CLIP-D: baseline in results/CLIP-D/pretrained[_<social>], FT in results/CLIP-D/lora_r4_qv[_<social>]
    sfx = f"_{social}" if social else ""
    base = Path(f"./results/{model}/pretrained{sfx}")
    if model == "CLIP-D":
        ft = _first_existing(f"./results/CLIP-D/lora_r4_qv{sfx}", f"./results/CLIP-D/lora_qv{sfx}")
    else:
        ft = base
    return base, ft


def improvements(model, ft_subdir, social, verbose=False, key=None):
    """{canonical dataset key: {metric: FT - baseline}} for the datasets present in both baseline and FT.
    If `key` is given, the raw (baseline BA, FT BA) of those datasets is also stored in RAW[key]."""
    tag = f"{model} split={social or 'pre'}"
    base_root, ft_root = roots(model, social)
    for r in (base_root, ft_root):
        if not r.is_dir():
            sib = sorted(p.name for p in r.parent.glob("*") if p.is_dir()) if r.parent.is_dir() else []
            print(f"  [warn] {tag}: {r} not found; folders in {r.parent}: {sib}")
    sink = io.StringIO()
    with (contextlib.nullcontext() if verbose else contextlib.redirect_stdout(sink)):
        base_raw = pmt.collect_results(base_root, f"{model}_pretrained")
        ft_raw = pmt.collect_results(ft_root, ft_subdir)

    def by_canon(d):
        out = {}
        for raw, v in d.items():
            c = canon(raw)
            if c in out:
                print(f"  [warn] {tag}: '{raw}' and another folder map to the same dataset key '{c}'")
            out[c] = v
            LABELS.setdefault(c, raw)
        return out

    base, ft = by_canon(base_raw), by_canon(ft_raw)

    def hint(root, sub):   # which sub-folders do exist inside the dataset folders?
        seen = set()
        for d in sorted(root.glob("*"))[:30] if root.is_dir() else []:
            if d.is_dir():
                seen |= {p.name for p in d.iterdir() if p.is_dir()}
        return sorted(seen)

    if not base:
        print(f"  [warn] {tag}: no baseline results ('{model}_pretrained' under {base_root}); "
              f"sub-folders seen: {hint(base_root, model)}")
    if not ft:
        print(f"  [warn] {tag}: no FT results ('{ft_subdir}' under {ft_root}); "
              f"sub-folders seen: {hint(ft_root, ft_subdir)}")
    common = sorted(set(base) & set(ft))
    only_b, only_f = sorted(set(base) - set(ft)), sorted(set(ft) - set(base))
    if base and ft and (only_b or only_f):
        show = lambda ks: [LABELS[k] for k in ks[:8]] + (["..."] if len(ks) > 8 else [])
        print(f"  [warn] {tag}: {len(common)} matched; {len(only_b)} only in baseline {show(only_b)}, "
              f"{len(only_f)} only in FT {show(only_f)}")
    if key is not None:
        RAW[key] = {c: (base[c][BA], ft[c][BA]) for c in common}
    return {c: {m: ft[c][m] - base[c][m] for m in METRICS} for c in common}


# ------------------------------------------------------------------ small helpers
def fmean(vals):
    v = [x for x in vals if np.isfinite(x)]
    return float(np.mean(v)) if v else np.nan


def mean_metrics(per_ds):            # {dataset: {metric: v}} -> {metric: mean over datasets}
    return {m: fmean([d[m] for d in per_ds.values()]) for m in METRICS}


def avg_metrics(list_of_dicts):      # [{metric: v}, ...] -> {metric: mean}
    return {m: fmean([d[m] for d in list_of_dicts]) for m in METRICS}


def fmt(x, p=4):
    return f"{x:+.{p}f}" if np.isfinite(x) else "   n/a"


# ------------------------------------------------------------------ terminal summary
def print_summary(imp, labels, splits, socials, title="each detector on its own datasets"):
    print(f"\n=== Mean {BA} improvement (FT - baseline), {title}; "
          f"PRE = aligned split, mean all = equal weight per split ===")
    head = f"{'detector':<24}" + "".join(f"{s:>12}" for s in splits) + \
           f"{'mean social':>14}{'mean all':>12}{'PRE-social':>12}"
    print(head); print("-" * len(head))
    cols = []
    for lab in labels:
        per_split = {s: fmean([d[BA] for d in imp[lab][s].values()]) for s in splits}
        social_mean = fmean([per_split[s] for s in socials])
        all_mean = fmean(list(per_split.values()))
        gap = per_split["pre"] - social_mean
        cols.append([per_split[s] for s in splits] + [social_mean, all_mean, gap])
        print(f"{lab:<24}" + "".join(f"{fmt(per_split[s]):>12}" for s in splits) +
              f"{fmt(social_mean):>14}{fmt(all_mean):>12}{fmt(gap):>12}")
        print(f"{'':<24}" + "".join(f"{'n=' + str(len(imp[lab][s])):>12}" for s in splits))
    if len(labels) > 1:   # mean over the detectors, column by column
        m = [fmean([c[i] for c in cols]) for i in range(len(cols[0]))]
        print("-" * len(head))
        print(f"{'mean detectors':<24}" + "".join(f"{fmt(x):>12}" for x in m[:len(splits)]) +
              f"{fmt(m[-3]):>14}{fmt(m[-2]):>12}{fmt(m[-1]):>12}")
    print("(values are absolute differences on the 0-1 scale; x100 for percentage points)\n")


# ------------------------------------------------------------------ csv
def write_csv(imp, labels, splits, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["detector", "split", "dataset"] + [f"d_{m}" for m in METRICS])
        for lab in labels:
            for s in splits:
                for ds, d in imp[lab][s].items():
                    w.writerow([lab, s, LABELS.get(ds, ds)] + [d[m] for m in METRICS])
    print(f"  -> saved: {path}")


# ------------------------------------------------------------------ compare table
def plot_compare(rows, row_labels, title, path, a, b):
    """rows: {row_label: (impA {metric: v}, impB {metric: v})}; cell = impA - impB."""
    n_rows, n_cols = len(row_labels), len(METRICS)
    diff = np.full((n_rows, n_cols), np.nan)
    for i, l in enumerate(row_labels):
        dA, dB = rows[l]
        for j, m in enumerate(METRICS):
            diff[i, j] = dA[m] - dB[m]
    finite = diff[np.isfinite(diff)]
    lim = max(np.max(np.abs(finite)) * 1.08, 0.01) if finite.size else 0.5
    norm = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)

    fig, ax = pmt._make_figure(n_rows, n_cols)
    for i, l in enumerate(row_labels):
        dA, dB = rows[l]
        for j, m in enumerate(METRICS):
            d = diff[i, j]
            x = LEFT_PAD + j * CELL_W
            y = TOP_PAD + (n_rows - 1 - i) * CELL_H
            if np.isfinite(d):
                rgba = CMAP_DIFF(norm(d))
                bg = mcolors.to_hex(rgba)
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                tc = "#1a1a2e" if lum > 0.52 else "#f0f0f0"
            else:
                bg, tc = "#2e2e4a", "#888888"
            ax.add_patch(FancyBboxPatch((x + 0.06, y + 0.06), CELL_W - 0.12, CELL_H - 0.12,
                                        boxstyle="round,pad=0.05", linewidth=0, facecolor=bg, clip_on=False))
            ax.text(x + CELL_W / 2, y + CELL_H * 0.63, f"{d:+.3f}" if np.isfinite(d) else "N/A",
                    ha="center", va="center", fontsize=11, fontweight="bold", color=tc, fontfamily="monospace")
            if np.isfinite(d):
                ax.text(x + CELL_W / 2, y + CELL_H * 0.27, f"{dA[m]:+.3f}/{dB[m]:+.3f}",
                        ha="center", va="center", fontsize=7, color=tc, fontfamily="monospace")
    pmt._draw_chrome(ax, row_labels, title, n_rows, n_cols)
    pmt._add_colorbar(fig, CMAP_DIFF, norm,
                      f"top: imp({a}) - imp({b});  bottom: imp({a}) / imp({b});  "
                      f"green = {a} improves more,  red = {b} improves more")
    pmt._save(fig, path)


# ------------------------------------------------------------------ family summary of Balanced Acc
FAMILY_ORDER = ["adaptive", "seasons", "style", "subject", "auto+", "portraits", "video"]


def fam_of(c):
    return LABELS[c].split("_", 1)[0]            # '<family>_<preset>_<code>' -> '<family>'


def family_summary(lab, splits, baseline_split, dispersion, common_only, out):
    """Rows = preset families (+ ALL), columns = baseline model and FT model on PRE / each social split.
    Cell = mean Balanced Acc (in %) over the presets of the family, with the sample variance (or std)
    over those presets in parentheses and the number of presets n."""
    cols = [(f"baseline ({baseline_split})", baseline_split, 0)] + [(f"FT {s}", s, 1) for s in splits]
    data = {}
    for name, s, i in cols:
        data[name] = {c: 100 * v[i] for c, v in RAW.get((lab, s), {}).items() if np.isfinite(v[i])}
    if not any(data.values()):
        print(f"  [skip] family summary {lab}: no data")
        return
    if common_only:
        keep = set.intersection(*[set(d) for d in data.values()])
        data = {n: {c: v for c, v in d.items() if c in keep} for n, d in data.items()}
    fams = sorted({fam_of(c) for d in data.values() for c in d},
                  key=lambda f: (FAMILY_ORDER.index(f) if f in FAMILY_ORDER else 99, f))
    rows = fams + ["ALL"]
    names = [n for n, _, _ in cols]
    M, V, N = (np.full((len(rows), len(names)), np.nan) for _ in range(3))
    spread = (lambda x: np.var(x, ddof=1)) if dispersion == "var" else (lambda x: np.std(x, ddof=1))
    for j, n in enumerate(names):
        for i, f in enumerate(rows):
            vals = [v for c, v in data[n].items() if f == "ALL" or fam_of(c) == f]
            N[i, j] = len(vals)
            if vals:
                M[i, j] = np.mean(vals)
                V[i, j] = spread(vals) if len(vals) > 1 else np.nan   # one preset -> no spread

    dn = "variance" if dispersion == "var" else "std"
    # terminal
    print(f"\n=== {lab}: Balanced Acc (%) by preset family, mean ({dn}) over presets ===")
    print(f"{'family':<12}" + "".join(f"{n:>22}" for n in names))
    for i, f in enumerate(rows):
        cells = [("   n/a" if not np.isfinite(M[i, j]) else
                  f"{M[i, j]:6.1f}" + (f" ({V[i, j]:6.1f})" if np.isfinite(V[i, j]) else " (   n/a)") +
                  f" n={int(N[i, j])}") for j in range(len(names))]
        print(f"{f:<12}" + "".join(f"{c:>22}" for c in cells))
    # csv (long format)
    path = out / f"family_summary_{lab}_BA.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["detector", "family", "column", "mean_BA_pct", dn, "n_presets"])
        for i, f in enumerate(rows):
            for j, n in enumerate(names):
                w.writerow([lab, f, n, M[i, j], V[i, j], int(N[i, j])])
    print(f"  -> saved: {path}")
    plot_family_table(M, V, N, rows, names, f"{lab}: Balanced Acc (%) by preset family",
                      out / f"family_summary_{lab}_BA.png", dn)


def plot_family_table(M, V, N, rows, cols, title, path, dn):
    n_rows, n_cols = len(rows), len(cols)
    fin = M[np.isfinite(M)]
    vmin, vmax = (fin.min() - 2, fin.max() + 1) if fin.size else (0, 100)
    norm = mcolors.Normalize(vmin=vmin, vmax=max(vmax, vmin + 1))
    fig, ax = pmt._make_figure(n_rows, n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            x = LEFT_PAD + j * CELL_W
            y = TOP_PAD + (n_rows - 1 - i) * CELL_H
            v = M[i, j]
            if np.isfinite(v):
                rgba = pmt.CMAP_REF(norm(v))
                bg = mcolors.to_hex(rgba)
                tc = "#1a1a2e" if 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2] > 0.52 else "#f0f0f0"
            else:
                bg, tc = "#2e2e4a", "#888888"
            ax.add_patch(FancyBboxPatch((x + 0.06, y + 0.06), CELL_W - 0.12, CELL_H - 0.12,
                                        boxstyle="round,pad=0.05", linewidth=0, facecolor=bg, clip_on=False))
            ax.text(x + CELL_W / 2, y + CELL_H * 0.63, f"{v:.1f}" if np.isfinite(v) else "N/A",
                    ha="center", va="center", fontsize=11, fontweight="bold", color=tc, fontfamily="monospace")
            if np.isfinite(v):
                sub = (f"({V[i, j]:.1f}) " if np.isfinite(V[i, j]) else "") + f"n={int(N[i, j])}"
                ax.text(x + CELL_W / 2, y + CELL_H * 0.27, sub, ha="center", va="center",
                        fontsize=7.5, color=tc, fontfamily="monospace")
    for j, c in enumerate(cols):                                   # column headers
        ax.text(LEFT_PAD + j * CELL_W + CELL_W / 2, TOP_PAD + n_rows * CELL_H + 0.10, c, ha="center",
                va="bottom", fontsize=10, fontweight="bold", color="#e0e0f8", fontfamily="monospace")
    for i, f in enumerate(rows):                                   # row labels
        ax.text(LEFT_PAD - 0.14, TOP_PAD + (n_rows - 1 - i) * CELL_H + CELL_H / 2, f, ha="right",
                va="center", fontsize=9.5, fontweight="bold", color="#e8e8f8", fontfamily="monospace")
    ax.text((LEFT_PAD + n_cols * CELL_W) / 2, TOP_PAD + n_rows * CELL_H + 0.50, title, ha="center",
            va="bottom", fontsize=13, fontweight="bold", color="#ffffff", fontfamily="monospace")
    pmt._add_colorbar(fig, pmt.CMAP_REF, norm,
                      f"cell: mean over the presets of the family  (parentheses: {dn}, n = number of presets)")
    pmt._save(fig, path)


# ------------------------------------------------------------------ table of improvements (one value per cell)
def plot_values(rows, row_labels, title, path):
    """rows: {row_label: {metric: improvement}}; green = FT better than baseline, red = worse."""
    n_rows, n_cols = len(row_labels), len(METRICS)
    vals = np.array([[rows[l][m] for m in METRICS] for l in row_labels], dtype=float)
    finite = vals[np.isfinite(vals)]
    lim = max(np.max(np.abs(finite)) * 1.08, 0.01) if finite.size else 0.5
    norm = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    fig, ax = pmt._make_figure(n_rows, n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            v = vals[i, j]
            x = LEFT_PAD + j * CELL_W
            y = TOP_PAD + (n_rows - 1 - i) * CELL_H
            if np.isfinite(v):
                rgba = CMAP_DIFF(norm(v))
                bg = mcolors.to_hex(rgba)
                tc = "#1a1a2e" if 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2] > 0.52 else "#f0f0f0"
            else:
                bg, tc = "#2e2e4a", "#888888"
            ax.add_patch(FancyBboxPatch((x + 0.06, y + 0.06), CELL_W - 0.12, CELL_H - 0.12,
                                        boxstyle="round,pad=0.05", linewidth=0, facecolor=bg, clip_on=False))
            ax.text(x + CELL_W / 2, y + CELL_H / 2, f"{v:+.3f}" if np.isfinite(v) else "N/A",
                    ha="center", va="center", fontsize=11, fontweight="bold", color=tc, fontfamily="monospace")
    pmt._draw_chrome(ax, row_labels, title, n_rows, n_cols)
    pmt._add_colorbar(fig, CMAP_DIFF, norm, "FT - baseline   (green = FT better  /  red = FT worse)")
    pmt._save(fig, path)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", nargs="+", default=["R50_nodown", "R50_TF", "CLIP-D"],
                    help="R50_nodown, R50_TF, CLIP-D (R50 FT = <model>_ft_unfreezeL4; 'R50_TF:ft' for <model>_ft)")
    ap.add_argument("--socials", nargs="+", default=["facebook", "telegram", "twitter"])
    ap.add_argument("--out", default="./results/metric_tables/comparison")
    ap.add_argument("--verbose", action="store_true", help="show the per-folder log of collect_results")
    ap.add_argument("--family-baseline-split", default="pre",
                    help="split whose BASELINE model gives the 'baseline' column of the family summary (default: pre)")
    ap.add_argument("--dispersion", default="var", choices=["var", "std"],
                    help="number in parentheses in the family summary: sample variance (default) or std")
    ap.add_argument("--family-common", action="store_true",
                    help="family summary: use only presets available in every column (default: all presets "
                         "available in each column; n is reported per cell)")
    a = ap.parse_args()

    out = Path(a.out)
    splits = ["pre"] + list(a.socials)
    specs = [parse_detector(s) for s in a.detectors]
    labels = [lab for _, _, lab in specs]

    imp = {}
    for model, ft_subdir, lab in specs:
        imp[lab] = {}
        for s in splits:
            print(f"collecting {lab} / {s} ...")
            imp[lab][s] = improvements(model, ft_subdir, None if s == "pre" else s, a.verbose, key=(lab, s))

    print_summary(imp, labels, splits, a.socials)
    write_csv(imp, labels, splits, out / "improvements.csv")

    # ---- per-detector family summary of Balanced Acc: baseline + FT on PRE / each social split
    for lab in labels:
        family_summary(lab, splits, a.family_baseline_split, a.dispersion, a.family_common, out)

    for A, B in itertools.combinations(labels, 2):
        # one table per split
        for s in splits:
            common = sorted(set(imp[A][s]) & set(imp[B][s]))
            if not common:
                print(f"  [skip] {A} vs {B}, split {s}: no common datasets")
                continue
            names = [LABELS[ds] for ds in common]
            rows = {LABELS[ds]: (imp[A][s][ds], imp[B][s][ds]) for ds in common}
            rows["MEAN"] = (mean_metrics({d: imp[A][s][d] for d in common}),
                            mean_metrics({d: imp[B][s][d] for d in common}))
            plot_compare(rows, names + ["MEAN"], f"{A} vs {B}: FT improvement, split = {s}",
                         out / f"compare_{A}_vs_{B}_{s}.png", A, B)

        # summary across the splits (rows = splits, averaged over the common datasets)
        srows, order = {}, []
        for s in splits:
            common = set(imp[A][s]) & set(imp[B][s])
            if common:
                lbl = "pre (aligned)" if s == "pre" else s
                srows[lbl] = (mean_metrics({d: imp[A][s][d] for d in common}),
                              mean_metrics({d: imp[B][s][d] for d in common}))
                order.append((s, lbl))
        soc = [srows[l] for s, l in order if s != "pre"]
        if soc:
            srows["mean social"] = (avg_metrics([x[0] for x in soc]), avg_metrics([x[1] for x in soc]))
        if srows:
            srows["mean all"] = (avg_metrics([x[0] for x in (srows[l] for _, l in order)]),
                                 avg_metrics([x[1] for x in (srows[l] for _, l in order)]))
            plot_compare(srows, list(srows), f"{A} vs {B}: FT improvement across splits",
                         out / f"compare_{A}_vs_{B}_summary.png", A, B)

    # ---- all detectors together, on the datasets that EVERY detector has (so rows are comparable)
    if len(labels) > 1:
        common = {s: set.intersection(*[set(imp[l][s]) for l in labels]) for s in splits}
        imp_c = {l: {s: {d: imp[l][s][d] for d in common[s]} for s in splits} for l in labels}
        print_summary(imp_c, labels, splits, a.socials, "datasets common to all detectors")
        nan_row = {m: np.nan for m in METRICS}
        pm = {l: {s: mean_metrics(imp_c[l][s]) if imp_c[l][s] else nan_row for s in splits} for l in labels}

        def table(pick, title, fname, n_txt):
            rows = {f"{l} ({n_txt})": pick(l) for l in labels}
            rows["mean detectors"] = avg_metrics(list(rows.values()))
            if not any(np.isfinite(v) for r in rows.values() for v in r.values()):
                print(f"  [skip] {fname}: no data")
                return
            plot_values(rows, list(rows), title, out / fname)

        for s in splits:
            if common[s]:
                table(lambda l, s=s: pm[l][s], f"FT improvement over baseline, split = {s}",
                      f"improvement_all_{s}.png", f"n={len(common[s])}")
            else:
                print(f"  [skip] split {s}: no datasets common to all detectors")
        table(lambda l: avg_metrics([pm[l][s] for s in a.socials]), "FT improvement, mean over social splits",
              "improvement_all_mean_social.png", "social")
        table(lambda l: avg_metrics([pm[l][s] for s in splits]), "FT improvement, mean over all splits",
              "improvement_all_mean_all.png", "all splits")

    print(f"\nDone -> {out.resolve()}/")


if __name__ == "__main__":
    main()