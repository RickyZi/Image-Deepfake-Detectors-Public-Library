"""
plot_ensemble_table.py

Produces a styled table for each ensemble report JSON, matching the dark-
background monospace style of plot_tables_singlerun.py exactly.

Layout:
  - Reference row  (aligned model, blue-scale, raw scores only)
  - Cross row(s)   (diff vs aligned, red-green colormap)
  - Ensemble rules (diff vs aligned, same colormap)

The aligned model is identified automatically: the baseline whose @preset
substring appears in the dataset tag.

Usage:
    # Single report
    python plot_ensemble_table.py ensemble_report_R50_nodown_blurbg_family__blurbg_strong_images.json

    # All reports at once → one PNG per report
    python plot_ensemble_table.py results/ensemble/*.json --out-dir plots/
"""

import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────
METRICS = ["TPR", "TNR", "Acc", "Balanced Acc", "F1", "AUC"]

CMAP_DIFF = mcolors.LinearSegmentedColormap.from_list(
    "rg_diverging", ["#b22222", "#f5f5f0", "#2a7a2a"]
)
CMAP_REF = mcolors.LinearSegmentedColormap.from_list(
    "blue_seq", ["#0d1b4b", "#1a3a8a", "#4a7fd4", "#a8c8f8"]
)

# Cell geometry — matches plot_tables_singlerun.py exactly
CELL_W   = 1.70
CELL_H   = 0.62
LEFT_PAD = 2.60    # slightly wider for longer ensemble row labels
TOP_PAD  = 0.75
BOT_PAD  = 0.65


# ── Figure helpers (verbatim from plot_tables_singlerun.py) ──────────────────
def _make_figure(n_rows, n_cols):
    fig_w = LEFT_PAD + n_cols * CELL_W + 0.3
    fig_h = TOP_PAD  + n_rows * CELL_H + BOT_PAD
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.set_axis_off()
    ax.set_xlim(0, LEFT_PAD + n_cols * CELL_W + 0.2)
    ax.set_ylim(0, TOP_PAD  + n_rows * CELL_H + 0.70)
    return fig, ax


def _draw_chrome(ax, row_labels, title, n_rows, n_cols):
    for j, metric in enumerate(METRICS):
        ax.text(
            LEFT_PAD + j * CELL_W + CELL_W / 2,
            TOP_PAD + n_rows * CELL_H + 0.10,
            metric, ha="center", va="bottom",
            fontsize=11, fontweight="bold",
            color="#e0e0f8", fontfamily="monospace",
        )
    for i, label in enumerate(row_labels):
        ax.text(
            LEFT_PAD - 0.14,
            TOP_PAD + (n_rows - 1 - i) * CELL_H + CELL_H / 2,
            label, ha="right", va="center",
            fontsize=9.0, fontweight="bold",
            color="#e8e8f8", fontfamily="monospace",
        )
    ax.text(
        (LEFT_PAD + n_cols * CELL_W) / 2,
        TOP_PAD + n_rows * CELL_H + 0.50,
        title, ha="center", va="bottom",
        fontsize=13, fontweight="bold",
        color="#ffffff", fontfamily="monospace",
    )
    for j in range(1, n_cols):
        x = LEFT_PAD + j * CELL_W
        ax.plot([x, x],
                [TOP_PAD - 0.02, TOP_PAD + n_rows * CELL_H + 0.02],
                color="#1a1a2e", linewidth=2, zorder=10)


def _add_colorbar(fig, cmap, norm, label):
    cbar_ax = fig.add_axes([0.12, 0.012, 0.78, 0.022])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(colors="#aaaacc", labelsize=7.5)
    cbar.set_label(label, color="#aaaacc", fontsize=7.5)
    cbar.outline.set_edgecolor("#444466")


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → saved: {path}")


# ── Row-label helpers ────────────────────────────────────────────────────────
def _short(model_id: str) -> str:
    """'R50_nodown@adaptive_blurbg_strong' → 'adaptive_blurbg_strong (aligned)'
       The '(aligned)' suffix is added by the caller for the reference row."""
    return model_id.split("@")[-1] if "@" in model_id else model_id


def _find_aligned(baselines: dict, dataset: str) -> str | None:
    """Return the baseline key whose preset substring appears in dataset tag."""
    dataset_norm = dataset.replace("-", "_").lower()
    for mid in baselines:
        preset = mid.split("@")[-1] if "@" in mid else mid
        if preset.replace("-", "_").lower() in dataset_norm:
            return mid
    # fallback: first key
    return next(iter(baselines), None)


# ── Main plotting function ───────────────────────────────────────────────────
def plot_ensemble(report: dict, output_path: Path):
    baselines  = report.get("baselines", {})
    results    = report.get("results",   {})
    dataset    = report.get("dataset",   "")
    detector   = report.get("detector",  "")

    aligned_key = _find_aligned(baselines, dataset)
    if aligned_key is None:
        print(f"  [skip] no baselines in {output_path.name}")
        return

    aligned_scores = baselines[aligned_key]

    # ── Row order ────────────────────────────────────────────────────────────
    # 1. Aligned (reference, blue-scale)
    # 2. Cross baselines (diffs vs aligned)
    # 3. Ensemble rules (diffs vs aligned)
    cross_keys    = [k for k in baselines if k != aligned_key]
    ensemble_keys = list(results.keys())

    row_keys = [aligned_key] + cross_keys + ensemble_keys
    row_labels = (
        [f"{_short(aligned_key)}\n(aligned)"] +
        [_short(k) for k in cross_keys] +
        [k for k in ensemble_keys]          # 'average', 'max', 'median' etc.
    )

    n_rows = len(row_labels)
    n_cols = len(METRICS)

    # ── Build raw / diff arrays ──────────────────────────────────────────────
    all_data = {**baselines, **results}
    raw   = np.full((n_rows, n_cols), np.nan)
    diffs = np.full((n_rows, n_cols), np.nan)

    for i, key in enumerate(row_keys):
        for j, metric in enumerate(METRICS):
            v   = all_data[key].get(metric, np.nan)
            ref = aligned_scores.get(metric, np.nan)
            raw[i, j] = v
            if i > 0 and np.isfinite(v) and np.isfinite(ref):   # skip aligned row
                diffs[i, j] = v - ref

    # ── Color norms ──────────────────────────────────────────────────────────
    finite_diffs = diffs[np.isfinite(diffs)]
    abs_max = max(np.max(np.abs(finite_diffs)) * 1.08, 0.01) if len(finite_diffs) else 0.5
    diff_norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    ref_vals = raw[0, np.isfinite(raw[0, :])]
    vmin = max(np.min(ref_vals) - 0.05, 0.0) if len(ref_vals) else 0.0
    vmax = min(np.max(ref_vals) + 0.02,  1.0) if len(ref_vals) else 1.0
    ref_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # ── Draw ─────────────────────────────────────────────────────────────────
    title = f"{detector}_ensemble  —  {dataset}"
    fig, ax = _make_figure(n_rows, n_cols)

    # horizontal separator between baselines and ensemble rules
    sep_y = TOP_PAD + (n_rows - 1 - len(cross_keys)) * CELL_H

    for i in range(n_rows):
        is_aligned = (i == 0)
        # draw separator line above first ensemble rule row
        if i == 1 + len(cross_keys) and ensemble_keys:
            ax.plot(
                [LEFT_PAD, LEFT_PAD + n_cols * CELL_W],
                [TOP_PAD + (n_rows - i) * CELL_H] * 2,
                color="#6666aa", linewidth=1.2, linestyle="--", zorder=11
            )

        for j in range(n_cols):
            val  = raw[i, j]
            diff = diffs[i, j]
            x = LEFT_PAD + j * CELL_W
            y = TOP_PAD  + (n_rows - 1 - i) * CELL_H

            if is_aligned:
                if np.isfinite(val):
                    rgba = CMAP_REF(ref_norm(val))
                    bg   = mcolors.to_hex(rgba)
                    lum  = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
                    txt_color = "#1a1a2e" if lum > 0.52 else "#f0f0f0"
                else:
                    bg, txt_color = "#2e2e4a", "#888888"
            else:
                if np.isfinite(diff):
                    rgba = CMAP_DIFF(diff_norm(diff))
                    bg   = mcolors.to_hex(rgba)
                    lum  = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
                    txt_color = "#1a1a2e" if lum > 0.52 else "#f0f0f0"
                else:
                    bg, txt_color = "#2e2e4a", "#888888"

            ax.add_patch(FancyBboxPatch(
                (x+0.06, y+0.06), CELL_W-0.12, CELL_H-0.12,
                boxstyle="round,pad=0.05", linewidth=0,
                facecolor=bg, clip_on=False,
            ))

            if is_aligned:
                txt = f"{val:.3f}" if np.isfinite(val) else "N/A"
                ax.text(x + CELL_W/2, y + CELL_H/2, txt,
                        ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color=txt_color, fontfamily="monospace")
            else:
                top_txt = f"{val:.3f}"  if np.isfinite(val)  else "N/A"
                bot_txt = f"{diff:+.3f}" if np.isfinite(diff) else ""
                ax.text(x + CELL_W/2, y + CELL_H*0.63, top_txt,
                        ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color=txt_color, fontfamily="monospace")
                ax.text(x + CELL_W/2, y + CELL_H*0.27, bot_txt,
                        ha="center", va="center",
                        fontsize=8.5, color=txt_color, fontfamily="monospace")

    _draw_chrome(ax, row_labels, title, n_rows, n_cols)
    _add_colorbar(fig, CMAP_DIFF, diff_norm,
                  "absolute diff vs aligned model   (green = higher  /  red = lower)")
    _save(fig, output_path)


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+",
                        help="One or more ensemble_report_*.json files")
    parser.add_argument("--out-dir", default="/second-disk/Image-Deepfake-Detectors-Public-Library/results/ensemble/tables/",
                        help="Directory to save plots (default: current dir)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for report_path in args.reports:
        if not os.path.exists(report_path):
            print(f"[warn] not found: {report_path}")
            continue
        with open(report_path) as f:
            report = json.load(f)

        base = os.path.splitext(os.path.basename(report_path))[0]
        out  = Path(args.out_dir) / f"{base}.png"
        plot_ensemble(report, out)


if __name__ == "__main__":
    main()