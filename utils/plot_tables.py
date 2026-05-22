"""
plot_metric_tables.py

Generates ONE color-coded table per dataset:
  - Comparison JSONs  → score + % change vs reference (red/green)
  - Reference JSON    → raw scores only, neutral blue palette

Usage:
Auto-discovery mode (no args):
    python plot_metric_tables.py
    Scans ./results/metrics_comparison/metric_comparison_*_vs_*.json
    Optionally also plots the reference if found at ./results/demo_images/aggregated_metrics.json

Explicit mode:
    python plot_metric_tables.py [--ref <ref.json>] [<comparison1.json> ...]

    --ref <path>   Plot the reference table from this file (raw scores, no diffs).
                   In auto-discovery mode the default reference path is used if it exists.

Output:
    ./results//metric_tables/<dataset_name>.png
"""

import json
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from pathlib import Path


# Constants

MODELS  = ["R50_nodown", "CLIP-D", "NPR", "R50_TF", "P2G"]
METRICS = ["TPR", "TNR", "Acc", "Balanced Acc", "F1", "AUC"]

# colors configurations
# Red → white → green  (comparison tables)
CMAP_DIFF = mcolors.LinearSegmentedColormap.from_list(
    "rg_diverging", ["#b22222", "#f5f5f0", "#2a7a2a"],
)

# Sequential blue palette for the reference table
CMAP_REF = mcolors.LinearSegmentedColormap.from_list(
    "blue_seq", ["#0d1b4b", "#1a3a8a", "#4a7fd4", "#a8c8f8"],
)

OUTPUT_DIR = Path("./results/metric_tables")
INPUT_DIR   = Path("./results/demo/")
DEFAULT_REF = Path("./results/demo/demo_images/demo_images_aggregated_metrics.json")


# Data loading helpers
def load_json(path):
    with open(path) as f:
        return json.load(f)


def parse_names(data, path):
    """Return (test_name, ref_name) from the 'dataset' field."""
    raw = data.get("dataset", Path(path).stem)
    parts = raw.split(" vs ")
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return raw, ""


def extract_comparison_table(data):
    """
    From a comparison JSON returns:
        values (n_models, n_metrics)  – absolute test scores
        pcts   (n_models, n_metrics)  – % change vs reference
    """
    values = np.full((len(MODELS), len(METRICS)), np.nan)
    pcts   = np.full((len(MODELS), len(METRICS)), np.nan)
    for i, model in enumerate(MODELS):
        md = data.get(model)
        if not md:
            continue
        for j, metric in enumerate(METRICS):
            v = md.get(metric)
            # p = md.get(f"{metric}_percent_diff")
            p = md.get(f"{metric}_abs_diff")
            if v is not None:
                values[i, j] = v
            if p is not None:
                pcts[i, j] = p
    return values, pcts


def extract_reference_table(data):
    """
    From a raw aggregated_metrics / reference JSON returns:
        values (n_models, n_metrics)  – absolute scores (no diffs)
    Handles both key styles:
        {"CLIP-D": {"TPR": ...}}  and  {"CLIP-D_demo_images": {"TPR": ...}}
    """
    values = np.full((len(MODELS), len(METRICS)), np.nan)
    for i, model in enumerate(MODELS):
        # try exact key first, then with any suffix
        md = data.get(model)
        if md is None:
            for key in data:
                if key.startswith(model):
                    md = data[key]
                    break
        if not md:
            continue
        for j, metric in enumerate(METRICS):
            v = md.get(metric)
            if v is not None:
                values[i, j] = v
    return values


# Table layout helpers 

def _base_figure():
    n_models, n_metrics = len(MODELS), len(METRICS)
    cell_w, cell_h = 1.6, 0.60 # shrinked from 2.2, 1.0
    left_pad, top_pad, bottom_pad = 1.90, 0.60, 0.55
    fig_w = left_pad + n_metrics * cell_w + 0.3
    fig_h = top_pad  + n_models  * cell_h + bottom_pad
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.set_axis_off()
    return fig, ax, cell_w, cell_h, left_pad, top_pad, n_models, n_metrics


def _draw_labels_and_title(ax, title, cell_w, cell_h, left_pad, top_pad, n_models, n_metrics):
    # column headers
    for j, metric in enumerate(METRICS):
        ax.text(
            left_pad + j * cell_w + cell_w / 2,
            top_pad + n_models * cell_h + 0.08,
            metric, ha="center", va="bottom",
            fontsize=11, fontweight="bold",
            color="#e0e0f8", fontfamily="monospace",
        )
    # row labels
    for i, model in enumerate(MODELS):
        ax.text(
            left_pad - 0.14,
            top_pad + (n_models - 1 - i) * cell_h + cell_h / 2,
            model, ha="right", va="center",
            fontsize=10, fontweight="bold",
            color="#e8e8f8", fontfamily="monospace",
        )
    # title
    ax.text(
        (left_pad + n_metrics * cell_w) / 2,
        top_pad + n_models * cell_h + 0.44,
        title, ha="center", va="bottom",
        fontsize=13, fontweight="bold",
        color="#ffffff", fontfamily="monospace",
    )
    # column dividers
    for j in range(1, n_metrics):
        x = left_pad + j * cell_w
        ax.plot([x, x],
                [top_pad - 0.02, top_pad + n_models * cell_h + 0.02],
                color="#1a1a2e", linewidth=2, zorder=10)

    ax.set_xlim(0, left_pad + n_metrics * cell_w + 0.2)
    ax.set_ylim(0, top_pad + n_models * cell_h + 0.65)


def _save(fig, output_path):
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved -> {output_path}")


# Comparison table (score + abs change, red/green) 

def plot_comparison_table(values, pcts, test_name, ref_name, output_path):
    fig, ax, cell_w, cell_h, left_pad, top_pad, n_models, n_metrics = _base_figure()

    finite = pcts[np.isfinite(pcts)]
    abs_max = min(max(np.max(np.abs(finite)) if len(finite) else 1.0, 1.0) * 1.08, 80.0)
    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    for i in range(n_models):
        for j in range(n_metrics):
            val, pct = values[i, j], pcts[i, j]
            x = left_pad + j * cell_w
            y = top_pad  + (n_models - 1 - i) * cell_h

            if np.isnan(pct):
                bg, top_txt, bot_txt, txt_color = "#2e2e4a", "N/A", "", "#888888"
            else:
                rgba = CMAP_DIFF(norm(pct))
                bg   = mcolors.to_hex(rgba)
                lum  = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
                txt_color = "#1a1a2e" if lum > 0.52 else "#f0f0f0"
                top_txt = f"{val:.3f}"
                # bot_txt = f"{'+'if pct>=0 else ''}{pct:.1f}"
                bot_txt = f"{pct:.3f}"

            ax.add_patch(FancyBboxPatch(
                (x+0.06, y+0.06), cell_w-0.12, cell_h-0.12,
                boxstyle="round,pad=0.05", linewidth=0, facecolor=bg, clip_on=False,
            ))
            ax.text(x+cell_w/2, y+cell_h*0.62, top_txt,
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color=txt_color, fontfamily="monospace")
            ax.text(x+cell_w/2, y+cell_h*0.28, bot_txt,
                    ha="center", va="center", fontsize=8.5,
                    color=txt_color, fontfamily="monospace")

    _draw_labels_and_title(ax, f"{test_name}  vs  {ref_name}",
                           cell_w, cell_h, left_pad, top_pad, n_models, n_metrics)

    cbar_ax = fig.add_axes([0.12, 0.015, 0.78, 0.025])
    sm = plt.cm.ScalarMappable(cmap=CMAP_DIFF, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(colors="#aaaacc", labelsize=7.5)
    cbar.set_label("% change vs reference  (green = improvement  /  red = degradation)",
                   color="#aaaacc", fontsize=7.5)
    cbar.outline.set_edgecolor("#444466")

    _save(fig, output_path)


# Reference table (raw scores only, blue palette)
def plot_reference_table(values, dataset_name, output_path):
    fig, ax, cell_w, cell_h, left_pad, top_pad, n_models, n_metrics = _base_figure()

    finite = values[np.isfinite(values)]
    vmin = max(np.min(finite) - 0.05, 0.0) if len(finite) else 0.0
    vmax = min(np.max(finite) + 0.02, 1.0) if len(finite) else 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for i in range(n_models):
        for j in range(n_metrics):
            val = values[i, j]
            x = left_pad + j * cell_w
            y = top_pad  + (n_models - 1 - i) * cell_h

            if np.isnan(val):
                bg, txt, txt_color = "#2e2e4a", "N/A", "#888888"
            else:
                rgba = CMAP_REF(norm(val))
                bg   = mcolors.to_hex(rgba)
                lum  = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
                txt_color = "#1a1a2e" if lum > 0.52 else "#f0f0f0"
                txt = f"{val:.3f}"

            ax.add_patch(FancyBboxPatch(
                (x+0.06, y+0.06), cell_w-0.12, cell_h-0.12,
                boxstyle="round,pad=0.05", linewidth=0, facecolor=bg, clip_on=False,
            ))
            ax.text(x+cell_w/2, y+cell_h/2, txt,
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color=txt_color, fontfamily="monospace")

    _draw_labels_and_title(ax, f"{dataset_name}  (reference)",
                           cell_w, cell_h, left_pad, top_pad, n_models, n_metrics)

    cbar_ax = fig.add_axes([0.12, 0.015, 0.78, 0.025])
    sm = plt.cm.ScalarMappable(cmap=CMAP_REF, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(colors="#aaaacc", labelsize=7.5)
    cbar.set_label("absolute metric score", color="#aaaacc", fontsize=7.5)
    cbar.outline.set_edgecolor("#444466")

    _save(fig, output_path)


# Auto-discover jsons files 
def discover_jsons(folder):
    files = sorted(folder.glob("metric_comparison_*_vs_*.json"))
    if not files:
        print(f"No comparison files found in {folder.resolve()}")
        print("Expected pattern: metric_comparison_<dataset>_vs_<reference>.json")
    return files


# ------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Plot metric comparison tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ref", metavar="REF_JSON", default=None,
        help="Path to reference JSON (raw scores). Plots a standalone reference table.",
    )
    parser.add_argument(
        "comparisons", nargs="*",
        help="Comparison JSON files. If omitted, auto-discovers from ./results/demo/",
    )
    args = parser.parse_args()


    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plotted = 0

    # reference table 
    ref_path = Path(args.ref) if args.ref else (DEFAULT_REF if DEFAULT_REF.exists() else None)
    if ref_path:
        if ref_path.exists():
            print(f"Plotting reference: {ref_path} ...")
            ref_data = load_json(ref_path)
            ref_name = ref_data.get("dataset", ref_path.stem)
            ref_values = extract_reference_table(ref_data)
            safe = ref_name.replace(" ", "_")
            plot_reference_table(ref_values, ref_name, OUTPUT_DIR / f"{safe}_reference.png")
            plotted += 1
        else:
            print(f"Warning: reference file not found at {ref_path}")
    elif args.ref is None and not DEFAULT_REF.exists():
        print("(No reference file found at default path — skipping reference table.)")
        print("  Pass --ref <path> to plot one explicitly.\n")

    # comparison tables
    if args.comparisons:
        paths = [Path(p) for p in args.comparisons]
        print(f"Using {len(paths)} file(s) from command-line arguments.\n")
    else:
        paths = discover_jsons(INPUT_DIR)
        print(f"Found {len(paths)} comparison file(s) in {INPUT_DIR.resolve()}\n")

    for path in paths:
        comp = load_json(path)
        test_name, ref_name = parse_names(comp, path)
        values, pcts = extract_comparison_table(comp)
        safe = test_name.replace(" ", "_")
        out  = OUTPUT_DIR / f"{test_name}_vs_{ref_name}.png"
        print(f"Plotting {test_name} ...")
        plot_comparison_table(values, pcts, test_name, ref_name, out)
        plotted += 1

    print(f"\nDone. {plotted} table(s) saved to  {OUTPUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()

