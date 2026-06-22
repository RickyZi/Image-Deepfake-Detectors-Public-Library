"""
Plot F1 scores from aggregated metrics JSON files.

Folder structure:
    results/
      pretrained/
        <dataset_name>/
          <model_name>/          (e.g. R50_nodown_pretrained)
            *_aggregated_metrics.json

The script walks the tree, collects overall F1 per dataset for each model,
and produces one bar chart per model saved as PNG under ./plots/.
"""

import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime
import argparse
import itertools

# -------------------------------------------------------- #
# Configuration
RESULTS_DIR = Path("./results/pretrained")
OUTPUT_DIR  = Path("./results/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REF_FOLDER = 'dataset' # ref run folder name
REF_LABEL = 'tf2k_dataset' # label ref run

# Colour palette
# PALETTE = [
#     "#4C72B0", "#DD8452", "#55A868", "#C44E52",
#     "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
#     "#CCB974", "#64B5CD",
# ]

PALETTE_REF      = "#4C72B0"   # blue  — reference bar
PALETTE   = [           # cycling palette for the other datasets
    "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3",
    "#8C8C8C", "#CCB974", "#64B5CD",
]


# --------------------------------------------------------- #
# Data Collection
# --------------------------------------------------------- #

def find_json(folder: Path):
    json_files = sorted(folder.glob("*_aggregated_metrics.json"))
    # There should be exactly one per run folder; take the first.
    return json_files[0] if json_files else None


def collect_f1(subdir_name: str) -> dict[str, float]:
    """
    Walk results/pretrained/<dataset>/<subdir_name>/
    Returns {dataset_label: f1_value} including the reference entry.
    """
    scores = {}
 
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Results root not found: {RESULTS_DIR.resolve()}")
 
    for dataset_dir in sorted(RESULTS_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue
 
        model_dir = dataset_dir / subdir_name
        if not model_dir.is_dir():
            continue
 
        json_path = find_json(model_dir)
        if json_path is None:
            print(f"  [skip] no aggregated_metrics.json in {model_dir}")
            continue
 
        with json_path.open() as f:
            payload = json.load(f)
 
        try:
            f1 = payload["overall"]["F1"]
        except KeyError:
            print(f"  [skip] 'overall.F1' missing in {json_path}")
            continue
 
        label = REF_LABEL if dataset_dir.name == REF_FOLDER else dataset_dir.name
        scores[label] = f1
        print(f"  [ok] model={model_dir.name!r:35s}  dataset={dataset_dir.name!r:25s}  F1={f1:.4f}")
 
    return scores



def plot_f1(scores: dict[str, float], title: str, out_path: Path):
    """
    Bar chart for one model variant (baseline or FT).
    Reference bar is always first and coloured distinctly.
    """
    if not scores:
        print(f"  [skip] no data for: {title}")
        return

    if len(scores) == 1 and 'tf2k_dataset' in scores:
        print(f" [skip] NO FT DATA, only 'tf2k_dataset' in f1 ft_scores")
        return
 
    # Reference bar first, then remaining datasets sorted alphabetically
    datasets = (
        [REF_LABEL] if REF_LABEL in scores else []
    ) + sorted(k for k in scores if k != REF_LABEL)
 
    f1_vals = [scores[d] for d in datasets]
    n = len(datasets)
 
    # Colours: reference = blue, others cycle through PALETTE_OTHERS
    palette_iter = itertools.cycle(PALETTE)
    colors = [
        PALETTE_REF if d == REF_LABEL else next(palette_iter)
        for d in datasets
    ]
 
    fig_w = max(8, n * 1.1 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    fig.patch.set_facecolor("#F7F7F7")
    ax.set_facecolor("#F7F7F7")
 
    bars = ax.bar(datasets, f1_vals, color=colors, width=0.6,
                  edgecolor="white", linewidth=0.8, zorder=3)
 
    # Value labels on top of each bar
    for bar, val in zip(bars, f1_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.012,
            f"{val:.3f}",
            ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color="#333333",
        )
 
    # Reference bar annotation
    if REF_LABEL in scores:
        ref_val = scores[REF_LABEL]
        ax.axhline(ref_val, color=PALETTE_REF, linewidth=1.0,
                   linestyle="--", alpha=0.5, zorder=2)
 
    # Gridlines
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.6,
            color="#CCCCCC", zorder=0)
    ax.grid(axis="y", which="minor", linestyle=":",  linewidth=0.4,
            color="#DDDDDD", zorder=0)
 
    # Axes styling
    ax.set_ylim(0, min(1.15, max(f1_vals) + 0.18))
    ax.set_ylabel("F1 Score", fontsize=11, labelpad=8)
    ax.set_xlabel("Dataset", fontsize=11, labelpad=8)
    ax.tick_params(axis="x", rotation=35, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#BBBBBB")
 
    # Legend for the reference bar
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=PALETTE_REF, label=f"reference ({REF_LABEL})")]
    # ax.legend(handles=legend_elements, fontsize=8.5, loc="lower right", framealpha=0.7)
 
    fig.suptitle(title, fontsize=14, fontweight="bold", color="#222222", y=1.02)
    ax.set_title("Overall F1 Score per Dataset", fontsize=10, color="#555555", pad=6)
 
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → saved: {out_path}")



# ------------------------------------- #
# ENTRY POINT #
# ------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Plot F1 scores per dataset for one detector model.")
    parser.add_argument("--model", type=str, default="R50_nodown", help="Model name prefix, e.g. R50_nodown or CLIP-D")
    parser.add_argument("--unfreezeL4", action = "store_true")
    parser.add_argument("--mlp", action = "store_true")
    parser.add_argument("--skipbase", action = "store_true")
    args = parser.parse_args()
 
    baseline_subdir = f"{args.model}_pretrained"
    # ft_subdir       = f"{args.model}_ft_unfreezeL4" if args.unfreezeL4 else f"{args.model}_ft"
    if args.unfreezeL4:
        ft_subdir       = f"{args.model}_ft_unfreezeL4"
    elif args.mlp:
        ft_subdir       = f"{args.model}_ft_MLP"
    else:
        ft_subdir       = f"{args.model}_ft"
    timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
 
    print(f"\nModel    : {args.model}")
    print(f"Baseline : {baseline_subdir}")
    print(f"FT       : {ft_subdir}\n")
 
    # Collect F1 scores
    print(f"── Collecting baseline F1 scores ({baseline_subdir}) ─────────────")
    baseline_scores = collect_f1(baseline_subdir)
    
    print(f"\n── Collecting FT F1 scores ({ft_subdir}) ─────────────────────────")
    ft_scores = collect_f1(ft_subdir)

    # Inject reference into FT scores if missing (dataset/ folder only has
    # the pretrained subfolder, so collect_f1 for ft_subdir won't find it)
    if REF_LABEL not in ft_scores and REF_LABEL in baseline_scores:
        ft_scores[REF_LABEL] = baseline_scores[REF_LABEL]
        print(f"  [ref] injected '{REF_LABEL}' from baseline into FT scores")

    # breakpoint()
    # Print summary
    print("\n── Baseline F1 summary ──────────────────────────────────────────")
    for d, v in sorted(baseline_scores.items()):
        marker = " ← ref" if d == REF_LABEL else ""
        print(f"  {d:<30s}  F1 = {v:.4f}{marker}")
 
    print("\n── FT F1 summary ────────────────────────────────────────────────")
    for d, v in sorted(ft_scores.items()):
        marker = " ← ref" if d == REF_LABEL else ""
        print(f"  {d:<30s}  F1 = {v:.4f}{marker}")
 
    # Plot
    print("\n── Plotting ─────────────────────────────────────────────────────")
    if not args.skipbase:
        plot_f1(
            scores   = baseline_scores,
            title    = f"{args.model} baseline",
            out_path = OUTPUT_DIR / f"{baseline_subdir}_f1_{timestamp}.png",
        )
    plot_f1(
        scores   = ft_scores,
        title    =  ft_subdir, # f"{args.model} FT" if not args.unfreezeL4 else f"{args.model} FT_unfreezeL4",
        out_path = OUTPUT_DIR / f"{ft_subdir}_f1_{timestamp}.png",
    )
 
    print(f"\nDone — plots saved to {OUTPUT_DIR.resolve()}/")
 
 
if __name__ == "__main__":
    main()