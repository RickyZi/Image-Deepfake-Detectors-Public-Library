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

# -------------------------------------------------------- #
# Configuration
RESULTS_DIR = Path("./results/pretrained")
OUTPUT_DIR  = Path("./results/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Colour palette
PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]
# -------------------------------------------------------- #

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # add timestamp to test run

parser = argparse.ArgumentParser(description='Plotting test-run tables for one detector.')

parser.add_argument('--model', type = str,  default = 'R50_nodown', help = 'Model run name, i.e. R50_nodown or CLIP-D')

args = parser.parse_args()

print(f"model: {args.model}")


# Collect data
# data[model_name][dataset_name] = f1_value
data: dict[str, dict[str, float]] = defaultdict(dict)

if not RESULTS_DIR.exists():
    raise FileNotFoundError(f"Results directory not found: {RESULTS_DIR.resolve()}")

# Walk:  pretrained/<dataset>/<model_folder>/*_aggregated_metrics.json
for dataset_dir in sorted(RESULTS_DIR.iterdir()):
    print(f"dataset_dir: {dataset_dir}")
    if not dataset_dir.is_dir():
        continue

    dataset_name = 'tf_dataset' if dataset_dir.name == 'dataset' else dataset_dir.name # e.g. "autumn_TM01"

    for model_dir in sorted(dataset_dir.iterdir()):
        print(f"model_dir: {model_dir}")
        if not model_dir.is_dir():
            continue
        
        if args.model in model_dir.name:
        #     if '_pretrained' in model_dir.name:
        #         model_name = model_dir.name.replace('_pretrained', '_baseline')
        #         print(f"model_name: {model_name}")
        #     else:
        #         model_name = model_dir.name
            model_name = f'{args.model}_baseline' if 'pretrained' in model_dir.name else model_dir.name # e.g. "R50_nodown_pretrained"
            print(f"model_name: {model_name}")
            # breakpoint()

            json_files = sorted(model_dir.glob("*_aggregated_metrics.json"))
            if not json_files:
                print(f"  [skip] no aggregated_metrics.json in {model_dir}")
                continue

            # There should be exactly one per run folder; take the first.
            fpath = json_files[0]
            with fpath.open() as f:
                payload = json.load(f)

            try:
                f1 = payload["overall"]["F1"]
            except KeyError:
                print(f"  [skip] 'overall.F1' missing in {fpath}")
                continue

            data[model_name][dataset_name] = f1
            print(f"  [ok] model={model_name!r:35s}  dataset={dataset_name!r:25s}  F1={f1:.4f}")

if not data:
    raise ValueError("No valid results found. Check the folder structure.")

# Pretty-print the collected dictionary
print("\n── Collected F1 scores ──────────────────────────────────────────────")
for model, scores in sorted(data.items()):
    print(f"\n  Model: {model}")
    for dataset, f1 in sorted(scores.items()):
        print(f"    {dataset:<30s}  F1 = {f1:.4f}")

# Plot one figure per model
for model, dataset_scores in sorted(data.items()):
    datasets = sorted(dataset_scores.keys())
    f1_vals  = [dataset_scores[d] for d in datasets]
    n        = len(datasets)

    fig_w = max(8, n * 1.0 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    fig.patch.set_facecolor("#F7F7F7")
    ax.set_facecolor("#F7F7F7")

    colors = [PALETTE[i % len(PALETTE)] for i in range(n)]
    bars   = ax.bar(datasets, f1_vals, color=colors, width=0.6,
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

    # Model name as main title
    fig.suptitle(model, fontsize=14, fontweight="bold", color="#222222", y=1.02)
    ax.set_title("Overall F1 Score per Dataset", fontsize=10,
                 color="#555555", pad=6)

    plt.tight_layout()

    out_path = OUTPUT_DIR / f"{model}_f1_{timestamp}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  → plot saved: {out_path}")

print(f"\nDone — {len(data)} plot(s) saved to {OUTPUT_DIR.resolve()}")