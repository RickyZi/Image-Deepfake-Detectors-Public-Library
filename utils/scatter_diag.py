"""
Scatter-plot version of the Balanced Accuracy comparisons per platform
(PRE, facebook, telegram, twitter).

For each platform, one figure is produced. Each point is one
(preset, detector) result:
  - x position         -> Standard accuracy (baseline / _pretrained on the preset)
  - y position         -> Adapted accuracy  (fine-tuned / _ft on the preset)
  - marker SHAPE+COLOR -> detector (R50_ND, R50_TF, CLIP-D), redundantly
                          encoded so the 3 groups are easy to tell apart

A dashed y=x diagonal marks "no change from fine-tuning": points above it
improved with fine-tuning, points below it got worse. A second reference
(one dashed vertical + horizontal line per detector) marks that detector's
accuracy on the CLEAN (no-preset) data, so you can see how both the
Standard and Adapted points compare to the un-adapted baseline.

Reuses the data-loading / preset-labeling helpers from plot_results.py so
both scripts stay consistent. Reads results_nested.json, produced by
collect_results.py.

Run:
    python plot_scatter.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
import matplotlib.font_manager as fm
import re
import os
from pathlib import Path

# from bar_plot.plot_results import (
#     load_results, extract_values, get_preset_list,
#     PLATFORM_CONFIG, DETECTORS, DETECTOR_LABELS, _resolve_font,
# )

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

# marker shape AND color per detector -- redundant encoding on purpose,
# makes the 3 detector groups easy to tell apart at a glance
DETECTOR_STYLE = {
    "R50_nodown": {"marker": "o", "color": "#4C72B0"},
    "R50_TF":     {"marker": "s", "color": "#DD8452"},
    "CLIP-D":     {"marker": "^", "color": "#55A868"},
}

AXIS_LIMITS = [45, 100]

MAP_SOCIAL_KEY = {
    'facebook': 'FB',
    'telegram': 'TL',
    'twitter': 'X'
}

PLATFORM_CONFIG = {
    "PRE": {
        # "baseline_platform": "tf2k",
        # "baseline_dataset": "dataset",
        # "exclude_datasets": set(),
        "baseline_platform": "PRE",
        "baseline_dataset": "dataset",
        "exclude_datasets": {"dataset"},
    },
    "facebook": {
        "baseline_platform": "social",
        "baseline_dataset": "Facebook",
        "exclude_datasets": set(),
    },
    "telegram": {
        "baseline_platform": "social",
        "baseline_dataset": "Telegram",
        "exclude_datasets": set(),
    },
    "twitter": {
        "baseline_platform": "social",
        "baseline_dataset": "Twitter",
        "exclude_datasets": set(),
    },
}

YLABELS_FLAG = True


RESULTS_FILE = "results_nested.json"

DETECTORS = ["R50_nodown", "R50_TF", "CLIP-D"]
DETECTOR_LABELS = {"R50_nodown": "R50-ND", "R50_TF": "R50-E2P", "CLIP-D": "CLIP-D"}

PREFERRED_FONT = "Liberation Mono"


def _resolve_font(preferred=PREFERRED_FONT):
    """Use the preferred font if installed, otherwise fall back to the
    system's generic monospace family so we don't spam 'findfont' warnings."""
    installed = {f.name for f in fm.fontManager.ttflist}
    if preferred in installed:
        return preferred
    print(f"[info] font '{preferred}' not found on this system, using 'monospace' instead")
    return "monospace"

def load_results(path=RESULTS_FILE):
    with open(path) as f:
        return json.load(f)


def extract_values(results, platform_key, dataset, variant, detectors=DETECTORS, metric="Balanced Acc"):
    """Values as percentages (0-100) for each detector; np.nan if missing.

    Since collect_results.py now stores the FULL metrics dict per variant
    (TPR, TNR, Acc, Balanced Acc, F1, AUC, ...), `metric` selects which one
    to pull. Defaults to "Balanced Acc" so existing callers are unaffected.
    """
    vals = []
    platform_entry = results.get(platform_key, {})
    dataset_entry = platform_entry.get(dataset, {})
    for det in detectors:
        variant_entry = dataset_entry.get(det, {}).get(variant)
        v = variant_entry.get(metric) if isinstance(variant_entry, dict) else None
        vals.append(v * 100 if v is not None else np.nan)
    return vals


def get_preset_list(results, platform_key, exclude, detectors=DETECTORS):
    datasets = results.get(platform_key, {})
    presets = []
    for d in sorted(datasets):
        if d in exclude:
            continue
        # keep only datasets that actually have some pretrained/ft value
        has_data = any(
            datasets[d].get(det, {}).get("pretrained") is not None
            or datasets[d].get(det, {}).get("ft") is not None
            for det in detectors
        )
        if has_data:
            presets.append(d)
    return presets

# ------------------------------------------------------------------

def build_legend_handles(detectors=DETECTORS):
    """Handles for detector marker/color + the two reference lines. Shared
    between the per-plot legend and the standalone legend image."""
    handles = [
        Line2D([0], [0], marker=DETECTOR_STYLE[d]["marker"], color=DETECTOR_STYLE[d]["color"],
               linestyle='', markersize=22, markeredgecolor='black', markeredgewidth=0.7,
               label=DETECTOR_LABELS.get(d, d))
        for d in detectors
    ]
    # handles.append(
    #     Line2D([0], [0], color='black', linestyle='--', linewidth=1.5,
    #            label='y = x (no change)')
    # )
    # handles.append(
    #     Line2D([0], [0], color='#666666', linestyle=':', linewidth=1.5,
    #            label='Baseline (clean data)')
    # )
    return handles
 
 
def save_standalone_legend(out_path="scatter_legend.png", detectors=DETECTORS):
    """Save just the legend (no axes/data) as its own image -- handy for a
    paper where one shared legend sits below/beside a grid of subplots."""
    plt.rcParams['font.family'] = _resolve_font()
    handles = build_legend_handles(detectors)
 
    fig = plt.figure(figsize=(8, 1))
    fig.legend(handles=handles, title="Detectors",
               loc='center', ncol=len(handles),
               fontsize=20, title_fontsize=20, frameon=False)
    plt.savefig(os.path.join("./scatter_plot/", out_path))
    plt.close(fig)
    print(f"Saved: {out_path}")



def plot_platform_scatter(results, platform_key, config, detectors=DETECTORS, out_path=None, show_legend = False):
    exclude = config["exclude_datasets"]
    baseline_platform = config["baseline_platform"]
    baseline_dataset = config["baseline_dataset"]

    presets = get_preset_list(results, platform_key, exclude, detectors)
    if not presets:
        print(f"[skip] no preset datasets found for platform '{platform_key}'")
        return

    baseline_clean = extract_values(results, baseline_platform, baseline_dataset,
                                     "pretrained", detectors)
    if any(np.isnan(v) for v in baseline_clean):
        print(f"[warn] missing clean-baseline value(s) for platform '{platform_key}': "
              f"{baseline_clean} (from {baseline_platform}/{baseline_dataset})")

    plt.rcParams['font.family'] = _resolve_font()
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_box_aspect(0.75)

    # diagonal: y = x, i.e. "fine-tuning made no difference"
    ax.plot(AXIS_LIMITS, AXIS_LIMITS, linestyle='--', color='black', linewidth=1.0, alpha=0.6, zorder=1)

    # clean-data baseline reference: one dashed cross (vertical + horizontal)
    # per detector, at that detector's accuracy on the un-preset clean data
    # for i, detector in enumerate(detectors):
    #     val = baseline_clean[i]
    #     if np.isnan(val):
    #         continue
    #     ax.axvline(val, linestyle=':', color='#666666', linewidth=1.0, alpha=0.6, zorder=1)
    #     ax.axhline(val, linestyle=':', color='#666666', linewidth=1.0, alpha=0.6, zorder=1)
    #     label = DETECTOR_LABELS.get(detector, detector)
    #     ax.text(val, AXIS_LIMITS[1] - 1, label, rotation=90, ha='right', va='top',
    #             fontsize=9, color='#666666', zorder=2)
    #     ax.text(AXIS_LIMITS[1] - 1, val, label, ha='right', va='bottom',
    #             fontsize=9, color='#666666', zorder=2)

    for preset in presets:
        for detector in detectors:
            style = DETECTOR_STYLE.get(detector, {"marker": "o", "color": "gray"})
            x = extract_values(results, platform_key, preset, "pretrained", [detector])[0]
            y = extract_values(results, platform_key, preset, "ft", [detector])[0]
            if np.isnan(x) or np.isnan(y):
                continue
            ax.scatter(x, y, color=style["color"], marker=style["marker"], s=200,
                       edgecolor='black', linewidth=0.7, alpha=0.8, zorder=3)

    ax.set_xlim(AXIS_LIMITS)
    ax.set_ylim(AXIS_LIMITS)
    ax.set_aspect('equal')
    ax.set_xlabel("BA - Standard", fontsize=35)
    ax.set_ylabel("BA - Adapted", fontsize=35)
    ax.tick_params(labelsize=25)
    ax.grid(True, linestyle='--', alpha=0.3)
    # title_text = platform_key if platform_key.isupper() else platform_key.capitalize()
    title_text = MAP_SOCIAL_KEY[platform_key] if platform_key in MAP_SOCIAL_KEY.keys() else platform_key.upper()
    ax.set_title(title_text, fontsize=35, weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # legend: detector -> marker shape + color, plus the two reference lines
    # detector_handles = [
    #     Line2D([0], [0], marker=DETECTOR_STYLE[d]["marker"], color=DETECTOR_STYLE[d]["color"],
    #            linestyle='', markersize=18, markeredgecolor='black', markeredgewidth=0.7,
    #            label=DETECTOR_LABELS.get(d, d))
    #     for d in detectors
    # ]
    # detector_handles.append(
    #     Line2D([0], [0], color='black', linestyle='--', linewidth=1.5,
    #            label='y = x (no change)')
    # )
    # detector_handles.append(
    #     Line2D([0], [0], color='#666666', linestyle=':', linewidth=1.5,
    #            label='Baseline (clean data)')
    # )
    # show legend only for PRE scatter plot
    # legend: detector -> marker shape + color, plus the two reference lines
    if show_legend:
        detector_handles = build_legend_handles(detectors)
        ax.legend(handles=detector_handles, title="Detector",
                  loc='upper center', bbox_to_anchor=(0.5, -0.1), 
                  fontsize=12, title_fontsize=13, ncol=len(detector_handles),
                  frameon=False)


    plt.tight_layout()
    out_path = out_path or f"./scatter_plot/"
    # check if path exists
    if not Path(out_path).is_dir():
        os.makedirs(out_path, exist_ok = True)
    plt.savefig(os.path.join(out_path, f'scatter_balanced_acc_{platform_key}.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    results = load_results()
    for platform_key, config in PLATFORM_CONFIG.items():
        plot_platform_scatter(results, platform_key, config) #, show_legend=(platform_key== 'PRE'))

    save_standalone_legend()