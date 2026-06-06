"""
plot_metric_tables.py

Walks results/pretrained/<dataset>/ and produces TWO color-coded metric tables:

  1. R50_nodown baseline  →  rows = datasets, cols = metrics
                             reference row  (tf2k_dataset) shows RAW scores
                             all other rows show RAW score + absolute diff vs ref

  2. R50_nodown FT        →  same layout, same reference, but reads the
                             R50_nodown_ft sub-folder instead

Folder conventions
------------------
  Baseline : results/pretrained/<dataset>/R50_nodown_pretrained/
             *_aggregated_metrics.json
  FT       : results/pretrained/<dataset>/R50_nodown_ft/
             *_aggregated_metrics.json
  Reference: dataset folder named exactly 'dataset'
             (displayed in the table as 'tf2k_dataset')

Usage
-----
    python plot_metric_tables.py

Output
------
    ./results/metric_tables/R50_nodown_baseline.png
    ./results/metric_tables/R50_nodown_FT.png
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# Constants
METRICS      = ["TPR", "TNR", "Acc", "Balanced Acc", "F1", "AUC"]
REF_FOLDER   = "dataset"          # folder name that is the reference run
REF_LABEL    = "tf2k_dataset"     # how to display it in the table

RESULTS_ROOT = Path("./results/pretrained")
OUTPUT_DIR   = Path("./results/metric_tables")

# Subfolder name inside each dataset dir
BASELINE_SUBDIR = "R50_nodown_pretrained"
FT_SUBDIR       = "R50_nodown_ft"

# Colormaps
CMAP_DIFF = mcolors.LinearSegmentedColormap.from_list(
    "rg_diverging", ["#b22222", "#f5f5f0", "#2a7a2a"]
)
CMAP_REF = mcolors.LinearSegmentedColormap.from_list(
    "blue_seq", ["#0d1b4b", "#1a3a8a", "#4a7fd4", "#a8c8f8"]
)

# ------------ # 
# Data loading #
# ------------ #
def load_overall(json_path: Path) -> dict | None:
    """Return the 'overall' dict from an aggregated_metrics JSON, or None."""
    try:
        with json_path.open() as f:
            data = json.load(f)
        return data.get("overall")
    except Exception as e:
        print(f"  [warn] could not read {json_path}: {e}")
        return None


def find_json(folder: Path) -> Path | None:
    """Return the first *_aggregated_metrics.json found in folder."""
    hits = sorted(folder.glob("*_aggregated_metrics.json"))
    return hits[0] if hits else None


def load_ref_scores() -> dict | None:
    """
    Load overall metrics from the reference folder (results/pretrained/dataset/).
    Returns {metric: value} or None if not found.
    """
    if not RESULTS_ROOT.exists():
        print(f"[error] Results root not found: {RESULTS_ROOT.resolve()}")
        return None

    ref_dir = RESULTS_ROOT / REF_FOLDER
    # Reference JSON may live under either model subfolder; try both
    for subdir in [BASELINE_SUBDIR, FT_SUBDIR]:
        model_dir = ref_dir / subdir
        if model_dir.is_dir():
            json_path = find_json(model_dir)
            if json_path:
                overall = load_overall(json_path)
                if overall:
                    scores = {m: overall.get(m, np.nan) for m in METRICS}
                    print(f"  [ref] {REF_LABEL:<30s}  F1={overall.get('F1', float('nan')):.4f}  (from {subdir})")
                    return scores
    print(f"  [warn] reference JSON not found under {ref_dir}")
    return None


def collect_results(subdir_name: str) -> dict[str, dict]:
    """
    Walk results/pretrained/<dataset>/<subdir_name>/ skipping the reference folder.
    Return {dataset_label: {metric: value, ...}}
    """
    results = {}
    if not RESULTS_ROOT.exists():
        print(f"[error] Results root not found: {RESULTS_ROOT.resolve()}")
        return results

    for dataset_dir in sorted(RESULTS_ROOT.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if dataset_dir.name == REF_FOLDER:          # skip — loaded separately
            continue

        model_dir = dataset_dir / subdir_name
        if not model_dir.is_dir():
            continue

        json_path = find_json(model_dir)
        if json_path is None:
            print(f"  [skip] no aggregated_metrics.json in {model_dir}")
            continue

        overall = load_overall(json_path)
        if overall is None:
            continue

        label = dataset_dir.name
        results[label] = {m: overall.get(m, np.nan) for m in METRICS}
        print(f"  [ok] {label:<30s}  F1={overall.get('F1', float('nan')):.4f}")

    return results

# -------------- #
# Figure helpers #
# -------------- #
CELL_W    = 1.70
CELL_H    = 0.62
LEFT_PAD  = 2.20   # space for row labels
TOP_PAD   = 0.75
BOT_PAD   = 0.65


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
    """Draw column headers, row labels, title, dividers."""
    # column headers
    for j, metric in enumerate(METRICS):
        ax.text(
            LEFT_PAD + j * CELL_W + CELL_W / 2,
            TOP_PAD + n_rows * CELL_H + 0.10,
            metric, ha="center", va="bottom",
            fontsize=11, fontweight="bold",
            color="#e0e0f8", fontfamily="monospace",
        )
    # row labels
    for i, label in enumerate(row_labels):
        ax.text(
            LEFT_PAD - 0.14,
            TOP_PAD + (n_rows - 1 - i) * CELL_H + CELL_H / 2,
            label, ha="right", va="center",
            fontsize=9.5, fontweight="bold",
            color="#e8e8f8", fontfamily="monospace",
        )
    # title
    ax.text(
        (LEFT_PAD + n_cols * CELL_W) / 2,
        TOP_PAD + n_rows * CELL_H + 0.50,
        title, ha="center", va="bottom",
        fontsize=13, fontweight="bold",
        color="#ffffff", fontfamily="monospace",
    )
    # vertical column dividers
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
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → saved: {path}")


# ── Main plotting function ────────────────────────────────────────────────────
def plot_table(results: dict[str, dict], ref_label: str, title: str, output_path: Path,
               ref_scores: dict | None = None):
    """
    results   : {dataset_label: {metric: value}}
    ref_label : the row that shows raw scores only (no diff)
    ref_scores: explicit reference metric dict used for computing diffs.
                If None, falls back to results[ref_label].
                Pass this so the FT table can use the baseline reference scores.
    """
    if not results:
        print(f"  [skip] no data for table: {title}")
        return

    # Inject the reference row into the table if it is not already present
    # (e.g. the FT results dict has no tf2k_dataset entry of its own)
    table_data = dict(results)
    if ref_label not in table_data and ref_scores:
        table_data[ref_label] = ref_scores

    # Row order: reference first, then all others sorted
    other_labels = sorted(k for k in table_data if k != ref_label)
    row_labels   = ([ref_label] if ref_label in table_data else []) + other_labels

    # Use explicit ref_scores if provided, else fall back to the ref row in table_data
    ref_row = ref_scores if ref_scores is not None else table_data.get(ref_label, {})

    results = table_data  # work on the (possibly augmented) copy

    n_rows = len(row_labels)
    n_cols = len(METRICS)

    # ── Build value / diff arrays ────────────────────────────────────────────
    raw   = np.full((n_rows, n_cols), np.nan)
    diffs = np.full((n_rows, n_cols), np.nan)   # NaN for ref row

    for i, label in enumerate(row_labels):
        for j, metric in enumerate(METRICS):
            v = results[label].get(metric, np.nan)
            raw[i, j] = v
            if label != ref_label:
                ref_v = ref_row.get(metric, np.nan)
                if np.isfinite(v) and np.isfinite(ref_v):
                    diffs[i, j] = v - ref_v

    # ── Color norms ─────────────────────────────────────────────────────────
    finite_diffs = diffs[np.isfinite(diffs)]
    abs_max = max(np.max(np.abs(finite_diffs)) * 1.08, 0.01) if len(finite_diffs) else 0.5
    diff_norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    finite_raw = raw[0, :][np.isfinite(raw[0, :])]   # ref row for blue scale
    vmin = max(np.min(finite_raw) - 0.05, 0.0) if len(finite_raw) else 0.0
    vmax = min(np.max(finite_raw) + 0.02,  1.0) if len(finite_raw) else 1.0
    ref_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # ── Draw ─────────────────────────────────────────────────────────────────
    fig, ax = _make_figure(n_rows, n_cols)

    for i, label in enumerate(row_labels):
        is_ref = (label == ref_label)
        for j in range(n_cols):
            val  = raw[i, j]
            diff = diffs[i, j]

            x = LEFT_PAD + j * CELL_W
            y = TOP_PAD  + (n_rows - 1 - i) * CELL_H

            # Background colour
            if is_ref:
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

            if is_ref:
                # Single centred value
                txt = f"{val:.3f}" if np.isfinite(val) else "N/A"
                ax.text(x + CELL_W/2, y + CELL_H/2, txt,
                        ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color=txt_color, fontfamily="monospace")
            else:
                # Top: raw score   Bottom: ± diff
                top_txt = f"{val:.3f}"  if np.isfinite(val)  else "N/A"
                bot_txt = (f"{diff:+.3f}" if np.isfinite(diff) else "")
                ax.text(x + CELL_W/2, y + CELL_H*0.63, top_txt,
                        ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color=txt_color, fontfamily="monospace")
                ax.text(x + CELL_W/2, y + CELL_H*0.27, bot_txt,
                        ha="center", va="center",
                        fontsize=8.5, color=txt_color, fontfamily="monospace")

    _draw_chrome(ax, row_labels, title, n_rows, n_cols)

    # Two colorbars: diff (main) and reference (blue)
    _add_colorbar(fig, CMAP_DIFF, diff_norm,
                  "absolute diff vs tf2k_dataset   (green = higher  /  red = lower)")

    _save(fig, output_path)


# ----------- # 
# Entry point #
# ----------- #
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load reference scores once from results/pretrained/dataset/
    print(f"\n── Loading reference scores ({REF_FOLDER} → displayed as {REF_LABEL}) ──")
    ref_scores = load_ref_scores()
    if ref_scores is None:
        print("  [error] Cannot continue without reference scores.")
        return

    # 2. Collect comparison rows (all folders except reference)
    print("\n── Collecting baseline results (R50_nodown_pretrained) ──────────")
    baseline = collect_results(BASELINE_SUBDIR)

    print("\n── Collecting FT results (R50_nodown_ft) ────────────────────────")
    ft = collect_results(FT_SUBDIR)

    # 3. Plot
    print("\n── Plotting ─────────────────────────────────────────────────────")
    plot_table(
        baseline,
        ref_label   = REF_LABEL,
        ref_scores  = ref_scores,
        title       = "R50_nodown_baseline results",
        output_path = OUTPUT_DIR / "R50_nodown_baseline.png",
    )
    plot_table(
        ft,
        ref_label   = REF_LABEL,
        ref_scores  = ref_scores,
        title       = "R50_nodown_FT results",
        output_path = OUTPUT_DIR / "R50_nodown_FT.png",
    )
    print(f"\nDone — tables saved to {OUTPUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()