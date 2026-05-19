"""
compare_ft.py

Compares two R50_nodown runs (base vs fine-tuned) across all data keys.
Reads aggregated_metrics.json from each run folder and plots a side-by-side
table with per-data-key metrics and % change from base to FT.

Usage:
    python compare_ft.py <base_run_dir> <ft_run_dir> [--output <out.png>]

    <base_run_dir>  path to the base model results folder
                    e.g. results/pretrained/season_TM01/R50_nodown
    <ft_run_dir>    path to the fine-tuned model results folder
                    e.g. results/pretrained_ft/season_TM01/R50_nodown

    --output        output image path (default: ./results/metric_tables/R50_nodown_base_vs_ft.png)
    --metric        which metric to colour-code (default: AUC)
                    choices: TPR TNR Acc "Balanced Acc" F1 AUC

Examples:
    python compare_ft.py \\
        results/pretrained/season_TM01/R50_nodown \\
        results/pretrained_ft/season_TM01/R50_nodown

    python compare_ft.py \\
        results/pretrained/season_TM01/R50_nodown \\
        results/pretrained_ft/season_TM01/R50_nodown \\
        --metric TPR --output results/metric_tables/tpr_comparison.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import numpy as np


# ── Config ────────────────────────────────────────────────────────────────────

METRICS = ["TPR", "TNR", "Acc", "Balanced Acc", "F1", "AUC"]

# Red → white → green for % change column
CMAP_DIFF = mcolors.LinearSegmentedColormap.from_list(
    "rg_diverging", ["#b22222", "#f0f0ec", "#2a7a2a"]
)
# Cool sequential blue for absolute values
CMAP_VAL = mcolors.LinearSegmentedColormap.from_list(
    "blue_seq", ["#0d1b4b", "#1a3a8a", "#4a7fd4", "#a8c8f8"]
)

OUTPUT_DIR = Path("./results/metric_tables")
AGGREGATED_FILENAME = "aggregated_metrics.json"

BG        = "#12121f"
HEADER_BG = "#1e1e35"
ROW_SEP   = "#2a2a44"
TEXT_MAIN = "#e8e8f8"
TEXT_DIM  = "#7878aa"
ACCENT    = "#4a7fd4"


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_aggregated(run_dir: Path) -> dict:
    path = run_dir / AGGREGATED_FILENAME
    if not path.exists():
        sys.exit(f"[ERROR] aggregated_metrics.json not found in {run_dir}\n"
                 f"  Run score_aggregator.py on this folder first.")
    with open(path) as f:
        return json.load(f)


def collect_per_key(data: dict) -> dict[str, dict]:
    """
    Returns {data_key: {metric: value}} from the per_data_key section.
    Also injects 'OVERALL' from the overall section.
    """
    out = {}
    for key, metrics in data.get("per_data_key", {}).items():
        out[key] = {m: metrics.get(m, float("nan")) for m in METRICS}
    overall = data.get("overall", {})
    if overall:
        out["OVERALL"] = {m: overall.get(m, float("nan")) for m in METRICS}
    return out


# ── Layout helpers ────────────────────────────────────────────────────────────

# Column layout: [key_label | base_M1 base_M2 ... | ft_M1 ft_M2 ... | Δ_M1 Δ_M2 ...]
# We draw three groups separated by thick dividers.

def _col_positions(n_metrics, key_w, cell_w, gap):
    """Return x-left positions for each of the 3*n_metrics value columns."""
    base_start = key_w + gap
    ft_start   = base_start + n_metrics * cell_w + gap
    diff_start = ft_start   + n_metrics * cell_w + gap
    positions  = []
    for start in (base_start, ft_start, diff_start):
        positions.extend([start + j * cell_w for j in range(n_metrics)])
    return positions, base_start, ft_start, diff_start


def _cell_color_val(val, norm, cmap):
    if np.isnan(val):
        return "#2e2e4a", TEXT_DIM
    rgba = cmap(norm(val))
    bg   = mcolors.to_hex(rgba)
    lum  = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
    txt  = "#12121f" if lum > 0.50 else "#f0f0f0"
    return bg, txt


def _cell_color_diff(pct, norm):
    if np.isnan(pct):
        return "#2e2e4a", TEXT_DIM
    rgba = CMAP_DIFF(norm(pct))
    bg   = mcolors.to_hex(rgba)
    lum  = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
    txt  = "#12121f" if lum > 0.50 else "#f0f0f0"
    return bg, txt


# ── Main plot ─────────────────────────────────────────────────────────────────

def plot_comparison(base_data, ft_data, base_label, ft_label, metric, output_path, model_name, dataset_name, overall_only = True):
    base_keys = collect_per_key(base_data)
    ft_keys   = collect_per_key(ft_data)
    
    print(f"base_keys: {base_keys}")
    print(f"ft_keys: {ft_keys}")
    
    

    if overall_only:
        row_labels = [base_label, ft_label, "Δ  (FT − Base) %"]
        base_row = base_keys.get("OVERALL", {})
        ft_row = ft_keys.get("OVERALL", {})
        diff_row = {}
        for m in METRICS:
            b = base_row.get(m, float("nan"))
            f = ft_row.get(m, float("nan"))
            if not np.isnan(b) and not np.isnan(f) and b != 0:
                diff_row[m] = (f - b) / abs(b) * 100
            else:
                diff_row[m] = float("nan")
        row_data = [base_row, ft_row, diff_row]
        n_rows = len(row_data)
    else:
        # # Union of all data keys, sorted; OVERALL always last
        all_keys = sorted(
            set(base_keys) | set(ft_keys) - {"OVERALL"},
            key=lambda k: (k == "OVERALL", k)
        )
        if "OVERALL" in base_keys or "OVERALL" in ft_keys:
            all_keys.append("OVERALL")
        n_rows    = len(all_keys)

    n_metrics = len(METRICS)

    # Dimensions
    key_w  = 2.2   # width of the row-label column
    cell_w = 1.3   # width of each metric cell
    cell_h = 0.52  # height of each row
    gap    = 0.25  # gap between groups
    pad_l  = 0.3
    pad_r  = 0.3
    pad_t  = 1.55  # top padding (headers)
    pad_b  = 0.55  # bottom padding (colour bar)

    if not overall_only:
        col_pos, base_start, ft_start, diff_start = _col_positions(
            n_metrics, key_w + pad_l, cell_w, gap
        )
        total_w = pad_l + key_w + gap + 3 * n_metrics * cell_w + 2 * gap + pad_r
    else:
        group_start = key_w + pad_l
        total_w = pad_l + key_w + gap + n_metrics * cell_w + pad_r

    total_h = pad_t + n_rows * cell_h + pad_b

    fig, ax = plt.subplots(figsize=(total_w, total_h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_axis_off()
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)

    # ── Colour normalisation ────────────────────────────────────────────────
    # Collect all absolute values for the chosen highlight metric across both runs
    if overall_only:
        all_abs = [
            v for d in (base_row, ft_row)
            for m, v in d.items()
            if not np.isnan(v)
        ]
    else:
        all_abs = [
            v for d in (base_keys, ft_keys)
            for k, row in d.items()
            if k != "OVERALL"
            for m, v in row.items()
            if not np.isnan(v)
        ]
    val_norm = mcolors.Normalize(
        vmin=max(min(all_abs) - 0.03, 0.0) if all_abs else 0.0,
        vmax=min(max(all_abs) + 0.02, 1.0) if all_abs else 1.0,
    )

    # Collect all % diffs for normalisation
    all_diffs = []
    if overall_only:
        for m in METRICS:
            v = diff_row.get(m, float("nan"))
            if not np.isnan(v):
                all_diffs.append(v)
    else:
        for key in all_keys:
            for m in METRICS:
                b = base_keys.get(key, {}).get(m, float("nan"))
                f = ft_keys.get(key, {}).get(m, float("nan"))
                if not np.isnan(b) and not np.isnan(f) and b != 0:
                    all_diffs.append((f - b) / abs(b) * 100)
    abs_max = max(max(abs(d) for d in all_diffs) * 1.1, 1.0) if all_diffs else 10.0
    diff_norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    # ── Title ───────────────────────────────────────────────────────────────
    ax.text(
        total_w / 2, total_h - 0.18,
        f"{model_name} - Base vs Fine-Tuned - {dataset_name}",
        ha="center", va="top", fontsize=14, fontweight="bold",
        color=TEXT_MAIN, fontfamily="monospace",
    )
    # ax.text(
    #     total_w / 2, total_h - 0.50,
    #     "season_TM01",
    #     ha="center", va="top", fontsize=9,
    #     color=TEXT_DIM, fontfamily="monospace",
    # )

    # ── Group headers ───────────────────────────────────────────────────────
    if overall_only:
        group_labels = [
            (group_start + n_metrics * cell_w / 2, "Overall metrics", ACCENT),
        ]
    else:
        group_labels = [
            (base_start + n_metrics * cell_w / 2, base_label, ACCENT),
            (ft_start   + n_metrics * cell_w / 2, ft_label,   "#6abf69"),
            (diff_start + n_metrics * cell_w / 2, "Δ  (FT − Base) %", "#cc8844"),
        ]
    header_y = pad_b + n_rows * cell_h + 0.60
    for x, label, color in group_labels:
        ax.text(x, header_y, label,
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color=color, fontfamily="monospace")

    # Metric sub-headers
    subheader_y = pad_b + n_rows * cell_h + 0.18
    if overall_only:
        for idx, m in enumerate(METRICS):
            x = group_start + idx * cell_w + cell_w / 2
            ax.text(x, subheader_y, m,
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                    color=TEXT_DIM, fontfamily="monospace")
    else:
        for idx, m in enumerate(METRICS):
            for group_start in (base_start, ft_start, diff_start):
                x = group_start + idx * cell_w + cell_w / 2
                ax.text(x, subheader_y, m,
                        ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                        color=TEXT_DIM, fontfamily="monospace")

    # ── Group separator lines ───────────────────────────────────────────────────────────────
    if not overall_only:
        for x in (ft_start - gap / 2, diff_start - gap / 2):
            ax.plot([x, x],
                    [pad_b - 0.05, pad_b + n_rows * cell_h + 0.75],
                    color=ROW_SEP, linewidth=1.5, zorder=10)

    # ── Rows ───────────────────────────────────────────────────────────────
    if overall_only:
        row_iter = list(zip(range(n_rows), row_labels, row_data))
    else:
        row_iter = [(row_idx, key, None) for row_idx, key in enumerate(all_keys)]

    for row_idx, row_label, row_data_item in row_iter:
        y = pad_b + (n_rows - 1 - row_idx) * cell_h
        is_highlight = overall_only and row_idx == 2

        row_bg = "#1c1c30" if row_idx % 2 == 0 else BG
        if overall_only and row_idx == 2:
            row_bg = "#1e2a1e"
        ax.add_patch(plt.Rectangle(
            (0, y), total_w, cell_h,
            facecolor=row_bg, linewidth=0, zorder=0
        ))

        ax.text(
            pad_l + key_w - 0.10,
            y + cell_h / 2,
            row_label,
            ha="right", va="center",
            fontsize=8.5 if not is_highlight else 9,
            fontweight="bold" if is_highlight else "normal",
            color=TEXT_MAIN if not is_highlight else "#aaffaa",
            fontfamily="monospace",
        )

        if overall_only:
            row = row_data_item
        else:
            key = row_label
            base_row = base_keys.get(key, {})
            ft_row = ft_keys.get(key, {})

        for m_idx, m in enumerate(METRICS):
            pad_inner = 0.05
            if overall_only:
                val = row.get(m, float("nan"))
                if row_idx < 2:
                    bg, txt_color = _cell_color_val(val, val_norm, CMAP_VAL)
                    text = f"{val:.3f}" if not np.isnan(val) else "—"
                else:
                    bg, txt_color = _cell_color_diff(val, diff_norm)
                    text = f"{'+' if not np.isnan(val) and val >= 0 else ''}{val:.1f}%" if not np.isnan(val) else "—"
                x = group_start + m_idx * cell_w
                ax.add_patch(FancyBboxPatch(
                    (x + pad_inner, y + pad_inner),
                    cell_w - 2 * pad_inner, cell_h - 2 * pad_inner,
                    boxstyle="round,pad=0.03", linewidth=0,
                    facecolor=bg, zorder=1, clip_on=False,
                ))
                ax.text(
                    x + cell_w / 2, y + cell_h / 2,
                    text,
                    ha="center", va="center",
                    fontsize=8, fontweight="bold",
                    color=txt_color, fontfamily="monospace", zorder=2,
                )
            else:
                b_val = base_row.get(m, float("nan"))
                f_val = ft_row.get(m, float("nan"))
                if not np.isnan(b_val) and not np.isnan(f_val) and b_val != 0:
                    pct = (f_val - b_val) / abs(b_val) * 100
                else:
                    pct = float("nan")

                for g_idx, (x_off, val) in enumerate([
                    (base_start, b_val),
                    (ft_start,   f_val),
                ]):
                    x = x_off + m_idx * cell_w
                    bg, txt_color = _cell_color_val(val, val_norm, CMAP_VAL)
                    ax.add_patch(FancyBboxPatch(
                        (x + pad_inner, y + pad_inner),
                        cell_w - 2 * pad_inner, cell_h - 2 * pad_inner,
                        boxstyle="round,pad=0.03", linewidth=0,
                        facecolor=bg, zorder=1, clip_on=False,
                    ))
                    ax.text(
                        x + cell_w / 2, y + cell_h / 2,
                        f"{val:.3f}" if not np.isnan(val) else "—",
                        ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color=txt_color, fontfamily="monospace", zorder=2,
                    )

                x = diff_start + m_idx * cell_w
                bg, txt_color = _cell_color_diff(pct, diff_norm)
                ax.add_patch(FancyBboxPatch(
                    (x + pad_inner, y + pad_inner),
                    cell_w - 2 * pad_inner, cell_h - 2 * pad_inner,
                    boxstyle="round,pad=0.03", linewidth=0,
                    facecolor=bg, zorder=1, clip_on=False,
                ))
                ax.text(
                    x + cell_w / 2, y + cell_h / 2,
                    f"{'+' if not np.isnan(pct) and pct >= 0 else ''}{pct:.1f}%" if not np.isnan(pct) else "—",
                    ha="center", va="center",
                    fontsize=7.5, fontweight="bold",
                    color=txt_color, fontfamily="monospace", zorder=2,
                )
    # ── Colour bars ─────────────────────────────────────────────────────────
    bar_y  = 0.06
    bar_h  = 0.18
    if overall_only:
        bar_start = group_start
        bar_w = (n_metrics * cell_w - 0.1) / 2
        diff_bar_x = bar_start + bar_w + 0.1
        diff_bar_w = bar_w
    else:
        bar_start = pad_l
        bar_w = (ft_start - gap / 2 - pad_l - 0.1)
        diff_bar_x = diff_start
        diff_bar_w = n_metrics * cell_w

    cbar_ax1 = fig.add_axes([
        bar_start / total_w,
        bar_y / total_h,
        bar_w / total_w,
        bar_h / total_h,
    ])
    sm1 = plt.cm.ScalarMappable(cmap=CMAP_VAL, norm=val_norm)
    sm1.set_array([])
    cb1 = fig.colorbar(sm1, cax=cbar_ax1, orientation="horizontal")
    cb1.ax.tick_params(colors=TEXT_DIM, labelsize=6.5)
    cb1.set_label("absolute score", color=TEXT_DIM, fontsize=6.5)
    cb1.outline.set_edgecolor(ROW_SEP)

    cbar_ax2 = fig.add_axes([
        diff_bar_x / total_w,
        bar_y / total_h,
        diff_bar_w / total_w,
        bar_h / total_h,
    ])
    sm2 = plt.cm.ScalarMappable(cmap=CMAP_DIFF, norm=diff_norm)
    sm2.set_array([])
    cb2 = fig.colorbar(sm2, cax=cbar_ax2, orientation="horizontal")
    cb2.ax.tick_params(colors=TEXT_DIM, labelsize=6.5)
    cb2.set_label("% change  (green = improvement)", color=TEXT_DIM, fontsize=6.5)
    cb2.outline.set_edgecolor(ROW_SEP)

    # ── Save ────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare base vs fine-tuned R50_nodown results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("base_dir", type=Path,
                        help="Path to the base model results folder "
                             "(must contain aggregated_metrics.json)")
    parser.add_argument("ft_dir", type=Path,
                        help="Path to the fine-tuned model results folder "
                             "(must contain aggregated_metrics.json)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output image path "
                             "(default: results/metric_tables/R50_nodown_base_vs_ft.png)")
    parser.add_argument("--metric", default="AUC", choices=METRICS,
                        help="Metric used for colour scale on absolute columns (default: AUC)")
    parser.add_argument("--base-label", default="Base", dest="base_label",
                        help="Display label for the base model column (default: Base)")
    parser.add_argument("--ft-label", default="Fine-Tuned", dest="ft_label",
                        help="Display label for the FT model column (default: Fine-Tuned)")
    parser.add_argument("--overall-only", action = 'store_true', 
                        help="select only overall results to be displayed")
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    ft_dir   = args.ft_dir.resolve()

    model_name = base_dir.name
    dataset_name = str(base_dir).split('/')[-2]
    print(f"model_name: {model_name}, dataset: {dataset_name}")
    # breakpoint()

    # base_dir = './results/demo/season_TM01/R50_nodown/R50_gen_aggregated_metrics.json'
    # ft_dir = './results/pretrained/season_TM01/R50_nodown/R50_FT_aggregated_metrics.json'

    print(f"Base : {base_dir}")
    print(f"FT   : {ft_dir}\n")

    base_data = load_aggregated(base_dir)
    ft_data   = load_aggregated(ft_dir)

    output_path = args.output or (OUTPUT_DIR / "R50_nodown_base_vs_ft.png")

    plot_comparison(
        base_data, ft_data,
        args.base_label, args.ft_label,
        args.metric, output_path,
        model_name,
        dataset_name,
        overall_only=args.overall_only,
    )


if __name__ == "__main__":
    main()
