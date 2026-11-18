#!/usr/bin/env python3
"""
Cross-preset generalization matrices.

Tree:  <root>/<det>/<subfolder>/<trn>[_vs_<tst>]/<detector_dir>/<datasets-dir>/metrics.json
       (no "_vs_" in the folder name = aligned case)

Choose which presets to map:
  --preset-set wifs        the 18 presets of the WIFS paper (default)
  --preset-set extended    the full extended list (portraits, auto+, video, ...)
  --preset-set all         every preset found on disk
  --presets style/bw_BW01 seasons/autumn_TM01 ...   custom subset (overrides --preset-set)

Examples:
  python3 plot_gen_matrix.py --list-keys
  python3 plot_gen_matrix.py --preset-set wifs --metric-key balanced_acc
  python3 plot_gen_matrix.py --preset-set extended --detectors R50_TF --summary
  python3 plot_gen_matrix.py --presets adaptive/blurbg_strong adaptive/blurbg_subtle style/bw_BW01

Cell annotations
  preset x preset matrices : value of the single run. With several detectors the all_detectors mean
                             matrices show  mean (variance across detectors).
  family x family (--summary): mean (variance) over all the preset pairs averaged in that block.
                             Variance is the sample variance (ddof=1, in percentage points^2 when the
                             metric is in %); a block with a single cell has no variance and shows the mean only.
                             Use --dispersion std for the standard deviation instead.

Row/column labels are the full preset slug  <family>_<preset_name>_<preset_code>  (e.g. seasons_autumn_TM01).

Outputs in <out>/<preset-set>/: all_results.csv, compare_delta_<a>_vs_<b>.png and, per detector,
<det>/<det>_{raw,delta,summary}.{csv,png}.
delta(trn, tst) = metric(trn, tst) - metric(trn, trn)   (row-wise, signed)
"""
import argparse, itertools, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def S(*items):  # "style/bw_BW01" -> folder slug "style_bw_BW01"
    return [i.replace("/", "_") for i in items]


WIFS_DATASET = S(
    "adaptive/blurbg_strong", "adaptive/blurbg_subtle", "adaptive/enhance_portrait",
    "adaptive/sky_bluedrama", "adaptive/subject_pop",
    "seasons/autumn_TM01", "seasons/spring_SP01", "seasons/summer_SM01", "seasons/winter_WN01",
    "style/bw_BW01", "style/cinematic_CN01", "style/cinematic2_CN11", "style/film_inspired_warmgold",
    "style/futuristic_FT01", "style/vintage_VN01",
    "subject/landscape_LN01", "subject/travel_TR01", "subject/travel2_TR11",
)

EXTENDED_DATASET = S(
    # adaptive
    "adaptive/blurbg_strong", "adaptive/blurbg_subtle", "adaptive/enhance_portrait",
    "adaptive/sky_bluedrama", "adaptive/subject_pop",
    # auto+
    "auto+/retro-AR01", "auto+/retro-AR09",
    # portraits
    "portraits/bw-PB01", "portraits/deepskin-PD01-red", "portraits/deepskin-PD01-orange",
    "portraits/deepskin-PD01-yellow", "portraits/edgy-PE01", "portraits/group-PG01",
    "portraits/group-PG06", "portraits/group-PG07", "portraits/lightskin-PL01",
    "portraits/light-skin-PL10", "portraits/mediumskin-PM01", "portraits/medium-skin-PM11",
    # seasons
    "seasons/autumn_TM01", "seasons/spring_SP01", "seasons/summer_SM01", "seasons/winter_WN01",
    # style
    "style/bw_BW01", "style/cinematic_CN01", "style/cinematic2_CN11", "style/film_inspired_warmgold",
    "style/film_inspired_boldbw", "style/film_inspired_coolbw", "style/futuristic_FT01", "style/vintage_VN01",
    # subject
    "subject/landscape_LN01", "subject/travel_TR01", "subject/travel2_TR11",
    "subject/concerts-CT01", "subject/concerts-CT09", "subject/concerts-CT10",
    "subject/food-FD01", "subject/food-FD04", "subject/lifestyle-LF01", "subject/lifestyle-LF08",
    "subject/urbanarchitecture-UA01", "subject/urban-architecture-UA08",
    # video
    "video/creative-VCR01", "video/creative-VCR08", "video/creative-VCR12",
)

PRESET_SETS = {"wifs": WIFS_DATASET, "extended": EXTENDED_DATASET}

FAMILY_ORDER = ["adaptive", "seasons", "style", "subject", "auto+", "portraits", "video"]

# name -> (top folder, sub folder, detector folder inside every run folder)
DETECTOR_LAYOUT = {
    "R50_TF": ("R50_TF", "pretrained", "R50_TF_ft_unfreezeL4"),
    "R50_nodown": ("R50_nodown", "pretrained", "R50_nodown_ft_unfreezeL4"),
    "CLIP-D": ("CLIP-D", "lora_r4_qv", "CLIP-D_ft"),
}

METRIC_KEY_ALIASES = {
    "accuracy": "Acc", "acc": "Acc",
    "balanced_acc": "Balanced Acc", "balanced_accuracy": "Balanced Acc",
    "auc": "AUC", "f1": "F1", "tpr": "TPR", "tnr": "TNR",
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
    for c in ("Balanced Acc", "Acc", "AUC", "F1"):
        if c in flat:
            return c
    sys.exit("cannot guess the metric, pass --metric-key. Available keys:\n  " + "\n  ".join(keys))


label = lambda s: s  # full slug <family>_<preset_name>_<preset_code>, e.g. seasons_autumn_TM01
family = lambda s: s.split("_", 1)[0]


def preset_sort_key(slug):
    fam = family(slug)
    return (FAMILY_ORDER.index(fam) if fam in FAMILY_ORDER else 99, slug)


def scan(run_root: Path, allowed=None):
    """Return [(trn, tst, dir)]; if `allowed` is given keep only runs with both presets inside it."""
    runs = []
    for d in sorted(p for p in run_root.iterdir() if p.is_dir()):
        parts = d.name.split("_vs_")
        if len(parts) == 1:
            trn = tst = parts[0]
        elif len(parts) == 2:
            trn, tst = parts
        else:
            continue
        if allowed is None or (trn in allowed and tst in allowed):
            runs.append((trn, tst, d))
    return runs


def family_matrix(M, include_aligned=False, dispersion="var"):
    """Average a preset x preset matrix into a family x family one.
    Returns (mean, dispersion, number of cells averaged, family order); dispersion is the sample
    variance (or std) over the cells of each block, NaN when a block has a single cell."""
    s = M.stack().dropna()
    s.index.names = ["trn", "test"]
    long = s.rename("v").reset_index()
    if not include_aligned:  # drop trn == tst so the family diagonal = same family, different preset
        long = long[long.trn != long.test]
    long["ft"], long["fs"] = long.trn.map(family), long.test.map(family)
    fams = sorted({family(p) for p in M.index},
                  key=lambda f: FAMILY_ORDER.index(f) if f in FAMILY_ORDER else 99)
    g = long.groupby(["ft", "fs"]).v
    G = g.mean().unstack().reindex(index=fams, columns=fams)
    V = (g.var(ddof=1) if dispersion == "var" else g.std(ddof=1)).unstack().reindex(index=fams, columns=fams)
    N = g.count().unstack().reindex(index=fams, columns=fams).fillna(0).astype(int)
    return G, V, N, fams


def draw(mat, presets, title, path, cmap, vmin=None, vmax=None, fmt="{:.1f}", cbar_label="",
         labels=None, disp=None, disp_label="variance"):
    """disp: optional DataFrame (same shape as mat) written under each value as '(x)'."""
    n = len(presets)
    summary = labels is not None
    cell, fs = (1.3, 11) if n <= 10 else (0.75, 6) if n <= 24 else (0.5, 4)
    tfs = 11 if n <= 10 else 8 if n <= 24 else 6
    lbls = labels or [label(p) for p in presets]
    if disp is not None:
        title += f"\ncell = value ({disp_label})"
    fig, ax = plt.subplots(figsize=(cell * n + 3, cell * n + 2))
    im = ax.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(lbls, rotation=60, ha="right", fontsize=tfs)
    ax.set_yticklabels(lbls, fontsize=tfs)
    ax.set_xlabel("Test preset"); ax.set_ylabel("Training preset"); ax.set_title(title)
    for i in range(n):
        for j in range(n):
            v = mat.values[i, j]
            if not np.isnan(v):
                txt = fmt.format(v)
                if disp is not None and not np.isnan(disp.values[i, j]):
                    txt += f"\n({disp.values[i, j]:.1f})"
                ax.text(j, i, txt, ha="center", va="center", fontsize=fs)
        ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False, ec="black", lw=1.5))
    for i in range(1, 0 if summary else n):  # family separators (not needed in the family summary)
        if family(presets[i]) != family(presets[i - 1]):
            ax.axhline(i - .5, color="white", lw=2.5); ax.axvline(i - .5, color="white", lw=2.5)
    fig.colorbar(im, ax=ax, fraction=0.04, label=cbar_label)
    fig.tight_layout(); fig.savefig(path, dpi=150 if n > 24 else 200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="generalization_results")
    ap.add_argument("--out", default="generalization_results/matrices")
    ap.add_argument("--detectors", nargs="+", default=list(DETECTOR_LAYOUT), choices=list(DETECTOR_LAYOUT))
    ap.add_argument("--preset-set", default="extended", choices=["wifs", "extended", "all"])
    ap.add_argument("--presets", nargs="+", default=None,
                    help="custom subset, e.g. style/bw_BW01 seasons/autumn_TM01 (overrides --preset-set)")
    ap.add_argument("--metric-key", default="balanced_accuracy", help="key in metrics.json (aliases accepted)")
    ap.add_argument("--list-keys", action="store_true")
    ap.add_argument("--summary", action="store_true",
                    help="also produce family x family matrices (mean over all presets of each family)")
    ap.add_argument("--include-aligned", action="store_true",
                    help="in the family summary keep the aligned cells (trn == tst) inside the diagonal blocks")
    ap.add_argument("--dispersion", default="var", choices=["var", "std"],
                    help="number shown in parentheses: sample variance (default) or standard deviation")
    a = ap.parse_args()
    dname = "variance" if a.dispersion == "var" else "std"
    disp_fn = np.var if a.dispersion == "var" else np.std

    # ---- which presets
    if a.presets:
        subset, set_name = S(*a.presets), "custom"
    elif a.preset_set == "all":
        subset, set_name = None, "all"
    else:
        subset, set_name = PRESET_SETS[a.preset_set], a.preset_set
    allowed = set(subset) if subset else None

    # ---- collect
    root = Path(a.root)
    rows, key = [], None
    for det in a.detectors:
        top, sub, det_folder = DETECTOR_LAYOUT[det]
        run_root = root / top / sub
        if not run_root.is_dir():
            print(f"[skip] {det}: {run_root} not found"); continue
        runs = scan(run_root, allowed)
        print(f"{det}: {len(runs)} runs ({set_name}) in {run_root}")
        for trn, tst, d in runs:
            f = find_metrics_file(d / det_folder)
            if not f:
                continue
            flat = flatten(json.load(open(f)))
            if a.list_keys:
                print(f)
                for k, v in flat.items():
                    print(f"  {k} = {v}")
                return
            key = key or pick_key(flat, a.metric_key)
            rows.append(dict(detector=det, trn=trn, test=tst, metric=flat.get(key, np.nan)))
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("no metrics.json found: check --root, --detectors and the preset selection")
    print(f"metric key: {key}")

    # ---- presets actually available, in the requested order
    seen = set(df.trn) | set(df.test)
    if subset:
        presets = [p for p in subset if p in seen]
        absent = [p for p in subset if p not in seen]
        if absent:
            print(f"[warn] no runs at all for: {', '.join(absent)}")
    else:
        presets = sorted(seen, key=preset_sort_key)
    print(f"{len(presets)} presets mapped")

    pct = df.metric.max() <= 1.0
    if pct:
        df["metric"] *= 100
    unit = "%" if pct else ""
    out = Path(a.out) / set_name
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "all_results.csv", index=False)

    # ---- per-detector matrices (one run per cell -> no variance)
    deltas, raws = {}, {}
    for det in a.detectors:
        d = df[df.detector == det]
        if d.empty:
            continue
        (out / det).mkdir(exist_ok=True)
        M = d.pivot(index="trn", columns="test", values="metric").reindex(index=presets, columns=presets)
        missing = int(M.isna().sum().sum())
        if missing:
            print(f"[warn] {det}: {missing} missing cells out of {len(presets) ** 2}")
            miss = [(t, s) for t in presets for s in presets if np.isnan(M.loc[t, s])]
            pd.DataFrame(miss, columns=["trn", "test"]).to_csv(out / det / f"{det}_missing_runs.csv", index=False)
        D = M - np.diag(M.values)[:, None]
        raws[det] = M
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

    # ---- all detectors together: mean over detectors, only for cells present in every detector
    Mm = Dm = None
    if len(raws) > 1:
        names = "+".join(raws)
        (out / "all_detectors").mkdir(exist_ok=True)
        wrap = lambda arr: pd.DataFrame(arr, index=presets, columns=presets)
        stack = lambda dd: np.stack([m.values for m in dd.values()])
        Mm, Dm = wrap(stack(raws).mean(axis=0)), wrap(stack(deltas).mean(axis=0))
        Mm.to_csv(out / "all_detectors" / "mean_raw.csv"); Dm.to_csv(out / "all_detectors" / "mean_delta.csv")
        draw(Mm, presets, f"Mean over {names}: {key}", out / "all_detectors" / "mean_raw.png", "viridis",
             cbar_label=f"{key} {unit}")
        lim = np.nanmax(np.abs(Dm.values))
        draw(Dm, presets, f"Mean over {names}: change in {key} vs aligned", out / "all_detectors" / "mean_delta.png",
             "RdYlGn", -lim, lim, fmt="{:+.1f}", cbar_label=f"delta {unit}")

    # ---- family-level summary: mean (variance) over all preset pairs of family A (train) vs family B (test)
    if a.summary:
        def save_family(Fr, Vr, Fd, Vd, fams, name, folder, cnt=None):
            folder.mkdir(parents=True, exist_ok=True)
            Fr.to_csv(folder / f"{name}_family_raw.csv"); Fd.to_csv(folder / f"{name}_family_delta.csv")
            Vr.to_csv(folder / f"{name}_family_raw_{dname}.csv"); Vd.to_csv(folder / f"{name}_family_delta_{dname}.csv")
            if cnt is not None:
                cnt.to_csv(folder / f"{name}_family_counts.csv")
            dl = f"{dname} over averaged presets"
            draw(Fr, fams, f"{name}: {key} by family (rows = training family)",
                 folder / f"{name}_family_raw_summary.png", "viridis", cbar_label=f"{key} {unit}",
                 labels=fams, disp=Vr, disp_label=dl)
            lim = np.nanmax(np.abs(Fd.values))
            draw(Fd, fams, f"{name}: change in {key} vs aligned, by family",
                 folder / f"{name}_family_delta_summary.png", "RdYlGn", -lim, lim, fmt="{:+.1f}",
                 cbar_label=f"delta {unit}", labels=fams, disp=Vd, disp_label=dl)

        for det in raws:
            Fr, Vr, cnt, fams = family_matrix(raws[det], a.include_aligned, a.dispersion)
            Fd, Vd, _, _ = family_matrix(deltas[det], a.include_aligned, a.dispersion)
            save_family(Fr, Vr, Fd, Vd, fams, det, out / det / "summary", cnt)
            print(f"\n{det} family summary ({key}, mean):\n{Fr.round(2)}\n({dname}):\n{Vr.round(2)}")
        if Mm is not None:  # family matrices of the detector-averaged preset matrices
            Fr, Vr, cnt, fams = family_matrix(Mm, a.include_aligned, a.dispersion)
            Fd, Vd, _, _ = family_matrix(Dm, a.include_aligned, a.dispersion)
            save_family(Fr, Vr, Fd, Vd, fams, "mean_" + "+".join(raws), out / "all_detectors" / "summary", cnt)

    # ---- pairwise comparison between detectors
    for (n1, D1), (n2, D2) in itertools.combinations(deltas.items(), 2):
        C = D1 - D2
        lim = np.nanmax(np.abs(C.values))
        draw(C, presets, f"Delta({n1}) - Delta({n2}): positive = {n1} generalizes better",
             out / f"compare_delta_{n1}_vs_{n2}.png", "PuOr", -lim, lim, fmt="{:+.1f}",
             cbar_label="difference of deltas")


if __name__ == "__main__":
    main()