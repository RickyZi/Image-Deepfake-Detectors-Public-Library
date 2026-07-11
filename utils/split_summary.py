"""
Generate a Markdown summary report comparing a "full_social_split.json"
(processed dataset samples for a given social platform) against a
"tf2k_SOCIAL_splits.json" (raw train/val/test split definition).

Usage:
    python generate_split_summary.py <full_social_split.json> <tf2k_SOCIAL_splits.json> [output.md]
"""
import json
import collections
import os
import sys


def which_split(entry, train_set, val_set, test_set):
    if entry in train_set:
        return "train"
    if entry in val_set:
        return "val"
    if entry in test_set:
        return "test"
    return None


def family_of(sample):
    """Family = everything between the Real/Fake label dir and the filename,
    e.g. 'Real/FFHQ/00001.jpg' -> ('Real', 'FFHQ')
         'Fake/StyleGAN/images-psi-0.7/00001.jpg' -> ('Fake', 'StyleGAN/images-psi-0.7')
    """
    parts = sample["relative_path"].split("/")
    label_dir = parts[0]
    fam = "/".join(parts[1:-1])
    return label_dir, fam


def build_report(full, tf2k):
    train_set, val_set, test_set = set(tf2k["train"]), set(tf2k["val"]), set(tf2k["test"])
    samples = full["samples"]

    fine_counts = collections.defaultdict(collections.Counter)
    unmatched = []

    for s in samples:
        entry = s["split_entry"]
        sp = which_split(entry, train_set, val_set, test_set)
        if sp is None:
            unmatched.append(entry)
            continue
        label_dir, fam = family_of(s)
        fine_counts[(label_dir, fam)][sp] += 1

    rows = []
    for (label_dir, fam), c in fine_counts.items():
        rows.append({
            "label": label_dir,
            "family": fam,
            "train": c.get("train", 0),
            "val": c.get("val", 0),
            "test": c.get("test", 0),
            "total": c.get("train", 0) + c.get("val", 0) + c.get("test", 0),
        })
    rows.sort(key=lambda r: (r["label"], r["family"]))

    # aggregated (top-level algo) counts
    agg = collections.defaultdict(collections.Counter)
    for r in rows:
        key = r["family"] if r["label"] == "Real" else r["family"].split("/")[0]
        c = agg[(r["label"], key)]
        c["train"] += r["train"]
        c["val"] += r["val"]
        c["test"] += r["test"]
        c["total"] += r["total"]
    agg_rows = [{"label": l, "family": f, **dict(c)} for (l, f), c in agg.items()]
    agg_rows.sort(key=lambda r: (r["label"], r["family"]))

    # matching / consistency checks
    full_entries = set(s["split_entry"] for s in samples)
    tf2k_entries = train_set | val_set | test_set
    only_in_full = full_entries - tf2k_entries
    only_in_tf2k = tf2k_entries - full_entries
    overlap_train_val = train_set & val_set
    overlap_train_test = train_set & test_set
    overlap_val_test = val_set & test_set

    sc = full.get("subfolder_counts", {})
    mismatches = []
    for r in rows:
        expected = sc.get("real" if r["label"] == "Real" else "fake", {}).get(r["family"])
        if expected != r["total"]:
            mismatches.append((r["label"], r["family"], expected, r["total"]))

    grand_train = sum(r["train"] for r in rows)
    grand_val = sum(r["val"] for r in rows)
    grand_test = sum(r["test"] for r in rows)
    grand_total = sum(r["total"] for r in rows)

    return {
        "rows": rows,
        "agg_rows": agg_rows,
        "train_set": train_set, "val_set": val_set, "test_set": test_set,
        "full_entries": full_entries, "tf2k_entries": tf2k_entries,
        "only_in_full": only_in_full, "only_in_tf2k": only_in_tf2k,
        "overlap_train_val": overlap_train_val,
        "overlap_train_test": overlap_train_test,
        "overlap_val_test": overlap_val_test,
        "unmatched": unmatched,
        "mismatches": mismatches,
        "grand_train": grand_train, "grand_val": grand_val,
        "grand_test": grand_test, "grand_total": grand_total,
    }


def pct(n, tot):
    return f"{(100 * n / tot):.1f}%" if tot else "0.0%"


def render_markdown(full, report):
    social = full.get("social", "N/A")
    n_real, n_fake = full.get("n_real", 0), full.get("n_fake", 0)
    r = report

    lines = []
    lines.append(f"# Split Summary Report — {social}")
    lines.append("")
    lines.append(f"Generated from `full_social_split.json` (social = **{social}**) and `tf2k_SOCIAL_splits.json`.")
    lines.append("")

    lines.append("## 1. Matching Check Between the Two Split Files")
    lines.append("")
    lines.append(f"- Total samples in `full_social_split.json`: **{len(r['full_entries'])}**")
    lines.append(f"- Total entries in `tf2k_SOCIAL_splits.json` (train+val+test): "
                  f"**{len(r['tf2k_entries'])}** (train={len(r['train_set'])}, "
                  f"val={len(r['val_set'])}, test={len(r['test_set'])})")
    lines.append(f"- Entries only in `full_social_split.json`: **{len(r['only_in_full'])}**")
    lines.append(f"- Entries only in `tf2k_SOCIAL_splits.json`: **{len(r['only_in_tf2k'])}**")
    lines.append(f"- Overlap between train/val: **{len(r['overlap_train_val'])}**, "
                  f"train/test: **{len(r['overlap_train_test'])}**, "
                  f"val/test: **{len(r['overlap_val_test'])}**")
    lines.append(f"- Samples that could not be assigned to any split: **{len(r['unmatched'])}**")
    ok = not (r["only_in_full"] or r["only_in_tf2k"] or r["unmatched"]
              or r["overlap_train_val"] or r["overlap_train_test"] or r["overlap_val_test"])
    lines.append("")
    lines.append(f"**Result: {'✅ PASS — files match perfectly' if ok else '⚠️ MISMATCH DETECTED — see details above'}**")
    lines.append("")

    lines.append("## 2. Expected Numerosity Check")
    lines.append("")
    lines.append(f"- Declared totals in `full_social_split.json`: n_real={n_real}, n_fake={n_fake} "
                  f"(total={n_real + n_fake})")
    lines.append(f"- Sum of train+val+test across all families: **{r['grand_total']}**")
    if r["mismatches"]:
        lines.append("")
        lines.append("⚠️ The following families do **not** match the expected `subfolder_counts`:")
        lines.append("")
        lines.append("| Label | Family | Expected | Found |")
        lines.append("|---|---|---|---|")
        for label, fam, expected, found in r["mismatches"]:
            lines.append(f"| {label} | {fam} | {expected} | {found} |")
    else:
        lines.append("- ✅ All family totals (train+val+test) match `subfolder_counts` exactly.")
    lines.append("")

    lines.append("## 3. Per-Family Counts (fine-grained)")
    lines.append("")
    lines.append("| Label | Family | Train | Val | Test | Total |")
    lines.append("|---|---|---|---|---|---|")
    for row in r["rows"]:
        lines.append(f"| {row['label']} | {row['family']} | {row['train']} | {row['val']} | "
                      f"{row['test']} | {row['total']} |")
    lines.append(f"| **GRAND TOTAL** | | **{r['grand_train']}** | **{r['grand_val']}** | "
                  f"**{r['grand_test']}** | **{r['grand_total']}** |")
    lines.append("")

    lines.append("## 4. Per-Algorithm Counts (aggregated across variants)")
    lines.append("")
    lines.append("| Label | Algo | Train | Val | Test | Total |")
    lines.append("|---|---|---|---|---|---|")
    for row in r["agg_rows"]:
        lines.append(f"| {row['label']} | {row['family']} | {row['train']} | {row['val']} | "
                      f"{row['test']} | {row['total']} |")
    lines.append("")

    lines.append("## 5. Split Ratio Overview")
    lines.append("")
    lines.append("| Split | Samples | Percentage of Total |")
    lines.append("|---|---|---|")
    lines.append(f"| Train | {r['grand_train']} | {pct(r['grand_train'], r['grand_total'])} |")
    lines.append(f"| Val | {r['grand_val']} | {pct(r['grand_val'], r['grand_total'])} |")
    lines.append(f"| Test | {r['grand_test']} | {pct(r['grand_test'], r['grand_total'])} |")
    lines.append(f"| **Total** | **{r['grand_total']}** | **100.0%** |")
    lines.append("")
    lines.append("_Note: individual small-count fake families cannot always hit an exact 60/20/20 "
                  "split due to integer rounding; check the aggregated totals above for the overall "
                  "split balance._")
    lines.append("")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    full_path, tf2k_path = sys.argv[1], sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else "split_summary_report.md"

    with open(full_path) as f:
        full = json.load(f)
    with open(tf2k_path) as f:
        tf2k = json.load(f)

    report = build_report(full, tf2k)
    md = render_markdown(full, report)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(md)

    print(f"Report saved to {out_path}")


if __name__ == "__main__":
    main()

    # python generate_split_summary.py full_social_split.json tf2k_SOCIAL_splits.json output.md