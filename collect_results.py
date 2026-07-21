"""
Collect metrics from the per-detector results layout, and, for adaptive_*
datasets, substitute ensemble results in place of the model's own "ft" run
(the real "pretrained" numbers are kept as-is).

Actual layout on disk:

  <BASE_DIR>/
      R50_nodown/
          pretrained/                    <- both pretrained AND ft variants live here
              <dataset_name>/
                  R50_nodown_pretrained/<dataset_keys_folder>/metrics.json
                  R50_nodown_ft_unfreezeL4/<dataset_keys_folder>/metrics.json
          pretrained_facebook/           <- same idea, facebook platform
          pretrained_telegram/
          pretrained_twitter/
          pretrained_social/             <- clean baseline datasets, pretrained ONLY
              Facebook/  Telegram/  Twitter/   (each: R50_nodown_pretrained/.../metrics.json)
          metric_tables/                 <- ignored
      R50_TF/
          ... identical structure to R50_nodown ...
      CLIP-D/
          pretrained/                    <- pretrained variant only
              <dataset_name>/CLIP-D_pretrained/<dataset_keys_folder>/metrics.json
          pretrained_facebook/ pretrained_telegram/ pretrained_twitter/ pretrained_social/
          lora_r4_qv/                    <- ft variant only (parallel tree)
              <dataset_name>/CLIP-D_ft/<dataset_keys_folder>/metrics.json
          lora_r4_qv_facebook/ lora_r4_qv_telegram/ lora_r4_qv_twitter/
          metric_tables/                 <- ignored

  <dataset_keys_folder> is the long concatenated-key folder name (e.g.
  "realFFHQ:pre&realFORLAB:pre&...&flux:pre") and is NOT a fixed name, so we
  glob for it rather than hardcoding it.

For adaptive_* datasets (in ANY platform: PRE, facebook, telegram, twitter):
  - "pretrained" is read from disk as normal.
  - "ft" is NOT read from disk. Instead we look up an ensemble report under:
      <ENSEMBLE_BASE>[/<platform>]/<family>/ensemble_report_<ft_model_tag>_<dataset>[_images].json
    e.g. ensemble_report_R50_nodown_ft_unfreezeL4_adaptive_blurbg_strong.json
    or   ensemble_report_CLIP-D_ft_adaptive_subject_pop_images.json
    (globbing for <family> and [/<platform>] since family folder naming is
    inconsistent -- sometimes "blurbg_family", sometimes "family_subject_pop",
    sometimes literally "adaptive_enhance_portrait")
    and use its results.average block as the "ft" entry.
  - If no ensemble report is found, "ft" is simply omitted for that combo (warned).
  - If the SAME (detector, dataset) matches under multiple family folders
    (this happens -- e.g. adaptive_subject_pop shows up under
    family_enhance_portrait, family_sky_bluedrama, AND family_subject_pop),
    prefer the family folder whose name topically matches the dataset (e.g.
    family_subject_pop for adaptive_subject_pop); only fall back to another
    match, with a loud warning, if none match by name.

Outputs (same as before, so downstream scripts stay compatible):
  - results_nested.json : {platform: {dataset: {detector: {variant: {metric: value}}}}}
  - results_flat.json   : tidy list of records, one row per (platform, dataset, detector, variant, metric)
  - results_by_detector.json (+ per-detector files under results_by_detector/)
"""

import os
import json
import glob
from collections import defaultdict

# ------------------------------------------------------------------

BASE_DIR = "results"  # top-level folder containing R50_nodown/, R50_TF/, CLIP-D/

# Ensemble report tree, sibling to the per-detector folders under BASE_DIR:
#   results/model_ensemble/ensemble_results/...
ENSEMBLE_BASE = os.path.join(BASE_DIR, "model_ensemble", "ensemble_results")

METRICS_FILENAME = "metrics.json"

DETECTORS = ["R50_nodown", "R50_TF", "CLIP-D"]

# Folder-name prefix (before the platform suffix) that holds each variant,
# per detector. R50 models keep both variants under the same "pretrained*"
# tree; CLIP-D splits them into "pretrained*" (pretrained) and "lora_r4_qv*" (ft).
DETECTOR_VARIANT_ROOT = {
    "R50_nodown": {"pretrained": "pretrained", "ft": "pretrained"},
    "R50_TF":     {"pretrained": "pretrained", "ft": "pretrained"},
    "CLIP-D":     {"pretrained": "pretrained", "ft": "lora_r4_qv"},
}

# Exact ft folder name per detector. NOT a glob -- both R50_nodown and R50_TF
# folders can contain OTHER ft runs with different strategies (e.g. some
# R50_TF_ft_<other_strategy>) which must NOT be picked up. Only "unfreezeL4"
# is the target ft model for the R50 detectors. CLIP-D only ever has the
# LoRA ft run, folder named "CLIP-D_ft". This same tag is also the "model"
# component of the ensemble_report_<tag>_<dataset>.json filenames.
FT_FOLDER_NAME = {
    "R50_nodown": "R50_nodown_ft_unfreezeL4",
    "R50_TF": "R50_TF_ft_unfreezeL4",
    "CLIP-D": "CLIP-D_ft",
}

# platform_key -> folder suffix appended to the variant root above.
# PRE has NO suffix (e.g. "pretrained" + "" = "pretrained").
PLATFORM_SUFFIXES = {
    "PRE": "",
    "facebook": "_facebook",
    "telegram": "_telegram",
    "twitter": "_twitter",
}

SOCIAL_PLATFORM_KEY = "social"
SOCIAL_SUFFIX = "_social"  # only ever appended to the "pretrained" root; no ft counterpart

ADAPTIVE_PREFIX = "adaptive_"

# Path prefix (under ENSEMBLE_BASE) holding each platform's ensemble runs.
# PRE has NO prefix folder -- its family folders (blurbg_family, etc.) sit
# directly under ENSEMBLE_BASE. Other platforms nest an extra folder first:
# ENSEMBLE_BASE/<platform>/<family>/ensemble_report_<tag>_<dataset>.json
ENSEMBLE_PLATFORM_PREFIX = {
    "PRE": "",
    "facebook": "facebook",
    "telegram": "telegram",
    "twitter": "twitter",
    # "social" intentionally omitted: the social platform only ever holds
    # clean baseline datasets (Facebook/Telegram/Twitter), never adaptive_*,
    # so ensemble substitution never applies there regardless.
}

REQUIRED_METRIC_KEY = "Balanced Acc"

# ------------------------------------------------------------------


def _normalize_dataset_name(raw_name):
    """make all preset match the PRE style, correct typo in enhance_portrait"""
    if "_" in raw_name:
        prefix, rest = raw_name.split("_", 1)
        rest = rest.replace("-", "_")
        normalized = f"{prefix}_{rest}"
    else:
        normalized = raw_name
    normalized = normalized.replace("portait", "portrait")  # known on-disk typo
    return normalized


def _family_core_token(family_name):
    """Normalize a family folder name down to its core topic token so it can
    be compared against a dataset's topic, e.g.:
      "family_subject_pop"      -> "subject_pop"
      "blurbg_family"           -> "blurbg"
      "adaptive_enhance_portrait" -> "enhance_portrait"  (family folder is
                                       literally the dataset name in some cases)
    """
    core = family_name
    if core.startswith("family_"):
        core = core[len("family_"):]
    elif core.endswith("_family"):
        core = core[: -len("_family")]
    if core.startswith(ADAPTIVE_PREFIX):
        core = core[len(ADAPTIVE_PREFIX):]
    return core


def _load_metrics_json(variant_dir):
    """variant_dir = .../<detector>_pretrained (or ft folder). It contains exactly
    one dataset-keys subfolder holding metrics.json; glob for it since the
    subfolder name isn't fixed."""
    pattern = os.path.join(variant_dir, "*", METRICS_FILENAME)
    matches = [m for m in glob.glob(pattern) if os.path.isfile(m)]

    if not matches:
        print(f"  [warn] no {METRICS_FILENAME} found under {variant_dir}")
        return None
    if len(matches) > 1:
        print(f"  [warn] multiple {METRICS_FILENAME} found under {variant_dir}: {matches} -- using first")

    with open(matches[0], "r") as f:
        return json.load(f)


def _find_variant_dir(dataset_dir, detector, variant):
    """variant: 'pretrained' or 'ft'. Both use EXACT folder names (no wildcard):
    pretrained is always "<detector>_pretrained"; ft is looked up from
    FT_FOLDER_NAME so that other/spurious ft strategies sitting alongside the
    target one (e.g. R50_TF_ft_<other_strategy> next to R50_TF_ft_unfreezeL4)
    are never picked up."""
    if variant == "pretrained":
        folder_name = f"{detector}_pretrained"
    else:
        folder_name = FT_FOLDER_NAME[detector]

    candidate = os.path.join(dataset_dir, folder_name)
    if os.path.isdir(candidate):
        return candidate

    # helpful warning if there ARE ft-like folders here but none match the target
    if variant == "ft":
        other_ft_dirs = glob.glob(os.path.join(dataset_dir, f"{detector}_ft*"))
        if other_ft_dirs:
            print(f"  [warn] found ft folder(s) in {dataset_dir} but none match expected "
                  f"'{folder_name}': {other_ft_dirs} -- skipping (not using these)")

    return None


def _find_ensemble_metrics(detector, dataset_name, platform_key):
    """Locate the flat ensemble_report_<ft_model_tag>_<dataset>[_images].json
    file for this detector/dataset/platform combo, under any family folder.
    Returns the results.average dict, or None."""
    prefix = ENSEMBLE_PLATFORM_PREFIX.get(platform_key)
    if prefix is None:
        # platform has no defined ensemble tree (e.g. "social") -- nothing to look up
        return None

    base = os.path.join(ENSEMBLE_BASE, prefix) if prefix else ENSEMBLE_BASE
    model_tag = FT_FOLDER_NAME[detector]

    # Filenames are inconsistent in up to two ways: an optional "_images" suffix,
    # and (occasionally, e.g. R50_nodown/adaptive_blurbg_subtle on facebook) the
    # "adaptive_" prefix is dropped from the dataset portion entirely. Check all
    # 4 combinations.
    dataset_topic = dataset_name[len(ADAPTIVE_PREFIX):] if dataset_name.startswith(ADAPTIVE_PREFIX) else dataset_name
    dataset_variants = [dataset_name] if dataset_topic == dataset_name else [dataset_name, dataset_topic]
    filename_candidates = [
        f"ensemble_report_{model_tag}_{d}{suffix}.json"
        for d in dataset_variants
        for suffix in ("", "_images")
    ]

    matches = []
    for family_dir in glob.glob(os.path.join(base, "*")):
        if not os.path.isdir(family_dir):
            continue
        for fname in filename_candidates:
            candidate = os.path.join(family_dir, fname)
            if os.path.isfile(candidate):
                matches.append(candidate)

    if not matches:
        print(f"  [warn] no ensemble_report_{model_tag}_{dataset_name}[_images].json found for "
              f"platform={platform_key} (searched under {base}/*)")
        return None

    def _family_of(match_path):
        return os.path.basename(os.path.dirname(match_path))

    if len(matches) == 1:
        chosen = matches[0]
    else:
        dataset_topic = dataset_name[len(ADAPTIVE_PREFIX):] if dataset_name.startswith(ADAPTIVE_PREFIX) else dataset_name
        preferred = []
        for m in matches:
            core = _family_core_token(_family_of(m))
            if dataset_topic.startswith(core) or core.startswith(dataset_topic):
                preferred.append(m)

        if len(preferred) == 1:
            chosen = preferred[0]
            ignored = [_family_of(m) for m in matches if m != chosen]
            print(f"  [info] multiple ensemble report matches for {detector}/{dataset_name} ({platform_key}); "
                  f"using family-matched '{_family_of(chosen)}', ignoring {ignored}")
        elif len(preferred) > 1:
            chosen = sorted(preferred)[0]
            print(f"  [warn] multiple FAMILY-MATCHED ensemble reports for {detector}/{dataset_name} "
                  f"({platform_key}): {[_family_of(m) for m in preferred]} -- using '{_family_of(chosen)}', "
                  f"pick may be arbitrary -- verify manually")
        else:
            chosen = sorted(matches)[0]
            print(f"  [warn] multiple ensemble reports for {detector}/{dataset_name} ({platform_key}) but "
                  f"none match the dataset by family name: {[_family_of(m) for m in matches]} -- falling back "
                  f"to '{_family_of(chosen)}', verify manually")

    with open(chosen, "r") as f:
        data = json.load(f)

    avg = data.get("results", {}).get("average")
    if avg is None:
        print(f"  [warn] {chosen} has no results.average block")
    return avg


def _add_records(records, platform_key, dataset_name, detector, variant, metrics, metric_label_override=None):
    for metric_name, metric_val in metrics.items():
        label = metric_label_override if (metric_label_override and metric_name == REQUIRED_METRIC_KEY) else metric_name
        records.append({
            "platform": platform_key,
            "dataset": dataset_name,
            "detector": detector,
            "variant": variant,
            "metric": label,
            "value": metric_val,
        })


def collect_results(base_dir=BASE_DIR):
    results = defaultdict(lambda: defaultdict(dict))
    records = []

    for detector in DETECTORS:
        variant_roots = DETECTOR_VARIANT_ROOT[detector]

        # ---- regular platforms: PRE, facebook, telegram, twitter ----
        for platform_key, suffix in PLATFORM_SUFFIXES.items():
            pretrained_top = os.path.join(base_dir, detector, variant_roots["pretrained"] + suffix)
            if not os.path.isdir(pretrained_top):
                print(f"[skip] folder not found: {pretrained_top}")
                continue

            dataset_names = sorted(
                d for d in os.listdir(pretrained_top) if os.path.isdir(os.path.join(pretrained_top, d))
            )

            for raw_dataset_name in dataset_names:
                dataset_name = _normalize_dataset_name(raw_dataset_name)
                if dataset_name != raw_dataset_name:
                    print(f"  [info] normalized dataset folder name '{raw_dataset_name}' -> '{dataset_name}' "
                          f"(platform={platform_key})")

                entry = {}
                is_adaptive = dataset_name.startswith(ADAPTIVE_PREFIX)

                # ---- pretrained ----
                # NOTE: use raw_dataset_name here -- that's the real on-disk folder name
                dataset_dir_pre = os.path.join(pretrained_top, raw_dataset_name)
                pre_variant_dir = _find_variant_dir(dataset_dir_pre, detector, "pretrained")
                if pre_variant_dir:
                    metrics = _load_metrics_json(pre_variant_dir)
                    if metrics is not None:
                        if REQUIRED_METRIC_KEY not in metrics:
                            print(f"  [warn] '{REQUIRED_METRIC_KEY}' not found in {pre_variant_dir} "
                                  f"(keys: {list(metrics.keys())})")
                        else:
                            entry["pretrained"] = metrics
                            _add_records(records, platform_key, dataset_name, detector, "pretrained", metrics)

                # ---- ft: ensemble for adaptive_*, ft models otherwise ----
                if is_adaptive:
                    ensemble_metrics = _find_ensemble_metrics(detector, dataset_name, platform_key)
                    if ensemble_metrics is not None:
                        entry["ft"] = ensemble_metrics
                        _add_records(records, platform_key, dataset_name, detector, "ft", ensemble_metrics)
                else:
                    ft_top = os.path.join(base_dir, detector, variant_roots["ft"] + suffix)
                    # NOTE: use raw_dataset_name here too 
                    dataset_dir_ft = os.path.join(ft_top, raw_dataset_name)
                    ft_variant_dir = _find_variant_dir(dataset_dir_ft, detector, "ft")
                    if ft_variant_dir:
                        metrics = _load_metrics_json(ft_variant_dir)
                        if metrics is not None:
                            if REQUIRED_METRIC_KEY not in metrics:
                                print(f"  [warn] '{REQUIRED_METRIC_KEY}' not found in {ft_variant_dir} "
                                      f"(keys: {list(metrics.keys())})")
                            else:
                                entry["ft"] = metrics
                                _add_records(records, platform_key, dataset_name, detector, "ft", metrics)

                if entry:
                    results[platform_key][dataset_name][detector] = entry

        # ---- social platform: pretrained-only clean baselines ----
        social_top = os.path.join(base_dir, detector, variant_roots["pretrained"] + SOCIAL_SUFFIX)
        if not os.path.isdir(social_top):
            print(f"[skip] folder not found: {social_top}")
            continue

        social_dataset_names = sorted(
            d for d in os.listdir(social_top) if os.path.isdir(os.path.join(social_top, d))
        )
        for raw_dataset_name in social_dataset_names:
            dataset_name = _normalize_dataset_name(raw_dataset_name)
            if dataset_name != raw_dataset_name:
                print(f"  [info] normalized dataset folder name '{raw_dataset_name}' -> '{dataset_name}' "
                      f"(platform={SOCIAL_PLATFORM_KEY})")

            dataset_dir = os.path.join(social_top, raw_dataset_name)
            variant_dir = _find_variant_dir(dataset_dir, detector, "pretrained")
            if not variant_dir:
                continue
            metrics = _load_metrics_json(variant_dir)
            if metrics is None:
                continue
            if REQUIRED_METRIC_KEY not in metrics:
                print(f"  [warn] '{REQUIRED_METRIC_KEY}' not found in {variant_dir} (keys: {list(metrics.keys())})")
                continue

            results[SOCIAL_PLATFORM_KEY][dataset_name][detector] = {"pretrained": metrics}
            _add_records(records, SOCIAL_PLATFORM_KEY, dataset_name, detector, "pretrained", metrics,
                         metric_label_override="Baseline Acc")

    results = {p: {d: dict(det) for d, det in ds.items()} for p, ds in results.items()}
    return results, records


def by_detector(records):
    """{detector: {platform: {dataset: {variant: {metric: value}}}}}"""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for r in records:
        out[r["detector"]][r["platform"]][r["dataset"]][r["variant"]][r["metric"]] = r["value"]

    def _to_plain(d):
        if isinstance(d, defaultdict):
            d = {k: _to_plain(v) for k, v in d.items()}
        return d

    return _to_plain(out)


def get_detector_records(records, detector):
    return [r for r in records if r["detector"] == detector]


if __name__ == "__main__":
    results, records = collect_results()

    with open("results_nested.json", "w") as f:
        json.dump(results, f, indent=2)

    with open("results_flat.json", "w") as f:
        json.dump(records, f, indent=2)

    detector_results = by_detector(records)
    with open("results_by_detector.json", "w") as f:
        json.dump(detector_results, f, indent=2)

    os.makedirs("results_by_detector", exist_ok=True)
    for detector, data in detector_results.items():
        safe_name = detector.replace("/", "_")
        with open(os.path.join("results_by_detector", f"{safe_name}.json"), "w") as f:
            json.dump(data, f, indent=2)

    print(f"\nCollected {len(records)} data points across {len(results)} platforms.")
    print(f"Detectors found: {sorted(detector_results.keys())}")
    print("Saved: results_nested.json, results_flat.json, results_by_detector.json")
    print("Saved per-detector files in: results_by_detector/")