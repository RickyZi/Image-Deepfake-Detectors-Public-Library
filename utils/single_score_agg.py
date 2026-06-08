"""
score_aggregator.py

Aggregates scores from a specific detector test run and computes overall metrics.

Expected folder layout:
    <results_dir>/gan1:fb/image_results.json
    <results_dir>/gan2:fb/image_results.json
    ...

Usage:
  # Aggregate all data-key subfolders:
  python score_aggregator.py <path/to/R50_nodown>

  # Filter to specific data keys (e.g. only Facebook results):
  python score_aggregator.py <path/to/R50_nodown> --filter fb

Usage:
  python score_aggregator.py results/pretrained/season_TM01/R50_nodown
"""

import sys
import argparse
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import os


# ------------------------------------------------------------------ #
# I/O                                                                #
# ------------------------------------------------------------------ #

def load_image_results(data_key_dir: Path):
    """Load image_results.json from a data-key folder. Returns [] on missing file."""
    path = data_key_dir / 'image_results.json'
    if not path.exists():
        print(f"  [WARN] image_results.json not found in {data_key_dir}, skipping.")
        return []
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------ #
# Score aggregation                                                  #
# ------------------------------------------------------------------ #

def aggregate_scores(run_dir: Path): #, key_filter=None):
    """
    Collect scores and labels from all run-dir folders
    """
    all_scores, all_labels = [], []
    image_results = {"Fake": {}, "Real": {}}
    per_key = {}

    dk_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir())

    for dk_dir in dk_dirs:
        key = dk_dir.name
        # if key_filter and key_filter not in key:
        #     continue

        items = load_image_results(dk_dir)
        if not items:
            continue

        key_scores, key_labels = [], []
        for item in items:
            if 'score_mix' in item:       # P2G format
                score = item['score_mix']
                label = item['binary_label']
            else:                         # all other detectors
                score = item['score']
                label = item['label']

            all_scores.append(score)
            all_labels.append(label)
            key_scores.append(score)
            key_labels.append(label)

            bucket = "Fake" if label == 1 else "Real"
            image_results[bucket][item['path']] = {"score": score, "label": label}

        per_key[key] = {"scores": key_scores, "labels": key_labels}
        print(f"  Loaded {len(key_scores):4d} samples from {key}")

    return all_scores, all_labels, image_results, per_key


# ------------------------------------------------------------------ #
# Metrics                                                            #
# ------------------------------------------------------------------ #

def calculate_metrics(scores, labels, is_p2g=False):
    scores = np.array(scores)
    labels = np.array(labels)

    if is_p2g:
        # TO BE FIXED!!!!
        # predictions = scores.astype(int)
        # auc_input   = scores.astype(float)
        print("P2G detector currently not supported! - SKIPPING METRICS COMPUTATION!")
        return
    else:
        predictions = (scores > 0).astype(int)
        import torch
        auc_input = torch.sigmoid(torch.tensor(scores)).numpy()

    total_accuracy = accuracy_score(labels, predictions)

    fake_mask = labels == 1
    tpr = accuracy_score(labels[fake_mask], predictions[fake_mask]) if fake_mask.sum() > 0 else 0.0

    real_mask = labels == 0
    tnr = accuracy_score(labels[real_mask], predictions[real_mask]) if real_mask.sum() > 0 else 0.0

    auc = 0.0
    if len(np.unique(labels)) > 1:
        try:
            auc = roc_auc_score(labels, auc_input)
        except Exception:
            pass

    f1 = 2 * tpr * tnr / (tpr + tnr) if (tpr + tnr) > 0 else 0.0
    balanced_accuracy = (tpr + tnr) / 2

    return {
        'TPR':          float(tpr),
        'TNR':          float(tnr),
        'Acc':          float(total_accuracy),
        'Balanced Acc': float(balanced_accuracy),
        'F1':           float(f1),
        'AUC':          float(auc),
        'num_images':   int(len(labels)),
    }


# ------------------------------------------------------------------ #
# Main                                                               #
# ------------------------------------------------------------------ #

def agg_scores(run_dir):
    all_scores, all_labels, image_results, per_key = aggregate_scores(run_dir)

    if not all_scores:
        print("[ERROR] No scores collected. Check that image_results.json files exist.")
        sys.exit(1)

    detector_name = run_dir.name
    is_p2g = 'P2G' in detector_name

    dataset_name = str(run_dir).split(os.sep)[-2]
    print(f"detector_name: {detector_name}")
    print(f"dataset_name: {dataset_name} ")
    # breakpoint()

    # Per-data-key metrics
    print()
    per_key_metrics = {}
    for key, data in sorted(per_key.items()):
        m = calculate_metrics(data['scores'], data['labels'], is_p2g=is_p2g)
        per_key_metrics[key] = m
        print(f"  {key:20s}  TPR={m['TPR']:.3f}  TNR={m['TNR']:.3f}  "
              f"AUC={m['AUC']:.3f}  n={m['num_images']}")

    # Overall metrics
    overall = calculate_metrics(all_scores, all_labels, is_p2g=is_p2g)
    print(f"\n  {'OVERALL':20s}  TPR={overall['TPR']:.3f}  TNR={overall['TNR']:.3f}  "
          f"AUC={overall['AUC']:.3f}  n={overall['num_images']}")

    # Save output
    output = {
        'run_dir':      str(run_dir),
        'detector':     detector_name,
        # 'key_filter':   args.key_filter,
        'overall':      overall,
        'per_data_key': per_key_metrics, # per generator data (flux:fb, ...)
    }

    # suffix   = f"_{args.key_filter}" if args.key_filter else ""
    out_path = run_dir / f"{detector_name}_{dataset_name}_aggregated_metrics.json"

    save_agg_scores(output, out_path)
    

def save_agg_scores(output, out_path):
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nAggregated metrics saved to:\n  {out_path}\n")


# ------------------------------------------------------------------ #
# Entry point                                                        #
# ------------------------------------------------------------------ #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate test-run scores for one detector.')

    parser.add_argument('--results-dir', type=Path, default = './results/pretrained/', help='Path to the detector run folder, e.g. results/pretrained/season_TM01/R50_nodown')

    parser.add_argument('--model', type = str,  default = 'R50_nodown_pretrained', help = 'Model run name, i.e. R50_nodown_pretrained or R50_nodown_ft')
    
    args = parser.parse_args()

    # example run: python3 utils/single_score_agg.py --model CLIP-D_ft 
    
    run_dir = args.results_dir.resolve()
    print(f"\nRun dir : {run_dir}")

    if not run_dir.is_dir():
        print(f"[ERROR] Not a directory: {run_dir}")
        sys.exit(1)

    # single folder run -> i.e. results/pretrained/filminspired_warmgold/R50_nodown_pretrained
    if run_dir != './resuls/pretrained/':
        print(f"Aggregating scores for\n\t model: {Path(run_dir).name}\n\t dataset: {str(run_dir).split(os.sep)[-2]} ")
        breakpoint()
        agg_scores(run_dir)

    else:
        # walk all dataset in the run directories
        for dataset_dir in sorted(Path(run_dir).iterdir()):
            # print(f"dataset_dir: {dataset_dir}")

            # check the models name inside dataset dir
            for model_dir in sorted(dataset_dir.iterdir()):
                # print(f"model_dir: {model_dir}")
                if model_dir.name == args.model:
                    print(f"\nAggregating scores for\n\t model: {model_dir.name}\n\t dataset: {dataset_dir.name}")
                    agg_scores(model_dir)
    