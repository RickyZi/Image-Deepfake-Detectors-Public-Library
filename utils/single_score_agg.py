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
from scipy.special import expit as sigmoid  # float64-safe sigmoid, no torch dependency
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score
import os
import torch

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

def calculate_metrics(scores, labels):
    # scores = np.array(scores)
    # labels = np.array(labels)

    # if is_p2g:
    #     # TO BE FIXED!!!!
    #     # predictions = scores.astype(int)
    #     # auc_input   = scores.astype(float)
    #     print("P2G detector currently not supported! - SKIPPING METRICS COMPUTATION!")
    #     return
    # else:
    #     predictions = (scores > 0).astype(int)
    #     import torch
    #     auc_input = torch.sigmoid(torch.tensor(scores)).numpy()

    # total_accuracy = accuracy_score(labels, predictions)

    # fake_mask = labels == 1
    # tpr = accuracy_score(labels[fake_mask], predictions[fake_mask]) if fake_mask.sum() > 0 else 0.0

    # real_mask = labels == 0
    # tnr = accuracy_score(labels[real_mask], predictions[real_mask]) if real_mask.sum() > 0 else 0.0

    # auc = 0.0
    # if len(np.unique(labels)) > 1:
    #     try:
    #         auc = roc_auc_score(labels, auc_input)
    #     except Exception:
    #         pass

    # f1 = 2 * tpr * tnr / (tpr + tnr) if (tpr + tnr) > 0 else 0.0
    # balanced_accuracy = (tpr + tnr) / 2

    """
    Replicates test.py metric computation exactly.
    AUC uses sigmoid on raw scores, matching:
        probabilities = torch.sigmoid(torch.tensor(all_scores)).numpy()
        auc = roc_auc_score(all_labels, probabilities)
    """

    all_scores = np.array(scores, dtype=np.float64)
    all_labels = np.array(labels, dtype=np.int32)  # labels are 1.0/0.0 floats from JSON
 
    # P2G score_mix values are in [0,1]; threshold at 0.5
    predictions = (all_scores > 0).astype(int)
 
    total_accuracy = accuracy_score(all_labels, predictions)
 
    fake_mask = all_labels == 1
    tpr = accuracy_score(all_labels[fake_mask], predictions[fake_mask]) if fake_mask.sum() > 0 else 0.0
 
    real_mask = all_labels == 0
    tnr = accuracy_score(all_labels[real_mask], predictions[real_mask]) if real_mask.sum() > 0 else 0.0
 
    # AUC: sigmoid on raw logits, matching test.py. scipy expit stays float64.
    if len(np.unique(all_labels)) > 1:
        # Apply sigmoid to convert scores to probabilities
        probabilities = torch.sigmoid(torch.tensor(all_scores)).numpy()
        auc = roc_auc_score(all_labels, probabilities) # sigmoid(all_scores))
    else:
        auc = 0.0
 
    # Standard precision-recall F1
    # f1 = f1_score(all_labels, predictions, zero_division=0.0)
    f1 = f1_score(all_labels, predictions, labels=[0, 1], zero_division=0.0)
 
    # balanced_accuracy = (tpr + tnr) / 2
    balanced_accuracy = balanced_accuracy_score(all_labels, predictions)  # adjusted=False by default
 
    # return tpr, tnr, total_accuracy, auc, f1, balanced_accuracy

    return {
        'TPR':          float(tpr),
        'TNR':          float(tnr),
        'Acc':          float(total_accuracy),
        'Balanced Acc': float(balanced_accuracy),
        'F1':           float(f1),
        'AUC':          float(auc),
        'num_images':   int(len(labels)),
    }


def p2g_calculate_metrics(scores, labels):
    """
    P2G variant. score_mix values are raw continuous scores (not binarized),
    so AUC uses sigmoid on them — same pipeline as calculate_metrics.
    Threshold for TPR/TNR/Acc remains at 0.5 (P2G scores are in [0,1] range
    after mix_top_mean aggregation; adjust if your scores use a different scale).
    """
    all_scores = np.array(scores)
    all_labels = np.array(labels)
 
    # P2G score_mix values are in [0,1]; threshold at 0.5
    predictions = (all_scores > 0.5).astype(int)
 
    total_accuracy = accuracy_score(all_labels, predictions)
 
    fake_mask = all_labels == 1
    tpr = accuracy_score(all_labels[fake_mask], predictions[fake_mask]) if fake_mask.sum() > 0 else 0.0
 
    real_mask = all_labels == 0
    tnr = accuracy_score(all_labels[real_mask], predictions[real_mask]) if real_mask.sum() > 0 else 0.0
 
    # AUC: sigmoid on continuous score_mix values. scipy expit stays float64.
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, sigmoid(all_scores))
    else:
        auc = 0.0
 
    # Standard precision-recall F1
    f1 = f1_score(all_labels, predictions, labels=[0, 1], zero_division=0.0)
 
    balanced_accuracy = (tpr + tnr) / 2
 
    # return tpr, tnr, total_accuracy, auc, f1, balanced_accuracy
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
        if is_p2g:
            m = p2g_calculate_metrics(data['scores'], data['labels'])
        m = calculate_metrics(data['scores'], data['labels'])
        per_key_metrics[key] = m
        print(f"  {key:20s}  TPR={m['TPR']:.3f}  TNR={m['TNR']:.3f}  "
              f"AUC={m['AUC']:.3f}  n={m['num_images']}")

    # Overall metrics

    if is_p2g:
        print("Aggregating P2G metrics")
        overall = p2g_calculate_metrics(all_scores, all_labels)
    else: 
        overall = calculate_metrics(all_scores, all_labels)
    print(f"\n  {'OVERALL':20s}  TPR={overall['TPR']:.3f}  TNR={overall['TNR']:.3f}  AUC={overall['AUC']:.3f}  n={overall['num_images']}")

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

    # parser.add_argument('--results-dir', type=Path, default = './results/pretrained/', help='Path to the detector run folder, e.g. results/pretrained/season_TM01/R50_nodown')

    # parser.add_argument('--model', type = str,  default = 'R50_nodown_pretrained', help = 'Model run name, i.e. R50_nodown_pretrained or R50_nodown_ft')
    
    # args = parser.parse_args()

    model = sys.argv[1] if len(sys.argv) > 2 else 'R50_nodown_pretrained'
    results_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('./results/pretrained/')
    

    # example run: python3 utils/single_score_agg.py --model CLIP-D_ft 
    # python3 utils/single_score_agg.py R50_TF ./results/pretrained/ (to compute all results in one-shot, 
    # otherwise specify the desired folder)
    
    # if not '_pretrained' in model: model += '_pretrained' 
    print(f"Model: {model}")
    # breakpoint()

    # run_dir = args.results_dir.resolve()
    run_dir = results_dir.resolve()
    print(f"\nRun dir : {run_dir}")
    
    # breakpoint()

    if not run_dir.is_dir():
        print(f"[ERROR] Not a directory: {run_dir}")
        sys.exit(1)

    
    
    # walk all dataset in the run directories
    for dataset_dir in sorted(Path(run_dir).iterdir()):
        print(f"dataset_dir: {dataset_dir}")

        # check if single run folder (model name instead dataset_dir)
        # single folder run -> i.e. results/pretrained/filminspired_warmgold/
        # dataset_dir = model name instead of filminspired_warmgold
        if dataset_dir.name == model:
            print(f"\n Aggregating SINGLE run scores for \n\t model: {dataset_dir.name}\n\t dataset: {run_dir.name}")
            # breakpoint()
            agg_scores(dataset_dir)
        else:
            # check the models name inside dataset dir
            for model_dir in sorted(dataset_dir.iterdir()):
                # print(f"model_dir: {model_dir.name}")
                if model_dir.name == model:
                    print(f"\nAggregating scores for\n\t model: {model_dir.name}\n\t dataset: {dataset_dir.name}")
                    agg_scores(model_dir)
    