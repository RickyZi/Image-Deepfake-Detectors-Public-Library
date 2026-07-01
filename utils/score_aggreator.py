# access the results folder and aggregates the score for each run, then calculates the metrics and saves them in a json file
import sys
import os
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from datetime import datetime
from scipy.special import expit as sigmoid  # float64-safe sigmoid, no torch dependency
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score,  balanced_accuracy_score
from datetime import datetime


# from __future__ import annotations

"""
metrics are saved in the result folder
results/
    model_name_dataset_name/
        subset_name/
            results.csv
            metrics.json
            image_results.json
 
results.csv contains name,pro,flag for each image
metrics.json contains the overall metrics for the run
image_results.json contains the scores for each image along with the path and label
 
Metrics calculated (replicating test.py):
- TPR (True Positive Rate) = TP / (TP + FN) = accuracy on fake images (label==1)
- TNR (True Negative Rate) = TN / (TN + FP) = accuracy on real images (label==0)
- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- AUC: sigmoid applied to raw scores before roc_auc_score (matches test.py exactly)
- F1: sklearn precision-recall F1 (standard definition)
- Balanced Accuracy: (TPR + TNR) / 2
 
-> collect scores and labels from image_results.json
-> aggregate scores and labels across all runs
-> compute overall metrics (TPR, TNR, Accuracy, AUC, F1, Balanced Acc)
-> save aggregated metrics to a new JSON file (e.g., aggregated_metrics.json)
"""


def aggregate_scores(results_dir):
    all_scores = []
    all_labels = []
    image_results = {"Fake": {}, "Real": {}}

    # Iterate through all runs in the results directory
    for run_name in os.listdir(results_dir):
        run_path = os.path.join(results_dir, run_name)
        if os.path.isdir(run_path):
            # Look for metrics.json and image_results.json in each run folder
            metrics_path = os.path.join(run_path, 'metrics.json')
            image_results_path = os.path.join(run_path, 'image_results.json')
            
            if os.path.exists(metrics_path) and os.path.exists(image_results_path):
                with open(image_results_path, 'r') as f:
                    run_image_results = json.load(f)
                    for item in run_image_results:
                        if 'score_mix' in item.keys():  # Check if it's a P2G result
                            # print("processing P2G scores (score_mix)") # for image:", item['path'])
                            path = item['path']
                            score = item['score_mix']
                            label = item['binary_label']
                        else:
                            path = item['path']
                            score = item['score']
                            label = item['label']

                        all_scores.append(score)
                        all_labels.append(label)
                        
                        if label == 1:  # Fake
                            image_results["Fake"][path] = {"score": score, "label": label}
                        else:  # Real
                            image_results["Real"][path] = {"score": score, "label": label}
            else:
                print(f"Warning: metrics.json or image_results.json not found in {run_path}. Skipping this run.")


    return all_scores, all_labels, image_results


def calculate_metrics(scores, labels):
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
        auc = roc_auc_score(all_labels, sigmoid(all_scores))
    else:
        auc = 0.0
 
    # # Standard precision-recall F1
    # f1 = f1_score(all_labels, predictions, zero_division=0.0)
 
    # balanced_accuracy = (tpr + tnr) / 2
 
    # return tpr, tnr, total_accuracy, auc, f1, balanced_accuracy
    # Standard precision-recall F1
    # f1 = f1_score(all_labels, predictions, zero_division=0.0)
    f1 = f1_score(all_labels, predictions, labels=[0, 1], zero_division=0.0)
 
    # balanced_accuracy = (tpr + tnr) / 2
    balanced_accuracy = balanced_accuracy_score(all_labels, predictions)  # adjusted=False by default
 
    return tpr, tnr, total_accuracy, auc, f1, balanced_accuracy

    # return {
    #     'TPR':          float(tpr),
    #     'TNR':          float(tnr),
    #     'Acc':          float(total_accuracy),
    #     'Balanced Acc': float(balanced_accuracy),
    #     'F1':           float(f1),
    #     'AUC':          float(auc),
    #     'num_images':   int(len(labels)),
    # }
 
 
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
    # f1 = f1_score(all_labels, predictions, zero_division=0.0)
    f1 = f1_score(all_labels, predictions, labels=[0, 1], zero_division=0.0)
 
    # balanced_accuracy = (tpr + tnr) / 2
    balanced_accuracy = balanced_accuracy_score(all_labels, predictions)  # adjusted=False by default
 
    return tpr, tnr, total_accuracy, auc, f1, balanced_accuracy

    # return {
    #     'TPR':          float(tpr),
    #     'TNR':          float(tnr),
    #     'Acc':          float(total_accuracy),
    #     'Balanced Acc': float(balanced_accuracy),
    #     'F1':           float(f1),
    #     'AUC':          float(auc),
    #     'num_images':   int(len(labels)),
    # }



def save_aggregated_metrics(metrics, output_path):
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":

    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('./results/demo/demo_images/')
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a valid directory.")
        sys.exit(1)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_name = results_dir.name
    output_metrics_path = results_dir / f'{test_name}_aggregated_metrics.json'  # Path to save the aggregated metrics

    # all_methods = ['R50_nodown', 'CLIP-D', 'R50_TF', 'P2G', 'NPR']

    # get subfoders in results_dir -> Path
    res_subfolders = [f for f in results_dir.iterdir() if f.is_dir()]  
    
    aggregated_metrics = {}
    aggregated_metrics['dataset'] = results_dir.name
    aggregated_metrics['timestamp'] = timestamp

    # # investigate the scores and labels for each method separately
    for subfolder in res_subfolders:
        method = subfolder.name
        print(f"Processing method: {method}")
        all_scores, all_labels, image_results = aggregate_scores(subfolder)
        
        # Calculate overall metrics
        if 'P2G' in method:
            # print(f"Calculating P2G-specific metrics for method: {method}")
            tpr, tnr, total_accuracy, auc, f1_score, balanced_accuracy = p2g_calculate_metrics(all_scores, all_labels)
        else:
            tpr, tnr, total_accuracy, auc, f1_score, balanced_accuracy = calculate_metrics(all_scores, all_labels)
        
        # Prepare metrics JSON
        metrics = {
            'TPR': float(tpr),
            'TNR': float(tnr),
            'Acc': float(total_accuracy),
            'Balanced Acc': float(balanced_accuracy),
            'F1': float(f1_score),
            'AUC': float(auc),
            'num_images': len(all_labels), # NOTE: P2G will have 150 imgs since some do not load correctly, the metrics are computed on the available imgs
            # 'analysis_timestamp': timestamp  # Using the last modified time of the subfolder as a timestamp
        }
        
        # update aggregated metrics
        aggregated_metrics[method] = metrics


    # Save aggregated metrics to a new JSON file (rewritten a)
    save_aggregated_metrics(aggregated_metrics, output_metrics_path)

    print(f'Aggregated metrics saved to {output_metrics_path}\n')


