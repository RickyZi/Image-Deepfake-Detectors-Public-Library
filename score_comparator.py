"""
summarize the aggregated score for each demo_run comparing the results with the golden standard of 'demo_images' dataset, 
and save the aggregated metrics to a new JSON file (e.g., aggregated_metrics.json) in the results directory.

---

compare the score of each run with the ref metrics and calculate the difference in metrics score

ref example:
{
    # "dataset": "demo_images",
    "R50_nodown_demo_images": {
        "TPR": 0.68125,
        "TNR": 1.0,
        "Acc total": 0.745,
        "AUC": 0.9192187499999999,
        "num_images": 200,
        "analysis_timestamp": "20260514_170750"
    },
    "CLIP-D_demo_images": {
        "TPR": 0.84375,
        "TNR": 0.975,
        "Acc total": 0.87,
        "AUC": 0.97328125,
        "num_images": 200,
        "analysis_timestamp": "20260514_170750"
    },
    "NPR_demo_images": {
        "TPR": 0.2375,
        "TNR": 1.0,
        "Acc total": 0.39,
        "AUC": 0.738125,
        "num_images": 200,
        "analysis_timestamp": "20260514_170750"
    },
    "R50_TF_demo_images": {
        "TPR": 0.5375,
        "TNR": 0.95,
        "Acc total": 0.62,
        "AUC": 0.7903125,
        "num_images": 200,
        "analysis_timestamp": "20260514_170750"
    },
    "P2G_demo_images": {
        "TPR": 0.7583333333333333,
        "TNR": 1.0,
        "Acc total": 0.8066666666666666,
        "AUC": 0.8791666666666667,
        "num_images": 150,
        "analysis_timestamp": "20260514_170750"
    }
}

output example: contains only the differences in metrics
{
    "dataset": "{current dataset} vs demo_images",
    "R50_nodown": {
        "TPR_diff": -0.05,
        "TNR_diff": 0.0,
        ...
    },
    ...


}

"""

import os
import json
import sys
from pathlib import Path


def load_metrics(metrics_path):
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    else:
        print(f"Error: Metrics file not found at {metrics_path}")
        sys.exit(1)

def compute_metric_differences(ref_metrics, new_metrics):
    result = {}
    skip_keys = {'num_images', 'analysis_timestamp'}
    
    for key in new_metrics.keys():
        if key in skip_keys:
            continue
        # Always include the test metric value
        result[key] = new_metrics[key]
        # Add diff if the key exists in reference too
        if key in ref_metrics:
            diff = new_metrics[key] - ref_metrics[key]
            result[key + '_diff'] = diff
            if ref_metrics[key] != 0:
                result[key + '_percent_diff'] = diff / abs(ref_metrics[key]) * 100
    
    return result

def write_metrics(metrics, output_path):
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":

    ref_metrics_path = './results/demo/demo_images/demo_images_aggregated_metrics.json'
    metrics2compare_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('./results/demo/seasons_SM01/seasons_SM01_aggregated_metrics.json')
    
    
    

    # load metrics from json files
    ref_metrics = load_metrics(ref_metrics_path)
    metrics2compare = load_metrics(metrics2compare_path)

    metric_comparison = {
        'dataset': f"{metrics2compare['dataset']} vs {ref_metrics['dataset']}",
        'R50_nodown': {},
        'CLIP-D': {},
        'NPR': {},
        'R50_TF': {},
        'P2G': {}
    }

    output_comparison_path = f'./results/demo/metric_comparison_{metrics2compare["dataset"]}_vs_{ref_metrics["dataset"]}.json'


    for method in ['R50_nodown', 'CLIP-D', 'NPR', 'R50_TF', 'P2G']:
        if method in ref_metrics and method in metrics2compare:
            print(f"Comparing metrics for method: {method}")
            print(f"Reference metrics (from {ref_metrics['dataset']}): {ref_metrics[method]}")
            print(f"New metrics (from {metrics2compare['dataset']}): {metrics2compare[method]}")
            metric_comparison[method] = compute_metric_differences(ref_metrics[method], metrics2compare[method])
            # preserve num_imgs from test dataset
            metric_comparison[method]['num_images'] = metrics2compare[method].get('num_images', None)
            # breakpoint()
        else:
            metric_comparison[method] = None  # or some default value indicating missing method

    # save the metric comparison to a new JSON file

    # os.makedirs(os.path.dirname(output_comparison_path), exist_ok=True)
    write_metrics(metric_comparison, output_comparison_path)


