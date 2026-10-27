"""
combination.py — pluggable score-combination rules for the C2PA-routed
detector ensemble.

Each rule takes the per-detector scores that were actually computed for a
given image (a dict, since routing means the set of detectors varies per
image - not every image gets all 3 models) and returns a single combined
probability that the image is fake.
"""

import numpy as np

def average_combine(scores):
    """scores: {'R50_TF': 0.83, 'CLIP-D': 0.61, ...} -> float in [0, 1]"""
    if not scores:
        raise ValueError("average_combine got no scores to combine")
    return float(np.mean(list(scores.values())))


def max_combine(scores):
    """Max of per-detector probabilities"""
    if not scores:
        raise ValueError("max_combine got no scores to combine")
    return float(max(scores.values()))


def median_combine(scores):
    """Median of per-detector probabilities"""
    if not scores:
        raise ValueError("median_combine got no scores to combine")
    return float(np.median(list(scores.values())))


def describe_subsets(records):
    """
    records: list of (scores_dict, label) - label 1.0 = fake, 0.0 = real
    Prints a table of how many records fall into each distinct routing subset
    (set of detectors that ran for that image), with class balance.
    """


    from collections import defaultdict

    missing = []

    by_subset = defaultdict(list)
    for scores, label in records:
        by_subset[frozenset(scores.keys())].append(label)

    print(f"{'subset':<40} {'n':>6} {'n_fake':>7} {'n_real':>7}")
    for subset, labels in sorted(by_subset.items(), key=lambda kv: -len(kv[1])):
        labels_arr = np.array(labels)
        name = " + ".join(sorted(subset))
        n_fake = int((labels_arr == 1).sum())
        n_real = int((labels_arr == 0).sum())
        print(f"{name:<40} {len(labels_arr):>6} {n_fake:>7} {n_real:>7}")


def evaluate_rules(records, rules, threshold=0.5):
    """
    run several combination rules over the same labeled dataset to compare their metrics directly
    instead of running separate experiments per rule.

    records: list of (scores_dict, label) - label 1.0 = fake, 0.0 = real
    rules:   {'rule_name': callable(scores_dict) -> float}
    Returns: {'rule_name': {'TPR':..., 'TNR':..., 'Acc':..., 'Balanced Acc':...,
                             'F1':..., 'AUC':..., 'num_images':...}}
    """
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, balanced_accuracy_score

    results = {}
    labels = np.array([label for _, label in records])

    for rule_name, rule_fn in rules.items():
        probs = np.array([rule_fn(scores) for scores, _ in records])
        preds = (probs > threshold).astype(int)

        # # Diagnostic: show first 3 images per rule 
        # print(f'\n  [evaluate_rules] rule={rule_name}:')
        # for i, ((scores, label), prob, pred) in enumerate(
        #         zip(records[:3], probs[:3], preds[:3])):
        #     scores_str = ', '.join(f'{k}={v:.4f}' for k, v in scores.items())
        #     print(f'    img{i+1}: scores=({scores_str})'
        #           f' → combined={prob:.4f} → pred={pred}  label={int(label)}'
        #           f'  {"✓" if pred == int(label) else "✗"}')

        fake_mask = labels == 1
        real_mask = labels == 0

        results[rule_name] = {
            'TPR': round(float(accuracy_score(labels[fake_mask], preds[fake_mask])) if fake_mask.sum() > 0 else 0.0, 4),
            'TNR': round(float(accuracy_score(labels[real_mask], preds[real_mask])) if real_mask.sum() > 0 else 0.0, 4),
            'Acc': round(float(accuracy_score(labels, preds)), 4),
            'Balanced Acc': round(float(balanced_accuracy_score(labels, preds)), 4),
            'F1': round(float(f1_score(labels, preds, labels=[0, 1], zero_division=0.0)), 4 ),
            'AUC': round(float(roc_auc_score(labels, probs)) if len(set(labels.tolist())) > 1 else 0.0, 4),
            'num_images': int(len(labels)),
        }

    return results