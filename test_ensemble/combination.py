"""
combination.py — pluggable score-combination rules for the C2PA-routed
detector ensemble.

Each rule takes the per-detector scores that were actually computed for a
given image (a dict, since routing means the set of detectors varies per
image - not every image gets all 3 models) and returns a single combined
probability that the image is fake.

None of this file depends on how you get the per-detector scores, on your
C2PA parser, or on your routing table - it only operates on
{'detector_name': probability} dicts, so it can be developed and unit
tested independently and then wired into your actual pipeline.
"""

import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression


# --------------------------------------------------------------------------
# Rule 1: simple average (Bonettini et al. 2020's choice - unweighted mean
# of sigmoid probabilities). No extra data needed; use as the baseline.
# --------------------------------------------------------------------------

def average_combine(scores):
    """scores: {'R50_TF': 0.83, 'CLIP-D': 0.61, ...} -> float in [0, 1]"""
    if not scores:
        raise ValueError("average_combine got no scores to combine")
    return float(np.mean(list(scores.values())))


# --------------------------------------------------------------------------
# Rule 2: weighted average, weights = per-(preset, detector) validation
# performance (e.g. AUC or accuracy) you already have from prior testing.
# Weights are renormalized over only the detectors actually present for
# this image, so it's well-defined regardless of how many models routing
# selected.
# --------------------------------------------------------------------------

def weighted_combine(scores, weights):
    """
    scores:  {'R50_TF': 0.83, 'CLIP-D': 0.61}
    weights: {'R50_TF': 0.91, 'CLIP-D': 0.87}  (e.g. per-preset val AUC)
             must have an entry for every key in `scores`.
    """
    if not scores:
        raise ValueError("weighted_combine got no scores to combine")
    missing = set(scores) - set(weights)
    if missing:
        raise KeyError(f"No weight provided for detector(s): {missing}")

    total_weight = sum(weights[d] for d in scores)
    if total_weight <= 0:
        # Degenerate weights (e.g. all zero) - fall back to simple average
        # rather than dividing by zero.
        return average_combine(scores)

    return float(sum(scores[d] * weights[d] for d in scores) / total_weight)


def max_combine(scores):
    """Max of per-detector probabilities - effectively an OR over 'fake':
    if even one detector is confident this is fake, the ensemble leans
    fake too. Appropriate when missing a real fake (false negative) is
    costlier to you than an extra false alarm on a real image - and
    particularly relevant here because each detector is fine-tuned on a
    different preset's manipulation. A detector whose training preset
    doesn't closely match this image's actual manipulation may under-
    react, while the detector that DOES match should catch it clearly.
    Averaging lets the under-reacting detector dilute that correct
    signal; max does not."""
    if not scores:
        raise ValueError("max_combine got no scores to combine")
    return float(max(scores.values()))


def median_combine(scores):
    """Median of per-detector probabilities - more robust to one outlier
    detector than the mean. Note it only behaves differently from
    average_combine when exactly 3 detectors are combined (median of 2
    numbers equals their mean by definition, and median of 1 number is
    just that number) - with routing typically selecting 1-3 detectors
    per image, this rule's distinct behavior only shows up on the 3-
    detector cases, where it becomes 'ignore the two most extreme
    opinions, keep the middle one' - narrower than it might sound, since
    it discards 2 of 3 models' information rather than weighting all
    three."""
    if not scores:
        raise ValueError("median_combine got no scores to combine")
    return float(np.median(list(scores.values())))


# --------------------------------------------------------------------------
# Deriving weights from data instead of guessing: each detector's own AUC,
# computed standalone (no combination involved) against the true labels on
# whatever records you pass in. Run this on val-split records - per preset
# for the most correct (but more data-hungry) weighting, or pooled across
# everything for a coarser global-per-detector weight.
# --------------------------------------------------------------------------

def compute_auc_weights(records):
    """
    records: [(scores_dict, label), ...] - all entries must have the same
    set of detector keys (true within one preset, or one routing subset).
    Returns {detector_name: auc}, each detector's own AUC standing alone.
    """
    from sklearn.metrics import roc_auc_score

    if not records:
        raise ValueError("compute_auc_weights got no records")
    detectors = set(records[0][0].keys())
    labels = np.array([label for _, label in records])
    if len(set(labels.tolist())) < 2:
        raise ValueError("compute_auc_weights needs both classes present in records")

    weights = {}
    for d in detectors:
        scores = np.array([s[d] for s, _ in records])
        weights[d] = float(roc_auc_score(labels, scores))
    return weights


# --------------------------------------------------------------------------
# Diagnostic: how many records fall into each distinct routing subset (set
# of detectors that ran for that image), with class balance. Run this
# before trusting SubsetStackingEnsemble - a subset with too few examples
# will silently fall back to average_combine (see min_samples_per_subset),
# which is safe but means the stacker isn't actually doing anything for
# that subset.
# --------------------------------------------------------------------------

def describe_subsets(records):
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


# --------------------------------------------------------------------------
# Rule 3: learned meta-classifier (stacking), one per distinct routing
# subset. A "routing subset" is the frozenset of detector names that ran
# for a given image (e.g. frozenset({'R50_TF', 'CLIP-D'})). Images routed
# to different subsets are fundamentally different learning problems (the
# stacker for 2-model subsets never has R50_nodown's score to look at), so
# they get separate small logistic regressions rather than one global model
# with imputed/zeroed missing inputs.
# --------------------------------------------------------------------------

class SubsetStackingEnsemble:
    """
    Fit:
        ensemble = SubsetStackingEnsemble()
        ensemble.fit(records)   # records: list of (scores_dict, label) pairs
                                 # label: 1.0 = fake, 0.0 = real

    Predict:
        ensemble.predict(scores_dict)  -> float in [0, 1]

    Internally trains one sklearn LogisticRegression per unique detector
    subset seen during fit(). A subset's classifier is only used at predict
    time if that exact subset was seen with enough examples during fit
    (see min_samples_per_subset) - otherwise predict() falls back to
    average_combine() for that subset and prints a warning, since a
    logistic regression trained on a handful of examples is likely to
    overfit worse than a plain average.
    """

    def __init__(self, min_samples_per_subset=30):
        self.min_samples_per_subset = min_samples_per_subset
        self.models = {}          # frozenset(detector_names) -> LogisticRegression
        self.detector_order = {}  # frozenset(detector_names) -> sorted tuple of names
        self.fallback_subsets = set()

    def fit(self, records):
        """records: iterable of (scores_dict, label) where label is 0/1."""
        by_subset = defaultdict(list)
        for scores, label in records:
            key = frozenset(scores.keys())
            by_subset[key].append((scores, label))

        for subset, items in by_subset.items():
            names = tuple(sorted(subset))
            self.detector_order[subset] = names

            if len(items) < self.min_samples_per_subset:
                self.fallback_subsets.add(subset)
                print(f"[SubsetStackingEnsemble] Only {len(items)} examples for "
                      f"subset {names} (< {self.min_samples_per_subset}) - "
                      f"will fall back to average_combine() for this subset at "
                      f"predict time instead of fitting a classifier on too "
                      f"little data.")
                continue

            X = np.array([[s[name] for name in names] for s, _ in items])
            y = np.array([label for _, label in items])
            if len(set(y.tolist())) < 2:
                self.fallback_subsets.add(subset)
                print(f"[SubsetStackingEnsemble] Subset {names} has only one "
                      f"class in the fit data - falling back to "
                      f"average_combine() for this subset.")
                continue

            clf = LogisticRegression()
            clf.fit(X, y)
            self.models[subset] = clf

        return self

    def predict(self, scores):
        subset = frozenset(scores.keys())
        names = self.detector_order.get(subset, tuple(sorted(subset)))

        if subset in self.models:
            X = np.array([[scores[name] for name in names]])
            return float(self.models[subset].predict_proba(X)[0, 1])

        # Unseen subset at fit time, or fell back due to too little data.
        return average_combine(scores)


# --------------------------------------------------------------------------
# Convenience: run several rules over the same labeled dataset and report
# metrics for each, so you can compare rules directly rather than running
# separate scripts per rule.
# --------------------------------------------------------------------------

def evaluate_rules(records, rules, threshold=0.5):
    """
    records: list of (scores_dict, label) - label 1.0 = fake, 0.0 = real
    rules:   {'rule_name': callable(scores_dict) -> float}
    Returns: {'rule_name': {'TPR':..., 'TNR':..., 'Acc':..., 'Balanced Acc':...,
                             'F1':..., 'AUC':..., 'num_images':...}}
    Metric set and key names match test.py's own metrics.json exactly, so
    ensemble results are directly comparable to each individual detector's
    own test output.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, balanced_accuracy_score

    results = {}
    labels = np.array([label for _, label in records])

    for rule_name, rule_fn in rules.items():
        probs = np.array([rule_fn(scores) for scores, _ in records])
        preds = (probs > threshold).astype(int)

        # ── Diagnostic: show first 3 images per rule ──────────────────────
        print(f'\n  [evaluate_rules] rule={rule_name}:')
        for i, ((scores, label), prob, pred) in enumerate(
                zip(records[:3], probs[:3], preds[:3])):
            scores_str = ', '.join(f'{k}={v:.4f}' for k, v in scores.items())
            print(f'    img{i+1}: scores=({scores_str})'
                  f' → combined={prob:.4f} → pred={pred}  label={int(label)}'
                  f'  {"✓" if pred == int(label) else "✗"}')
            
        # breakpoint()
        # ──────────────────────────────────────────────────────────────────


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