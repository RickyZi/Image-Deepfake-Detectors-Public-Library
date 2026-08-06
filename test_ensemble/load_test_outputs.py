"""
load_test_outputs.py — load one or more test.py-format image_results.json
files (self and cross-preset runs you've already produced) and align them
into per-image {model_id: score} records ready for combination.py.

test.py stores RAW LOGITS in image_results.json:
    scores = model(data).squeeze(1)          # pre-sigmoid
    image_results.append({'path': path, 'score': score.item(), 'label': label_val})
combination.py's rules all assume probability-like scores in [0, 1] (e.g.
average_combine treats a mean of raw logits very differently than a mean
of probabilities, and AUC/thresholding both assume [0,1]) - so every score
is passed through sigmoid here, once, on load. Don't sigmoid twice if
you've already converted a file yourself.

Usage:
    from load_test_outputs import align_from_files

    records = align_from_files({
        'R50nd@blurbg_strong': 'results/self_blurbg_strong/image_results.json',
        'R50nd@blurbg_subtle': 'results/cross_strong_ckpt_on_subtle_images/image_results.json',
    })
    # records: [(scores_dict, label), ...] - same shape align_scores()
    # produces in run_ensemble_test.py, ready for combination.py's
    # average_combine / max_combine / weighted_combine / compute_auc_weights
    # / SubsetStackingEnsemble / evaluate_rules.
"""
import json
import math


def _sigmoid(x):
    # Plain-Python sigmoid so this file has no torch dependency - useful
    # if you're aligning results on a machine without the training env set up.
    return 1.0 / (1.0 + math.exp(-x))


def load_image_results(path, apply_sigmoid=True):
    """Returns {path: (probability_or_raw_score, label)}."""
    with open(path) as f:
        results = json.load(f)

    out = {}
    for r in results:
        score = _sigmoid(r['score']) if apply_sigmoid else r['score']
        out[r['path']] = (score, r['label'])
    return out


def align_from_files(model_result_files, apply_sigmoid=True):
    """
    model_result_files: {model_id: path_to_image_results_json}
        e.g. {'R50nd@blurbg_strong': '.../self/image_results.json',
              'R50nd@blurbg_subtle': '.../cross/image_results.json'}
        model_id can be any string you want to show up as a key in
        combination.py's scores dicts / weights dicts - "{architecture}@{preset}"
        is just a convention, not required.

    Returns: list of (scores_dict, label) - only for image paths present
    in every file (an image missing from one run is dropped with a
    warning, same policy as run_ensemble_test.py's align_scores()).
    """
    per_model = {mid: load_image_results(p, apply_sigmoid)
                 for mid, p in model_result_files.items()}

    model_ids = list(per_model.keys())
    common_paths = set(per_model[model_ids[0]].keys())
    for m in model_ids[1:]:
        common_paths &= set(per_model[m].keys())

    all_paths = set()
    for m in model_ids:
        all_paths |= set(per_model[m].keys())
    dropped = all_paths - common_paths
    if dropped:
        print(f"[load_test_outputs] warning: {len(dropped)} image(s) missing from at "
              f"least one file - dropped from evaluation. Since self and cross runs "
              f"should share the exact same --data_root, a large drop count here "
              f"usually means the data_root or data_keys didn't actually match "
              f"between runs - worth checking before trusting the aligned results.")

    records = []
    for path in common_paths:
        scores = {}
        label = None
        for m in model_ids:
            score, lbl = per_model[m][path]
            scores[m] = score
            label = lbl  # same image, same label regardless of which checkpoint scored it
        records.append((scores, label))
    return records


if __name__ == '__main__':
    """
    CLI usage:

        python load_test_outputs.py test_files.json
        python load_test_outputs.py test_files.json --val-files val_files.json

    test_files.json / val_files.json format (same shape for both):
        {"R50nd@blurbg_strong": "results/.../self/image_results.json",
         "R50nd@blurbg_subtle": "results/.../cross/image_results.json"}

    --val-files is optional. Without it, only average/max/median are
    reported (they need no fitting). With it, weighted (weights computed
    from val-split AUC) and stacked (SubsetStackingEnsemble fit on val)
    are added too - both fit on val, evaluated on the test_files data, so
    nothing is evaluated on data it was fit on.
    """
    import argparse
    from combination import (average_combine, max_combine, median_combine, weighted_combine,
                              compute_auc_weights, describe_subsets,
                              SubsetStackingEnsemble, evaluate_rules)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('test_files', help='JSON file: {model_id: path_to_image_results_json}')
    parser.add_argument('--val-files', default=None,
                         help='JSON file: {model_id: path_to_val_image_results_json} - '
                              'used to compute weights and fit the stacked rule')
    parser.add_argument('--scores-are-probs', action='store_true',
                         help='Set when score files come from infer_r50nodown.py / '
                              'infer_r50tf.py / infer_clipd.py, which already apply '
                              'sigmoid and write probabilities. Without this flag, '
                              'sigmoid is applied on load (correct for test.py output, '
                              'which stores raw logits).')
    parser.add_argument('--detector-name', default=None,
                         help='Detector identifier used in the output filename, '
                              'e.g. R50_nodown_ft')
    parser.add_argument('--dataset-name', default=None,
                         help='Dataset/preset identifier used in the output filename, '
                              'e.g. adaptive_blurbg_strong')
    parser.add_argument('--out-dir', default='.',
                         help='Directory to write the ensemble report JSON '
                              '(default: current directory)')
    args = parser.parse_args()
    apply_sigmoid = not args.scores_are_probs

    with open(args.test_files) as f:
        test_file_map = json.load(f)
    eval_records = align_from_files(test_file_map, apply_sigmoid=apply_sigmoid)
    print(f'{len(eval_records)} aligned test records')
    print('test records by routing subset:')
    describe_subsets(eval_records)
    print()

    rules = {
        'average': average_combine,
        'max':     max_combine,
        'median':  median_combine,
    }

    computed_weights = None
    val_file_map = None

    if args.val_files:
        with open(args.val_files) as f:
            val_file_map = json.load(f)
        fit_records = align_from_files(val_file_map, apply_sigmoid=apply_sigmoid)
        print(f'{len(fit_records)} aligned val records')
        print('val records by routing subset:')
        describe_subsets(fit_records)
        print()

        weights = compute_auc_weights(fit_records)
        computed_weights = weights
        print(f'weights computed from val-split AUC: {weights}\n')
        rules['weighted'] = lambda s: weighted_combine(s, weights)

        stacker = SubsetStackingEnsemble(min_samples_per_subset=30).fit(fit_records)
        rules['stacked'] = stacker.predict
    else:
        print('No --val-files given - skipping weighted and stacked rules '
              '(they need val-split data to fit on).\n')

    results = evaluate_rules(eval_records, rules)

    # Per-checkpoint baselines: each model's solo performance on the same
    # images, so the report is self-contained for comparison without having
    # to cross-reference separate metrics.json files.
    # Aligned detector = same checkpoint as data source; cross detector = the
    # other checkpoint in the family. Both are labelled by their model_id key.
    model_ids = list(test_file_map.keys())
    baselines = evaluate_rules(
        eval_records,
        {mid: (lambda s, m=mid: s[m]) for mid in model_ids}
    )

    print('\nBaselines (single-checkpoint):')
    print(json.dumps(baselines, indent=2))
    print('\nEnsemble results:')
    print(json.dumps(results, indent=2))

    # Save report to JSON if --detector-name and --dataset-name are provided.
    # Filename: ensemble_report_{detector}_{dataset}.json
    if args.detector_name and args.dataset_name:
        import os
        os.makedirs(args.out_dir, exist_ok=True)
        report_filename = f"ensemble_report_{args.detector_name}_{args.dataset_name}.json"
        report_path = os.path.join(args.out_dir, report_filename)
        report = {
            'detector':    args.detector_name,
            'dataset':     args.dataset_name,
            'test_files':  test_file_map,
            'val_files':   val_file_map if args.val_files else None,
            'weights':     computed_weights if args.val_files else None,
            'baselines':   baselines,   # per-checkpoint solo performance
            'results':     results,     # ensemble combination rules
        }
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f'\nEnsemble report saved to: {report_path}')
    elif args.detector_name or args.dataset_name:
        print('\n[warn] Both --detector-name and --dataset-name are needed to save a '
              'report - provide both or neither.')