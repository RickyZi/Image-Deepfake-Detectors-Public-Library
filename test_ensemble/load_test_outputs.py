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

try:
    import torch as _torch
    def _sigmoid(x):
        # Uses torch.sigmoid to match test.py's computation exactly:
        #   probs = torch.sigmoid(torch.tensor(all_scores)).numpy()
        return float(_torch.sigmoid(_torch.tensor(x, dtype=_torch.float32)).item())
except ImportError:
    # Fallback for environments without PyTorch — mathematically identical
    # for all practical score ranges but kept as a safety net only.
    def _sigmoid(x):
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

    # ── Diagnostic: per-file image counts before alignment ─────────────────
    print('[align_from_files] per-file image counts:')
    for mid, results in per_model.items():
        print(f'  {mid}: {len(results)} images')

    model_ids = list(per_model.keys())
    common_paths = set(per_model[model_ids[0]].keys())
    for m in model_ids[1:]:
        common_paths &= set(per_model[m].keys())

    all_paths = set()
    for m in model_ids:
        all_paths |= set(per_model[m].keys())
    dropped = all_paths - common_paths
    if dropped:
        print(f"\n[align_from_files] WARNING: {len(dropped)} image(s) missing from at "
              f"least one file — dropped from evaluation.")
        print(f"  total unique paths across all files : {len(all_paths)}")
        print(f"  paths present in every file (kept)  : {len(common_paths)}")
        print(f"  paths missing from ≥1 file (dropped): {len(dropped)}")

        # Show which file each dropped image is missing from
        print("\n  Per-file breakdown of missing images:")
        for mid, results in per_model.items():
            missing_from_this = dropped - set(results.keys())
            if missing_from_this:
                print(f"    {mid}: missing {len(missing_from_this)} images")
                for p in sorted(missing_from_this)[:5]:
                    print(f"      {p}")
                if len(missing_from_this) > 5:
                    print(f"      ... and {len(missing_from_this)-5} more")

        # Save full list of dropped paths for inspection
        dropped_save_path = 'dropped_images.json'
        with open(dropped_save_path, 'w') as f:
            json.dump({
                'n_dropped': len(dropped),
                'dropped_paths': sorted(dropped),
                'per_file_missing': {
                    mid: sorted(dropped - set(results.keys()))
                    for mid, results in per_model.items()
                }
            }, f, indent=2)
        print(f"\n  Full list saved to: {dropped_save_path}")
        print("  Check which file each path is missing from — if one file has "
              "fewer entries than the others, that run likely failed silently "
              "on some batches. Re-running test.py for that checkpoint should "
              "produce a complete image_results.json.\n")

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
    parser.add_argument('--data-root', default=None,
                         help='Dataset/preset identifier used in the output filename, '
                              'e.g. adaptive_blurbg_strong')
    parser.add_argument('--out-dir', default='.',
                         help='Directory to write the ensemble report JSON '
                              '(default: current directory)')
    args = parser.parse_args()
    apply_sigmoid = not args.scores_are_probs

    with open(args.test_files) as f:
        test_file_map = json.load(f)

    # # ── Diagnostic: show raw logits before sigmoid ─────────────────────────
    # print('\n=== RAW SCORES FROM image_results.json (logits, pre-sigmoid) ===')
    # for model_id, fpath in test_file_map.items():
    #     with open(fpath) as f:
    #         raw = json.load(f)
    #     scores_only = [r['score'] for r in raw]
    #     labels_only = [r['label'] for r in raw]
    #     n_fake = sum(1 for l in labels_only if l == 1.0)
    #     n_real = sum(1 for l in labels_only if l == 0.0)
    #     print(f'  {model_id}:')
    #     print(f'    file       : {fpath}')
    #     print(f'    n_images   : {len(raw)}  (fake={n_fake}, real={n_real})')
    #     print(f'    score range: [{min(scores_only):.4f}, {max(scores_only):.4f}]')
    #     print(f'    mean score : {sum(scores_only)/len(scores_only):.4f}')
    #     # Show first 3 images as concrete examples
    #     for r in raw[:3]:
    #         sig = 1.0 / (1.0 + math.exp(-r['score']))
    #         pred = 1 if sig > 0.5 else 0
    #         print(f'    example: logit={r["score"]:+.4f} → sigmoid={sig:.4f} '
    #               f'→ pred={pred}  label={int(r["label"])}  '
    #               f'{"✓" if pred == int(r["label"]) else "✗"}')
    # print()

    eval_records = align_from_files(test_file_map, apply_sigmoid=apply_sigmoid)

    # ── Diagnostic: show aligned records after sigmoid ─────────────────────
    print(f'=== ALIGNED RECORDS (after sigmoid, {len(eval_records)} images) ===')
    labels_all = [label for _, label in eval_records]
    n_fake_aligned = sum(1 for l in labels_all if l == 1.0)
    n_real_aligned = sum(1 for l in labels_all if l == 0.0)
    print(f'  aligned images: {len(eval_records)} '
          f'(fake={n_fake_aligned}, real={n_real_aligned})')
    print(f'  model_ids in each record: {list(eval_records[0][0].keys())}')
    print('  first 3 aligned records:')
    for scores, label in eval_records[:3]:
        scores_str = ', '.join(f'{k}={v:.4f}' for k, v in scores.items())
        avg = sum(scores.values()) / len(scores)
        mx  = max(scores.values())
        print(f'    label={int(label)}  scores=({scores_str})')
        print(f'           → average={avg:.4f} pred={1 if avg>0.5 else 0} '
              f'| max={mx:.4f} pred={1 if mx>0.5 else 0}')
    print()

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
        print(f"detector: {args.detector_name}")
        print(f"dataset_name: {args.dataset_name}")
        print(f"data_root: {args.data_root}")
        # breakpoint()
        import os
        os.makedirs(args.out_dir, exist_ok=True)
        report_filename = f"ensemble_report_{args.dataset_name}.json"
        # ensemble_avg_metrics = f"ensemble_metrics.json"
        report_path = os.path.join(args.out_dir, report_filename)
        # metrics_path = os.path.join(args.out_dir, args.detector_name, args.dataset_name, ensemble_avg_metrics)
        # os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        report = {
            'detector':    args.detector_name,
            'dataset':     args.dataset_name,
            'data-root':   args.data_root,
            'test_files':  test_file_map,
            'val_files':   val_file_map if args.val_files else None,
            'weights':     computed_weights if args.val_files else None,
            'baselines':   baselines,   # per-checkpoint solo performance
            'results':     results,     # ensemble combination rules
        }

        # metrics = {
        #     'detector':    args.detector_name,
        #     'data-root':   args.data_root,
        #     'results':     results,
        # }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f'\nEnsemble report saved to: {report_path}')

        # with open(metrics_path, 'w') as f:
        #         json.dump(metrics, f, indent=2)
        # print(f'\nEnsemble report saved to: {metrics_path}')
    
    elif args.detector_name or args.dataset_name:
        print('\n[warn] Both --detector-name and --dataset-name are needed to save a '
              'report - provide both or neither.')