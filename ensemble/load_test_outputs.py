"""
alignes the image_results.json files produced by test.py for one or more checkpoints 
(self and cross-preset) into a single list of per-image records, 
each containing {model_id: score} and the ground truth label, ready for combination.py's ensemble rules.
"""

import json
import math

try:
    import torch as _torch
    def _sigmoid(x):
        return float(_torch.sigmoid(_torch.tensor(x, dtype=_torch.float32)).item())
except ImportError:
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
    make sure all models_results_files have the same image paths, 
    and return a list of aligned (scores_dict, label) records for 
    computing the ensamble combination rules in combination.py.
    """
    per_model = {mid: load_image_results(p, apply_sigmoid)
                 for mid, p in model_result_files.items()}

    # Diagnostic: per-file image counts before alignment
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
              "on some batches. Re-running test.py for that model should "
              "produce a complete image_results.json.\n")

    records = []
    for path in common_paths:
        scores = {}
        label = None
        for m in model_ids:
            score, lbl = per_model[m][path]
            scores[m] = score
            label = lbl  # same image, same label regardless of which model scored it (ground truth per image)
        records.append((scores, label))
    return records


if __name__ == '__main__':

    import argparse
    from combination import average_combine, max_combine, median_combine, describe_subsets, evaluate_rules

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

    eval_records = align_from_files(test_file_map, apply_sigmoid=apply_sigmoid)

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
        # 'max':     max_combine,
        # 'median':  median_combine,
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
    else:
        print('No --val-files given - skipping weighted and stacked rules '
              '(they need val-split data to fit on).\n')

    results = evaluate_rules(eval_records, rules)

    # Per-detector results
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
    if args.detector_name and args.dataset_name:
        print(f"detector: {args.detector_name}")
        print(f"dataset_name: {args.dataset_name}")
        print(f"data_root: {args.data_root}")
        # breakpoint()
        import os
        os.makedirs(args.out_dir, exist_ok=True)
        report_filename = f"ensemble_report_{args.dataset_name}.json"
        report_path = os.path.join(args.out_dir, report_filename)
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

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f'\nEnsemble report saved to: {report_path}')
    
    elif args.detector_name or args.dataset_name:
        print('\n[warn] Both --detector-name and --dataset-name are needed to save a '
              'report - provide both or neither.')