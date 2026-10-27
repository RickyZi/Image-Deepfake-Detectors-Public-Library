"""
model ensemble inference subprocesses orchestrator
for one or more groups of known-ambiguous images, aligns their per-image
scores, and evaluates every combination rule from combination.py side by
side.
"""
import os
import sys
import json
import subprocess
import tempfile

from combination import average_combine, max_combine, median_combine, evaluate_rules, describe_subsets

DETECTOR_DIRS = {
    'R50_nodown': '/home/rz/TB_WP3/Image-Deepfake-Detectors-Public-Library/detectors/R50_nodown/',
    'R50_TF':     '/home/rz/TB_WP3/Image-Deepfake-Detectors-Public-Library/detectors/R50_TF/',
}
DETECTOR_SCRIPTS = {
    'R50_nodown': 'infer_r50nodown.py',
    'R50_TF':     'infer_r50tf.py',
}


def load_group_config(config_path):
    with open(config_path) as f:
        return json.load(f)


def run_checkpoint(architecture, preset, name, data_root, split_file, data_keys,
                    tf2k, device, ft, r50unfreezeL4, split):
    if architecture not in DETECTOR_DIRS:
        raise NotImplementedError(
            f"No inference script wired up yet for architecture '{architecture}' "
            f"(only {list(DETECTOR_DIRS)} are implemented)."
        )

    detector_dir = DETECTOR_DIRS[architecture]
    script = DETECTOR_SCRIPTS[architecture]

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        out_path = tmp.name

    cmd = [
        sys.executable, script,
        '--name', name,
        '--dataset', preset,
        '--data_root', data_root,
        '--split_file', split_file,
        '--data_keys', data_keys,
        '--tf2k', str(tf2k),
        '--device', device,
        '--split', split,
        '--out', out_path,
    ]
    if ft:
        cmd.append('--ft')
    if r50unfreezeL4:
        cmd.append('--r50unfreezeL4')

    print(f'[orchestrator] running {architecture}@{preset} (split={split})')
    subprocess.run(cmd, cwd=detector_dir, check=True)

    with open(out_path) as f:
        results = json.load(f)
    os.unlink(out_path)
    return {r['path']: (r['score'], r['label']) for r in results}


def align_scores(per_model_results):
    """
    per_model_results: {'R50_nodown@presetA': {path: (score, label)}, ...}
    Returns a list of (scores_dict, label) - only for paths every queried
    checkpoint actually returned a score for. An image missing from one
    checkpoint's output is dropped with a warning rather than silently
    evaluated as a smaller routing subset than was actually requested.
    """
    model_ids = list(per_model_results.keys())
    common_paths = set(per_model_results[model_ids[0]].keys())
    for m in model_ids[1:]:
        common_paths &= set(per_model_results[m].keys())

    all_paths = set()
    for m in model_ids:
        all_paths |= set(per_model_results[m].keys())
    dropped = all_paths - common_paths
    if dropped:
        print(f'[orchestrator] warning: {len(dropped)} image(s) missing from at least '
              f'one checkpoint\'s output - dropped from evaluation.')

    records = []
    for path in common_paths:
        scores = {}
        label = None
        for m in model_ids:
            score, lbl = per_model_results[m][path]
            scores[m] = score
            label = lbl  # same image, same label regardless of which checkpoint scored it
        records.append((scores, label))
    return records


def collect_records(config, split):
    """Run every configured checkpoint against its group's shared image
    set for the given split ('val' or 'test'), and return the aligned
    (scores_dict, label) records across all groups combined."""
    all_records = []
    for group_name, cfg in config.items():
        per_model = {}
        for ckpt in cfg['checkpoints']:
            model_id = f"{ckpt['architecture']}@{ckpt['preset']}"
            per_model[model_id] = run_checkpoint(
                architecture=ckpt['architecture'],
                preset=ckpt['preset'],
                name=cfg.get('name', 'pretrained'),
                data_root=cfg['data_root'],
                split_file=cfg['split_file'],
                data_keys=cfg['data_keys'],
                tf2k=cfg.get('tf2k', True),
                device=cfg.get('device', 'cuda:0'),
                ft=ckpt.get('ft', True),
                r50unfreezeL4=ckpt.get('r50unfreezeL4', False),
                split=split,
            )
        all_records.extend(align_scores(per_model))
    return all_records


if __name__ == '__main__':
    config = load_group_config(sys.argv[1])

    print('\n[orchestrator] === collecting TEST split scores (for evaluating every rule) ===')
    eval_records = collect_records(config, split='test')
    print(f'[orchestrator] {len(eval_records)} aligned test-split records')
    print('[orchestrator] test-split records by routing subset:')
    describe_subsets(eval_records)
    print()

    rules = {
        'average':  average_combine,
        # 'max':      max_combine,
        # 'median':   median_combine
    }

    results = evaluate_rules(eval_records, rules)
    print(json.dumps(results, indent=2))