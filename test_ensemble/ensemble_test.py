"""
run_ensemble_test.py — orchestrates per-checkpoint inference subprocesses
for one or more groups of known-ambiguous images, aligns their per-image
scores, and evaluates every combination rule from combination.py side by
side.

Config shape (see ambiguous_groups.example.json): each "group" is ONE
shared set of images (data_root/split_file/data_keys) - the images whose
C2PA manifest is ambiguous between several candidate presets - plus a list
of (preset, architecture) CHECKPOINTS to score those same images against.
This supports both:
  - cross-family combination: different architectures, same preset
  - same-family combination: the same architecture fine-tuned on more than
    one candidate preset (e.g. two R50_nodown checkpoints)
  - or any mix of both, since every checkpoint is just scored against the
    same shared image set and combined by model-instance identifier
    ("{architecture}@{preset}"), not by architecture name alone.

Each image's scores dict is keyed by "{architecture}@{preset}", e.g.
{"R50_nodown@presetA": 0.83, "R50_nodown@presetB": 0.61, "CLIP-D@presetA": 0.71}
- this is what actually fixes the same-family collision: two checkpoints
of the same architecture (R50_nodown under presetA vs presetB) get
distinct keys instead of overwriting each other. combination.py's
functions don't care what the keys mean, so nothing there needed to
change - only the orchestrator's keying did.

Uses your existing train/val/test split: the stacked-classifier rule is
fit on the VAL split (never train, since train images already shaped the
underlying detectors' own weights), and every rule is evaluated on the
TEST split, with the same metric set as test.py (TPR/TNR/Acc/Balanced
Acc/F1/AUC) for direct comparability.

Why subprocesses: see infer_r50nodown.py / infer_r50tf.py docstrings -
R50_nodown, R50_TF, and (eventually) CLIP-D define same-named local
modules (networks.py, parser.py, utils/dataset.py), so importing more
than one into this process would silently collide via sys.modules
caching. Each checkpoint's inference runs as its own subprocess instead.

STILL MISSING (flagging explicitly rather than guessing):
  - infer_clipd.py does not exist yet - need CLIP-D's current test.py.
  - Whether --r50unfreezeL4 applies is set per checkpoint entry in the
    config - confirm this matches what's actually deployed.
  - compute_auc_weights() below is pooled across everything queried for a
    group. For per-(preset, architecture) weights instead of pooled, call
    it separately per checkpoint's own solo scores - happy to wire that in
    once you confirm you want that granularity.

Usage:
    python run_ensemble_test.py ambiguous_groups.json
"""
import os
import sys
import json
import subprocess
import tempfile

from combination import (average_combine, max_combine, median_combine, weighted_combine,
                          compute_auc_weights, describe_subsets,
                          SubsetStackingEnsemble, evaluate_rules)

# --------------------------------------------------------------------------
# Fill these in for your environment.
# --------------------------------------------------------------------------
DETECTOR_DIRS = {
    'R50_nodown': '/home/rz/TB_WP3/Image-Deepfake-Detectors-Public-Library/detectors/R50_nodown/',
    'R50_TF':     '/home/rz/TB_WP3/Image-Deepfake-Detectors-Public-Library/detectors/R50_TF/',
    # 'CLIP-D':   '/path/to/detectors/CLIP-D',   # TODO: once infer_clipd.py exists
}
DETECTOR_SCRIPTS = {
    'R50_nodown': 'infer_r50nodown.py',
    'R50_TF':     'infer_r50tf.py',
    # 'CLIP-D':   'infer_clipd.py',
}


def load_group_config(config_path):
    """Stand-in for the real C2PA routing table. Returns:
    {group_name: {'data_root':..., 'split_file':..., 'data_keys':...,
                   'name':..., 'tf2k':..., 'device':...,
                   'checkpoints': [{'preset':..., 'architecture':...,
                                     'ft':..., 'r50unfreezeL4':...}, ...]}}
    """
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

    print('[orchestrator] === collecting VAL split scores (for fitting the stacked rule '
          'and computing data-driven weights) ===')
    fit_records = collect_records(config, split='val')
    print(f'[orchestrator] {len(fit_records)} aligned val-split records')
    print('[orchestrator] val-split records by routing subset:')
    describe_subsets(fit_records)

    print('\n[orchestrator] === collecting TEST split scores (for evaluating every rule) ===')
    eval_records = collect_records(config, split='test')
    print(f'[orchestrator] {len(eval_records)} aligned test-split records')
    print('[orchestrator] test-split records by routing subset:')
    describe_subsets(eval_records)
    print()

    # Weights computed from each checkpoint's own val-split AUC, pooled
    # across all configured groups. See the module docstring for how to
    # upgrade to per-(preset, architecture) weights instead.
    computed_weights = compute_auc_weights(fit_records)
    print(f'[orchestrator] weights computed from val-split AUC: {computed_weights}\n')

    stacker = SubsetStackingEnsemble(min_samples_per_subset=30).fit(fit_records)

    rules = {
        'average':  average_combine,
        'max':      max_combine,
        'median':   median_combine,
        'weighted': lambda s: weighted_combine(s, computed_weights),
        'stacked':  stacker.predict,
    }
    results = evaluate_rules(eval_records, rules)
    print(json.dumps(results, indent=2))