"""
launch_ensemble.py — ensemble evaluation launcher.

All configuration comes from a single JSON config file.
The only required CLI argument is --ensemble-config.

Config file format:

    {
      "meta": {
        "detector":      "R50_nodown",    // detector name → loads configs/<detector>.yaml
        "weights_name":  "pretrained",    // used in checkpoint paths and report output dir
        "config_dir":    "configs",       // path to YAML configs dir (default: configs)
        "results_root":  "/path/to/results",  // base dir for test.py output (optional,
                                          // defaults to results/ next to this script)
        "report_tag":    "blurbg_family"  // tag for the output report filename
      },
      "blurbg_strong_images": {
        "data_root":    "/path/to/blurbg_strong/",
        "split_file":   "/path/to/test2k_splits.json",
        "data_keys":    "realFFHQ:pre&...",
        "tf2k":         true,
        "device":       "cuda:0",
        "num_threads":  4,
        "batch_size":   32,
        "checkpoints":  [
          {"architecture": "R50_nodown", "preset": "adaptive_blurbg_strong",
           "ft": true, "r50unfreezeL4": true},
          {"architecture": "R50_nodown", "preset": "blurbg_subtle",
           "ft": true, "r50unfreezeL4": true}
        ]
      }
    }

Usage:
    python3 launch_ensemble.py --ensemble-config ensemble/r50nd_blurbg.json
"""
import os
import sys
import json
import subprocess
import tempfile
import shutil
import argparse
import yaml


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


# Map architecture names to their detector subdirectory.
DETECTOR_DIRS = {
    'R50_nodown': 'R50_nodown',
    'R50_TF':     'R50_TF',
    'CLIP-D':     'CLIP-D'
}

# test.py output_dir tag per architecture — mirrors each detector's test.py
# construction so launch_ensemble.py can find the image_results.json it wrote.
DETECTOR_RESULT_TAG = {
    'R50_nodown': 'R50_nodown',
    'R50_TF':     'R50_TF',
    'CLIP-D':     'CLIP-D'
}


def image_results_path(results_root, weights_name, preset, arch, ft, unfreeze, data_keys):
    """Reconstruct the path where test.py wrote image_results.json.
    Mirrors the output_dir logic in each detector's test.py."""
    dataset_tag  = preset.replace(os.sep, '_').replace('-', '_')
    tag  = 'ft' if ft else 'pretrained'
    tag += '_unfreezeL4' if unfreeze else ''
    det_tag = DETECTOR_RESULT_TAG.get(arch, arch)
    return os.path.join(
        results_root, weights_name, dataset_tag,
        f'{det_tag}_{tag}', data_keys,
        'image_results.json'
    )


def run_ensemble(ensemble_cfg, project_root):
    meta = ensemble_cfg.get('meta', {})

    detector     = meta.get('detector')
    weights_name = meta.get('weights_name', 'pretrained')
    config_dir   = meta.get('config_dir',   'configs')
    report_tag   = meta.get('report_tag',   'ensemble')
    results_root = meta.get('results_root', "/second-disk/Image-Deepfake-Detectors-Public-Library/results/ensemble/")

    if not detector:
        raise ValueError("ensemble config must have a 'meta.detector' field")

    # Load detector YAML for detector_args (--arch, --cropSize, --blur_sig, etc.)
    yaml_path = os.path.join(config_dir, f'{detector}.yaml')
    if not os.path.exists(yaml_path):
        # Try relative to project root
        yaml_path = os.path.join(project_root, config_dir, f'{detector}.yaml')
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f'Detector config not found: {yaml_path}')

    det_config = load_yaml(yaml_path)
    detector_args = det_config.get('detector_args', [])
    global_cfg = det_config.get('global', {})
    split_file_fallback = os.path.abspath(global_cfg.get('split_file', 'test2k_splits.json'))

    detectors_root       = os.path.join(project_root, 'detectors')
    ensemble_scripts_dir = os.path.join(project_root, 'test_ensemble')

    # Iterate over groups (skip the reserved 'meta' key).
    # Each group produces its OWN report — one per source preset image set.
    # (Previously all_test_files was accumulated across groups and one merged
    # report was written, which caused the second group's scores to silently
    # overwrite the first's when both groups share the same checkpoint names.)
    for group_name, group in ensemble_cfg.items():
        if group_name == 'meta':
            continue

        print(f'\n[ensemble] === group: {group_name} ===')

        group_data_root   = group['data_root']
        group_split_file  = group.get('split_file', split_file_fallback)
        group_data_keys   = group['data_keys']
        group_device      = group.get('device', 'cuda:0')
        group_name_arg    = group.get('name', weights_name)
        group_num_threads = str(group.get('num_threads', 4))
        # group_batch_size  = str(group.get('batch_size', 32))
        group_tf2k        = str(group.get('tf2k', True))
        # group_ensemble    = True
        # group_social      = group['social']

        socials = ["facebook", "telegram", "twitter"]
        group_social = ""
        for s in socials:
            if s in group_data_root:
                group_social = s
                break

        # if s:
        #     group_data_root = group_data_root.replace()

        print(f"group_social: {group_social}")

        group_test_files = {}    # scores for THIS group only

        for ckpt in group['checkpoints']:
            arch     = ckpt['architecture']
            preset   = ckpt['preset']
            ft       = ckpt.get('ft', True)
            unfreeze = ckpt.get('r50unfreezeL4', False)
            # social = group_social

            if arch not in DETECTOR_DIRS:
                print(f'[ensemble] WARNING: unknown architecture {arch} — skipping')
                continue

            det_dir  = os.path.join(detectors_root, DETECTOR_DIRS[arch])
            model_id = f'{arch}@{preset}'

            cmd_parts = [
                sys.executable, 'test.py',
                '--name',        group_name_arg,
                '--dataset',     preset,
                '--data_root',   group_data_root,
                '--split_file',  group_split_file,
                '--data_keys',   group_data_keys,
                '--tf2k',        group_tf2k,
                '--device',      group_device,
                '--num_threads', group_num_threads,
                '--task',        'test',
                '--ensemble', 
            ]
            if ft:
                cmd_parts.append('--ft')
            if unfreeze:
                cmd_parts.append('--r50unfreezeL4')
            cmd_parts.extend(detector_args)

            print(f'[ensemble] running test.py for {model_id}')
            subprocess.run(cmd_parts, cwd=det_dir, check=True)

            img_path = image_results_path(
                results_root, group_name_arg, preset,
                arch, ft, unfreeze, group_data_keys
            )
            if not os.path.exists(img_path):
                raise FileNotFoundError(
                    f'Expected image_results.json at:\n  {img_path}\n'
                    f'Check that results_root in meta matches where test.py '
                    f'actually writes results, or set meta.results_root explicitly.'
                )

            group_test_files[model_id] = img_path
            print(f'[ensemble] scores found at {img_path}')

        # ── One report per group ───────────────────────────────────────────
        # report_tag + group_name gives a unique filename per source preset:
        # ensemble_report_R50_nodown_blurbg_family__blurbg_strong_images.json
        group_report_tag = f'{report_tag}_{group_name}'
        print(f"group_report_tag: {group_report_tag}")
        # breakpoint()
        if group_social!="":
            out_dir = os.path.join(project_root, 'results', 'ensemble', 'ensemble_results', group_social, report_tag) 
        else:
            out_dir = os.path.join(project_root, 'results', 'ensemble', 'ensemble_results', report_tag)
        os.makedirs(out_dir, exist_ok=True)

        tmpdir = tempfile.mkdtemp(prefix='ensemble_maps_')
        try:
            test_map_path = os.path.join(tmpdir, 'test_files.json')
            with open(test_map_path, 'w') as f:
                json.dump(group_test_files, f, indent=2)

            loader_script = os.path.join(ensemble_scripts_dir, 'load_test_outputs.py')
            load_cmd = [
                sys.executable, loader_script,
                test_map_path,
                '--detector-name', detector,
                '--dataset-name',  group_report_tag,
                '--out-dir',       out_dir,
                '--data-root', group_data_root
            ]
            print(f'\n[ensemble] combining scores for {group_name} → {out_dir}')
            subprocess.run(load_cmd, check=True)
            print(f'[ensemble] report: ensemble_report_{detector}_{group_report_tag}.json')
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description='Ensemble evaluation — all config from JSON, '
                    'calls test.py per checkpoint then load_test_outputs.py'
    )
    parser.add_argument('--ensemble-config', type=str, required=True,
                        help='Path to ensemble config JSON '
                             '(must contain a "meta" block, see docstring)')
    args = parser.parse_args()

    if not os.path.exists(args.ensemble_config):
        parser.error(f'ensemble config not found: {args.ensemble_config}')

    with open(args.ensemble_config) as f:
        ensemble_cfg = json.load(f)

    project_root = os.path.abspath(os.path.dirname(__file__))
    run_ensemble(ensemble_cfg, project_root)


if __name__ == '__main__':
    main()

    # /second-disk/Image-Deepfake-Detectors-Public-Library/results/ensemble/pretrained/adaptive_blurbg_subtle/R50_nodown_ft_unfreezeL4/realFFHQ:pre&realFORLAB:pre&gan1:pre&gan2:pre&gan3:pre&sd15:pre&sd2:pre&sd3:pre&sdXL:pre&flux:pre/image_results.json