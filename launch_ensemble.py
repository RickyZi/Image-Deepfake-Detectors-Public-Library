"""
launch_ensemble_clipd.py — ensemble evaluation launcher for CLIP-D.

Mirrors how the CLIP-D launcher.py calls test.py (shell=True, string args)
and reconstructs the exact output_dir path test.py builds so
load_test_outputs.py can find image_results.json.

Key CLIP-D specifics vs R50_nodown:
  - output path base is hardcoded to /second-disk/.../results/ in test.py
    → set meta.results_root to override it here
  - no --r50unfreezeL4 flag
  - --arch is NOT passed — test.py derives it from --name internally
  - checkpoint: checkpoint/{name}/ft_weights/{dataset_name}/best.pt
  - tag is 'ft' or 'pretrained' (no unfreezeL4 suffix)

Config format:

    {
      "meta": {
        "detector":      "CLIP-D",
        "weights_name":  "pretrained",
        "config_dir":    "configs",
        "results_root":  "/second-disk/Image-Deepfake-Detectors-Public-Library/results",
        "report_tag":    "blurbg_family"
      },
      "blurbg_strong_images": {
        "data_root":    "/path/to/blurbg_strong/",
        "split_file":   "/path/to/test2k_splits.json",
        "data_keys":    "realFFHQ:pre&...",
        "tf2k":         true,
        "device":       "cuda:0",
        "num_threads":  4,
        "checkpoints": [
          {"architecture": "CLIP-D", "preset": "adaptive_blurbg_strong", "ft": true},
          {"architecture": "CLIP-D", "preset": "blurbg_subtle",          "ft": true}
        ]
      }
    }

Usage:
    python3 launch_ensemble_clipd.py --ensemble-config ensemble/clipd_blurbg.json
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


def image_results_path(results_root, weights_name, preset, ft, data_keys):
    """Reconstruct the path where CLIP-D test.py writes image_results.json.

    Mirrors test.py's output_dir construction (non-social branch):
        f'{results_root}/{name}/{dataset_name}/CLIP-D_{tag}/{data_keys}/image_results.json'
    """
    dataset_name = (preset.replace(os.sep, '_')
                         .replace('-', '_')
                         .replace('bw01', 'bw_BW01')
                         .replace('portait', 'portrait'))
    tag = 'ft' if ft else 'pretrained'
    return os.path.join(
        results_root, weights_name,
        dataset_name, f'CLIP-D_{tag}',
        data_keys, 'image_results.json'
    )


def run_ensemble(ensemble_cfg, project_root):
    meta = ensemble_cfg.get('meta', {})

    detector     = meta.get('detector', 'CLIP-D')
    weights_name = meta.get('weights_name', 'pretrained')
    config_dir   = meta.get('config_dir',   'configs')
    report_tag   = meta.get('report_tag',   'ensemble')
    # CLIP-D test.py hardcodes /second-disk/ — override via meta.results_root
    results_root = meta.get('results_root', "/second-disk/Image-Deepfake-Detectors-Public-Library/results/ensemble/")

    yaml_path = os.path.join(config_dir, f'{detector}.yaml')
    if not os.path.exists(yaml_path):
        yaml_path = os.path.join(project_root, config_dir, f'{detector}.yaml')
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f'Detector config not found: {yaml_path}')

    det_config    = load_yaml(yaml_path)
    detector_args = det_config.get('detector_args', [])
    global_cfg    = det_config.get('global', {})
    split_file_fallback = os.path.abspath(global_cfg.get('split_file', 'test2k_splits.json'))

    det_dir = os.path.join(project_root, 'detectors', 'CLIP-D')
    ensemble_scripts_dir = os.path.join(project_root, 'test_ensemble')

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
        group_tf2k        = group.get('tf2k', True)
        group_ensemble    = True

    
        group_test_files = {}    # scores for THIS group only

        for ckpt in group['checkpoints']:
            arch   = ckpt['architecture']
            preset = ckpt['preset']
            ft     = ckpt.get('ft', True)

            if arch != 'CLIP-D':
                print(f'[ensemble] WARNING: this launcher only handles CLIP-D, '
                      f'skipping {arch}@{preset}')
                continue

            model_id = f'CLIP-D@{preset}'

            # Build the command as a string, matching CLIP-D launcher.py
            # exactly (shell=True, quoted strings, same arg order).
            # NOTE: --arch is intentionally NOT passed — test.py derives it
            # from --name internally to handle lora_r4_qv vs standard CLIP.
            cmd_args = [
                f'--name "{group_name_arg}"',
                f'--dataset {preset}',
                f'--data_root {group_data_root}',
                f'--split_file {group_split_file}',
                f'--data_keys "{group_data_keys}"',
                f'--tf2k {str(group_tf2k)}',
                f'--ensemble',
                f'--device {group_device}',
                f'--num_threads {group_num_threads}',
                f'--task test',
            ]
            if ft:
                cmd_args.append('--ft')
            # Pass detector_args from YAML (--lora_r, etc.) but NOT --arch
            # (CLIP-D test.py sets it itself based on --name)
            for arg in detector_args:
                # print(arg)
                if not str(arg).startswith('--arch') or 'opencliplinearloranext_clipL14commonpool_r4_qv' not in str(arg):
                    cmd_args.append(str(arg))
                # if str(arg).startswith('--arch') or :
                #     continue
                # else:
                #     cmd_args.append(str(arg))

            cmd_str = 'python -u test.py ' + ' '.join(cmd_args)

            print(f'[ensemble] running CLIP-D@{preset}')
            print(f'[ensemble] cmd: {cmd_str}')
            # breakpoint()
            subprocess.run(cmd_str, shell=True, cwd=det_dir, check=True)

            img_path = image_results_path(
                results_root, group_name_arg, preset, ft, group_data_keys
            )
            if not os.path.exists(img_path):
                raise FileNotFoundError(
                    f'Expected image_results.json at:\n  {img_path}\n'
                    f'Check meta.results_root matches where CLIP-D test.py '
                    f'actually writes results.\n'
                    f'CLIP-D test.py hardcodes /second-disk/ — set '
                    f'meta.results_root to override.'
                )

            group_test_files[model_id] = img_path
            print(f'[ensemble] scores found at {img_path}')

        # ── One report per group ───────────────────────────────────────────
        group_report_tag = f'{report_tag}__{group_name}'
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
            ]
            print(f'\n[ensemble] combining scores for {group_name} → {out_dir}')
            subprocess.run(load_cmd, check=True)
            print(f'[ensemble] report: ensemble_report_{detector}_{group_report_tag}.json')
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description='CLIP-D ensemble evaluation — all config from JSON'
    )
    parser.add_argument('--ensemble-config', type=str, required=True,
                        help='Path to ensemble config JSON (must contain a "meta" block)')
    args = parser.parse_args()

    if not os.path.exists(args.ensemble_config):
        parser.error(f'ensemble config not found: {args.ensemble_config}')

    with open(args.ensemble_config) as f:
        ensemble_cfg = json.load(f)

    project_root = os.path.abspath(os.path.dirname(__file__))
    run_ensemble(ensemble_cfg, project_root)


if __name__ == '__main__':
    main()

    # python3 launch_ensemble.py --ensemble-config test_ensemble/configs/clipd/blurbg_family.json