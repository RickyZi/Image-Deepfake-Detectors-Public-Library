"""
ensemble evaluation launcher for CLIP-D.

Mirrors how the CLIP-D launcher.py calls test.py (shell=True, string args)
to launch different tests with the same detector, then collects the resulting image_results.json files
in the ensamble_report_{detector}_{group_report_tag}.json file, which is written to the specified out_dir.

Config format:

    {
      "metadata": {
        "detector":      "CLIP-D",
        "weights_name":  "pretrained",
        "config_dir":    "configs",
        "results_root":  "/second-disk/Image-Deepfake-Detectors-Public-Library/results",
        "report_tag":    "blurbg_family"
      },
      "blurbg_strong_images": {
        "data_root":    "/path/to/blurbg_strong/",
        "split_file":   "/path/to/tf2k_splits.json",
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


ensemble config are defined as json files in ensemble/configs/clipd/*.json

output is a json file averaging the result of all the different detectrors 

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
    """
    Reconstruct the path where CLIP-D test.py writes image_results.json.
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
    experiment = ensemble_cfg.get('experiment', {})

    detector     = experiment.get('detector', 'CLIP-D')
    weights_name = experiment.get('weights_name', 'pretrained')
    config_dir   = experiment.get('config_dir',   'configs')
    report_tag   = experiment.get('report_tag',   'ensemble')
    
    results_root = experiment.get('results_root', "/second-disk/Image-Deepfake-Detectors-Public-Library/results/ensemble/")

    yaml_path = os.path.join(config_dir, f'{detector}.yaml')
    if not os.path.exists(yaml_path):
        yaml_path = os.path.join(project_root, config_dir, f'{detector}.yaml')
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f'Detector config not found: {yaml_path}')

    det_config    = load_yaml(yaml_path)
    detector_args = det_config.get('detector_args', [])
    global_cfg    = det_config.get('global', {})
    split_file_fallback = os.path.abspath(global_cfg.get('split_file', 'tf2k_splits.json'))

    det_dir = os.path.join(project_root, 'detectors', 'CLIP-D')
    ensemble_scripts_dir = os.path.join(project_root, 'ensemble')

    

    for group_name, group in ensemble_cfg.items():
        if group_name == 'experiment':
            continue

        print(f'\n[ensemble] === group: {group_name} ===')

        group_data_root   = group['data_root']
        group_split_file  = group.get('split_file', split_file_fallback)
        group_data_keys   = group['data_keys']
        group_device      = group.get('device', 'cuda:0')
        group_name_arg    = group.get('name', weights_name)
        group_num_threads = str(group.get('num_threads', 4))
        group_tf2k        = group.get('tf2k', True)
        # group_ensemble    = True
        socials = ["facebook", "telegram", "twitter"]

        

        group_social = ""
        for s in socials:
                if s in group_data_root:
                    group_social = s
                    break
    
        group_test_files = {}    
        model_name = None
        model_id = None
        
        for ckpt in group['checkpoints']:
            arch   = ckpt['architecture']
            preset = ckpt['preset']
            ft     = ckpt.get('ft', True)

            if arch != 'CLIP-D':
                print(f'[ensemble] WARNING: this launcher only handles CLIP-D, '
                      f'skipping {arch}@{preset}')
                continue
            
            if ft:
                model_id = f'CLIP-D_ft@{preset}'
                model_name = "CLIP-D_ft"
            else:
                model_id = f'CLIP-D@{preset}'

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

            if group_social:
                cmd_args.append(f'--social {group_social}')
            
            for arg in detector_args:
                # print(arg)
                if not str(arg).startswith('--arch') or 'opencliplinearloranext_clipL14commonpool_r4_qv' not in str(arg):
                    cmd_args.append(str(arg))
                

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

        # One report per group
        # group_report_tag = f'{report_tag}__{group_name}'
        group_report_tag = f'{model_name}_{group_name}'
        print(f"group_report_tag: {group_report_tag}")
        # breakpoint()
        if group_social!="":
            out_dir = os.path.join(project_root, 'results', 'model_ensemble', 'ensemble_results', group_social, report_tag) 
        else:
            out_dir = os.path.join(project_root, 'results', 'model_ensemble', 'ensemble_results', report_tag)
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
        description='CLIP-D ensemble evaluation — all config from JSON'
    )
    parser.add_argument('--ensemble-config', type=str, required=True,
                        help='Path to ensemble config JSON (must contain a "experiment" block)')
    args = parser.parse_args()

    if not os.path.exists(args.ensemble_config):
        parser.error(f'ensemble config not found: {args.ensemble_config}')

    with open(args.ensemble_config) as f:
        ensemble_cfg = json.load(f)

    project_root = os.path.abspath(os.path.dirname(__file__))
    run_ensemble(ensemble_cfg, project_root)


if __name__ == '__main__':
    main()