
import os
import subprocess
import time
import argparse
import yaml
import glob
import shutil


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# smi vampire function, busy waiting for a free-enough GPU, use min_vram to set the threshold
def get_gpus():
    from numpy import argwhere, asarray, diff
    import re
    smi = os.popen('nvidia-smi').readlines()
    div = re.compile('[+]-{3,}[+]|[|]={3,}[|]')
    dividers = argwhere([div.match(line) != None for line in smi])[-2:, 0]
    processes = [line for line in smi[dividers[0]+1:dividers[1]] if ' C ' in line]
    free = list(set([process.split()[1] for process in processes]) ^ set([str(0), str(1)]))

    udiv = re.compile('[|]={3,}[+]={3,}[+]={3,}[|]')
    ldiv = re.compile('[+]-{3,}[+]-{3,}[+]-{3,}[+]')
    divider_up = argwhere([udiv.match(line) != None for line in smi])[0,0]
    divider_down = argwhere([ldiv.match(line) != None for line in smi])[-1, 0]

    gpus = [line for line in smi[divider_up+1:divider_down] if '%' in line and 'MiB' in line]
    gpus = [gpu.split('|')[2].replace(' ', '').replace('MiB', '').split('/') for gpu in gpus]
    memory = diff(asarray(gpus).astype(int), axis=1).squeeze()

    return free, memory

def autotest(train_list, data_list, detector_name, checkpoint_name):
    """Generate task list from training and testing configurations."""
    assert type(data_list) == list
    task_list = []

    for train_config in train_list:
        
        train_dict = {'detector': detector_name, 'model': None, 'data': train_config['data']}
        task_list.append({'type':'train', 'details':train_dict})
        
        for data in data_list:
            
            name = checkpoint_name #train_dict['data']
            task_list.append({'type':'test', 'details':{'detector': detector_name, 'model': name, 'data': data}})
    
    return task_list


def parse_phases(phases_str):
    """Parse phases string into list."""
    if phases_str.lower() == 'both':
        return ['train', 'test']
    elif phases_str.lower() == 'train':
        return ['train']
    elif phases_str.lower() == 'test':
        return ['test']
    else:
        raise ValueError(f"Invalid phases: {phases_str}. Must be 'train', 'test', or 'both'")


def run_demo(args):
    import json
    import torch

    project_root = os.path.abspath(os.path.dirname(__file__))
    demo_root = os.path.join(project_root, 'demo_images')
    assert os.path.isdir(demo_root), f"Demo folder not found: {demo_root}"

    # Build split file from demo_images
    def build_demo_split_json(root_path, out_path):
        test_entries = []
        for mod in ['PreSocial', 'Facebook', 'Telegram', 'X']:
            mod_path = os.path.join(root_path, mod)
            if not os.path.isdir(mod_path):
                continue
            for dirpath, dirnames, filenames in os.walk(mod_path, topdown=True, followlinks=True):
                if len(dirnames):
                    continue
                rel_dir = f"{dirpath}/".replace(mod_path + os.sep, '')
                parts = rel_dir.split(os.sep)[:3]
                if len(parts) < 3:
                    continue
                label, gen, sub = parts
                for fname in sorted(filenames):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext not in ['.png', '.jpg', '.jpeg']:
                        continue
                    stem = os.path.splitext(fname)[0]
                    test_entries.append(os.path.join(gen, sub, stem))

        with open(out_path, 'w') as f:
            json.dump({'test': sorted(list(set(test_entries)))}, f)

    split_demo_file = os.path.join(project_root, 'split_demo.json')
    build_demo_split_json(demo_root, split_demo_file)

    def prepare_best_checkpoint(detector_dir, preferred_path=None):
        weights_dir = os.path.join(detector_dir, 'checkpoint', 'pretrained', 'weights')
        src_weight = None

        if preferred_path:
            src_weight = preferred_path if os.path.isabs(preferred_path) else os.path.normpath(os.path.join(detector_dir, preferred_path))
            if not os.path.isfile(src_weight):
                print(f"[demo] Preferred weights not found at {src_weight}, falling back to search")
                src_weight = None

        if src_weight is None:
            if not os.path.isdir(weights_dir):
                return None
            candidates = []
            for ext in ('*.pt', '*.pth'):
                candidates.extend(glob.glob(os.path.join(weights_dir, ext)))
            if not candidates:
                return None
            src_weight = sorted(candidates)[0]

        run_dir = os.path.join(detector_dir, 'checkpoint', 'demo', 'weights')
        os.makedirs(run_dir, exist_ok=True)
        dst_weight = os.path.join(run_dir, 'best.pt')
        shutil.copy2(src_weight, dst_weight)
        return dst_weight

    
    device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    name = 'demo'

    detectors_root = os.path.join(project_root, 'detectors')
    all_methods = ['R50_nodown', 'CLIP-D', 'R50_TF', 'P2G', 'NPR']
    methods = all_methods if args.demo_detector == 'all' else [args.demo_detector]

    os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)

    for method in methods:
        det_dir = os.path.join(detectors_root, method)
        if not os.path.isdir(det_dir):
            continue

        preferred_weights = args.weights_name or './checkpoint/pretrained/weights/best.pt'
        best_path = prepare_best_checkpoint(det_dir, preferred_weights)
        if best_path is None:
            print(f"[demo] Skipping {method}: no pretrained weights found under checkpoint/pretrained/weights/")
            continue

        config_path = os.path.join(args.config_dir, f'{method}.yaml')
        config = load_config(config_path) if os.path.exists(config_path) else {}
        detector_args = config.get('detector_args', [])
        testing_keys = config.get('testing', []) or ['all:all']
        global_cfg = config.get('global', {})
        num_threads = global_cfg.get('num_threads', 8)

        for data_keys in testing_keys:
            args_list = [
                f'--name "{name}"',
                f'--task test',
                f'--device {device}',
                f'--split_file {split_demo_file}',
                f'--data_root {demo_root}',
                f'--data_keys "{data_keys}"',
                f'--num_threads {num_threads}',
            ] + detector_args

            cmd_args = ' '.join(args_list)
            log_file = os.path.join(project_root, 'logs', f'demo_{method}_{data_keys.replace(":","-")}.log')
            with open(log_file, 'w') as f:
                cwd = os.getcwd()
                os.chdir(det_dir)
                try:
                    print(f"[demo] Running {method} test with args: {cmd_args}")
                    runner = 'test.py'
                    subprocess.run(f'python -u {runner} {cmd_args}', shell=True)#, stdout=f, stderr=f)
                finally:
                    os.chdir(cwd)            
        shutil.rmtree(os.path.join(det_dir, 'checkpoint', 'demo'))

    print('[demo] Completed. Results saved under detectors/<method>/results/demo/<scenario>/results.csv')

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Launcher for deepfake detector training and testing')
    parser.add_argument('--detector', type=str, required=False,
                        choices=['R50_TF', 'R50_nodown', 'CLIP-D', 'P2G', 'NPR'],
                        help='Detector to use')
    parser.add_argument('--phases', type=str, default='both',
                        choices=['train', 'test', 'both'],
                        help='Phases to run: train, test, or both (default: both)')
    parser.add_argument('--config-dir', type=str, default='configs',
                        help='Path to configs directory (default: configs/)'),
    parser.add_argument('--weights_name', type=str, default=None, 
                        help='Name of the weights directory')
    parser.add_argument('--demo', action='store_true', help='Run demo on demo_images across detectors')
    parser.add_argument('--demo-detector', type=str, default='all', choices=['all', 'R50_TF', 'R50_nodown', 'CLIP-D', 'P2G', 'NPR'], help='Which detector to demo (default: all)')
    
    # Add detect mode arguments
    detect_group = parser.add_argument_group('detect', 'Single image detection options')
    detect_group.add_argument('--detect', action='store_true', help='Run single image detection mode')
    detect_group.add_argument('--image', type=str, help='Path to image file for detection')
    detect_group.add_argument('--weights', type=str, default='pretrained', help='Path to model weights for detection')
    detect_group.add_argument('--output', type=str, help='Path to save detection results')
    detect_group.add_argument('--dry-run', action='store_true', help='Print commands without executing')
    
    args = parser.parse_args()

    if args.demo:
        return run_demo(args)
        
    if args.detect:
        if args.detector is None:
            parser.error('--detector is required for detect mode')
        if args.image is None:
            parser.error('--image is required for detect mode')
        from support.detect import run_detect
        return run_detect(args)

    if args.detector is None:
        parser.error('--detector is required unless --demo is specified')

    # Load configuration from YAML
    config_path = os.path.join(args.config_dir, f'{args.detector}.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = load_config(config_path)
    
    # Extract configuration values
    global_config = config.get('global', {})
    dataset_path = global_config.get('dataset_path')
    device_override = global_config.get('device_override')  # Can be None
    if args.weights_name is not None:
        global_config['name'] = args.weights_name
    else:
        global_config['name'] = config.get('training', [])[0]['data']
    model_name = global_config.get('name')
    # Handle string "null" as None
    if device_override == "null" or device_override == "":
        device_override = None
    min_vram = global_config.get('min_vram', 16000)
    split_file = os.path.abspath(global_config.get('split_file', 'split.json'))
    num_threads = global_config.get('num_threads', 8)
    dry_run = global_config.get('dry_run', False)
    only_list = global_config.get('only_list', False)
    phases = parse_phases(args.phases)
    
    detector_args = config.get('detector_args', [])
    training_configs = config.get('training', [])
    test_list = config.get('testing', [])
    
    os.makedirs('logs', exist_ok=True)
    
    # Generate tasks
    tasks = []
    if training_configs:
        tasks.extend(autotest(training_configs, test_list, args.detector, model_name))
    
    print('Number of tasks:', len(tasks))
    for task in tasks:
        print(task)
    
    if only_list:
        return
    
    # From here the launcher will create all the arguments to use when calling the train script
    for task in tasks:
        if task['type'] not in phases:
            continue
        
        cmd_args = []
        
        if task['type'] == 'train':
            cmd_args.append(f'--name "{model_name}"')#{task["details"]["model"]}"')
        else:
            cmd_args.append(f'--name "{task["details"]["model"]}"')
        
        cmd_args.append(f'--split_file {split_file}')
        cmd_args.append(f'--task {task["type"]}')
        cmd_args.append(f'--num_threads {num_threads}')
        cmd_args.append(f'--data_keys "{task["details"]["data"]}"')
        cmd_args.append(f'--data_root {dataset_path}')
        
        device = None
        if device_override is not None:
            device = device_override
        else:
            if not dry_run:
                print('Waiting for GPU')
                while device is None:
                    free, memory = get_gpus()
                    if len(free):
                        device = "cuda:" + free[0]
                    elif max(memory) > min_vram:
                        device = "cuda:" + str([i for i, mem in enumerate(memory) if mem == max(memory)][0])
                    time.sleep(1)
                print('GPU found')
        
        cmd_args.append(f'--device {device}')
        
        # Add detector-specific arguments
        for arg in detector_args:
            cmd_args.append(arg)
        
        cmd_args_str = ' '.join(cmd_args)
        
        # Call train.py or test.py
        if not dry_run:
            #log_file = f'logs/{task["type"]}_{task["details"]["detector"]}_{task["details"]["model"]}_{task["details"]["data"]}.log'
            log_file = f'logs/{task["type"]}_{task["details"]["detector"]}_{model_name}_{task["details"]["data"]}.log'
            with open(log_file, 'w') as f:
                cwd = os.getcwd()
                os.chdir(f'./detectors/{task["details"]["detector"]}')
                
                start_time = time.time()
                
                runner = f'{task["type"]}.py'
                print(f'Call to {runner} with: {cmd_args_str}')
                
                subprocess.run(f'python -u {runner} {cmd_args_str}', shell=True)#, stdout=f, stderr=f)
                
                end_time = time.time()
                print(f'Execution time: {end_time-start_time:.2f} seconds')
                
                print('#'*80)
                print('#'*80)
                
                os.chdir(cwd)


if __name__ == '__main__':
    main()



