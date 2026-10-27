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


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Launcher for deepfake detector training and testing')
    parser.add_argument('--detector', type=str, required=False, choices=['R50_E2P', 'R50_ND', 'CLIP-D'], help='Detector to use')
    parser.add_argument('--phases', type=str, default='both', choices=['train', 'test', 'both'], help='Phases to run: train, test, or both (default: both)')
    parser.add_argument('--config-dir', type=str, default='configs', help='Path to configs directory (default: configs/)'),
    parser.add_argument('--weights-name', type=str, default='pretrained', help='Name of the weights directory')
    parser.add_argument('--arch', type=str, help = 'Detector architecture (used for apply LoRA on CLIP-D)')
    # --------------------------- #
    parser.add_argument('--ft', action = 'store_true', help = 'Path to pretrained model to load') ## for FT model (os just --ft flag?)
    parser.add_argument('--tf2k', type = bool, default = False, help = 'Use 2k dataset and splits for training and testing') ## for FT model (os just --ft flag?)
    parser.add_argument('--r50unfreezeL4', action='store_true', help='Unfreeze layer 4 when FT ResNet models')
    # --------------------------- #
    parser.add_argument('--demo', action='store_true', help='Run demo on demo_images across detectors')
    # --------------------------- #
    parser.add_argument('--dataset', type = str, default = 'dataset', help = 'Which dataset to use (default: dataset)') # add custom dataset for demo
    # Add detect mode arguments
    detect_group = parser.add_argument_group('detect', 'Single image detection options')
    detect_group.add_argument('--detect', action='store_true', help='Run single image detection mode')
    detect_group.add_argument('--image', type=str, help='Path to image file for detectionpython3 ')
    detect_group.add_argument('--weights', type=str, default='pretrained', help='Path to model weights for detection')
    detect_group.add_argument('--output', type=str, help='Path to save detection results')
    detect_group.add_argument('--dry-run', action='store_true', help='Print commands without executing')
    parser.add_argument('--resume', action='store_true', help='Resume training from the last saved per-epoch checkpoint in the run\'s checkpoint dir')
    parser.add_argument('--social', type = str, default= "", help = 'Which dataset from a social to load (default: niet)') # add custom dataset for demo
    
    args = parser.parse_args()
        
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
    
    # set architecture for lora
    if args.detector == 'CLIP-D' and args.weights_name == 'lora_r4_qv':
        args.arch = 'opencliplinearloranext_clipL14commonpool_r4_qv'
    
    print(f"model arch: {args.arch}")
    
    if  args.social:
            args.tf2k = True # used to remove mod from dataset preprocessing
            print(f"Dataset TF2K: setting --tf2k flag to True")
            print(f"loading social data - {args.social} - {args.dataset}")
            print(args.dataset)
            # /dataset-disk/tb_dataset/tf2k_lr/social
            dataset_path = os.path.join(global_config.get('dataset_path', ''), 'social', args.social, args.dataset)
            split_file = './Image-Deepfake-Detectors-Public-Library/splits/tf2k_SOCIAL_splits.json'

    elif 'seasons' in args.dataset or 'style' in args.dataset or 'adaptive' in args.dataset or 'subject' in args.dataset:
            args.tf2k = True # used to remove mod from dataset preprocessing
            print(f"Dataset TF2K: setting --tf2k flag to True")
            print(args.dataset)
            dataset_path = os.path.join(global_config.get('dataset_path', ''), 'tf_dataset', args.dataset)
            split_file = os.path.abspath(global_config.get('split_file', 'tf2k_splits.json'))

    elif 'Facebook' in args.dataset or 'Telegram' in args.dataset or 'Twitter' in args.dataset:
            args.tf2k = True # used to remove mod from dataset preprocessing
            print(f"Dataset TF2K: setting --tf2k flag to True")
            print(f"loading social data - {args.social} - {args.dataset}")
            print(args.dataset)
            dataset_path = os.path.join('/dataset-disk/tb_dataset/test_split_social', args.dataset)
            split_file = './Image-Deepfake-Detectors-Public-Library/splits/tf2k_SOCIAL_splits.json'
    else:
        dataset_path = './Image-Deepfake-Detectors-Public-Library/truefake_2k/dataset'
        args.tf2k = True
        split_file = os.path.abspath(global_config.get('split_file', 'tf2k_splits.json'))
            
    print(f"Using dataset path: {dataset_path}")
     
    
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

   
    num_threads = global_config.get('num_threads', 8) 
    dry_run = global_config.get('dry_run', False)
    only_list = global_config.get('only_list', False)
    phases = parse_phases(args.phases)
    
    detector_args = config.get('detector_args', [])
    training_configs = config.get('training', [])
    test_list = config.get('testing', [])

    print(f"training_configs: {training_configs}")
    print(f"test_list: {test_list}")
    
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
        cmd_args.append(f'--dataset {args.dataset}')
        
        if args.tf2k:
            cmd_args.append(f'--tf2k {args.tf2k}') # pass tf2k flag to train/test scripts for correct dataset handling

        if args.ft: 
                cmd_args.append(f'--ft')

        if args.r50unfreezeL4:
            cmd_args.append(f'--r50unfreezeL4')

        if args.mlp:
            cmd_args.append(f'--mlp')

        if args.resume:
            cmd_args.append(f'--resume')
        
        if args.arch:
            cmd_args.append(f'--arch {args.arch}')

        if args.social:
            cmd_args.append(f'--social {args.social}')

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
        
        print(f"cmd_args: {cmd_args}")
        # 

        # Add detector-specific arguments
        for arg in detector_args:
            cmd_args.append(arg)
        
        cmd_args_str = ' '.join(cmd_args)
        
        # Call train.py or test.py
        if not dry_run:
            log_file = f'logs/{task["type"]}_{task["details"]["detector"]}_{model_name}_{task["details"]["data"]}.log'
            if args.ft:
                log_file = f'logs/FT_{task["details"]["detector"]}_{model_name}_{task["details"]["data"]}.log'
            else:
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



