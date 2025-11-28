import os
import json
import sys
import yaml
import subprocess
from datetime import datetime

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

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_detect(args):
    """Run single image detection."""
    if not args.image:
        raise ValueError("--image is required for detect mode")
    
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
        
    # Load detector config
    config_path = os.path.join(args.config_dir, f'{args.detector}.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = load_config(config_path)
    global_config = config.get('global', {})
    
    # Set device
    device = global_config.get('device_override')
    if not device or device == "null":
        _, memory = get_gpus()
        if len(memory) > 0:
            device = f"cuda:{memory.argmax()}"
        else:
            device = "cpu"
            
    # Set up output path for results
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Extract image filename without extension
        image_name = os.path.splitext(os.path.basename(args.image))[0]
        output_dir = os.path.join('detection_results', args.detector, 'detect')
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f'{timestamp}_{image_name}.json')
    
    # Call detector's detect.py
    detector_path = os.path.join('detectors', args.detector)
    detect_script = os.path.join(detector_path, 'detect.py')
    
    if not os.path.exists(detect_script):
        raise FileNotFoundError(f"Detector {args.detector} does not support single image detection")
    
    cmd_args = [
        sys.executable,
        detect_script,
        f'--image "{args.image}"',
        f'--device {device}',
        f'--output "{args.output}"'
    ]
    
    # Add model path if specified
    if args.weights:
        cmd_args.append(f'--model "{args.weights}"')
    
    cmd = ' '.join(cmd_args)
    print(f"Running detection with {args.detector}...")
    
    if not args.dry_run:
        subprocess.run(cmd, shell=True)#, check=True)
        
        # Print results if available
        if os.path.exists(args.output):
            with open(args.output, 'r') as f:
                result = json.load(f)
            print("\nDetection Results:")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Time: {result['elapsed_time']:.3f}s")
            print(f"\nFull results saved to: {args.output}")