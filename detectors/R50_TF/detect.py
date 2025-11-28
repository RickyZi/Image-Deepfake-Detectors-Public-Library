# ----------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------
import os
import argparse
import json
import time
import yaml
import torch
from PIL import Image
import torchvision.transforms.v2 as Tv2

from networks import ImageClassifier
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from support.detect_utils import format_result, save_result, get_device


# ----------------------------------------------------------------------------
# IMAGE PREPROCESSING
# ----------------------------------------------------------------------------
def preprocess_image(image_path):
    """
    Load and preprocess a single image for model input.
    Uses the same normalization as test.py (ImageNet stats).
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Apply transforms (same as test split without augmentation)
    transform = Tv2.Compose([
        Tv2.ToImage(),
        Tv2.ToDtype(torch.float32, scale=True),
        Tv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Apply transforms and add batch dimension
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    
    return tensor


# ----------------------------------------------------------------------------
# CONFIG LOADING AND PARSING
# ----------------------------------------------------------------------------
def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_detector_args(detector_args, default_num_centers=1):
    """
    Parse detector_args list (e.g., ["--arch", "nodown", "--prototype", "--freeze"])
    into a settings object.
    """
    class Settings:
        def __init__(self):
            self.arch = "nodown"
            self.freeze = False
            self.prototype = False
            self.num_centers = default_num_centers
    
    settings = Settings()
    
    i = 0
    while i < len(detector_args):
        arg = detector_args[i]
        
        if arg == "--arch":
            if i + 1 < len(detector_args):
                settings.arch = detector_args[i + 1]
                i += 2
            else:
                i += 1
        elif arg == "--freeze":
            settings.freeze = True
            i += 1
        elif arg == "--prototype":
            settings.prototype = True
            i += 1
        elif arg == "--num_centers":
            if i + 1 < len(detector_args):
                settings.num_centers = int(detector_args[i + 1])
                i += 2
            else:
                i += 1
        else:
            i += 1
    
    return settings


def resolve_config_path(config_path):
    """
    Resolve config path. If relative, resolve relative to project root
    (two levels up from detect.py location).
    """
    if os.path.isabs(config_path):
        return config_path
    
    # Get directory of detect.py (detectors/R50_TF/)
    detect_dir = os.path.dirname(os.path.abspath(__file__))
    # Go two levels up to project root
    project_root = os.path.dirname(os.path.dirname(detect_dir))
    # Join with config path
    return os.path.join(project_root, config_path)


# ----------------------------------------------------------------------------
# INFERENCE
# ----------------------------------------------------------------------------
def run_inference(model, image_path, device):
    """
    Run inference on a single image.
    Returns: (probability, label, runtime_ms)
    """
    start_time = time.time()
    
    # Preprocess image
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        raw_score_tensor = model(image_tensor).squeeze(1)  # shape [1]
    
    # Convert to probability using sigmoid
    probability = torch.sigmoid(raw_score_tensor).item()
    
    # Determine label (fake if probability > 0.5, else real)
    label = "fake" if probability > 0.5 else "real"
    
    # Calculate runtime in milliseconds
    runtime_ms = int((time.time() - start_time) * 1000)
    
    return probability, label, runtime_ms


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Single image inference for R50_TF detector')
    parser.add_argument('--input', type=str, required=False, help='Path to input image (alias: --image)')
    parser.add_argument('--image', type=str, required=False, help='Path to input image (alias for --input)')
    parser.add_argument('--output', type=str, default='/tmp/result.json', help='Path to output JSON file')
    parser.add_argument('--checkpoint', type=str, required=False, help='Path to model checkpoint file')
    parser.add_argument('--model', type=str, required=False, help='Model name or checkpoint directory (alias for --checkpoint)')
    parser.add_argument('--config', type=str, default='configs/R50_TF.yaml', help='Path to YAML config file')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (cuda:0, cpu, etc.)')
    
    args = parser.parse_args()

    # Normalize image argument: prefer --image over --input if provided
    if args.image:
        args.input = args.image

    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif getattr(args, 'model', None):
        detect_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(detect_dir, 'checkpoint', args.model, 'weights', 'best.pt')
        if os.path.exists(candidate):
            checkpoint_path = candidate
        else:
            # If model refers directly to a file path, accept it
            if os.path.isabs(args.model) and os.path.exists(args.model):
                checkpoint_path = args.model
            else:
                # Try resolving relative to project root
                project_root = os.path.dirname(os.path.dirname(detect_dir))
                candidate2 = os.path.join(project_root, args.model)
                if os.path.exists(candidate2):
                    checkpoint_path = candidate2

    # If still not found, keep existing behavior (will raise later)
    if checkpoint_path:
        args.checkpoint = checkpoint_path
    
    # Resolve config path
    config_path = resolve_config_path(args.config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load config
    config = load_config(config_path)
    
    # Parse detector_args from config
    detector_args = config.get('detector_args', [])
    settings = parse_detector_args(detector_args)
    
    # Get device from config if available, else use argument
    device_str = args.device
    if config.get('global', {}).get('device_override'):
        device_override = config['global']['device_override']
        if device_override and device_override != "null" and device_override != "":
            device_str = device_override
    
    # Determine device
    if device_str.startswith('cuda') and not torch.cuda.is_available():
        print(f"Warning: CUDA requested but not available. Using CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = ImageClassifier(settings)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    
    # Run inference
    print(f"Running inference on {args.input}")
    probability, label, runtime_ms = run_inference(model, args.input, device)
    
    # Format result to match other detectors (prediction/confidence/elapsed_time)
    elapsed_time = runtime_ms / 1000.0
    formatted = format_result(label, float(round(probability, 4)), elapsed_time)

    # Save using shared utility
    if args.output:
        save_result(formatted, args.output)
        print(f"Results saved to {args.output}")

    # Print concise output for user
    print(f"Prediction: {formatted['prediction']}")
    print(f"Confidence: {formatted['confidence']:.4f}")
    print(f"Time: {formatted['elapsed_time']:.3f}s")


if __name__ == '__main__':
    main()

