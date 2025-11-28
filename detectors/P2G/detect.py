import os
import sys
import time
import torch
import argparse
import json
import pickle
from PIL import Image

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
# Also add this detector's src folder so `models` package inside P2G can be imported
this_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(this_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from support.detect_utils import format_result, save_result, get_device, load_image

# Import P2G model (now that src is on sys.path)
from models.slinet_det import SliNet

def parse_args():
    parser = argparse.ArgumentParser(description='P2G single image detector')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default=None, help='Path to model checkpoint (can be relative to detectors/P2G)')
    parser.add_argument('--output', type=str, help='Path to save detection result JSON')
    parser.add_argument('--device', type=str, help='Device to run on (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--config', type=str, default='configs/test.json', help='Path to config file (relative to detectors/P2G)')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from JSON file. Accepts absolute or relative path (relative to detectors/P2G)."""
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    args = parse_args()

    # Setup device
    if args.device:
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            print("CUDA is not available, falling back to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device(args.device)
    else:
        device = get_device()

    # Load config (JSON used by P2G)
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config '{args.config}': {e}")
        return

    # Resolve model checkpoint path (allow passing relative path or None)
    if args.model is None:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoint', 'pretrained', 'weights', 'best.pt')
    else:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoint', str(args.model), 'weights', 'best.pt')


    if not os.path.exists(model_path):
        print(f"Model checkpoint not found: {model_path}")
        return
    # Load checkpoint early so we can populate config/args expected by SliNet
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"Error loading checkpoint '{model_path}': {e}")
        return

    # populate required fields for SliNet from checkpoint (as prepare_model does in eval.py)
    # Use checkpoint values when available, otherwise fall back to config or sensible defaults
    try:
        config['K'] = checkpoint.get('K', config.get('K', 5))
        config['topk_classes'] = checkpoint.get('topk_classes', config.get('topk_classes', 1))
        # eval.py used 'ensembling_flags' in checkpoint and assigned to args['ensembling']
        config['ensembling'] = checkpoint.get('ensembling_flags', config.get('ensembling', [False, False, False, False]))
        # number of tasks
        if 'tasks' in checkpoint:
            config['num_tasks'] = checkpoint['tasks'] + 1
            config['task_name'] = range(config['num_tasks'])
    except Exception:
        # If any key is missing we continue with defaults — SliNet will raise more specific errors if needed
        pass

    # ensure device in config is a torch.device
    config['device'] = device

    # Instantiate model and load state dict
    try:
        model = SliNet(config)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model = model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load and preprocess image
    try:
        image_tensor, _ = load_image(args.image, size=224)
        image_tensor = image_tensor.to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Load object labels from classes.pkl
    try:
        pkl_path = os.path.join(src_dir, 'utils', 'classes.pkl')
        with open(pkl_path, 'rb') as f:
            object_labels_dict = pickle.load(f)
        # Normalize to a relative path key like those stored in classes.pkl
        rel_path = os.path.relpath(args.image, config.get('data_path', project_root)).replace(os.sep, '/')
        candidates = [rel_path, rel_path.lstrip('/'), '/' + rel_path]
        found_key = None
        for k in candidates:
            if k in object_labels_dict:
                found_key = k
                break
        if found_key is None:
            basename = os.path.basename(rel_path)
            for k in object_labels_dict.keys():
                if k.endswith('/' + basename) or k.endswith(basename):
                    found_key = k
                    break
        def ensure_topk_tuples(label_list, topk=5):
            # Convert to tuples if needed
            tuples = [(lbl, 1.0) if isinstance(lbl, str) else lbl for lbl in label_list]
            # Remove duplicates, keep order
            seen = set()
            unique = []
            for t in tuples:
                if t[0] not in seen:
                    unique.append(t)
                    seen.add(t[0])
            # Pad or truncate to topk
            while len(unique) < topk:
                unique.append(('unknown', 1.0))
            return unique[:topk]

        if found_key is None:
            fallback_val = next(iter(object_labels_dict.values()))
            object_label = ensure_topk_tuples(fallback_val, topk=5)
            print(f"[warn] object label not found for '{rel_path}' (requested '{args.image}'), using fallback label")
        else:
            val = object_labels_dict[found_key]
            object_label = ensure_topk_tuples(val, topk=5)
    except Exception as e:
        print(f"Error loading object labels: {e}")
        return

    # Run detection
    start_time = time.time()
    with torch.no_grad():
        try:
            # Always wrap as batch size 1
            object_label = [object_label]
            outputs = model(image_tensor, object_label[0])

            # Robust output handling: model may return logits, 2-class scores, or tensors of different shapes.
            confidence = None
            if isinstance(outputs, dict) and 'logits' in outputs:
                out = torch.as_tensor(outputs['logits']).detach().cpu()
                # If logits is 2-class, use softmax
                if out.ndim == 2 and out.shape[1] == 2:
                    probs = torch.softmax(out, dim=1)
                    confidence = float(probs[0, 1])
                else:
                    confidence = float(torch.sigmoid(out.mean()).item())
            elif torch.is_tensor(outputs):
                out = outputs.detach().cpu()
                if out.ndim == 0:
                    confidence = float(torch.sigmoid(out).item())
                elif out.ndim == 1:
                    if out.numel() == 2:
                        probs = torch.softmax(out, dim=0)
                        confidence = float(probs[1])
                    else:
                        confidence = float(torch.sigmoid(out.mean()).item())
                elif out.ndim == 2:
                    if out.shape[0] >= 1 and out.shape[1] == 2:
                        probs = torch.softmax(out, dim=1)
                        confidence = float(probs[0, 1])
                    else:
                        confidence = float(torch.sigmoid(out.mean()).item())
                else:
                    confidence = float(torch.sigmoid(out.mean()).item())
            else:
                try:
                    confidence = float(outputs)
                except Exception:
                    confidence = 0.0

            if confidence is None:
                confidence = 0.0

            label = 'fake' if confidence > 0.5 else 'real'

            result = format_result(label, confidence, time.time() - start_time)
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Time: {result['elapsed_time']:.3f}s")

            # Save result if output path provided
            if args.output:
                save_result(result, args.output)
        except Exception as e:
            print(f"Error during detection: {e}")
            return

if __name__ == '__main__':
    main()
