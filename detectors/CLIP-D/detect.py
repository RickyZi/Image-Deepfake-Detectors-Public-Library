# ----------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------
import os
import sys
import time
import torch
import argparse
from PIL import Image
import torchvision.transforms as transforms

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from support.detect_utils import format_result, save_result, get_device

from networks import create_architecture

# ----------------------------------------------------------------------------
# IMAGE PREPROCESSING
# ----------------------------------------------------------------------------
def preprocess_image(image_path, size=224):
    """Load and preprocess a single image for model input."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# ----------------------------------------------------------------------------
# ARGUMENT PARSING
# ----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='CLIP-D single image detector')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='pretrained', help='Name of the model checkpoint directory')
    parser.add_argument('--output', type=str, help='Path to save detection result JSON')
    parser.add_argument('--device', type=str, help='Device to run on (e.g., cuda:0, cuda:1, cpu)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup device
    device = torch.device(args.device) if args.device else get_device()
    
    # Load model
    try:
        load_path = f'./detectors/CLIP-D/checkpoint/{args.model}/weights/best.pt'
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model weights not found at: {load_path}")
            
        checkpoint = torch.load(load_path, map_location=device)
        # Initialize model and load state
        model =  create_architecture("opencliplinearnext_clipL14commonpool", pretrained=False, num_classes=1).to(device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load and preprocess image
    try:
        image_tensor = preprocess_image(args.image)
        image_tensor = image_tensor.to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Run detection
    start_time = time.time()
    with torch.no_grad():
        try:
            score = model(image_tensor)
            prediction = torch.sigmoid(score)
            
            confidence = prediction.item()
            
            result = format_result(
                'fake' if confidence>0.5 else 'real',
                confidence,
                time.time() - start_time
            )
            
            # Print result
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