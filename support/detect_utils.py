import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import json

def load_image(image_path, size=224):
    """Load and preprocess an image for detection.
    
    Args:
        image_path (str): Path to the image file
        size (int): Size to resize the image to (default: 224)
        
    Returns:
        torch.Tensor: Preprocessed image tensor
        PIL.Image: Original loaded image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    image = Image.open(image_path).convert('RGB')
    
    # Standard normalization used by most models
    preprocess = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return preprocess(image).unsqueeze(0), image

def format_result(prediction, confidence, elapsed_time):
    """Format detection results.
    
    Args:
        prediction (str): 'real' or 'fake'
        confidence (float): Confidence score (0-1)
        elapsed_time (float): Detection time in seconds
        
    Returns:
        dict: Formatted results
    """
    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "elapsed_time": float(elapsed_time)
    }

def save_result(result, output_path):
    """Save detection result to JSON file.
    
    Args:
        result (dict): Detection result to save
        output_path (str): Path to save the JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

def get_device():
    """Get the best available device (CUDA or CPU).
    
    Returns:
        torch.device: Device to use for computation
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path, device):
    """Load a PyTorch model from checkpoint.
    
    Args:
        model_path (str): Path to model checkpoint
        device (torch.device): Device to load the model to
        
    Returns:
        dict: Loaded checkpoint
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
    return torch.load(model_path, map_location=device)