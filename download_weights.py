"""
Download model weights for Hugging Face Spaces deployment.
This script downloads model weights on first run if they're not present.
"""
import os
import urllib.request
import ssl

# Bypass SSL verification for downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def download_file(url, dest_path):
    """Download a file from URL to destination path."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    if os.path.exists(dest_path):
        print(f"✓ {dest_path} already exists")
        return
    
    print(f"Downloading {os.path.basename(dest_path)}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"✓ Downloaded {dest_path}")
    except Exception as e:
        print(f"✗ Failed to download {dest_path}: {e}")


# Model weights URLs (update these with actual URLs)
WEIGHTS_URLS = {
    "R50_TF": "https://drive.google.com/uc?export=download&id=YOUR_GOOGLE_DRIVE_ID",  # Replace
    "R50_nodown": "https://drive.google.com/uc?export=download&id=YOUR_GOOGLE_DRIVE_ID",  # Replace
    "CLIP-D": "https://drive.google.com/uc?export=download&id=YOUR_GOOGLE_DRIVE_ID",  # Replace
    "P2G": "https://drive.google.com/uc?export=download&id=YOUR_GOOGLE_DRIVE_ID",  # Replace
    "NPR": "https://drive.google.com/uc?export=download&id=YOUR_GOOGLE_DRIVE_ID",  # Replace
}


def download_all_weights():
    """Download all model weights if not present."""
    print("Checking model weights...")
    
    for model_name, url in WEIGHTS_URLS.items():
        dest_path = f"detectors/{model_name}/checkpoint/pretrained/weights/best.pt"
        
        # Skip if URL not configured
        if "YOUR_GOOGLE_DRIVE_ID" in url:
            print(f"⚠ Skipping {model_name}: URL not configured")
            continue
        
        download_file(url, dest_path)
    
    # Download P2G classes.pkl
    classes_url = "https://github.com/laitifranz/Prompt2Guard/raw/main/src/utils/classes.pkl"
    classes_path = "detectors/P2G/src/utils/classes.pkl"
    download_file(classes_url, classes_path)
    
    print("\nWeight check complete!")


if __name__ == "__main__":
    download_all_weights()
