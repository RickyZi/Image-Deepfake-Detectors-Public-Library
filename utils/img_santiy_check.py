import os
from PIL import Image

image_dir = "/home/rz/TB_WP3/Image-Deepfake-Detectors-Public-Library/truefake_2k/test/" #Fake_StableDiffusion3_general_00374.jpg"
corrupted_files = []

# Walk through your dataset directory
for root, dirs, files in os.walk(image_dir):
    for file in files:
        # Filter for common extensions if needed
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp')):
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify() # Verify that the image isn't corrupted
            except (IOError, SyntaxError, Image.UnidentifiedImageError) as e:
                print(f"Bad file detected: {file_path}")
                corrupted_files.append(file_path)

print(f"Scan complete. Found {len(corrupted_files)} corrupted files.")
# Optional: Uncomment to automatically delete them
# for path in corrupted_files:
#     os.remove(path)