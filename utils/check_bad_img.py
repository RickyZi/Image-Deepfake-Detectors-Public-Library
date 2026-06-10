import os
from PIL import Image


bad_file_path = "/home/rz/TB_WP3/Image-Deepfake-Detectors-Public-Library/truefake_2k/tf2k_lr_org/style/cinematic_CN01/Fake/StableDiffusion3/general/00374.jpg"

with open(bad_file_path, "rb") as f:
    first_bytes = f.read(16)
    print(f"Hex representation: {first_bytes.hex()}")
    print(f"Text representation: {first_bytes}")


# ------------------------------------------------------------- #
# import cv2
# from PIL import Image

# bad_file_path = "/home/rz/TB_WP3/Image-Deepfake-Detectors-Public-Library/truefake_2k/tf2k_lr_org/style/cinematic_CN01/Fake/StableDiffusion3/general/00374.jpg"

# # Try reading with OpenCV
# img_cv = cv2.imread(bad_file_path)

# if img_cv is not None:
#     # If OpenCV successfully read it, overwrite it as a clean JPEG
#     cv2.imwrite(bad_file_path, img_cv)
#     print("Fixed the image using OpenCV re-encoding!")
    
#     # Verify PIL can open it now
#     with Image.open(bad_file_path) as img:
#         img.verify()
# else:
#     print("OpenCV couldn't save it either. The file is completely unreadable.")