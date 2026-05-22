"""
helper for counting the number of images store in a folder.
Sanity check for trn-tst-val splits
"""

import os
import sys
from pathlib import Path

# img_path = './demo_images/demo_images/'
img_path = './demo_images/season_TM01'

sub_folders = [f.path for f in os.scandir(img_path) if f.is_dir()]

real_imgs = fake_imgs = 0

for sub_folder in sub_folders:
    counter = 0
    for root, dirs, files in os.walk(sub_folder):
        if files:
            print(f"root: {root}")
            print(f"files: {files}")
            if 'real' in root.lower():
                real_imgs += len(files)
            elif 'fake' in root.lower():
                fake_imgs += len(files)
            # print(f"Number of images in {root}: {len(files)}")
            counter += len(files)
    print(f"Total number of images in {sub_folder}: {counter}")

print(f"Total number of real images: {real_imgs}")
print(f"Total number of fake images: {fake_imgs}")

# counter = 0
# for root, dirs, files in os.walk(img_path):
#     if files:
#         print(f"Number of images in {root}: {len(files)}")
#         counter += len(files)
# print(f"Total number of images in {img_path}: {counter}")

# Total number of images in ./demo_images/demo_images: 200
# Total number of images in ./demo_images/tb_preset: 0
# Total number of images in ./demo_images/season_SM01: 200
# Total number of images in ./demo_images/season_SP01: 200
# Total number of images in ./demo_images/season_TM01: 200
# Total number of images in ./demo_images/season_WN01: 200
# Total number of images in ./demo_images/style_BW01: 200
# Total number of images in ./demo_images/style_CN01: 200
# Total number of images in ./demo_images/style_CN11: 200
# Total number of images in ./demo_images/style_FT01: 200
# Total number of images in ./demo_images/style_VN01: 200
# Total number of images in ./demo_images/style_warmgold: 200

# --------------------------------------------------------------- #
# # img_path = './demo_images/'

# # rename imgs according to TB dataset

# img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('./demo_images/season_SM01/')

# sub_folders = [f.path for f in os.scandir(img_path) if f.is_dir()]


# for sub_folder in sub_folders:
#     # counter = 0
#     for root, dirs, files in os.walk(sub_folder):
#         if 'FLUX' in root:
#             # root = root.replace('FLUX', 'FLUX.1') # replace FLUX with FLUX.1
#             print(f"Renamed FLUX to FLUX.1 in path: {root}")
#             # rename the folder
#             os.rename(root, root.replace('FLUX', 'FLUX.1'))
    
#         elif '1p5' in root:
#             # root = root.replace('1p5', '1.5') # replace 1p5 with 1.5
#             print(f"Renamed 1p5 to 1.5 in path: {root}")
#             # rename the folder
#             os.rename(root, root.replace('1p5', '1.5'))

