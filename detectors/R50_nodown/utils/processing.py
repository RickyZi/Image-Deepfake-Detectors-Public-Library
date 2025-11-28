'''                                        
Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.
                        
Licensed under the Apache License, Version 2.0 (the "License");       
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at                    
                                           
    http://www.apache.org/licenses/LICENSE-2.0
                                                      
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,    
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                         
See the License for the specific language governing permissions and
limitations under the License.
''' 

import random
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.v2 as Tv2
from PIL import Image
import torch

def make_processing(opt):
    opt = parse_arguments(opt)
    transforms_list = list()  # list of transforms

    if opt.task == 'train':
        transforms_aug = make_aug(opt)  # make data-augmentation transforms
        if transforms_aug is not None:
            transforms_list.append(transforms_aug)

        transforms_post = make_post(opt)  # make post-data-augmentation transforms
        if transforms_post is not None:
            transforms_list.append(transforms_post)

    if opt.task == 'test' and 'realFORLAB:pre' in opt.data_keys:
        transforms_list.append(Tv2.CenterCrop(1024)) 
    if opt.task == 'test' and 'realFORLAB:fb' in opt.data_keys:
        transforms_list.append(Tv2.CenterCrop(720)) 
    if opt.task == 'test' and 'realFORLAB:tw' in opt.data_keys:
        transforms_list.append(Tv2.CenterCrop(1200)) 
    if opt.task == 'test' and 'realFORLAB:tl' in opt.data_keys:
        transforms_list.append(Tv2.CenterCrop(800)) 
    transforms_list.append(make_normalize(opt))  # make normalization

    return Tv2.Compose(transforms_list)


def add_processing_arguments(parser):
    # parser is an argparse.ArgumentParser
    #
    # ICASSP2023: --cropSize 96 --loadSize -1 --resizeSize -1 --norm_type resnet --resize_prob 0.2 --jitter_prob 0.8 --colordist_prob 0.2 --cutout_prob 0.2 --noise_prob 0.2 --blur_prob 0.5 --cmp_prob 0.5 --rot90_prob 1.0 --hpf_prob 0.0 --blur_sig 0.0,3.0 --cmp_method cv2,pil --cmp_qual 30,100 --resize_size 256 --resize_ratio 0.75
    #

    parser.add_argument("--cropSize",type=int,default=-1,help="crop images to this size post augumentation")

    # data-augmentation probabilities
    parser.add_argument("--resize_prob", type=float, default=0.0)
    parser.add_argument("--jitter_prob", type=float, default=0.0)
    parser.add_argument("--colordist_prob", type=float, default=0.0)
    parser.add_argument("--cutout_prob", type=float, default=0.0)
    parser.add_argument("--noise_prob", type=float, default=0.0)
    parser.add_argument("--blur_prob", type=float, default=0.0)
    parser.add_argument("--cmp_prob", type=float, default=0.0)

    # data-augmentation parameters
    parser.add_argument("--blur_sig", default="0.5")
    parser.add_argument("--cmp_qual", default="75")
    parser.add_argument("--resize_size", type=int, default=256)
    parser.add_argument("--resize_ratio", type=float, default=1.0)

    # other
    parser.add_argument("--norm_type", type=str, default="resnet")  # normalization type

    return parser


def parse_arguments(opt):
    if not isinstance(opt.blur_sig, list):
        opt.blur_sig = [float(s) for s in opt.blur_sig.split(",")]
    if not isinstance(opt.cmp_qual, list):
        opt.cmp_qual = [int(s) for s in opt.cmp_qual.split(",")]
        
    print(opt.cmp_qual)
    return opt

def make_post(opt):
    transforms_list = list()

    if opt.cropSize > 0:
        print("\nUsing Post Random Crop\n")
        transforms_list.append(Tv2.RandomCrop(opt.cropSize, pad_if_needed=True, padding_mode="symmetric"))

    if len(transforms_list) == 0:
        return None
    else:
        return Tv2.Compose(transforms_list)


def make_aug(opt):
    # AUG
    transforms_list_aug = list()

    if (opt.resize_size > 0) and (opt.resize_prob > 0):  # opt.resized_ratio
        transforms_list_aug.append(
            Tv2.RandomChoice(
                [
                    Tv2.RandomResizedCrop(
                        size=opt.resize_size,
                        scale=(0.08, 1.0),
                        ratio=(opt.resize_ratio, 1.0 / opt.resize_ratio),
                    ),
                    Tv2.RandomCrop([opt.resize_size])
                ],
                p=[opt.resize_prob, 1 - opt.resize_prob],
            )
        )

    if opt.jitter_prob > 0:
        transforms_list_aug.append(
            Tv2.RandomApply(
                [
                    Tv2.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], 
                p=opt.jitter_prob
            )
        )

    if opt.colordist_prob > 0:
        transforms_list_aug.append(Tv2.RandomGrayscale(p=opt.colordist_prob))

    if opt.cutout_prob > 0:
        transforms_list_aug.append(create_cutout_transforms(opt.cutout_prob))

    if opt.noise_prob > 0:
        transforms_list_aug.append(
            Tv2.Compose([
                Tv2.ToImage(),
                Tv2.ToDtype(torch.float32, scale=False),
                Tv2.RandomApply(
                    [   
                        Tv2.GaussianNoise(sigma=0.44)
                    ],
                    p=opt.noise_prob
                ),
                Tv2.ToDtype(torch.uint8, scale=False),
                Tv2.ToPILImage(),
            ])
        )

    if opt.blur_prob > 0:
        transforms_list_aug.append(
            Tv2.RandomApply(
                [
                    Tv2.GaussianBlur(
                        kernel_size=15,
                        sigma=opt.blur_sig,
                    )
                ],
                p=opt.blur_prob
            )
        )

    if opt.cmp_prob > 0:
        transforms_list_aug.append(
            Tv2.RandomApply(
                [
                    Tv2.JPEG(
                        opt.cmp_qual
                    )
                ],
                opt.cmp_prob,
            )
        )


    transforms_list_aug.append(Tv2.Compose([Tv2.RandomHorizontalFlip(), Tv2.RandomVerticalFlip()]))

    if len(transforms_list_aug) > 0:
        return Tv2.Compose(transforms_list_aug)
    else:
        return None


def make_normalize(opt):
    transforms_list = list()
    if opt.norm_type == "resnet":
        print("normalize RESNET")

        transforms_list.append(Tv2.ToTensor())
        transforms_list.append(
            Tv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    else:
        assert False

    return Tv2.Compose(transforms_list)

def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return random.choice(s)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random.random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")
              
def create_cutout_transforms(p):
    from albumentations import CoarseDropout
    aug = CoarseDropout(
        num_holes_range=(1,1),
        hole_height_range=(1, 48),
        hole_width_range=(1, 48),
        fill=128,
        p=p,
    )
    return transforms.Lambda(
        lambda img: Image.fromarray(aug(image=np.array(img))["image"])
    )