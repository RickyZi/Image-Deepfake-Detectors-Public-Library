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
import torchvision.transforms.v2 as Tv2

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

    transforms_list.append(make_normalize(opt))  # make normalization

    t = Tv2.Compose(transforms_list)

    return t


def add_processing_arguments(parser):
    # parser is an argparse.ArgumentParser
    #
    # ICASSP2023: --cropSize 96 --loadSize -1 --resizeSize -1 --norm_type resnet --resize_prob 0.2 --jitter_prob 0.8 --colordist_prob 0.2 --cutout_prob 0.2 --noise_prob 0.2 --blur_prob 0.5 --cmp_prob 0.5 --rot90_prob 1.0 --hpf_prob 0.0 --blur_sig 0.0,3.0 --cmp_method cv2,pil --cmp_qual 30,100 --resize_size 256 --resize_ratio 0.75
    # ICME2021  : --cropSize 96 --loadSize -1 --resizeSize -1 --norm_type resnet --resize_prob 0.0 --jitter_prob 0.0 --colordist_prob 0.0 --cutout_prob 0.0 --noise_prob 0.0 --blur_prob 0.5 --cmp_prob 0.5 --rot90_prob 1.0 --hpf_prob 0.0 --blur_sig 0.0,3.0 --cmp_method cv2,pil --cmp_qual 30,100
    #

    parser.add_argument("--resizeSize", type=int, default=224, help="scale images to this size post augumentation")

    # data-augmentation probabilities
    parser.add_argument("--resize_prob", type=float, default=0.0)
    parser.add_argument("--cmp_prob", type=float, default=0.0)

    # data-augmentation parameters
    parser.add_argument("--cmp_qual", default="75")
    parser.add_argument("--resize_size", type=int, default=256)
    parser.add_argument("--resize_ratio", type=float, default=1.0)

    # other
    parser.add_argument("--norm_type", type=str, default="clip") 

    return parser


def parse_arguments(opt):
    if not isinstance(opt.cmp_qual, list):
        opt.cmp_qual = [int(s) for s in opt.cmp_qual.split(",")]
    return opt


def make_post(opt):
    transforms_list = list()
    if opt.resizeSize > 0:
        print("\nUsing Post Resizing\n")
        transforms_list.append(Tv2.Resize(opt.resizeSize, interpolation=Tv2.InterpolationMode.BICUBIC))
        transforms_list.append(Tv2.CenterCrop((opt.resizeSize, opt.resizeSize)))

    if len(transforms_list) == 0:
        return None
    else:
        return Tv2.Compose(transforms_list)


def make_aug(opt):
    # AUG
    transforms_list_aug = list()

    if (opt.resize_size > 0) and (opt.resize_prob > 0):  # opt.resized_ratio
        transforms_list_aug.append(
            Tv2.RandomApply(
                [
                    Tv2.RandomResizedCrop(
                        size=opt.resize_size,
                        scale=(5/8, 1.0),
                        ratio=(opt.resize_ratio, 1.0 / opt.resize_ratio),
                    )
                ],
                opt.resize_prob,
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

    if len(transforms_list_aug) > 0:
        return Tv2.Compose(transforms_list_aug)
    else:
        return None


def make_normalize(opt):
    transforms_list = list()

    if opt.norm_type == "clip":
        print("normalize CLIP")
        transforms_list.append(Tv2.ToTensor())
        transforms_list.append(
            Tv2.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            )
        )
    else:
        assert False

    return Tv2.Compose(transforms_list)
