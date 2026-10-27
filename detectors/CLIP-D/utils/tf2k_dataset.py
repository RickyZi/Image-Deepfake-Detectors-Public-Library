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

import os
import json
import torch
import bisect
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler
from torchvision import datasets
from .processing import make_processing

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def tf2k_create_dataloader(opt, split=None):
    """
    Create a dataloader for the TF2K dataset.
    """
    if split == "train":
        opt.split = 'train'
        is_train=True

    elif split == "val":
        opt.split = 'val'
        is_train=False
    
    elif split == "test":
        opt.split = 'test'
        opt.batch_size = 2
        is_train=False
    
    else:
        raise ValueError(f"Unknown split {split}")

    dataset = TrueFake_dataset(opt)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=is_train,
        num_workers=int(opt.num_threads),
    )
    return data_loader

def parse_tf2k_dataset(settings):
    """
    Parese the TF2K dataset based on the provided settings and return a list of dataset configurations.
    Combines all generators keys into a single list for easier processing.
    """
    
    gen_keys = {
        'gan1':['StyleGAN'],
        'gan2':['StyleGAN2'],
        'gan3':['StyleGAN3'],
        'sd15':['StableDiffusion1.5'],
        'sd2':['StableDiffusion2'],
        'sd3':['StableDiffusion3'],
        'sdXL':['StableDiffusionXL'],
        'flux':['FLUX.1'],
        'realFFHQ':['FFHQ'],
        'realFORLAB':['FORLAB']
    }

    gen_keys['all'] =   [gen_keys[key][0] for key in gen_keys.keys()]
    gen_keys['real'] =  [gen_keys[key][0] for key in gen_keys.keys() if 'real'  in key]

    need_real = (settings.split in ['train', 'val'] and not len([data for data in settings.data_keys.split('&') if 'real' in data.split(':')[0]]))

    assert not need_real, 'Train task without real data, this will not get handeled automatically, terminating'

    dataset_list = []
    for gen in settings.data_keys.split('&'):
       
        dataset_list.append({'gen':gen_keys[gen]}) 
    
    return dataset_list


class TrueFake_dataset(datasets.DatasetFolder):
    """
    A custom dataset class for the TrueFake dataset, inheriting from torchvision's DatasetFolder.
    It loads images based on the provided settings and applies transformations.
    """
    
    def __init__(self, settings):
        self.data_root = settings.data_root
        self.split = settings.split

        with open(settings.split_file, "r") as f:
            split_list = sorted(json.load(f)[self.split])
            
        self.split_set = set(split_list)
        
        dataset_list = parse_tf2k_dataset(settings)

        self.samples = []
        self.info = []
        for dict in dataset_list:
            generators = dict['gen']
            for dataset_root, dataset_dirs, dataset_files in os.walk(self.data_root, topdown=True, followlinks=True):
                if len(dataset_dirs):
                    continue

                # Compute path relative to mod root, e.g.:
                rel = os.path.relpath(dataset_root, self.data_root)
                parts = rel.split(os.sep)
                if len(parts) < 2:
                    continue

                label, gen = parts[0], parts[1]
                sub = parts[2] if len(parts) > 2 else None

                if gen not in generators:
                    continue

                for filename in sorted(dataset_files):
                    if os.path.splitext(filename)[1].lower() not in ['.png', '.jpg', '.jpeg']:
                        continue
                    stem = os.path.splitext(filename)[0]
                   
                    key = os.path.join(gen, sub, stem) if sub else os.path.join(gen, stem)
                    if self._in_list(split_list, key):
                        self.samples.append(os.path.join(dataset_root, filename))
                        self.info.append((label, gen, sub))


        if len(self.samples) == 0:
            print(f'Warning: No samples found for {self.split} split with the given settings. Please check your data_keys and split_file.')
            breakpoint()
        else:
            print(f'Loaded {len(self.samples)} samples for {self.split}')
        self.transform = make_processing(settings)
        print(self.transform)

    def _in_list(self, split, elem):
        i = bisect.bisect_left(split, elem)
        return i != len(split) and split[i] == elem
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path = self.samples[index]
        label, gen, sub = self.info[index] 

        import time
        last_exc = None
        sample = None
        for attempt in range(3):
            try:
                sample = Image.open(path).convert('RGB')
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                if attempt < 2:
                    print(f"[WARN] Attempt {attempt+1}/3 failed for {path}: {e} — retrying...")
                    time.sleep(0.1 * (attempt + 1))  
 
        if sample is None:
            raise RuntimeError(
                f"Failed to load image after 3 attempts: {path}"
            ) from last_exc

        # sample = Image.open(path).convert('RGB')
        sample = self.transform(sample)

        target = 1.0 if label == 'Fake' else 0.0

        return {'img': sample, 'target': target, 'path': path}