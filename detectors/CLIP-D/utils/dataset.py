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

def create_dataloader(opt, split=None):
    if split == "train":
        opt.split = 'train'
        is_train=True

    elif split == "val":
        opt.split = 'val'
        is_train=False
    
    elif split == "test":
        opt.split = 'test'
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

def parse_dataset(settings):
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
    # gen_keys['gan'] =   [gen_keys[key][0] for key in gen_keys.keys() if 'gan'   in key]
    # gen_keys['sd'] =    [gen_keys[key][0] for key in gen_keys.keys() if 'sd'    in key]
    gen_keys['real'] =  [gen_keys[key][0] for key in gen_keys.keys() if 'real'  in key]

    mod_keys = {
        'pre':  ['PreSocial'],
        'fb':   ['Facebook'],
        'tl':   ['Telegram'],
        'tw':   ['X'],
    }

    mod_keys['all'] = [mod_keys[key][0] for key in mod_keys.keys()]
    mod_keys['shr'] = [mod_keys[key][0] for key in mod_keys.keys() if key in ['fb', 'tl', 'tw']]

    need_real = (settings.split in ['train', 'val'] and not len([data for data in settings.data_keys.split('&') if 'real' in data.split(':')[0]]))

    assert not need_real, 'Train task without real data, this will not get handeled automatically, terminating'

    dataset_list = []
    for data in settings.data_keys.split('&'):
        gen, mod = data.split(':')
        dataset_list.append({'gen':gen_keys[gen], 'mod':mod_keys[mod]})
    
    return dataset_list

class TrueFake_dataset(datasets.DatasetFolder):
    def __init__(self, settings):
        self.data_root = settings.data_root
        self.split = settings.split

        with open(settings.split_file, "r") as f:
            split_list = sorted(json.load(f)[self.split])
        
        dataset_list = parse_dataset(settings)
        
        self.samples = []
        self.info = []
        for dict in dataset_list:
            generators = dict['gen']
            modifiers = dict['mod']

            for mod in modifiers:
                for dataset_root, dataset_dirs, dataset_files in os.walk(os.path.join(self.data_root, mod), topdown=True, followlinks=True):
                    if len(dataset_dirs):
                        continue
                    (label, gen, sub)  = f'{dataset_root}/'.replace(os.path.join(self.data_root, mod) + os.sep, '').split(os.sep)[:3][:3]
                    
                    if gen in generators:
                        for filename in sorted(dataset_files):
                            if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
                                if self._in_list(split_list, os.path.join(gen, sub, os.path.splitext(filename)[0])):
                                    self.samples.append(os.path.join(dataset_root, filename))
                                    self.info.append((mod, label, gen, sub))

        self.transform = make_processing(settings)
        print(self.transform)

    def _in_list(self, split, elem):
        i = bisect.bisect_left(split, elem)
        return i != len(split) and split[i] == elem
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path = self.samples[index]
        mod, label, gen, sub = self.info[index]

        sample = Image.open(path).convert('RGB')
        sample = self.transform(sample)

        target = 1.0 if label == 'Fake' else 0.0
        
        return {'img':sample, 'target':target, 'path':path}