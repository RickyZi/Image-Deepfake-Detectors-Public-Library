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
# from .processing import make_processing
import torchvision.transforms.v2 as Tv2
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# def get_bal_sampler(dataset):
#     targets = []
#     for d in dataset.datasets:
#         targets.extend(d.targets)

#     ratio = np.bincount(targets)
#     w = 1.0 / torch.tensor(ratio, dtype=torch.float)
#     if torch.all(w==w[0]):
#         print(f"RandomSampler: # {ratio}")
#         sampler = RandomSampler(dataset, replacement = False)
#     else:
#         w = w / torch.sum(w)
#         print(f"WeightedRandomSampler: # {ratio}, Weightes {w}")
#         sample_weights = w[targets]
#         sampler = WeightedRandomSampler(
#             weights=sample_weights, num_samples=len(sample_weights)
#         )
#     return sampler

def tf2k_create_dataloader(opt, split=None):
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

    # mod_keys = {
    #     'pre':  ['PreSocial'],
    #     'fb':   ['Facebook'],
    #     'tl':   ['Telegram'],
    #     'tw':   ['X'],
    # }

    # mod_keys['all'] = [mod_keys[key][0] for key in mod_keys.keys()]
    # mod_keys['shr'] = [mod_keys[key][0] for key in mod_keys.keys() if key in ['fb', 'tl', 'tw']]

    need_real = (settings.split in ['train', 'val'] and not len([data for data in settings.data_keys.split('&') if 'real' in data.split(':')[0]]))

    assert not need_real, 'Train task without real data, this will not get handeled automatically, terminating'

    dataset_list = []
    for data in settings.data_keys.split('&'):
        # # gen = data.split(':')
        # dataset_list.append({'gen':gen_keys[data]}) #, 'mod':mod_keys[mod]})
        gen, mod = data.split(':')
        dataset_list.append({'gen':gen_keys[gen]}) #, 'mod':mod_keys[mod]})
        # removed mod because we have just Real/FORLAB and no social media processing (mod)
    
    return dataset_list


class TrueFake_dataset(datasets.DatasetFolder):
    def __init__(self, settings):
        self.data_root = settings.data_root
        self.split = settings.split

        with open(settings.split_file, "r") as f:
            split_list = sorted(json.load(f)[self.split])
        
        # # check FORLAB in split_file
        # forlab_keys = [k for k in split_list if 'FORLAB' in k]
        # print(f"FORLAB keys in split file: {forlab_keys[:5]}")
        self.split_set = set(split_list)
        
        dataset_list = parse_tf2k_dataset(settings)
        
        self.samples = []
        self.info = []
        for dict in dataset_list:
            generators = dict['gen']
            
            # print("NO MOD!!!")
            # dataset_roo = os.path.join(self.data_root, mod)
            # print(f"\n\tR50_nodown - dataset.py - mod_root: {dataset_root}")
            for dataset_root, dataset_dirs, dataset_files in os.walk(self.data_root, topdown=True, followlinks=True):
                if len(dataset_dirs):
                    continue

                # Compute path relative to mod root, e.g.:
                #   Real/FFHQ (2 parts — no sub, Real images)
                #   Fake/StyleGAN3/conf-t (3 parts — has sub, Fake images)
                rel = os.path.relpath(dataset_root, self.data_root)
                parts = rel.split(os.sep)
                # print(f"rel: {rel}")
                # print(f"parts: {parts}")
                # breakpoint()
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
                    # Key must match with test_json_split.py:
                    #   gen/stem for Real  (e.g. "FFHQ/00098")
                    #   gen/sub/stem for Fake  (e.g. "StyleGAN3/conf-t-psi-0.5/00098")
                    key = os.path.join(gen, sub, stem) if sub else os.path.join(gen, stem)
                    if self._in_list(split_list, key):
                        self.samples.append(os.path.join(dataset_root, filename))
                        self.info.append((label, gen, sub))

        # --------------------------------------- #
        # APPLY IMG TRANSFORMATIONS 
        # --------------------------------------- #

        self.transform_start = Tv2.Compose(
            [
                Tv2.ToImage()
            ]
        )

        self.transform_end = Tv2.Compose(
            [
                Tv2.CenterCrop(1024)    if self.split == 'test' and 'realFORLAB:pre'    in settings.data_keys  else Tv2.Identity(),
                Tv2.CenterCrop(720)     if self.split == 'test' and 'realFORLAB:fb'     in settings.data_keys  else Tv2.Identity(),
                Tv2.CenterCrop(1200)    if self.split == 'test' and 'realFORLAB:tw'     in settings.data_keys  else Tv2.Identity(),
                Tv2.CenterCrop(800)     if self.split == 'test' and 'realFORLAB:tl'     in settings.data_keys  else Tv2.Identity(),
                Tv2.ToDtype(torch.float32, scale=True),
                Tv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        if self.split in ['train', 'val']:
            self.transform_aug = {
                'light': Tv2.Compose(
                                        [
                                            Tv2.RandomChoice([Tv2.RandomResizedCrop([300], (0.5, 1.5), (0.5, 2)), Tv2.RandomCrop([300])], p=[0.3, 0.7]),
                                            Tv2.Compose([Tv2.RandomHorizontalFlip(p=0.5), Tv2.RandomVerticalFlip(p=0.5)]),
                                            Tv2.RandomCrop(96, pad_if_needed=True) if self.split == 'train' else Tv2.Identity(),
                                        ]
                                    ),
                'heavy': Tv2.Compose(
                                        [
                                            Tv2.RandomChoice([Tv2.RandomResizedCrop([300], (0.5, 1.5), (0.5, 2)), Tv2.RandomCrop([300])], p=[0.3, 0.7]),

                                            Tv2.RandomApply([Tv2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.3),
                                            Tv2.RandomApply([Tv2.GaussianBlur(kernel_size=11, sigma=(0.1,3))], p=0.3),
                                            Tv2.RandomApply([Tv2.JPEG((65, 95))], p=0.3),

                                            Tv2.Compose([Tv2.RandomHorizontalFlip(p=0.5), Tv2.RandomVerticalFlip(p=0.5)]),

                                            Tv2.RandomCrop(96, pad_if_needed=True) if self.split == 'train' else Tv2.Identity(),
                                        ]
                                    )
            }

        else:
            self.transform_aug = None
        
        print()
        print(f'Transforms for {self.split}:')
        print(self.transform_start)
        if self.transform_aug:
            print(self.transform_aug['light'])
            print(self.transform_aug['heavy'])
        print(self.transform_end)

        if len(self.samples) == 0:
            print(f'Warning: No samples found for {self.split} split with the given settings. Please check your data_keys and split_file.')
            breakpoint()
        else:
            print(f'Loaded {len(self.samples)} samples for {self.split}')


        # # --------------------------------------------------------------- #
        # # IMAGE VALIDATION (TO AVOID ERROR WHEN LOADING IMGS)
        # # --------------------------------------------------------------- #
        # # Validate all collected images upfront. img.verify() does a cheap
        # # header-only parse without decoding pixels, so this is fast even for
        # # large splits. Corrupt files are removed before any DataLoader worker
        # # is spawned, preventing mid-epoch crashes.
        # # Any skipped file is logged to a JSON under logs/skipped_images/ for
        # # traceability across runs, models, and datasets.
        # valid_samples, valid_info = [], []
        # skipped = []
        # for path, info in zip(self.samples, self.info):
        #     try:
        #         with Image.open(path) as img:
        #             img.verify()
        #         valid_samples.append(path)
        #         valid_info.append(info)
        #     except Exception as e:
        #         print(f"[WARN] Skipping corrupt image: {path}  ({e})")
        #         skipped.append({'path': path, 'error': str(e)})
 
        # if skipped:
        #     print(f"[WARN] Removed {len(skipped)} corrupt file(s) "
        #           f"from dataset (split='{self.split}').")
        #     self._write_skipped_log(settings, skipped)
 
        # self.samples = valid_samples
        # self.info    = valid_info
 
    def _write_skipped_log(self, settings, skipped):
        """Append skipped-image records to a per-run JSON log file.
 
        Filename pattern:
            logs/skipped_images/<model_name>__<dataset>__<phase>.json
 
        where:
            model_name  = settings.name   (e.g. "pretrained")
            dataset     = last component of settings.data_root
                          (e.g. "cinematic_CN01")
            phase       = "ft"         if settings.ft is True
                          "train"      if self.split == "train"
                          "val"        if self.split == "val"
                          "test"       otherwise
        """
        from datetime import datetime
 
        dataset_name = os.path.basename(os.path.normpath(settings.data_root))
        phase = 'ft' if getattr(settings, 'ft', False) else self.split
 
        log_dir = os.path.join('logs', 'skipped_images')
        os.makedirs(log_dir, exist_ok=True)
 
        model_name = getattr(settings, 'name', 'unknown')
        log_file = os.path.join(
            log_dir,
            f"{model_name}__{dataset_name}__{phase}.json"
        )
 
        # Load existing log if present (accumulate across re-runs)
        if os.path.exists(log_file):
            with open(log_file) as f:
                existing = json.load(f)
        else:
            existing = []
 
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for entry in skipped:
            existing.append({
                'timestamp':  timestamp,
                'model':      model_name,
                'dataset':    dataset_name,
                'phase':      phase,
                'split':      self.split,
                'data_keys':  getattr(settings, 'data_keys', ''),
                'path':       entry['path'],
                'error':      entry['error'],
            })
 
        with open(log_file, 'w') as f:
            json.dump(existing, f, indent=2)
 
        print(f"[INFO] Skipped-image log written to: {log_file}")


    def _in_list(self, split, elem):
        i = bisect.bisect_left(split, elem)
        return i != len(split) and split[i] == elem
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        label, gen, sub = self.info[index]

        image = Image.open(path).convert('RGB')
        sample = self.transform_start(image)
        if self.transform_aug:
            # sample = self.transform_aug['heavy' if mod == 'PreSocial' else 'light'](sample)
            # tf2k data are only 'PreSocial'
            sample = self.transform_aug['heavy'](sample)
        sample = self.transform_end(sample)

        target = 1.0 if label == 'Fake' else 0.0
        
        return sample, target, path
    
