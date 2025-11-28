import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import dataset_folder
from torchvision.datasets import DatasetFolder
import json
import bisect
from PIL import Image
import torchvision.transforms.v2 as Tv2
'''
def get_dataset(opt):
    dset_lst = []
    for cls in opt.classes:
        root = opt.dataroot + '/' + cls
        dset = dataset_folder(opt, root)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)
'''

import os
# def get_dataset(opt):
#     classes = os.listdir(opt.dataroot) if len(opt.classes) == 0 else opt.classes
#     if '0_real' not in classes or '1_fake' not in classes:
#         dset_lst = []
#         for cls in classes:
#             root = opt.dataroot + '/' + cls
#             dset = dataset_folder(opt, root)
#             dset_lst.append(dset)
#         return torch.utils.data.ConcatDataset(dset_lst)
#     return dataset_folder(opt, opt.dataroot)

# def get_bal_sampler(dataset):
#     targets = []
#     for d in dataset.datasets:
#         targets.extend(d.targets)

#     ratio = np.bincount(targets)
#     w = 1. / torch.tensor(ratio, dtype=torch.float)
#     sample_weights = w[targets]
#     sampler = WeightedRandomSampler(weights=sample_weights,
#                                     num_samples=len(sample_weights))
#     return sampler


# def create_dataloader(opt):
#     shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
#     dataset = get_dataset(opt)
#     sampler = get_bal_sampler(dataset) if opt.class_bal else None

#     data_loader = torch.utils.data.DataLoader(dataset,
#                                               batch_size=opt.batch_size,
#                                               shuffle=shuffle,
#                                               sampler=sampler,
#                                               num_workers=int(opt.num_threads))
#     return data_loader


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

    need_real = (settings.task == 'train' and not len([data.split(':')[0] for data in settings.data_keys.split('&') if 'real' in data.split(':')[0]]))

    assert not need_real, 'Train task without real data, this will not get handeled automatically, terminating'

    dataset_list = []
    for data in settings.data_keys.split('&'):
        gen, mod = data.split(':')
        dataset_list.append({'gen':gen_keys[gen], 'mod':mod_keys[mod]})
    
    return dataset_list

class TrueFake_dataset(DatasetFolder):
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

                    (label, gen, sub)  = f'{dataset_root}/'.replace(os.path.join(self.data_root, mod) + os.sep, '').split(os.sep)[:3]
                    
                    if gen in generators:
                        for filename in sorted(dataset_files):
                            if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
                                if self._in_list(split_list, os.path.join(gen, sub, os.path.splitext(filename)[0])):
                                    self.samples.append(os.path.join(dataset_root, filename))
                                    self.info.append((mod, label, gen, sub))

        if settings.isTrain:
            crop_func = Tv2.RandomCrop(settings.cropSize)
        elif settings.no_crop:
            crop_func = Tv2.Identity()
        else:
            crop_func = Tv2.CenterCrop(settings.cropSize)

        if settings.isTrain and not settings.no_flip:
            flip_func = Tv2.RandomHorizontalFlip()
        else:
            flip_func = Tv2.Identity()

        if not settings.isTrain and settings.no_resize:
            rz_func = Tv2.Identity()
        else:
            rz_func = Tv2.Resize((settings.loadSize, settings.loadSize))

        self.transform = Tv2.Compose([
                            rz_func,
                            crop_func,
                            flip_func,
                            Tv2.ToTensor(),
                            Tv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
    
                    
    def _in_list(self, split, elem):
        i = bisect.bisect_left(split, elem)
        return i != len(split) and split[i] == elem
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path = self.samples[index]
        mod, label, gen, sub = self.info[index]

        image = Image.open(path).convert('RGB')
        sample = self.transform(image)

        target = 1.0 if label == 'Fake' else 0.0
        
        return sample, target, path
    

def create_dataloader(settings, split=None):
    if split == "train":
        settings.split = 'train'
        is_train=True

    elif split == "val":
        settings.split = 'val'
        settings.batch_size = settings.batch_size//4
        is_train=False
    
    elif split == "test":
        settings.split = 'test'
        settings.batch_size = settings.batch_size//4
        is_train=False
    
    else:
        raise ValueError(f"Unknown split {split}")

    dataset = TrueFake_dataset(settings)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=settings.batch_size,
        num_workers=int(settings.num_threads),
        shuffle = is_train,
        collate_fn=None,
    )
    return data_loader
