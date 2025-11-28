import os
import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
import json
import bisect

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class CDDB_benchmark(object):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = args["class_order"]
        self.class_order = class_order

    def download_data(self):

        train_dataset = []
        test_dataset = []
        for id, name in enumerate(self.args["task_name"]):
            root_ = os.path.join(self.args["data_path"], name, "train")
            sub_classes = os.listdir(root_) if self.args["multiclass"][id] else [""]
            for cls in sub_classes:
                for imgname in os.listdir(os.path.join(root_, cls, "0_real")):
                    train_dataset.append(
                        (os.path.join(root_, cls, "0_real", imgname), 0 + 2 * id)
                    )
                for imgname in os.listdir(os.path.join(root_, cls, "1_fake")):
                    train_dataset.append(
                        (os.path.join(root_, cls, "1_fake", imgname), 1 + 2 * id)
                    )

        for id, name in enumerate(self.args["task_name"]):
            root_ = os.path.join(self.args["data_path"], name, "val")
            sub_classes = os.listdir(root_) if self.args["multiclass"][id] else [""]
            for cls in sub_classes:
                for imgname in os.listdir(os.path.join(root_, cls, "0_real")):
                    test_dataset.append(
                        (os.path.join(root_, cls, "0_real", imgname), 0 + 2 * id)
                    )
                for imgname in os.listdir(os.path.join(root_, cls, "1_fake")):
                    test_dataset.append(
                        (os.path.join(root_, cls, "1_fake", imgname), 1 + 2 * id)
                    )

        self.train_data, self.train_targets = split_images_labels(train_dataset)
        self.test_data, self.test_targets = split_images_labels(test_dataset)


def parse_dataset(data_keys):
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

    dataset_list = []
    for data in data_keys.split('&'):
        gen, mod = data.split(':')
        dataset_list.append({'gen':gen_keys[gen], 'mod':mod_keys[mod]})
    
    return dataset_list

class TrueFake_benchmark(object):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = args["class_order"]
        self.class_order = class_order

    def _in_list(self, split, elem):
        i = bisect.bisect_left(split, elem)
        return i != len(split) and split[i] == elem
    
    def download_data(self):

        with open(self.args["split_file"], "r") as f:
            splits = json.load(f)
            train_split = sorted(splits["train"])
            val_split = sorted(splits["val"])

        train_dataset = []
        test_dataset = []
        

        for id, name in enumerate(self.args["task_name"]):
            dataset_list = parse_dataset(name)

            for dict in dataset_list:
                generators = dict['gen']
                modifiers = dict['mod']

                for mod in modifiers:
                    for dataset_root, dataset_dirs, dataset_files in os.walk(os.path.join(self.args["data_path"], mod), topdown=True, followlinks=True):
                        if len(dataset_dirs):
                            continue

                        (label, gen, sub)  = f'{dataset_root}/'.replace(os.path.join(self.args["data_path"], mod) + os.sep, '').split(os.sep)[:3]
                        
                        if gen in generators:
                            for filename in sorted(dataset_files):
                                if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
                                    if self._in_list(train_split, os.path.join(gen, sub, os.path.splitext(filename)[0])):
                                        train_dataset.append((os.path.join(dataset_root, filename), (1 if label == 'Fake' else 0) + 2 * id))

                                    if self._in_list(val_split, os.path.join(gen, sub, os.path.splitext(filename)[0])):
                                        test_dataset.append((os.path.join(dataset_root, filename), (1 if label == 'Fake' else 0) + 2 * id))

        self.train_data, self.train_targets = split_images_labels(train_dataset)
        self.test_data, self.test_targets = split_images_labels(test_dataset)
