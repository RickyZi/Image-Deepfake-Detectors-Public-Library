import sys
import time
import os
import csv
import torch
from util import Logger, printSet
from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
import networks.resnet as resnet
import numpy as np
import random
from data import create_dataloader

from tqdm import tqdm
import pandas as pd

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
seed_torch(100)
# DetectionTests = {
#                 'ForenSynths': { 'dataroot'   : '/opt/data/private/DeepfakeDetection/ForenSynths/',
#                                  'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
#                                  'no_crop'    : True,
#                                },

#            'GANGen-Detection': { 'dataroot'   : '/opt/data/private/DeepfakeDetection/GANGen-Detection/',
#                                  'no_resize'  : True,
#                                  'no_crop'    : True,
#                                },

#          'DiffusionForensics': { 'dataroot'   : '/opt/data/private/DeepfakeDetection/DiffusionForensics/',
#                                  'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
#                                  'no_crop'    : True,
#                                },

#         'UniversalFakeDetect': { 'dataroot'   : '/opt/data/private/DeepfakeDetection/UniversalFakeDetect/',
#                                  'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
#                                  'no_crop'    : True,
#                                },

#                  }


opt = TestOptions().parse(print_options=False)
opt.model_path = f'./train/{opt.name}/models/best.pt'
print(f'Model_path {opt.model_path}')


# get model
model = resnet50(num_classes=1)
model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=True)
model.to(opt.device)
model.eval()

opt.no_resize = False
opt.no_crop   = True

os.makedirs(f'./train/{opt.name}/data/{opt.data_keys}', exist_ok=True)
test_dataloader = create_dataloader(opt, split='test')


model.eval()

csv_filename = f'./train/{opt.name}/data/{opt.data_keys}/results.csv'
# df = pd.DataFrame(columns=['name', 'pro','flag'])
with open(csv_filename, 'w') as f:
    f.write(f"{','.join(['name', 'pro', 'flag'])}\n")

with torch.no_grad():
    with tqdm(test_dataloader, unit='batch', mininterval=0.5) as tbatch:
        tbatch.set_description(f'Validation')
        for (data, labels, paths) in tbatch:
            data = data.to(opt.device)
            labels = labels.to(opt.device)

            scores = model(data).squeeze(1)

            with open(csv_filename, 'a') as f:
                for score, label, path in zip(scores, labels, paths):
                    f.write(f"{path}, {score.item()}, {label.item()}\n")
                    # df = df._append({'name': path,'pro': score.item(),'flag':label.item()}, ignore_index=True)

# df.to_csv(csv_filename, index=False)


# for testSet in DetectionTests.keys():
#     dataroot = DetectionTests[testSet]['dataroot']
#     printSet(testSet)

#     accs = [];aps = []
#     print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
#     for v_id, val in enumerate(os.listdir(dataroot)):
#         opt.dataroot = '{}/{}'.format(dataroot, val)
#         opt.classes  = '' #os.listdir(opt.dataroot) if multiclass[v_id] else ['']
#         opt.no_resize = DetectionTests[testSet]['no_resize']
#         opt.no_crop   = DetectionTests[testSet]['no_crop']
#         acc, ap, _, _, _, _ = validate(model, opt)
#         accs.append(acc);aps.append(ap)
#         print("({} {:12}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
#     print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) 

