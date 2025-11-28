import os
import glob
import pandas
import random
from torch.utils.data import DataLoader, random_split
import json
import bisect

dataset_path = os.path.join(os.sep, 'path', 'to', 'dataset') 

datasets = []
for dataset_root, dataset_dirs, dataset_files in os.walk(os.path.join(dataset_path), topdown=True, followlinks=True):
    if len(dataset_dirs):
        continue

    # if 'Telegram' not in dataset_root:
    #     continue

    id = dataset_root.split('Real/')[-1].split('Fake/')[-1]
    #print(id)

    shr = dataset_root.replace(dataset_path + os.sep, '').split('/')[0]
    #print(shr)
    if 'FORLAB' in id or 'FFHQ' in id:
        files = sorted([file.replace('.jpg', '').replace('.png', '') for file in dataset_files])[:40000]
    else:
        files = sorted([file.replace('.jpg', '').replace('.png', '') for file in dataset_files])
    #print(len(files))
    datasets.append({'id': id, 'shared': shr,  'root': dataset_root, 'files': files})


split = []

train_set = []
val_set = []
test_set = []
breakpoint()
for dataset in [dataset for dataset in datasets if (dataset['shared'] == 'Telegram')]:
    print(dataset['id'])
    files_pre = [dataset_com for dataset_com in datasets if dataset_com['shared'] == 'PreSocial' and dataset_com['id'] == dataset['id']][0]['files']
    files_post = dataset['files']

    train_set_post, val_set_post, test_set_post = random_split(files_post, [0.7, 0.15, 0.15])
    
    residual_pre = [file for file in files_pre if file not in files_post]
    residual_pre_neg = [file for file in files_pre if file in files_post]

    train_set_pre, val_set_pre, test_set_pre = random_split(residual_pre, [0.7, 0.15, 0.15])

    train_set = train_set + [os.path.join(dataset['id'], file) for file in train_set_post] + [os.path.join(dataset['id'], file) for file in train_set_pre]
    val_set = val_set + [os.path.join(dataset['id'], file) for file in val_set_post] + [os.path.join(dataset['id'], file) for file in val_set_pre]
    test_set = test_set + [os.path.join(dataset['id'], file) for file in test_set_post] + [os.path.join(dataset['id'], file) for file in test_set_pre]

    print(len(train_set_post), len(val_set_post), len(test_set_post), ':', len(train_set_post)+len(val_set_post)+len(test_set_post))
    print(len(train_set_pre), len(val_set_pre), len(test_set_pre), ':', len(train_set_pre)+len(val_set_pre)+len(test_set_pre))
    print(len(train_set_pre)+len(train_set_post), len(val_set_pre)+len(val_set_post), len(test_set_pre)+len(test_set_post), ':', len(train_set_pre)+len(train_set_post)+len(val_set_pre)+len(val_set_post)+len(test_set_pre)+len(test_set_post))
    #print(val_set)
    #print(test_set)
    #train_set = train_set + [os.path.join(dataset['id'], file) for file in train_set_pre]
    #val_set = val_set + [os.path.join(dataset['id'], file) for file in val_set_pre]
    #test_set = test_set + [os.path.join(dataset['id'], file) for file in test_set_pre]

print(len(train_set), len(val_set), len(test_set), ':', len(train_set)+len(val_set)+len(test_set))

#with open("train.json", "w") as f:
#    json.dump(train_set, f)

#with open("val.json", "w") as f:
#    json.dump(val_set, f)

#with open("test.json", "w") as f:
#    json.dump(test_set, f)

with open("split.json", "w") as f:
    json.dump({'train': sorted(train_set), 'val': sorted(val_set), 'test': sorted(test_set)}, f)