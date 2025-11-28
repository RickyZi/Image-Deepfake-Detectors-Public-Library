import sys
import time
import os
import csv
import torch
import json
from util import Logger, printSet
from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
import networks.resnet as resnet
import numpy as np
import random
from data import create_dataloader
from sklearn.metrics import roc_auc_score, accuracy_score

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

opt = TestOptions().parse(print_options=False)
opt.model_path = os.path.join(f'./checkpoint/{opt.name}/weights/best.pt')
print(f'Model_path {opt.model_path}')


# get model
model = resnet50(num_classes=1)
model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=True)
model.to(opt.device)
model.eval()

opt.no_resize = False
opt.no_crop   = True

output_dir = f'./results/{opt.name}/data/{opt.data_keys}'
os.makedirs(output_dir, exist_ok=True)

test_dataloader = create_dataloader(opt, split='test')

model.eval()

# File paths
csv_filename = os.path.join(output_dir, 'results.csv')
metrics_filename = os.path.join(output_dir, 'metrics.json')
image_results_filename = os.path.join(output_dir, 'image_results.json')

# Extract training dataset keys from model name (format: "training_keys_freeze_down" or "training_keys")
training_dataset_keys = []
model_name = opt.name
if '_freeze_down' in model_name:
    training_name = model_name.replace('_freeze_down', '')
else:
    training_name = model_name
if '&' in training_name:
    training_dataset_keys = training_name.split('&')
else:
    training_dataset_keys = [training_name]

# Collect all results
all_scores = []
all_labels = []
all_paths = []
image_results = []

start_time = time.time()

# Write CSV header
with open(csv_filename, 'w') as f:
    f.write(f"{','.join(['name', 'pro', 'flag'])}\n")

with torch.no_grad():
    with tqdm(test_dataloader, unit='batch', mininterval=0.5) as tbatch:
        tbatch.set_description(f'Validation')
        for (data, labels, paths) in tbatch:
            data = data.to(opt.device)
            labels = labels.to(opt.device)

            scores = model(data).squeeze(1)

            # Collect results
            for score, label, path in zip(scores, labels, paths):
                score_val = score.item()
                label_val = label.item()
                
                all_scores.append(score_val)
                all_labels.append(label_val)
                all_paths.append(path)
                
                image_results.append({
                    'path': path,
                    'score': score_val,
                    'label': label_val
                })
            
            # Write to CSV (maintain backward compatibility)
            with open(csv_filename, 'a') as f:
                for score, label, path in zip(scores, labels, paths):
                    f.write(f"{path}, {score.item()}, {label.item()}\n")

# Calculate metrics
all_scores = np.array(all_scores)
all_labels = np.array(all_labels)

# Convert scores to probabilities using sigmoid (as done in validate.py)
probabilities = torch.sigmoid(torch.tensor(all_scores)).numpy()

# Convert probabilities to predictions using threshold 0.5 (as done in validate.py)
predictions = (probabilities > 0.5).astype(int)

# Calculate overall metrics
total_accuracy = accuracy_score(all_labels, predictions)

# TPR (True Positive Rate) = TP / (TP + FN) = accuracy on fake images (label==1)
fake_mask = all_labels == 1
if fake_mask.sum() > 0:
    tpr = accuracy_score(all_labels[fake_mask], predictions[fake_mask])
else:
    tpr = 0.0

# Calculate TNR on real images (label==0) in the test set
real_mask = all_labels == 0
if real_mask.sum() > 0:
    # Overall TNR calculated on all real images in the test set
    tnr = accuracy_score(all_labels[real_mask], predictions[real_mask])
else:
    tnr = 0.0

# AUC calculation (using probabilities)
if len(np.unique(all_labels)) > 1:  # Need both classes for AUC
    auc = roc_auc_score(all_labels, probabilities)
else:
    auc = 0.0

execution_time = time.time() - start_time

# Prepare metrics JSON
metrics = {
    'TPR': float(tpr),
    'TNR': float(tnr),
    'Acc total': float(total_accuracy),
    'AUC': float(auc),
    'execution time': float(execution_time)
}

# Write metrics JSON
with open(metrics_filename, 'w') as f:
    json.dump(metrics, f, indent=2)

# Write individual image results JSON
with open(image_results_filename, 'w') as f:
    json.dump(image_results, f, indent=2)

print(f'\nMetrics saved to {metrics_filename}')
print(f'Image results saved to {image_results_filename}')
print(f'\nMetrics:')
print(f'  TPR: {tpr:.4f}')
print(f'  TNR: {tnr:.4f}')
print(f'  Accuracy: {total_accuracy:.4f}')
print(f'  AUC: {auc:.4f}')
print(f'  Execution time: {execution_time:.2f} seconds')


