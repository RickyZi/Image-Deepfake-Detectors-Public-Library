import os
from tqdm import tqdm
import torch
import pandas as pd
import json
import time
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from networks import create_architecture, count_parameters
from utils.dataset import create_dataloader
from utils.processing import add_processing_arguments
from parser import get_parser

def test(loader, model, settings, device):
    model.eval()
    
    start_time = time.time()
    
    # File paths
    output_dir = f'./results/{settings.name}/{settings.data_keys}/data/'
    os.makedirs(output_dir, exist_ok=True)
    
    csv_filename = os.path.join(output_dir, 'results.csv')
    metrics_filename = os.path.join(output_dir, 'metrics.json')
    image_results_filename = os.path.join(output_dir, 'image_results.json')
    
    # Collect all results
    all_scores = []
    all_labels = []
    all_paths = []
    image_results = []
    
    # Extract training dataset keys from model name (format: "training_keys_freeze_down" or "training_keys")
    # The model name typically contains the training dataset keys used for training
    training_dataset_keys = []
    model_name = settings.name
    # Remove common suffixes like "_freeze_down"
    if '_freeze_down' in model_name:
        training_name = model_name.replace('_freeze_down', '')
    else:
        training_name = model_name
    # Split by & to get individual training dataset keys
    if '&' in training_name:
        training_dataset_keys = training_name.split('&')
    else:
        training_dataset_keys = [training_name]
    
    # Write CSV header
    with open(csv_filename, 'w') as f:
        f.write(f"{','.join(['name', 'pro', 'flag'])}\n")
    
    with torch.no_grad():
        with tqdm(loader, unit='batch', mininterval=0.5) as tbatch:
            tbatch.set_description(f'Validation')
            for data_dict in tbatch:
                data = data_dict['img'].to(device)
                labels = data_dict['target'].to(device)
                paths = data_dict['path']

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
    
    # Convert scores to predictions (threshold at 0, as used in train.py)
    predictions = (all_scores > 0).astype(int)
    
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
    
    # AUC calculation (needs probabilities, so we'll use sigmoid on scores)
    if len(np.unique(all_labels)) > 1:  
        # Apply sigmoid to convert scores to probabilities
        probabilities = torch.sigmoid(torch.tensor(all_scores)).numpy()
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

if __name__ == '__main__':
    parser = get_parser()
    parser = add_processing_arguments(parser)
    settings = parser.parse_args()
    
    device = torch.device(settings.device if torch.cuda.is_available() else 'cpu')

    test_dataloader = create_dataloader(settings, split='test')

    model = create_architecture(settings.arch, pretrained=True, num_classes=1).to(device)
    num_parameters = count_parameters(model)
    print(f"Arch: {settings.arch} with #parameters {num_parameters}")
    load_path = f'./checkpoint/{settings.name}/weights/best.pt'
    
    print('loading the model from %s' % load_path)
    model.load_state_dict(torch.load(load_path, map_location=device)['model'])
    model.to(device)

    test(test_dataloader, model, settings, device)
