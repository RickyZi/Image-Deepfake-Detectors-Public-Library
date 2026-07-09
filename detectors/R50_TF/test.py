# ----------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------
import os
import torch
import pandas as pd
from tqdm import tqdm
import json
import time
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

from networks import ImageClassifier
from parser import get_parser
from utils.dataset import create_dataloader
from utils.tf2k_dataset import tf2k_create_dataloader

from datetime import datetime

from utils.logger import create_logger

def test(loader, model, settings, device):
    model.eval()
    
    start_time = time.time()
    
    # # File paths
    # output_dir = f'./results/{settings.name}/{settings.data_keys}/data/'
    # os.makedirs(output_dir, exist_ok=True)

    # --------------------------- #
    # # File paths update
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") #
    # # output_dir = f'./results/{settings.name}/data_{timestamp}/{settings.data_keys}'
    # dataset_dir_name = settings.data_root.split('/')[-1]  # Extract dataset directory name from path
    # tag = 'ft' if settings.ft else 'pretrained'
    # output_dir = f'/home/rz/TB_WP3/Image-Deepfake-Detectors-Public-Library/results/{settings.name}/{dataset_dir_name}/R50_TF_{tag}/{settings.data_keys}' # change path to be outside detector folder
    # os.makedirs(output_dir, exist_ok=True)
    dataset_dir_name = settings.dataset.replace(os.sep, '_')
    # settings.data_root.split('/')[-1]  # Extract dataset directory name from path
    print(f"dataset_name: {dataset_dir_name}")
    tag = 'ft' if settings.ft else 'pretrained'
    tag += '_unfreezeL4' if settings.r50unfreezeL4 else ''
    print(f"test_tag: {tag}")
    # breakpoint()
    # if dataset_dir_name in ['Facebook', 'Telegram', 'Twitter']:
    if settings.social:
        output_dir = f'/home/rz/TB_WP3/Image-Deepfake-Detectors-Public-Library/results/{settings.social}/{dataset_dir_name}/R50_TF_{tag}/{settings.data_keys}'
        logger_path = os.path.join(f'/home/rz/TB_WP3/Image-Deepfake-Detectors-Public-Library/results/{settings.social}/{dataset_dir_name}/R50_TF_{tag}/', 'test_log.log')
    else:
        output_dir = f'/home/rz/TB_WP3/Image-Deepfake-Detectors-Public-Library/results/{settings.name}/{dataset_dir_name}/R50_TF_{tag}/{settings.data_keys}' # change path to be outside detector folder
        logger_path = os.path.join(f'/home/rz/TB_WP3/Image-Deepfake-Detectors-Public-Library/results/{settings.name}/{dataset_dir_name}/R50_TF_{tag}/', 'test_log.txt')
    os.makedirs(output_dir, exist_ok=True)
    # --------------------------- #

    logger = create_logger(logger_path)
    
    csv_filename = os.path.join(output_dir, 'results.csv')
    metrics_filename = os.path.join(output_dir, 'metrics.json')
    image_results_filename = os.path.join(output_dir, 'image_results.json')
    
    # Collect all results
    all_scores = []
    all_labels = []
    all_paths = []
    image_results = []
    
    # Parse dataset keys from settings.data_keys (format: "key1&key2&..." or single "key")
    dataset_keys = settings.data_keys.split('&') if '&' in settings.data_keys else [settings.data_keys]
    
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
    

    # add info on the testing set
    logger.info("=== Test settings ===")
    
    logger.info(f"Model name: {settings.name}_{tag}")
    logger.info(f"Dataset: {dataset_dir_name}")
    logger.info(f"Dataset keys: {dataset_keys}")
    logger.info(f"Training dataset keys: {training_dataset_keys}")

    with torch.no_grad():
        with tqdm(loader, unit='batch', mininterval=0.5) as tbatch:
            tbatch.set_description(f'Validation')
            for (data, labels, paths) in tbatch:
                data = data.to(device)
                labels = labels.to(device)

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

    logger.info(f'\nMetrics saved to {metrics_filename}')
    logger.info(f'Image results saved to {image_results_filename}')
    logger.info(f'\nMetrics:')
    logger.info(f'  TPR: {tpr:.4f}')
    logger.info(f'  TNR: {tnr:.4f}')
    logger.info(f'  Accuracy: {total_accuracy:.4f}')
    logger.info(f'  AUC: {auc:.4f}')
    logger.info(f'  Execution time: {execution_time:.2f} seconds')

if __name__ == "__main__":
    parser = get_parser()
    settings = parser.parse_args()
    
    device = torch.device(settings.device if torch.cuda.is_available() else 'cpu')

    if settings.tf2k:
        test_dataloader = tf2k_create_dataloader(settings, split='test')
    else:
        test_dataloader = create_dataloader(settings, split='test')

    model = ImageClassifier(settings)
    # breakpoint()
    model.to(device)
    # fix load path!!!!
    if settings.ft and settings.r50unfreezeL4 and settings.social:
        load_path = f'./checkpoint/{settings.name}/social/{settings.social}/ft_unfreezeL4_weights/{settings.dataset.replace(os.sep, '_')}_ft_unfreezeL4/best.pt'
    elif settings.ft and settings.r50unfreezeL4:
        load_path = f'./checkpoint/{settings.name}/ft_unfreezeL4_weights/{settings.dataset.replace(os.sep, '_')}_ft_unfreezeL4/best.pt'
    elif settings.ft:
        load_path = f'./checkpoint/{settings.name}/ft_weights/{settings.dataset.replace(os.sep, '_')}/best.pt'
    else:
        load_path = f'./checkpoint/{settings.name}/weights/best.pt'
    # load_path = f'./checkpoint/{settings.name}/weights/best.pt' if not settings.ft else f'./checkpoint/{settings.name}/ft_weights/{settings.dataset.replace(os.sep, '_')}/best.pt'
    print('loading the model from %s' % load_path)
    # breakpoint()
    # path_weight = f'./checkpoint/{settings.name}/weights/best.pt' 
    state_dict = torch.load(load_path, map_location=device)
    # breakpoint()
    # RuntimeError: Attempting to deserialize object on CUDA device 1 but torch.cuda.device_count() is 1. 
    # Please use torch.load with map_location to map your storages to an existing device.
    model.load_state_dict(state_dict)
    test(test_dataloader, model, settings, device)