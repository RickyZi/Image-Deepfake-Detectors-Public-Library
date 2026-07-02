import os
from tqdm import tqdm
import torch
import pandas as pd
import json
import time
from datetime import datetime
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from networks import create_architecture, count_parameters
from utils.dataset import create_dataloader
from utils.tf2k_dataset import tf2k_create_dataloader
from utils.processing import add_processing_arguments
from parser import get_parser
from networks.openclipnet import MLPHead

from utils.logger import create_logger

def test(loader, model, settings, device):
    model.eval()
    
    start_time = time.time()

    # --------------------------- #
    # File paths update
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # add timestamp to test run
    # output_dir = f'./results/{settings.name}/data_{timestamp}/{settings.data_keys}'
    # tag = 'ft' if settings.ft else 'pretrained'
    if settings.ft and settings.mlp:
        tag = 'ft_MLP'
    elif settings.ft:
        tag = 'ft'
    else:
        tag = 'pretrained'

    dataset_dir_name = settings.data_root.split('/')[-1]  # Extract dataset directory name from path

    if any(sub in str(settings.data_root) for sub in ['Facebook', 'Telegram', 'Twitter']):
        output_dir = f'/home/rz/TB_WP3/Image-Deepfake-Detectors-Public-Library/results/{settings.name}_social/{dataset_dir_name}/CLIP-D_{tag}/{settings.data_keys}'
        logger = create_logger(os.path.join(f'/home/rz/TB_WP3/Image-Deepfake-Detectors-Public-Library/results/{settings.name}_social/{dataset_dir_name}/CLIP-D_{tag}/', 'test.log'))
    else:
        output_dir = f'/home/rz/TB_WP3/Image-Deepfake-Detectors-Public-Library/results/{settings.name}/{dataset_dir_name}/CLIP-D_{tag}/{settings.data_keys}' # change path to be outside detector folder
        logger = create_logger(os.path.join(f'/home/rz/TB_WP3/Image-Deepfake-Detectors-Public-Library/results/{settings.name}/{dataset_dir_name}/CLIP-D_{tag}/', 'test.log'))
    
    os.makedirs(output_dir, exist_ok=True)
    # --------------------------- #

    # logger = create_logger(os.path.join(output_dir, 'test.log'))
    
    csv_filename = os.path.join(output_dir, 'results.csv')
    metrics_filename = os.path.join(output_dir, 'metrics.json')
    image_results_filename = os.path.join(output_dir, 'image_results.json')
    
    # Collect all results
    all_scores = []
    all_labels = []
    all_paths = []
    image_results = []
    
    # Extract training dataset keys from model name (format: "training_keys_freeze_down" or "training_keys")
    training_dataset_keys = []
    model_name = settings.name
    if '_freeze_down' in model_name:
        training_name = model_name.replace('_freeze_down', '')
    else:
        training_name = model_name
    if '&' in training_name:
        training_dataset_keys = training_name.split('&')
    else:
        training_dataset_keys = [training_name]
    
    # Write CSV header
    with open(csv_filename, 'w') as f:
        f.write(f"{','.join(['name', 'pro', 'flag'])}\n")
    

    # log info on the test run
    logger.info(f"Model name: {settings.name}")
    logger.info(f"Test dataset: {settings.data_keys}")
    logger.info(f"Test dataset keys: {settings.data_keys.split('&')}")

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
    
    # Convert scores to predictions (threshold at 0, as used in train.py: y_pred > 0.0)
    predictions = (all_scores > 0).astype(int)
    
    # Calculate overall metrics
    total_accuracy = accuracy_score(all_labels, predictions)
    
    # TPR (True Positive Rate) = TP / (TP + FN) = accuracy on fake images (label==1)
    fake_mask = all_labels == 1
    if fake_mask.sum() > 0:
        tpr = accuracy_score(all_labels[fake_mask], predictions[fake_mask])
    else:
        tpr = 0.0
    
    # TNR per dataset key (True Negative Rate) = TN / (TN + FP) = accuracy on real images (label==0)
    tnr_per_dataset = {}
    
    # Calculate TNR on real images (label==0) in the test set
    real_mask = all_labels == 0
    if real_mask.sum() > 0:
        # Overall TNR calculated on all real images in the test set
        tnr = accuracy_score(all_labels[real_mask], predictions[real_mask])
    else:
        tnr = 0.0
        
        # Map TNR to training dataset keys (as shown in the example JSON structure)
        # The TNR is calculated on the test set, but organized by training dataset keys
        #for training_key in training_dataset_keys:
        #    tnr_per_dataset[training_key] = overall_tnr
    
    # AUC calculation (needs probabilities, so we'll use sigmoid on scores)
    if len(np.unique(all_labels)) > 1:  # Need both classes for AUC
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
    # breakpoint()

    # log results
    logger.info(f"Metrics saved to {metrics_filename}")
    logger.info(f"Image results saved to {image_results_filename}")
    logger.info(f"Metrics:")
    logger.info(f"  TPR: {tpr:.4f}")
    logger.info(f"  TNR: {tnr:.4f}")
    logger.info(f"  Accuracy: {total_accuracy:.4f}")
    logger.info(f"  AUC: {auc:.4f}")
    logger.info(f"  Execution time: {execution_time:.2f} seconds")

if __name__ == '__main__':
    parser = get_parser()
    parser = add_processing_arguments(parser)
    settings = parser.parse_args()
    
    device = torch.device(settings.device if torch.cuda.is_available() else 'cpu')
    
    if settings.tf2k:
        test_dataloader = tf2k_create_dataloader(settings, split = 'test')
    else:
        test_dataloader = create_dataloader(settings, split='test')

    model = create_architecture(settings.arch, pretrained=True, num_classes=1).to(device)

    if settings.ft and settings.mlp:
        in_features = model.num_features
        hidden_dim  = settings.mlp_hidden # 256
        dropout     = settings.mlp_dropout # 0.3
        print(f"[test] Replacing fc with MLPHead (in={in_features}, hidden={hidden_dim}, dropout={dropout})")
        model.fc = MLPHead(
            in_features=in_features,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_classes=1,
        ).to(device)

    num_parameters = count_parameters(model)
    print(f"Arch: {settings.arch} with #parameters {num_parameters}")

    # breakpoint()
    
    # load_path = f'./checkpoint/{settings.name}/weights/best.pt'
    # load_path = f'./checkpoint/{settings.name}/weights/best.pt' if not settings.ft else f'./checkpoint/{settings.name}/ft_weights/{settings.dataset.replace(os.sep, '_')}/best.pt'
    if settings.ft and settings.mlp:
        load_path = f'./checkpoint/{settings.name}/ft_MLP_weights/{settings.dataset.replace(os.sep, '_')}/best.pt'
    elif settings.ft:
       load_path = f'./checkpoint/{settings.name}/ft_weights/{settings.dataset.replace(os.sep, '_')}/best.pt'
    else:
       load_path = f'./checkpoint/{settings.name}/weights/best.pt'
    
    print('loading the model from %s' % load_path)
    # breakpoint()
    model.load_state_dict(torch.load(load_path, map_location=device)['model'])
    model.to(device)
    
    # breakpoint()

    test(test_dataloader, model, settings, device)
