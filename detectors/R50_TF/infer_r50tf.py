"""
infer_r50tf.py — score a test set with a specific R50_TF FT checkpoint,
for use as one leg of the C2PA-routed ensemble.

Must run with cwd set to the R50_TF detector directory - see
infer_r50nodown.py's docstring and run_ensemble_test.py's docstring for
why this needs to be an isolated subprocess rather than a direct import
alongside the other detectors.

Mirrors test_r50tf.py's exact checkpoint path construction, including its
dataset-name sanitization. Note the unfreezeL4 suffix differs from
R50_nodown's ('_ft_unfreezeL4' here vs '_unfreezeL4' there) - this is not
a typo, it's matching what's actually on disk for each detector.

Usage:
    python infer_r50tf.py \\
        --name pretrained --dataset seasons/autumn_TM01 --ft \\
        --data_root /path/to/preset/images --split_file /path/to/split.json \\
        --data_keys "..." --tf2k True --device cuda:0 --out scores_r50tf.json
"""
import os
import json
import torch
from tqdm import tqdm

from networks import ImageClassifier
from utils.dataset import create_dataloader
from utils.tf2k_dataset import tf2k_create_dataloader
from utils.processing import add_processing_arguments
from parser import get_parser


def build_load_path(settings):
    dataset_name = (settings.dataset.replace(os.sep, '_')
                     .replace('-', '_')
                     .replace('bw01', 'bw_BW01')
                     .replace('portait', 'portrait'))
    if settings.ft and settings.r50unfreezeL4:
        return f'./checkpoint/{settings.name}/ft_unfreezeL4_weights/{dataset_name}_ft_unfreezeL4/best.pt'
    elif settings.ft:
        return f'./checkpoint/{settings.name}/ft_weights/{dataset_name}/best.pt'
    else:
        return f'./checkpoint/{settings.name}/weights/best.pt'


if __name__ == '__main__':
    parser = get_parser()
    parser = add_processing_arguments(parser)
    parser.add_argument('--out', type=str, required=True,
                         help='Path to write per-image scores as JSON')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'],
                         help="Which split to score - 'val' when fitting a "
                              "combination rule, 'test' for final evaluation")
    settings, _ = parser.parse_known_args()

    device = torch.device(settings.device if torch.cuda.is_available() else 'cpu')

    if settings.tf2k:
        loader = tf2k_create_dataloader(settings, split=settings.split)
    else:
        loader = create_dataloader(settings, split=settings.split)

    model = ImageClassifier(settings)
    model.to(device)

    load_path = build_load_path(settings)
    print(f'[R50_TF] loading {load_path} (split={settings.split})')
    state_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    results = []
    with torch.no_grad():
        for (data, labels, paths) in tqdm(loader, desc='R50_TF inference'):
            data = data.to(device)

            scores = model(data).squeeze(1)
            probs = torch.sigmoid(scores)

            for prob, label, path in zip(probs.cpu().tolist(), labels.tolist(), paths):
                results.append({'path': path, 'score': float(prob), 'label': float(label)})

    with open(settings.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[R50_TF] wrote {len(results)} scores to {settings.out}')