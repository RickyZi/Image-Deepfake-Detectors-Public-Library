"""
infer_r50nodown.py — score a test set with a specific R50_nodown FT
checkpoint, for use as one leg of the C2PA-routed ensemble.

Must run with cwd set to the R50_nodown detector directory, so its local
`networks`, `utils`, `parser` modules resolve normally - this is exactly
what test.py does. This script skips metrics/CSV output and instead writes
raw per-image probabilities as JSON, so an orchestrator can combine them
with other detectors' scores for the same images.

Run via subprocess, not imported directly into a process that also
imports R50_TF's or CLIP-D's same-named `networks`/`utils`/`parser`
modules - see run_ensemble_test.py's docstring for why that's unsafe
(sys.modules caches by bare module name, so importing more than one
same-named module into one process silently gives the wrong one to
whichever detector's code imports it second).

Mirrors test.py's exact checkpoint path construction, including its
dataset-name sanitization (bw01 -> bw_BW01, portait -> portrait, - -> _)
so it finds the same checkpoint file test.py would.

Usage:
    python infer_r50nodown.py \\
        --name pretrained --dataset seasons/autumn_TM01 --ft \\
        --data_root /path/to/preset/images --split_file /path/to/split.json \\
        --data_keys "..." --tf2k True --device cuda:0 --out scores_r50nd.json
"""
import os
import json
import torch
from tqdm import tqdm

from networks import create_architecture
from utils.dataset import create_dataloader
from utils.tf2k_dataset import tf2k_create_dataloader
from utils.processing import add_processing_arguments
from parser import get_parser


def build_load_path(settings):
    dataset_dir_name = (settings.dataset.replace(os.sep, '_')
                         .replace('-', '_')
                         .replace('bw01', 'bw_BW01')
                         .replace('portait', 'portrait'))
    if settings.ft and settings.r50unfreezeL4:
        return f'./checkpoint/{settings.name}/ft_unfreezeL4_weights/{dataset_dir_name}_unfreezeL4/best.pt'
    elif settings.ft:
        return f'./checkpoint/{settings.name}/ft_weights/{dataset_dir_name}/best.pt'
    else:
        return f'./checkpoint/{settings.name}/weights/best.pt'


if __name__ == '__main__':
    # Tell PyTorch's CUDA allocator to use expandable segments — prevents
    # fragmentation-induced OOM on the T4 (15.9 GiB) where the allocator
    # sometimes can't find a contiguous block even when total free > request.
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    parser = get_parser()
    parser = add_processing_arguments(parser)   # registers --blur_sig, --blur_prob,
                                                # --cmp_qual, --jitter_prob, etc. so
                                                # processing.py's parse_arguments()
                                                # finds them on settings rather than
                                                # raising AttributeError
    parser.add_argument('--out', type=str, required=True,
                         help='Path to write per-image scores as JSON')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'],
                         help="Which split to score - 'val' when fitting a "
                              "combination rule, 'test' for final evaluation")
    settings, _ = parser.parse_known_args()

    # Mirror the flags test.py sets before building the dataloader so the
    # dataset class applies the correct (test-time) transforms: center crop
    # instead of random crop, no horizontal flip, resize active.
    # Without these, isTrain is undefined and no_crop/no_resize may be
    # missing from the namespace, causing wrong transforms or AttributeError.
    settings.isTrain        = False
    settings.task           = 'test'   # controls make_processing — without this
                                       # the training augmentation path is used,
                                       # producing large random-cropped tensors
                                       # instead of the 96×96 center-crop
    settings.no_crop        = False   # use CenterCrop(cropSize), not Identity
    settings.no_resize      = False   # apply Resize(loadSize)
    settings.serial_batches = True    # no shuffling at test time

    device = torch.device(settings.device if torch.cuda.is_available() else 'cpu')

    if settings.tf2k:
        loader = tf2k_create_dataloader(settings, split=settings.split)
    else:
        loader = create_dataloader(settings, split=settings.split)

    model = create_architecture(settings.arch, pretrained=True, num_classes=1).to(device)
    load_path = build_load_path(settings)
    print(f'[R50_nodown] loading {load_path} (split={settings.split})')
    model.load_state_dict(torch.load(load_path, map_location=device)['model'])
    model.eval()

    results = []
    with torch.no_grad():
        for data_dict in tqdm(loader, desc='R50_nodown inference'):
            data = data_dict['img'].to(device)
            labels = data_dict['target']
            paths = data_dict['path']

            scores = model(data).squeeze(1)
            probs = torch.sigmoid(scores)

            for prob, label, path in zip(probs.cpu().tolist(), labels.tolist(), paths):
                results.append({'path': path, 'score': float(prob), 'label': float(label)})

    with open(settings.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[R50_nodown] wrote {len(results)} scores to {settings.out}')