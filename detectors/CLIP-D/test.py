"""
test.py — inference / evaluation entry point for CLIP-D.

Called by launcher.py via:
    python -u test.py --name <name> --arch <arch> --task test [args...]

Checkpoint lookup:
  --ft not set: checkpoint/<name>/weights/best.pt
  --ft set:     checkpoint/<name>/ft_weights/best.pt

Log file
────────
  logs/test_<name>_<dataset_dir>.log     (appended on each run)

Log content:
  • Run config (arch, weights path, data_root, data_keys)
  • Per-class breakdown (real count, fake count)
  • Final metrics: TPR, TNR, Accuracy, AUC, elapsed time
"""

import os
import time
import json
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

from networks import create_architecture, count_parameters
from utils.logger import create_logger
from utils.processing import add_processing_arguments
from parser import get_parser


def test(loader, model, output_dir, device, logger):

    # print("TESTING MODEL")
    # 

    model.eval()
    start_time = time.time()

    # ── Output directory ──────────────────────────────────────────────────
    # dataset_dir_name = settings.data_root.rstrip("/").split("/")[-1]
    # output_dir = os.path.join(
    #     # "/home/rz/TB_WP3/Image-Deepfake-Detectors-Public-Library/results",
    #     "./Image-Deepfake-Detectors-Public-Library/results",
    #     settings.name,
    #     dataset_dir_name,
    #     "CLIP-D",
    #     settings.data_keys,
    # )
    

    csv_filename           = os.path.join(output_dir, "results.csv")
    metrics_filename       = os.path.join(output_dir, "metrics.json")
    image_results_filename = os.path.join(output_dir, "image_results.json")

    logger.info(f"Results directory: {output_dir}")

    all_scores, all_labels, all_paths = [], [], []
    image_results = []

    with open(csv_filename, "w") as f:
        f.write("name,pro,flag\n")

    # ── Inference loop ────────────────────────────────────────────────────
    with torch.no_grad():
        with tqdm(loader, unit="batch", mininterval=0.5) as tbatch:
            tbatch.set_description("Test")
            for data_dict in tbatch:
                imgs   = data_dict["img"].to(device)
                labels = data_dict["target"].to(device)
                paths  = data_dict["path"]

                scores = model(imgs).squeeze(1)

                for score, label, path in zip(scores, labels, paths):
                    sv, lv = score.item(), label.item()
                    all_scores.append(sv)
                    all_labels.append(lv)
                    all_paths.append(path)
                    image_results.append({"path": path, "score": sv, "label": lv})

                with open(csv_filename, "a") as f:
                    for score, label, path in zip(scores, labels, paths):
                        f.write(f"{path},{score.item()},{label.item()}\n")

    # ── Metrics ───────────────────────────────────────────────────────────
    all_scores  = np.array(all_scores)
    all_labels  = np.array(all_labels)
    predictions = (all_scores > 0).astype(int)

    total_accuracy = accuracy_score(all_labels, predictions)

    fake_mask = all_labels == 1
    tpr = (accuracy_score(all_labels[fake_mask], predictions[fake_mask])
           if fake_mask.sum() > 0 else 0.0)

    real_mask = all_labels == 0
    tnr = (accuracy_score(all_labels[real_mask], predictions[real_mask])
           if real_mask.sum() > 0 else 0.0)

    if len(np.unique(all_labels)) > 1:
        probs = torch.sigmoid(torch.tensor(all_scores)).numpy()
        auc   = roc_auc_score(all_labels, probs)
    else:
        auc = 0.0

    elapsed = time.time() - start_time

    metrics = {
        "TPR":            float(tpr),
        "TNR":            float(tnr),
        "Acc total":      float(total_accuracy),
        "AUC":            float(auc),
        "execution time": float(elapsed),
    }

    # ── Write output files ────────────────────────────────────────────────
    with open(metrics_filename, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(image_results_filename, "w") as f:
        json.dump(image_results, f, indent=2)

    # ── Log results ───────────────────────────────────────────────────────
    n_real = int(real_mask.sum())
    n_fake = int(fake_mask.sum())
    n_tot  = len(all_labels)

    logger.info("-" * 60)
    logger.info(f"Dataset breakdown: total={n_tot}  real={n_real}  fake={n_fake}")
    logger.info(f"TPR  (acc on fake)  : {tpr:.4f}   ({int(tpr*n_fake)}/{n_fake} correct)")
    logger.info(f"TNR  (acc on real)  : {tnr:.4f}   ({int(tnr*n_real)}/{n_real} correct)")
    logger.info(f"Accuracy (overall)  : {total_accuracy:.4f}")
    logger.info(f"AUC                 : {auc:.4f}")
    logger.info(f"Elapsed             : {elapsed:.1f}s")
    logger.info(f"CSV                 : {csv_filename}")
    logger.info(f"Metrics JSON        : {metrics_filename}")
    logger.info("-" * 60)

    return metrics


if __name__ == "__main__":
    parser = get_parser()
    parser = add_processing_arguments(parser)
    settings = parser.parse_args()
    settings.task = "test"

    # ── Logger ────────────────────────────────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    dataset_name = settings.dataset.replace(os.sep, '_').replace('-', '_').replace('bw01', 'bw_BW01').replace('portait', 'portrait')
    print(f"dataset_name: {dataset_name}")
    # if settings.ft and settings.mlp:
    #     tag = 'ft_MLP'
    # elif settings.ft:
    #     tag = 'ft'
    # else:
    #     tag = 'pretrained'
    # result_folder = f'/second-disk/Image-Deepfake-Detectors-Public-Library/results/{settings.name}/{dataset_name}/CLIP-D_{tag}/'
    # log_path = os.path.join(result_folder,"logs")
    # logger   = create_logger(os.path.join(log_path, f"test_{settings.name}_{dataset_name}_{settings.data_keys}.log"))

    if settings.name == 'lora_r4_qv':
        settings.arch = 'opencliplinearloranext_clipL14commonpool_r4_qv'
    else:
        settings.arch = 'opencliplinearnext_clipL14commonpool'

    print(f'arch: {settings.arch}')
    # breakpoint()

    # ── Create output folder ──────────────────────────────────────────────────────────
    if settings.ft and settings.mlp:
        tag = 'ft_MLP'
    elif settings.ft:
        tag = 'ft'
    else:
        tag = 'pretrained'

    # dataset_dir_name = settings.data_root.split('/')[-1]  # Extract dataset directory name from path
    if settings.social:
        output_dir = f'/second-disk/Image-Deepfake-Detectors-Public-Library/results/{settings.name}_{settings.social}/{dataset_name}/CLIP-D_{tag}/{settings.data_keys}'
        logger_path = os.path.join(f'/second-disk/Image-Deepfake-Detectors-Public-Library/results/{settings.name}_{settings.social}/{dataset_name}/CLIP-D_{tag}/', 'test_log.log')
    else:
        output_dir = f'/second-disk/Image-Deepfake-Detectors-Public-Library/results/{settings.name}/{dataset_name}/CLIP-D_{tag}/{settings.data_keys}' # change path to be outside detector folder
        logger_path = os.path.join(f'/second-disk/Image-Deepfake-Detectors-Public-Library/results/{settings.name}/{dataset_name}/CLIP-D_{tag}/', 'test_log.txt')

    print(f"output_dir: {output_dir}")
    # 
    
    os.makedirs(output_dir, exist_ok=True)
    logger = create_logger(logger_path)

    # ── Log test header ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"TEST START  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"name       = {settings.name}")
    logger.info(f"arch       = {settings.arch}")
    logger.info(f"fine-tune  = {settings.ft}")
    logger.info(f"data_keys  = {settings.data_keys}")
    logger.info(f"data_root  = {settings.data_root}")
    logger.info(f"split_file = {settings.split_file}")
    logger.info(f"device     = {settings.device}")

    device = torch.device(settings.device if torch.cuda.is_available() else "cpu")

    # ── Dataset ───────────────────────────────────────────────────────────
    if settings.tf2k:
        from utils.tf2k_dataset import tf2k_create_dataloader
        test_dataloader = tf2k_create_dataloader(settings, split="test")
    else:
        from utils.dataset import create_dataloader
        test_dataloader = create_dataloader(settings, split="test")

    logger.info(f"test batches = {len(test_dataloader)}")

    # ── Model ─────────────────────────────────────────────────────────────
    # print(f"settings.arch: {settings.arch}")
    # 
    if settings.name == 'lora_r4_qv':
            settings.arch = 'opencliplinearloranext_clipL14commonpool_r4_qv'
    model = create_architecture(settings.arch, pretrained=True, num_classes=1)
    n_params = count_parameters(model)
    logger.info(f"arch trainable params = {n_params:,}")

    # ── Load checkpoint ───────────────────────────────────────────────────
    # load_path = (
    #     os.path.join("checkpoint", settings.name, "ft_weights",dataset_name, "best.pt")
    #     if settings.ft
    #     else os.path.join("checkpoint", settings.name, "weights",  "best.pt")
    # )

    # for testing model on the social dataset used for FT
    # if settings.ft and settings.social:
    #     load_path =  os.path.join("checkpoint", settings.name, "social", settings.social,"ft_weights", dataset_name, "best.pt")
    
    
    if settings.ft:
        load_path = os.path.join("checkpoint", settings.name, "ft_weights", dataset_name, "best.pt")
    else:
        load_path = os.path.join("checkpoint", settings.name, "weights",  "best.pt")
    # load_path = f'./checkpoint/{settings.name}/weights/best.pt' if not settings.ft else f'./checkpoint/{settings.name}/ft_weights/{settings.dataset.replace(os.sep, '_')}/best.pt'
    print('loading the model from %s' % load_path)

    # breakpoint()

    # logger.info(f"Loading weights from: {load_path}")

    # state = torch.load(load_path, map_location=device)
    # sd    = state["model"] if "model" in state else state
    
    # try:
    #     model.load_state_dict(sd, strict=True)
    #     logger.info("Weights loaded (strict=True)")
    # except RuntimeError as e:
    #     logger.warning(f"strict=True failed: {e}")
    #     missing, unexpected = model.load_state_dict(sd, strict=False)
    #     logger.warning(
    #         f"Loaded with strict=False — missing={len(missing)}, "
    #         f"unexpected={len(unexpected)}"
    #     )

    # print('loading the model from %s' % load_path)
    model.load_state_dict(torch.load(load_path, map_location=device, weights_only=True)['model'])
    model.to(device)

    

    # ── Run test ──────────────────────────────────────────────────────────
    metrics = test(test_dataloader, model, output_dir, device, logger)

    # ── Footer ────────────────────────────────────────────────────────────
    logger.info(f"TEST END  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file : {os.path.abspath(logger_path)}")
    logger.info("=" * 60)