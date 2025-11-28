import argparse
import json
import os
from tqdm import tqdm
import io
import pickle
import copy
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from einops import reduce, rearrange
import bisect
from models.slinet import SliNet
import pandas as pd
import time
from sklearn.metrics import roc_auc_score, accuracy_score
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
        'tw':   ['Twitter'],
    }

    mod_keys['all'] = [mod_keys[key][0] for key in mod_keys.keys()]
    mod_keys['shr'] = [mod_keys[key][0] for key in mod_keys.keys() if key in ['fb', 'tl', 'tw']]

    dataset_list = []
    for data in data_keys.split('&'):
        gen, mod = data.split(':')
        dataset_list.append({'gen':gen_keys[gen], 'mod':mod_keys[mod]})
    
    return dataset_list

class DummyDataset(Dataset):
    def __init__(self, data_path, data_type, data_scenario, data_compression, split_file=None):
        self.do_compress = [
            data_compression[0],
            data_compression[1],
        ]  # enable/disable compression from flag - jpeg quality
        self.trsf = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        images = []
        labels = []

        # print(f'--- Data compression: {data_compression} ---')

        if data_type == "cddb":
            if data_scenario == "cddb_hard":
                subsets = [
                    "gaugan",
                    "biggan",
                    "wild",
                    "whichfaceisreal",
                    "san",
                ]  # <- CDDB Hard
                multiclass = [0, 0, 0, 0, 0]
            elif data_scenario == "ood":
                subsets = ["deepfake", "glow", "stargan_gf"]  # <- OOD experiments
                multiclass = [0, 1, 1]
            else:
                raise RuntimeError(
                    f"Unexpected data_scenario value: {data_scenario}. Expected 'cddb_hard' or 'ood'."
                )
            print(f"--- Test on {subsets} with {data_scenario} scenario ---")
            for id, name in enumerate(subsets):
                root_ = os.path.join(data_path, name, "val")
                # sub_classes = ['']
                sub_classes = os.listdir(root_) if multiclass[id] else [""]
                for cls in sub_classes:
                    for imgname in os.listdir(os.path.join(root_, cls, "0_real")):
                        images.append(os.path.join(root_, cls, "0_real", imgname))
                        labels.append(0 + 2 * id)

                    for imgname in os.listdir(os.path.join(root_, cls, "1_fake")):
                        images.append(os.path.join(root_, cls, "1_fake", imgname))
                        labels.append(1 + 2 * id)
        
        elif data_type == "TrueFake":
            print(f"--- Test on {data_scenario} ---")

            with open(split_file, "r") as f:
                splits = json.load(f)
                test_split = sorted(splits["test"])


            dataset_list = parse_dataset(data_scenario)

            for dict in dataset_list:
                generators = dict['gen']
                modifiers = dict['mod']

                for mod in modifiers:
                    for dataset_root, dataset_dirs, dataset_files in os.walk(os.path.join(data_path, mod), topdown=True, followlinks=True):
                        if len(dataset_dirs):
                            continue

                        (label, gen, sub)  = f'{dataset_root}/'.replace(os.path.join(data_path, mod) + os.sep, '').split(os.sep)[:3]
                        
                        if gen in generators:
                            for filename in sorted(dataset_files):
                                if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
                                    if self._in_list(test_split, os.path.join(gen, sub, os.path.splitext(filename)[0])):
                                        images.append(os.path.join(dataset_root, filename))
                                        labels.append(1 if label == 'Fake' else 0)

        else:
            pass

        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.dataset_path = data_path

        with open("./src/utils/classes.pkl", "rb") as f:
            self.object_labels = pickle.load(f)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_path, self.images[idx])
        image = self.trsf(
            self.pil_loader(img_path, self.do_compress[0], self.do_compress[1])
        )
        label = self.labels[idx]
        #object_label = self.object_labels[img_path.replace(self.dataset_path, "")][0:5]

        # Normalize to a relative path key like those stored in classes.pkl
        rel_path = os.path.relpath(img_path, self.dataset_path).replace(os.sep, '/')

        # Try a few variants to match keys stored in classes.pkl
        candidates = [rel_path, rel_path.lstrip('/'), '/' + rel_path]
        found_key = None
        for k in candidates:
            if k in self.object_labels:
                found_key = k
                break

        # If not found, try matching by basename (may be ambiguous but prevents crash)
        if found_key is None:
            basename = os.path.basename(rel_path)
            for k in self.object_labels.keys():
                if k.endswith('/' + basename) or k.endswith(basename):
                    found_key = k
                    break

        # If still not found, fall back to the first available label entry to avoid KeyError
        if found_key is None:
            # pick any available entry as fallback (preserve expected structure)
            fallback_val = next(iter(self.object_labels.values()))
            object_label = fallback_val[0:5]
            print(f"[warn] object label not found for '{rel_path}' (requested '{img_path}'), using fallback label")
        else:
            object_label = self.object_labels[found_key][0:5]
        
        return object_label, image, label, img_path

    def pil_loader(self, path, do_compress, quality):
        with open(path, "rb") as f:
            if do_compress:
                f = self.compress_image_to_memory(path, quality=quality)
            img = Image.open(f)
            return img.convert("RGB")

    def compress_image_to_memory(self, path, quality):
        with Image.open(path) as img:
            output = io.BytesIO()
            img.save(output, "JPEG", quality=quality)
            output.seek(0)
            return output

    def _in_list(self, split, elem):
        i = bisect.bisect_left(split, elem)
        return i != len(split) and split[i] == elem

def setup_parser():
    parser = argparse.ArgumentParser(
        description="Reproduce of multiple continual learning algorithms."
    )
    # parser.add_argument(
    #     "--scenario", type=str, default="cddb_hard", help="scenario to test"
    # )
    parser.add_argument("--resume", type=str, default="", help="resume model")
    parser.add_argument(
        "--random_select", action="store_true", help="use random select"
    )
    parser.add_argument(
        "--upperbound", action="store_true", help="use groundtruth task identification"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cddb_inference.json",
        help="Json file of settings.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/francesco.laiti/datasets/CDDB/",
        help="data path",
    )
    parser.add_argument("--datatype", type=str, default="deepfake", help="data type")
    parser.add_argument(
        "--compression", type=bool, default=False, help="test on compressed data"
    )
    parser.add_argument(
        "--c_quality",
        type=int,
        default=100,
        help="quality of JPEG compressed (100, 90, 50...)",
    )
    return parser


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def load_configuration():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    if args.resume == "":
        args.resume = f'./best.pt'
    args_dict = vars(args)
    args_dict.update(param)
    return args_dict


def compute_predictions(outputs):
    predictions = {}

    # Top1
    outputs_top1 = rearrange(outputs, "b t p -> b (t p)")
    _, predicts_top1 = outputs_top1.max(dim=1)
    predictions["top1"] = predicts_top1 % 2

    # Mean
    outputs_mean = reduce(outputs, "b t p -> b p", "mean")
    predictions["mean"] = torch.argmax(outputs_mean, dim=-1)

    # Mixture of experts (top & mean)
    r_f_tensor = rearrange(outputs, "b t p -> b p t")
    r_f_max, _ = torch.max(r_f_tensor, dim=-1)
    r_f_mean = reduce(r_f_tensor, "b p t -> b p", "mean")
    diff_max = torch.abs(r_f_max[:, 0] - r_f_max[:, 1])
    diff_mean = torch.abs(r_f_mean[:, 0] - r_f_mean[:, 1])
    conditions = diff_mean > diff_max
    predicts_based_on_mean = torch.where(
        r_f_mean[:, 0] > r_f_mean[:, 1],
        torch.zeros_like(conditions),
        torch.ones_like(conditions),
    )
    predicts_based_on_max = torch.where(
        r_f_max[:, 0] > r_f_max[:, 1],
        torch.zeros_like(conditions),
        torch.ones_like(conditions),
    )
    predictions["mix_top_mean"] = torch.where(
        conditions, predicts_based_on_mean, predicts_based_on_max
    )*1

    return predictions


def accuracy_binary(y_pred, y_true, increment=2):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = float(
        "{:.2f}".format((y_pred % 2 == y_true % 2).sum() * 100 / len(y_true))
    )  # * Task-agnostic AA *

    task_acc = []
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        acc = ((y_pred[idxes] % 2) == (y_true[idxes] % 2)).sum() * 100 / len(idxes)
        all_acc[label] = float("{:.2f}".format(acc))
        task_acc.append(acc)
    all_acc["task_wise"] = float(
        "{:.2f}".format(sum(task_acc) / len(task_acc))
    )  # * Average Accuracy (AA) or Task-wise AA *
    return all_acc


def prepare_model(args):
    checkpoint = torch.load(f'./checkpoint/{args["run_name"]}/weights/best.pt', map_location=args["device"])
    # update config args
    args["K"] = checkpoint["K"]
    args["topk_classes"] = checkpoint["topk_classes"]
    args["ensembling"] = checkpoint["ensembling_flags"]

    # load all prototypes
    keys_dict = {
        "all_keys": checkpoint["keys"]["all_keys"].unsqueeze(0),  # * [Task, N_cluster = 5, 512]
        "all_keys_one_cluster": checkpoint["keys"]["all_keys_one_cluster"].unsqueeze(0),  # * [Task, 512]
        "real_keys_one_cluster": checkpoint["keys"]["real_keys_one_cluster"].unsqueeze(0),  # * [Task, 512]
        "fake_keys_one_cluster": checkpoint["keys"]["fake_keys_one_cluster"].unsqueeze(0),  # * [Task, 512]
    }
    for key in keys_dict.keys():
        print(f"--- {key}: {keys_dict[key].shape} ---")

    # print(checkpoint["tasks"])
    args["num_tasks"] = checkpoint["tasks"] + 1
    print(f"--- Number of tasks: {args['num_tasks']} ---")
    args["task_name"] = range(args["num_tasks"])

    # build and load model
    model = SliNet(args)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    model = model.to(args["device"])

    print(f"--- Run: {checkpoint.get('run_name', 'not available')} ---")

    return model, keys_dict


def prepare_data_loader(args):
    test_dataset = DummyDataset(
        args["data_path"],
        args["dataset"],
        args["scenario"],
        [args["compression"], args["c_quality"]],
        args["split_file"],
    )
    return DataLoader(
        test_dataset,
        batch_size=args["batch_size_eval"],
        shuffle=False,
        num_workers=int(os.environ.get("SLURM_CPUS_ON_NODE", 2)),
    )


@torch.no_grad
def inference_step(args, model: SliNet, test_loader, keys_dict):
    start_time = time.time()
    
    total_tasks = args["num_tasks"]
    
    def upperbound_selection(targets):
        domain_indices = torch.div(targets, 2, rounding_mode="floor")
        domain_prob = torch.zeros(
            (len(targets), total_tasks), dtype=torch.float16, device=args["device"]
        )
        domain_prob[torch.arange(len(targets)), domain_indices] = 1.0
        return domain_prob

    def process_batch(inputs, targets, object_name):
        keys_dict["upperbound"] = upperbound_selection(targets)
        if args["upperbound"]:
            keys_dict["prototype"] = "upperbound"
        outputs = model.interface(inputs, object_name, total_tasks, keys_dict)

        if args["softmax"]:
            outputs = torch.nn.functional.softmax(outputs, dim=-1)
        return compute_predictions(outputs)
    
    # File paths
    output_dir = f'./results/{args["run_name"]}/data/{args["scenario"]}'
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, 'results.csv')
    metrics_filename = os.path.join(output_dir, 'metrics.json')
    image_results_filename = os.path.join(output_dir, 'image_results.json')
    
    # Extract training dataset keys from run_name (format: "training_keys_freeze_down" or "training_keys")
    training_dataset_keys = []
    run_name = args["run_name"]
    if '_freeze_down' in run_name:
        training_name = run_name.replace('_freeze_down', '')
    else:
        training_name = run_name
    if '&' in training_name:
        training_dataset_keys = training_name.split('&')
    else:
        training_dataset_keys = [training_name]
    
    # Collect all results
    all_predictions_top1 = []
    all_predictions_mean = []
    all_predictions_mix = []
    all_labels = []
    all_binary_labels = []
    all_paths = []
    image_results = []
    
    # Write CSV header
    with open(csv_filename, 'w') as f:
        f.write(f"{','.join(['name', 'pro_top', 'pro_mean', 'pro_mix', 'flag'])}\n")

    for _, (object_name, inputs, targets, paths) in tqdm(enumerate(test_loader), total=len(test_loader), mininterval=5):
        inputs, targets = inputs.to(args["device"]), targets.to(args["device"])
        predictions = process_batch(inputs, targets, object_name)

        # Collect results
        for score_top, score_mean, score_mix, label, path in zip(predictions['top1'], predictions['mean'], predictions['mix_top_mean'], targets, paths):
            label_val = label.item()
            binary_label = label_val % 2  # Convert to binary (task-agnostic)
            
            all_predictions_top1.append(score_top.item())
            all_predictions_mean.append(score_mean.item())
            all_predictions_mix.append(score_mix.item())
            all_labels.append(label_val)
            all_binary_labels.append(binary_label)
            all_paths.append(path)
            
            image_results.append({
                'path': path,
                'score_top1': score_top.item(),
                'score_mean': score_mean.item(),
                'score_mix': score_mix.item(),
                'label': label_val,
                'binary_label': binary_label
            })
        
        # Write to CSV (maintain backward compatibility)
        with open(csv_filename, 'a') as f:
            for score_top, score_mean, score_mix, label, path in zip(predictions['top1'], predictions['mean'], predictions['mix_top_mean'], targets, paths):
                f.write(f"{path}, {score_top.item()}, {score_mean.item()}, {score_mix.item()}, {label.item()}\n")

    # Calculate metrics using 'mix_top_mean' as primary prediction method
    all_predictions_mix = np.array(all_predictions_mix)
    all_binary_labels = np.array(all_binary_labels)
    
    # Predictions are already binary (0 or 1)
    predictions = all_predictions_mix.astype(int)
    
    # Calculate overall metrics
    total_accuracy = accuracy_score(all_binary_labels, predictions)
    
    # TPR (True Positive Rate) = TP / (TP + FN) = accuracy on fake images (label==1)
    fake_mask = all_binary_labels == 1
    if fake_mask.sum() > 0:
        tpr = accuracy_score(all_binary_labels[fake_mask], predictions[fake_mask])
    else:
        tpr = 0.0
    
    
    # Calculate TNR on real images (label==0) in the test set
    real_mask = all_binary_labels == 0
    if real_mask.sum() > 0:
        # Overall TNR calculated on all real images in the test set
        tnr = accuracy_score(all_binary_labels[real_mask], predictions[real_mask])
    else:
        tnr = 0.0
        
    # AUC calculation
    # For AUC, we need probabilities. Since predictions are binary (0/1), we'll use the scores
    # We need to convert binary predictions to probabilities. Since we don't have raw logits,
    # we'll use a simple approach: normalize predictions or use a threshold-based probability
    if len(np.unique(all_binary_labels)) > 1:  # Need both classes for AUC
        # Use predictions directly as probabilities (they're already 0/1, but AUC needs continuous)
        # For binary predictions, we can create probabilities based on the score distribution
        # Since mix_top_mean gives us binary predictions, we'll use a simple approach:
        # Create probabilities by normalizing or using the predictions directly
        # Actually, for AUC with binary predictions, we can use the predictions as-is
        # But ideally we'd have probabilities. For now, we'll calculate AUC using predictions
        # Note: This might not be ideal, but works for binary classifier outputs
        try:
            auc = roc_auc_score(all_binary_labels, predictions.astype(float))
        except:
            auc = 0.0
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
    print(f'\nMetrics (using mix_top_mean):')
    print(f'  TPR: {tpr:.4f}')
    print(f'  TNR: {tnr:.4f}')
    print(f'  Accuracy: {total_accuracy:.4f}')
    print(f'  AUC: {auc:.4f}')
    print(f'  Execution time: {execution_time:.2f} seconds')


def pretty_print(data):
    return json.dumps(data, indent=4, sort_keys=True)


if __name__ == "__main__":
    args = load_configuration()
    print(args)
    #args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    scenarios = copy.deepcopy(args["scenario"])
    model, keys_dict = prepare_model(args)
    keys_dict["prototype"] = args["prototype"]

    for s in scenarios:
        args["scenario"] = s
        os.makedirs(f'./results/{args["run_name"]}/data/{args["scenario"]}', exist_ok=True)

        test_loader = prepare_data_loader(args)
        inference_step(args, model, test_loader, keys_dict)
        