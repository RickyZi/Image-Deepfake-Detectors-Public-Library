'''
Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import os
import torch
import numpy as np
import tqdm
from networks import create_architecture, count_parameters

def _get_visual(inner_model):
    # LoRA variant -> properly registered attribute
    if hasattr(inner_model, "backbone"):
        return getattr(inner_model.backbone, "visual", None)
    # Original frozen variant -> stored as a list
    if hasattr(inner_model, "bb") and isinstance(inner_model.bb, list) and inner_model.bb:
        return getattr(inner_model.bb[0], "visual", None)
    return None

def freeze_clip_backbone(ft_model, unfreeze_last_n_blocks=2):
    inner = ft_model.model   # OpenClipLinear 

    for param in inner.parameters():
        param.requires_grad_(False)

    visual = _get_visual(inner)
    if visual is None:
        raise RuntimeError(
            "freeze_clip_backbone: could not locate the visual encoder inside "
            f"model.model ({type(inner).__name__}). "
            "Inspect model.model and adjust _get_visual()."
        )

    resblocks    = visual.transformer.resblocks
    total_blocks = len(resblocks)
    first_open   = total_blocks - unfreeze_last_n_blocks

    print(f"[freeze] CLIP visual encoder: {total_blocks} transformer blocks total")
    print(f"[freeze] Unfreezing blocks {first_open}–{total_blocks - 1} "
          f"({unfreeze_last_n_blocks} blocks)")

    for i, block in enumerate(resblocks):
        if i >= first_open:
            for param in block.parameters():
                param.requires_grad_(True)

    for attr in ("ln_post", "proj"):
        obj = getattr(visual, attr, None)
        if obj is None:
            continue
        if isinstance(obj, torch.Tensor):   # proj is a raw Parameter in open_clip
            obj.requires_grad_(True)
        else:
            for param in obj.parameters():
                param.requires_grad_(True)
        print(f"[freeze] Unfreezing visual.{attr}")

    for head_attr in ("head", "classifier", "fc", "linear"):
        head = getattr(inner, head_attr, None)
        if head is not None:
            for param in head.parameters():
                param.requires_grad_(True)
            print(f"[freeze] Unfreezing head: model.model.{head_attr}")
            break

    trainable = sum(p.numel() for p in inner.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in inner.parameters())
    print(f"[freeze] Trainable: {trainable:,} / {total:,}  ({100 * trainable / total:.2f} %)")


class FTModel(torch.nn.Module):
    """
    Fine-tuning wrapper.  Behaviour depends on --arch and --ft:
    - arch starts with 'opencliplinearloranext_'  (LoRA mode)      
        - OpenClipLinearLoRA freezes backbone and injects adapters 
        inside __init__, so no extra freeze step is needed here 
        - Only LoRA A/B matrices and the fc head are trained

    - arch starts with 'opencliplinearnext_'   
        - OpenClipLinear builds with a fully-frozen backbone     
        - freeze_clip_backbone() unfreezes the last N blocks                         
    
    Checkpoint saved to: checkpoint/<name>/ft_weights/   

    Checkpoint format (same for both modes):
        {'model': state_dict, 'optimizer': state_dict, 'total_steps': int}
    """

    def __init__(self, opt):
        super().__init__()

        self.opt         = opt
        self.total_steps = 0
        self.device      = torch.device(
            opt.device if torch.cuda.is_available() else "cpu"
        )

        # Detect fine-tuning mode
        if opt.name == 'lora_r4_qv':
            opt.arch = 'opencliplinearloranext_clipL14commonpool_r4_qv'
        self._is_lora = opt.arch.startswith("opencliplinearloranext_")

        
        dataset_name = opt.dataset.replace(os.sep, '_')
        print(f"dataset_name: {dataset_name}")
        
        if opt.ft and opt.social:
            self.save_dir = os.path.join("checkpoint", opt.name, "social", opt.social , "ft_weights", dataset_name)

        elif opt.ft:
            self.save_dir = os.path.join("checkpoint", opt.name, "ft_weights" , dataset_name)
        else:
            self.save_dir = os.path.join("checkpoint", opt.name, "weights" , dataset_name)
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"save_dir: {self.save_dir}")
        
        self.model = create_architecture(opt.arch, pretrained=True, num_classes=1)
        print(f"[FTModel] Arch: {opt.arch}")
        print(f"[FTModel] Initial trainable params: {count_parameters(self.model):,}")

        if opt.ft:
            load_path = os.path.join("checkpoint", "pretrained", "weights", "best.pt")
            if os.path.isfile(load_path):
                print(f"[FTModel] Loading from {load_path}")
                state = torch.load(load_path, map_location=self.device, weights_only=True)
                sd    = state["model"] if "model" in state else state

                try:
                    self.model.load_state_dict(sd, strict=True)
                    print("[FTModel] Weights loaded (strict=True)")
                except RuntimeError as e:
                    print(f"[FTModel] strict=True failed ({e}).")
                    print("[FTModel] Retrying with strict=False "
                          "(LoRA matrices will use random init).")
                    miss, unex = self.model.load_state_dict(sd, strict=False)
                    print(f"[FTModel]   missing={len(miss)}, unexpected={len(unex)}")

                self.total_steps = (
                    state.get("total_steps", 0)
                    if isinstance(state, dict) else 0
                )
            else:
                print(f"[FTModel] No checkpoint at {load_path}. "
                      "Starting from CLIP pretrained weights.")

        fc_norm_after = self.model.fc.weight.norm().item()

        if opt.ft and not self._is_lora:
            freeze_clip_backbone(self, unfreeze_last_n_blocks=2)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        print(f"[FTModel] Params passed to optimizer: "
              f"{sum(p.numel() for p in trainable):,}")

        self.loss_fn   = torch.nn.BCEWithLogitsLoss().to(self.device)
        self.optimizer = torch.optim.Adam(
            trainable,
            lr=opt.lr,
            betas=(opt.beta1, 0.999),
            weight_decay=opt.weight_decay,
        )

        self.model.to(self.device)

    def adjust_learning_rate(self, min_lr=1e-6):
        for pg in self.optimizer.param_groups:
            pg["lr"] /= 10.0
            if pg["lr"] < min_lr:
                return False
        return True

    def get_learning_rate(self):
        for pg in self.optimizer.param_groups:
            return pg["lr"]

    def train_on_batch(self, data):
        self.total_steps += 1
        self.model.train()

        inp   = data["img"].to(self.device)
        label = data["target"].to(self.device).float()
        out   = self.model(inp)

        if len(out.shape) == 4:
            ss   = out.shape
            loss = self.loss_fn(
                out,
                label[:, None, None, None].repeat(1, ss[1], ss[2], ss[3]),
            )
        else:
            loss = self.loss_fn(out.squeeze(1), label)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.cpu()

    def predict(self, data_loader):
            self.model.eval()
            y_true, y_pred, y_path = [], [], []
            with torch.no_grad():
                for data in tqdm.tqdm(data_loader):
                    img   = data["img"]
                    label = data["target"].cpu().numpy()
                    paths = list(data["path"])
                    out   = self.model(img.to(self.device)).cpu().numpy()[:, -1]
                    assert label.shape == out.shape
                    y_pred.extend(out.tolist())
                    y_true.extend(label.tolist())
                    y_path.extend(paths)
            return np.array(y_true), np.array(y_pred), y_path

 
    def _build_state(self, epoch, early_stopping=None):
        state = {
            "model":       self.model.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "epoch":       epoch,
        }
        if early_stopping is not None:
            state["early_stopping"] = {
                "best_score": early_stopping.best_score,
                "count_down": early_stopping.count_down,
                "early_stop": early_stopping.early_stop,
            }
        return state
 
    def save_networks(self, epoch, early_stopping=None):
        """
        Save checkpoint to <save_dir>/<epoch>.pt.
        
        Args:
            epoch          : int (epoch number) or "best"
            early_stopping : live EarlyStopping object — pass it on every
                             call so last.pt contains full resumable state
        """
        state = self._build_state(epoch, early_stopping)
 
        named_path = os.path.join(self.save_dir, f"{epoch}.pt")
        torch.save(state, named_path)
 
        # last.pt — overwritten every epoch for --resume
        if isinstance(epoch, int):
            last_path = os.path.join(self.save_dir, "last.pt")
            torch.save(state, last_path)

 
    def load_for_resume(self):
        """
        Find the highest-numbered <N>.pt checkpoint in save_dir and load it.
        After loading, returns the full state dict so train.py can:
        - set start_epoch = state["epoch"]
        - restore EarlyStopping from state["early_stopping"]
        """
        # Collect all files whose stem is a pure integer
        candidates = []
        for fname in os.listdir(self.save_dir):
            stem, ext = os.path.splitext(fname)
            if ext == ".pt" and stem.isdigit():
                candidates.append((int(stem), fname))
 
        if not candidates:
            raise FileNotFoundError(
                f"\n--resume requested but no numbered checkpoint found in:\n"
                f"  {self.save_dir}\n"
                f"Files present: {os.listdir(self.save_dir)}\n"
                "Run without --resume to start training from scratch."
            )
 
        # Pick the highest epoch number (int comparison, not string comparison)
        last_epoch, last_fname = max(candidates, key=lambda x: x[0])
        last_path = os.path.join(self.save_dir, last_fname)
 
        print(f"[FTModel] Resuming from {last_path}  "
              f"(highest numbered checkpoint: epoch {last_epoch})")
 
        state = torch.load(last_path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.total_steps = state.get("total_steps", 0)
 
        print(f"[FTModel] Loaded: epoch={state.get('epoch', last_epoch)}  "
              f"total_steps={self.total_steps}")
        return state