## ---------------------------------------------------- ##

import os
import torch
import numpy as np
import tqdm

from networks import create_architecture, count_parameters


class FTModel(torch.nn.Module):
    def __init__(self, opt):
        super(FTModel, self).__init__()
        self.opt = opt
        self.total_steps = 0
        # tag = 'unfreezeL4' if opt.R50unfreeL4 else None
        dataset = opt.dataset.replace(os.sep, '_')
        # dataset += '_unfreezeL4' if opt.r50unfreezeL4 else ''
        if opt.r50unfreezeL4:
            dataset += '_r50unfreezeL4'

        print(f"opt: {self.opt}")
        print(f"dataset: {dataset}")
        # breakpoint()
        if opt.ft and opt.r50unfreezeL4:
            self.save_dir = os.path.join('checkpoint', opt.name, 'ft_unfreezeL4_weights', dataset)
        else:
            self.save_dir = os.path.join('checkpoint', opt.name, 'ft_weights', dataset)
        # else:
            # print("no valid option for saving FT model")
        # self.save_dir = os.path.join('checkpoint', opt.name, 'ft_weights', dataset)
        os.makedirs(self.save_dir, exist_ok=True)
        self.device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')

        print(f"opt.arch: {opt.arch}")
        self.model = create_architecture(opt.arch, pretrained=True, num_classes=1)
        print(f"Arch: {opt.arch} with #trainable params: {count_parameters(self.model)}")
        # breakpoint()

        self.loss_fn = torch.nn.BCEWithLogitsLoss().to(self.device)
        self.optimizer = self._build_optimizer()
        self.model.to(self.device)

    def _build_optimizer(self):
        # Only include parameters that require gradients (e.g., after freezing)
        trainable = filter(lambda p: p.requires_grad, self.model.parameters())
        return torch.optim.Adam(
            trainable, lr=self.opt.lr,
            betas=(self.opt.beta1, 0.999),
            weight_decay=self.opt.weight_decay
        )

    def reinitialize_optimizer(self):
        """Redefine optimizer to make sure it operates only on frozen layers"""
        self.optimizer = self._build_optimizer()
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Optimizer reinitialized — trainable params: {trainable_count}")

    def freeze_backbone(self, r50unfreezeL4 = False):
        """Freeze all layers except the final fc head."""
        for param in self.model.parameters():
            param.requires_grad = False

        if r50unfreezeL4:
            print("\n\tUnfreezing Layer 4 in FT R50 model\n")
            # may consided to unfreeze layer4 to give feature extractor some capacity to adapt
            for param in self.model.layer4.parameters():
                param.requires_grad = True

        for param in self.model.fc.parameters():
            param.requires_grad = True

        self.reinitialize_optimizer() # optimizer recomputed on unfrozen parameters

        # Sanity check
        trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
        print(f"Trainable layers: {trainable}")
        if r50unfreezeL4:
            assert all("fc" in n or "layer4" in n for n in trainable), "Unexpected trainable params outside fc and layer4!"
        else:
            assert all("fc" in n for n in trainable), "Unexpected trainable params outside fc!"

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] /= 10.0
            if param_group["lr"] < min_lr:
                return False
        return True

    def get_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def load_networks(self, checkpoint_path): #, r50unfreezeL4):
        """Load model (and optionally optimizer) from a checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(state_dict['model'])
        print(f"Loaded model weights from {checkpoint_path}")

        # breakpoint()

        # # Optionally restore optimizer and step count
        # if 'optimizer' in state_dict:
        #     self.optimizer.load_state_dict(state_dict['optimizer'])
        #     print("Restored optimizer state")
        # if 'total_steps' in state_dict:
        #     self.total_steps = state_dict['total_steps']
        #     print(f"Restored total_steps: {self.total_steps}")
    
    def load_checkpoint(self, checkpoint_path):
        """Fully restore model, op  timizer, and step count from a checkpoint
        written by save_networks, for resuming an interrupted run. Unlike
        load_networks (which only restores weights, for bootstrapping a
        fine-tune from a pretrained backbone), this restores everything
        needed to continue training exactly where it left off. Returns the
        raw checkpoint dict so the caller can pull out any extra keys (e.g.
        early-stopping state) that were stashed in it."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
 
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict) or 'model' not in checkpoint:
            raise ValueError(
                f"{checkpoint_path} doesn't look like a checkpoint written by save_networks "
                f"(expected a dict with a 'model' key) - refusing to resume from it."
            )
        self.model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'total_steps' in checkpoint:
            self.total_steps = checkpoint['total_steps']
        print(f"Resumed model, optimizer, and total_steps from {checkpoint_path}")
        return checkpoint


    def train_on_batch(self, data):
        self.total_steps += 1
        self.model.train()

        img = data['img'].to(self.device)
        label = data['target'].to(self.device).float()

        self.optimizer.zero_grad() # zero grad before forward step
        output = self.model(img)

        if len(output.shape) == 4:
            ss = output.shape
            loss = self.loss_fn(
                output,
                label[:, None, None, None].repeat(1, ss[1], ss[2], ss[3])
            )
        else:
            loss = self.loss_fn(output.squeeze(1), label)

        loss.backward()
        self.optimizer.step()
        return loss.cpu()

    def save_networks(self, epoch, extra=None):
        # add extra to save epoch - related info for resume training
        save_path = os.path.join(self.save_dir, f'{epoch}.pt')
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
        }
        if extra:
            state.update(extra)
            
        torch.save(state, save_path)


    def predict(self, data_loader):
        self.model.eval()
        y_true, y_pred, y_path = [], [], []

        with torch.no_grad():
            for data in tqdm.tqdm(data_loader):
                img = data['img'].to(self.device)
                label = data['target'].cpu().numpy()
                paths = list(data['path'])

                out = self.model(img).cpu().numpy().squeeze(1)  # [B, 1] → [B]

                assert label.shape == out.shape, \
                    f"Shape mismatch: label {label.shape} vs output {out.shape}"

                y_pred.extend(out.tolist())
                y_true.extend(label.tolist())
                y_path.extend(paths)

        return np.array(y_true), np.array(y_pred), y_path