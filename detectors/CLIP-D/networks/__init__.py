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


def create_architecture(name_arch, pretrained=False, num_classes=1):
    """
    Factory function for all CLIP-D network variants.

    Arch string format
    ──────────────────
    Frozen backbone (original):
        opencliplinear_<pretrain>           e.g. opencliplinear_clipL14commonpool
        opencliplinearnext_<pretrain>       e.g. opencliplinearnext_clipL14commonpool

    LoRA fine-tuning (new):
        opencliplinearloranext_<pretrain>_r<rank>_<targets>
        e.g.  opencliplinearloranext_clipL14commonpool_r4_qv
              opencliplinearloranext_clipL14commonpool_r8_qvo
              opencliplinearloranext_clipL14commonpool_r4_qkvo

    <targets> is any combination of q / k / v / o (attention projections).
    Recommended starting point: qv  (query + value, as per Hu et al. 2021).

    ResNet baselines:
        res50nodown
        res50
    """

    # ── ResNet baselines ─────────────────────────────────────────────────
    if name_arch == "res50nodown":
        from .resnet_mod import resnet50
        model = (resnet50(pretrained=True, stride0=1, dropout=0.5).change_output(num_classes)
                 if pretrained else resnet50(num_classes=num_classes, stride0=1, dropout=0.5))

    elif name_arch == "res50":
        from .resnet_mod import resnet50
        model = (resnet50(pretrained=True, stride0=2).change_output(num_classes)
                 if pretrained else resnet50(num_classes=num_classes, stride0=2))

    # ── Frozen-backbone CLIP-D (original) ────────────────────────────────
    elif name_arch.startswith("opencliplinear_"):
        from .openclipnet import OpenClipLinear
        model = OpenClipLinear(
            num_classes=num_classes,
            pretrain=name_arch[len("opencliplinear_"):],
            normalize=True,
            next_to_last=False,
        )

    elif name_arch.startswith("opencliplinearnext_"):
        from .openclipnet import OpenClipLinear
        model = OpenClipLinear(
            num_classes=num_classes,
            pretrain=name_arch[len("opencliplinearnext_"):],
            normalize=True,
            next_to_last=True,
        )

    # ── LoRA CLIP-D ───────────────────────────────────────────────────────
    # Format: opencliplinearloranext_<pretrain>_r<rank>_<targets>
    # Example: opencliplinearloranext_clipL14commonpool_r4_qv
    elif name_arch.startswith("opencliplinearloranext_"):
        from .openclipnet import OpenClipLinearLoRA

        # Strip prefix, then parse  <pretrain>_r<rank>_<targets>
        suffix = name_arch[len("opencliplinearloranext_"):]
        parts  = suffix.split("_")

        # The last token is <targets>, the second-to-last is r<rank>,
        # and everything before is the pretrain key.
        # This works for pretrain keys without underscores (common case)
        # and for keys that do contain underscores (e.g. clipL14datacomp).
        targets_str = parts[-1]    # e.g. "qv"
        rank_str    = parts[-2]    # e.g. "r4"
        pretrain    = "_".join(parts[:-2])  # e.g. "clipL14commonpool"

        if not (rank_str.startswith("r") and rank_str[1:].isdigit()):
            raise ValueError(
                f"Cannot parse rank from arch name '{name_arch}'. "
                f"Expected: opencliplinearloranext_<pretrain>_r<int>_<targets>"
            )

        lora_r       = int(rank_str[1:])
        lora_targets = tuple(c for c in targets_str if c in ("q", "k", "v", "o"))

        if not lora_targets:
            raise ValueError(
                f"No valid targets found in '{targets_str}'. "
                f"Use any combination of q/k/v/o."
            )

        model = OpenClipLinearLoRA(
            num_classes=num_classes,
            pretrain=pretrain,
            normalize=True,
            next_to_last=True,          # always use 1024-d next-to-last features
            lora_r=lora_r,
            lora_alpha=float(lora_r),   # alpha == r  →  effective scale = 1
            lora_targets=lora_targets,
        )

    else:
        raise ValueError(
            f"Unknown architecture: '{name_arch}'. "
            "Valid prefixes: res50, res50nodown, opencliplinear_, "
            "opencliplinearnext_, opencliplinearloranext_"
        )

    return model


def count_parameters(model):
    """Return the number of trainable (requires_grad=True) parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_weights(model, model_path):
    """Load a checkpoint into model, handling various saved formats."""
    from torch import load
    dat = load(model_path, map_location="cpu")

    if "model" in dat:
        sd = dat["model"]
        # Strip DataParallel "module." prefix if present
        if any(k.startswith("module.") for k in sd):
            sd = {k[7:]: v for k, v in sd.items()}
        model.load_state_dict(sd)
    elif "state_dict" in dat:
        model.load_state_dict(dat["state_dict"])
    elif "net" in dat:
        model.load_state_dict(dat["net"])
    else:
        # Bare state dict
        model.load_state_dict(dat)

    return model