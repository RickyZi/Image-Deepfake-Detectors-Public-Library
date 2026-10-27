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

    # ResNet baselines
    if name_arch == "res50nodown":
        from .resnet_mod import resnet50
        model = (resnet50(pretrained=True, stride0=1, dropout=0.5).change_output(num_classes)
                 if pretrained else resnet50(num_classes=num_classes, stride0=1, dropout=0.5))

    elif name_arch == "res50":
        from .resnet_mod import resnet50
        model = (resnet50(pretrained=True, stride0=2).change_output(num_classes)
                 if pretrained else resnet50(num_classes=num_classes, stride0=2))

    # CLIP-D baselines
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

    # LoRA CLIP-D
    elif name_arch.startswith("opencliplinearloranext_"):
        from .openclipnet import OpenClipLinearLoRA

        suffix = name_arch[len("opencliplinearloranext_"):]
        parts  = suffix.split("_")

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
            next_to_last=True,          
            lora_r=lora_r,
            lora_alpha=float(lora_r),  
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