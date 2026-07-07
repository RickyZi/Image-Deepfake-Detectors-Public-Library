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

LoRA extension: OpenClipLinearLoRA — see bottom of file.
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from .resnet_mod import ChannelLinear


# ---------------------------------------------------------------------------
# Pretrained model registry
# ---------------------------------------------------------------------------
dict_pretrain = {
    'clipL14openai'     : ('ViT-L-14', 'openai'),
    'clipL14laion400m'  : ('ViT-L-14', 'laion400m_e32'),
    'clipL14laion2B'    : ('ViT-L-14', 'laion2b_s32b_b82k'),
    'clipL14datacomp'   : ('ViT-L-14', 'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K', 'open_clip_pytorch_model.bin'),
    'clipL14commonpool' : ('ViT-L-14', 'laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K', 'open_clip_pytorch_model.bin'),
    'clipaL14datacomp'  : ('ViT-L-14-CLIPA', 'datacomp1b'),
    'cocaL14laion2B'    : ('coca_ViT-L-14', 'laion2b_s13b_b90k'),
    'clipg14laion2B'    : ('ViT-g-14', 'laion2b_s34b_b88k'),
    'eva2L14merged2b'   : ('EVA02-L-14', 'merged2b_s4b_b131k'),
    'clipB16laion2B'    : ('ViT-B-16', 'laion2b_s34b_b88k'),
}


def _build_backbone(pretrain: str):
    """Create an open_clip backbone from a dict_pretrain key."""
    spec = dict_pretrain[pretrain]
    if len(spec) == 2:
        return open_clip.create_model(spec[0], pretrained=spec[1])
    from huggingface_hub import hf_hub_download
    return open_clip.create_model(spec[0], pretrained=hf_hub_download(*spec[1:]))


# ---------------------------------------------------------------------------
# Original frozen-backbone model (unchanged)
# ---------------------------------------------------------------------------

class OpenClipLinear(nn.Module):
    """
    CLIP-D detector with a permanently frozen backbone.
    Only the ChannelLinear fc head is trainable.
    This is the original GRIP-UNINA implementation.
    """

    def __init__(self, num_classes=1, pretrain='clipL14commonpool',
                 normalize=True, next_to_last=False):
        super().__init__()

        backbone = _build_backbone(pretrain)

        if next_to_last:
            self.num_features = backbone.visual.proj.shape[0]
            backbone.visual.proj = None
        else:
            self.num_features = backbone.visual.output_dim

        # Store in a plain list so PyTorch does NOT register it as a submodule.
        # This keeps it out of model.parameters() and model.state_dict() —
        # the intentional design choice from the original paper code.
        self.bb        = [backbone]
        self.normalize = normalize
        self.fc        = ChannelLinear(self.num_features, num_classes)
        nn.init.normal_(self.fc.weight.data, 0.0, 0.02)

    def to(self, *args, **kwargs):
        self.bb[0].to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward_features(self, x):
        with torch.no_grad():
            self.bb[0].eval()
            return self.bb[0].encode_image(x, normalize=self.normalize)

    def forward_head(self, x):
        return self.fc(x)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


# ===========================================================================
# LoRA building blocks
# ===========================================================================

class LoRALinear(nn.Module):
    """
    Linear layer whose effective weight is:

        W_eff = W₀  +  (alpha / r) · B @ A

    W₀ is stored as a frozen buffer (never updated).
    A (r × d_in) and B (d_out × r) are the only trainable parameters.

    Initialisation follows Hu et al. 2021:
        A ~ Kaiming uniform
        B  = 0   →  delta_W = 0 at step 0
    """

    def __init__(self, weight: torch.Tensor, bias, r: int, lora_alpha: float = 1.0):
        super().__init__()
        self.scale = lora_alpha / r
        d_out, d_in = weight.shape

        self.register_buffer("weight", weight.detach().clone())
        self.has_bias = bias is not None
        if self.has_bias:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)

        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_eff = self.weight + self.scale * (self.lora_B @ self.lora_A)
        return F.linear(x, W_eff, self.bias if self.has_bias else None)


class LoRAMultiheadAttention(nn.Module):
    """
    Drop-in replacement for torch.nn.MultiheadAttention inside an
    open_clip ResidualAttentionBlock.

    open_clip packs Q, K, V into a single in_proj_weight [3d, d].
    We unpack this into four explicit projections (q, k, v, o) and
    apply LoRALinear only to the projections listed in lora_targets.

    Call signature matches what open_clip's ResidualAttentionBlock.attention()
    passes:
        self.attn(q_x, k_x, v_x, need_weights=False, attn_mask=...)
    Returns: (output, None)  — consistent with nn.MultiheadAttention.

    Args:
        mha          : existing nn.MultiheadAttention to replace
        r            : LoRA rank
        lora_alpha   : scaling factor  (scale = lora_alpha / r)
        lora_targets : tuple of projections to LoRA-ify:
                       any subset of ('q', 'k', 'v', 'o')
    """

    def __init__(self, mha: nn.MultiheadAttention, r: int = 4, lora_alpha: float = 1.0, lora_targets: tuple = ('q', 'v')):
        super().__init__()
        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.head_dim  = mha.embed_dim // mha.num_heads
        d = self.embed_dim

        # ── Unpack packed QKV ─────────────────────────────────────────────
        W = mha.in_proj_weight.detach()
        b = mha.in_proj_bias.detach() if mha.in_proj_bias is not None else None

        slices = {
            "q": (W[:d].clone(),    b[:d].clone()    if b is not None else None),
            "k": (W[d:2*d].clone(), b[d:2*d].clone() if b is not None else None),
            "v": (W[2*d:].clone(),  b[2*d:].clone()  if b is not None else None),
            "o": (mha.out_proj.weight.detach().clone(),
                  mha.out_proj.bias.detach().clone()
                  if mha.out_proj.bias is not None else None),
        }

        def _proj(key):
            W_s, b_s = slices[key]
            if key in lora_targets:
                return LoRALinear(W_s, b_s, r=r, lora_alpha=lora_alpha)
            lin = nn.Linear(W_s.shape[1], W_s.shape[0], bias=b_s is not None)
            lin.weight = nn.Parameter(W_s, requires_grad=False)
            if b_s is not None:
                lin.bias = nn.Parameter(b_s, requires_grad=False)
            return lin

        self.q_proj = _proj("q")
        self.k_proj = _proj("k")
        self.v_proj = _proj("v")
        self.o_proj = _proj("o")

    def forward(self, q_x, k_x=None, v_x=None, need_weights=False, attn_mask=None):
        if k_x is None: k_x = q_x
        if v_x is None: v_x = q_x

        B, N, C = q_x.shape
        q = self.q_proj(q_x)
        k = self.k_proj(k_x)
        v = self.v_proj(v_x)

        def _reshape(t):
            return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = _reshape(q), _reshape(k), _reshape(v)
        scale = self.head_dim ** -0.5
        attn  = (q @ k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            attn = attn + attn_mask.to(q.dtype)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.o_proj(x), None   # (output, attn_weights)


# ===========================================================================
# LoRA-equipped CLIP-D
# ===========================================================================

class OpenClipLinearLoRA(nn.Module):
    """
    CLIP-D detector with LoRA adapters in the visual encoder.

    Architecture
    ─────────────
    image
      → CLIP ViT-L/14 visual encoder
          frozen pretrained weights  +  trainable LoRA deltas (A, B per block)
      → 1024-d CLS embedding  (next-to-last layer, after ln_post, before proj)
      → ChannelLinear fc  (1024 → 1)
      → logit  (BCEWithLogitsLoss during training)

    Key differences from OpenClipLinear
    ─────────────────────────────────────
    1. backbone stored as self.backbone (proper nn.Module attribute), so
       state_dict, .to(device), and .parameters() all cover it automatically.
    2. forward_features() has NO torch.no_grad() — LoRA A/B need gradients.
    3. Frozen base weights have requires_grad=False individually, so they
       never appear in the optimizer even though backbone is registered.
    4. LoRA is injected only into the visual transformer; the text encoder
       is untouched (this detector never calls encode_text()).

    Parameter budget  (ViT-L/14, 24 blocks, r=4, targets=('q','v'))
    ─────────────────────────────────────────────────────────────────
      LoRA A+B:  24 × 2 × (4×1024 + 1024×4) = 393 216
      fc:        1024 + 1                    =   1 025
      Total      trainable                   = 394 241   (≈ 0.10 % of model)

    Checkpoint compatibility
    ────────────────────────
    state_dict() includes frozen backbone weights AND LoRA A/B weights.
    Loading with strict=True works when the arch string is identical between
    save and load.  Loading an older fc-only checkpoint via strict=False
    is supported in FTModel (LoRA A/B keep their zero/Kaiming init, which
    is correct: delta_W = 0 before any fine-tuning step).

    Args:
        num_classes   : output logits (1 for binary real/fake detection)
        pretrain      : key in dict_pretrain
        normalize     : L2-normalise the CLIP embedding
        next_to_last  : use 1024-d pre-projection feature (recommended: True)
        lora_r        : LoRA rank.  r=4 strongly recommended for ≤2k images.
        lora_alpha    : LoRA scaling (lora_alpha=lora_r → scale=1, recommended)
        lora_targets  : attention projections to adapt: subset of ('q','k','v','o')
    """

    def __init__(
        self,
        num_classes: int = 1,
        pretrain: str = 'clipL14commonpool',
        normalize: bool = True,
        next_to_last: bool = True,
        lora_r: int = 4,
        lora_alpha: float = 4.0,
        lora_targets: tuple = ('q', 'v'),
    ):
        super().__init__()
        self.normalize    = normalize
        self.lora_r       = lora_r
        self.lora_alpha   = lora_alpha
        self.lora_targets = lora_targets

        # ── 1. Build backbone ─────────────────────────────────────────────
        backbone = _build_backbone(pretrain)

        if next_to_last:
            self.num_features = backbone.visual.proj.shape[0]   # 1024 for ViT-L/14
            backbone.visual.proj = None
        else:
            self.num_features = backbone.visual.output_dim      # 768

        # ── 2. Freeze everything ──────────────────────────────────────────
        for p in backbone.parameters():
            p.requires_grad_(False)

        # ── 3. Inject LoRA into visual transformer blocks ─────────────────
        #       Text encoder is deliberately left untouched.
        resblocks = backbone.visual.transformer.resblocks
        print(f"[LoRA] Injecting into {len(resblocks)} visual transformer blocks …")
        for block in resblocks:
            block.attn = LoRAMultiheadAttention(
                block.attn,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_targets=lora_targets,
            )

        # ── 4. Register backbone as a real nn.Module attribute ────────────
        #       Unlike OpenClipLinear (which uses self.bb = [backbone]),
        #       we register it properly so state_dict and .to() work
        #       correctly for all parameters including the LoRA matrices.
        self.backbone = backbone

        # ── 5. Classification head ────────────────────────────────────────
        self.fc = ChannelLinear(self.num_features, num_classes)
        nn.init.normal_(self.fc.weight.data, 0.0, 0.02)

        # ── 6. Report ─────────────────────────────────────────────────────
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"[LoRA] r={lora_r}, alpha={lora_alpha}, targets={lora_targets}")
        print(f"[LoRA] Trainable: {trainable:,} / {total:,}  "
              f"({100 * trainable / total:.3f} %)")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # No torch.no_grad(): gradients flow through LoRA A and B.
        # Frozen base weights are safe because requires_grad=False was set
        # individually on each of them during construction.
        return self.backbone.encode_image(x, normalize=self.normalize)

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))

    # ------------------------------------------------------------------
    # Optional weight merging for inference-time cleanup
    # ------------------------------------------------------------------

    @torch.no_grad()
    def merge_lora_weights(self) -> None:
        """
        Merge  ΔW = (alpha/r) · B @ A  into the frozen base weight in-place,
        then zero A and B.  After this the model is equivalent to a plain
        OpenClipLinear checkpoint and has zero overhead at inference.

        Call only after training is complete — not during training.
        """
        for block in self.backbone.visual.transformer.resblocks:
            for proj in (block.attn.q_proj, block.attn.k_proj,
                         block.attn.v_proj, block.attn.o_proj):
                if isinstance(proj, LoRALinear):
                    proj.weight.add_(proj.scale * (proj.lora_B @ proj.lora_A))
                    proj.lora_A.zero_()
                    proj.lora_B.zero_()
        print("[LoRA] Weights merged into base matrices.")