# Reconstruction-Aware Detector Adaptation (SDA)

This repository contains the official implementation for detector Supervised Domain Adaptation (SDA) described in Section V ("Experimental Evaluation") of our paper:

> **Enhanced image forensics through partial C2PA-based provenance information**  
> *Riccardo Ziglio, Eleonora Marchesini, Andrea Montibeller, Giulia Boato, Silvio Ranise, Cecilia Pasquini*  
> Accepted to IEEE Workshop on Information Forensics and Security 2026

The codebase builds upon the original [TrueBees Image-Deepfake-Detectors-Public-Library](https://github.com/truebees-ai/Image-Deepfake-Detectors-Public-Library). 

Detector-specific source projects and original papers are linked inside the respective READMEs in `detectors/`.

---

## Detectors & Resources

We adapt the three top-performing classifiers from [2] as standard baseline detectors:

| Detector | Summary | Documentation |
| --- | --- | --- |
| `CLIP-D` [1] | CLIP-based detector with added support for LoRA fine-tuning. | [`detectors/CLIP-D/README.md`](detectors/CLIP-D/README.md) |
| `R50_E2P` [2] | ResNet50-based detector developed by the TrueFake authors. | [`detectors/R50_E2P/README.md`](detectors/R50_E2P/README.md) |
| `R50_ND` [3] | ResNet50-based detector variant omitting early downsampling. | [`detectors/R50_ND/README.md`](detectors/R50_ND/README.md) |

### Downloads

- **Pre-trained Checkpoints**: Download from [SharePoint Release Link](https://fbk-my.sharepoint.com/my?id=%2Fpersonal%2Frziglio%5Ffbk%5Feu%2FDocuments%2FSharedMaterial%2FReplicationPackages%2FWIFS2026%5FReplicationPackage%2FWIFS2026%5FFTmodels&ga=1&share=cgqiQOCr3hxKSIPfO6eEDcXTEgUCI%2D10cYG3HBP9NMapILfHOw). Place weights inside `./detectors/<detector_name>/checkpoint/`.

- **Dataset**: Download the experimental dataset from [SharePoint Data Link](https://fbk-my.sharepoint.com/my?id=%252Fpersonal%252Frziglio_fbk_eu%252FDocuments%252FSharedMaterial%252FReplicationPackages%252FWIFS2026_ReplicationPackage%252FWIFS2026_C2PAdataset&ga=1).

---

## Adaptation Strategies

Detectors load pre-trained weights (from the original TrueBees repository) and adapt via:

- **`R50_ND` & `R50_E2P`**: Fine-tuned by unfreezing Layer 4 and the fully-connected classification head (`--ft --unfreezeL4R50`).
- **`CLIP-D`**: Fine-tuned using Low-Rank Adaptation (LoRA) applied exclusively to the Query and Value visual encoder projections (`--ft --weights-name lora_r4_qv`).

Fine-tuning and model paths are automatically resolved via the YAML configs in `configs/` and JSON files in `ensemble/configs/`.

---

## Repository Structure

```text
├── launcher.py                   # Main entry point for training, testing, and single-image inference
├── configs/                      # Detector YAML experiment configurations
├── splits/                       # Benchmark split definitions (e.g., tf2k_splits.json, tf2k_SOCIAL_splits.json)
├── ensemble/                     # Output alignment, score fusion rules, and ensemble evaluations
│   ├── launch_ensemble_r50.py    # Launcher for ResNet-50 ensemble evaluations
│   └── launch_ensemble_clipd.py  # Launcher for CLIP-D ensemble evaluations
├── support/                      # Shared inference routines and split parsers
└── utils/                        # Dataset validation and split generation scripts
```

---

## Setup

### Prerequisites

- `Ubuntu >= 22.04`
- `Python >= 3.10`

### Installation

```bash
python -m venv IDFD_VENV
source IDFD_VENV/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

### Training & Testing

Run automated training and evaluation using `launcher.py`:

```bash
python launcher.py \
  --detector <CLIP-D|R50_ND|R50_E2P> \
  --phases <train|test|both> \
  --dataset <dataset_relative_path> \
  [--ft] \
  [--unfreezeL4R50] \
  [--weights-name lora_r4_qv]
```

**Examples:**

- **Fine-tune ResNet-50 (`R50_ND`)**:

  ```bash
  python launcher.py --detector R50_ND --phases both --dataset adaptive/blurbg_strong --ft --unfreezeL4R50
  ```

- **Fine-tune CLIP-D with LoRA**:

  ```bash
  python launcher.py --detector CLIP-D --phases both --dataset adaptive/blurbg_strong --ft --weights-name lora_r4_qv
  ```

<!-- ### Single-Image Detection

```bash
python launcher.py --detect \
  --detector CLIP-D \
  --image /path/to/image.jpg \
  --weights-name lora_r4_qv \
  --ft \
  --output detection_results/prediction.json 
```
-->

### Ensemble Evaluation

Ensemble setups evaluate per-checkpoint probability predictions and combine scores via arithmetic mean. Configure the dataset root and checkpoints in `ensemble/configs/`, then run:

```bash
python launch_ensemble_r50.py --ensemble-config ensemble/configs/r50nd/blurbg_family.json
python launch_ensemble_clipd.py --ensemble-config ensemble/configs/clipd/blurbg_family.json
```

Reports are exported to `results/model_ensemble/ensemble_results/`.

---

## Dataset & Split Utilities

The repository includes split files under `splits/` (e.g., `tf2k_splits.json` and `tf2k_SOCIAL_splits.json`). Use the utilities in `utils/` and `support/json_compile.py` to inspect directory structures or compile new custom JSON split files.

---

<!-- ## Citation & References

If you use this code or dataset in your research, please cite our paper:

```bibtex
@inproceedings{yourpaper2026,
  author    = {Author, A. and Author, B.},
  title     = {Reconstruction-Aware Detector Adaptation},
  booktitle = {IEEE Workshop on Information Forensics and Security (WIFS)},
  year      = {2026}
}
``` -->

### Additional References

[1] D. Cozzolino et al., "Raising the Bar of AI-generated Image Detection with CLIP," *CVPRW*, 2024.  
[2] S. Dell'Anna et al., "TrueFake: A Real World Case Dataset of Last Generation Fake Images also Shared on Social Networks," *arXiv:2504.20658*, 2025.  
[3] R. Corvi et al., "On The Detection of Synthetic Images Generated by Diffusion Models," *ICASSP*, 2023.