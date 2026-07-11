import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # ── Identity ──────────────────────────────────────────────────────────
    parser.add_argument("--name", type=str, default="test",
                        help="Run name — used as checkpoint directory key")
    parser.add_argument("--arch", type=str,
                        default="opencliplinearnext_clipL14commonpool",
                        help=(
                            "Architecture name.  "
                            "Frozen-backbone:  opencliplinearnext_<pretrain>  "
                            "LoRA fine-tuning: opencliplinearloranext_<pretrain>_r<rank>_<targets>  "
                            "  e.g. opencliplinearloranext_clipL14commonpool_r4_qv"
                        ))
    parser.add_argument("--task", type=str, choices=["train", "test"],
                        help="Task to execute: train or test")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CUDA device string, e.g. 'cuda:0' or 'cpu'")

    # ── Dataset / splits ─────────────────────────────────────────────────
    parser.add_argument("--split_file", type=str,
                        help="Path to train/val/test split JSON file")
    parser.add_argument("--data_root",  type=str,
                        help="Root path to the image dataset")
    parser.add_argument("--data_keys",  type=str,
                        help="Dataset specification string, e.g. 'all:pre&real:pre'")
    # Passed by launcher for book-keeping; not used inside train/test scripts directly
    parser.add_argument("--dataset",    type=str, default="dataset",
                        help="Dataset sub-folder name (used by launcher for path assembly)")

    # ── DataLoader ───────────────────────────────────────────────────────
    parser.add_argument("--batch_size",   type=int,   default=32,
                        help="DataLoader batch size (32 recommended for LoRA with 2k images)")
    parser.add_argument("--num_threads",  type=int,   default=8,
                        help="Number of DataLoader worker threads")

    # ── Optimiser ────────────────────────────────────────────────────────
    parser.add_argument("--lr",           type=float, default=1e-4,
                        help="Initial learning rate (1e-4 works well for LoRA)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="AdamW weight decay (1e-4 recommended for LoRA)")
    parser.add_argument("--beta1",        type=float, default=0.9,
                        help="Adam beta1 momentum term")

    # ── Training schedule ────────────────────────────────────────────────
    parser.add_argument("--num_epoches",     type=int, default=1000,
                        help="Maximum number of training epochs")
    parser.add_argument("--earlystop_epoch", type=int, default=5,
                        help="Patience: epochs without improvement before LR drop")

    # ── Fine-tuning flags ─────────────────────────────────────────────────
    parser.add_argument("--ft", action="store_true",
                        help=(
                            "Fine-tuning mode.  "
                            "- For LoRA arch: loads fc weights from weights/best.pt (strict=False), "
                            "  then trains LoRA + fc.  Saves to ft_weights/.  "
                            "- For classic arch: loads weights/best.pt and unfreezes last N blocks."
                        ))

    # ── Dataset variant ──────────────────────────────────────────────────
    parser.add_argument("--tf2k", type=bool, default=False,
                        help="Use TF2K dataset loader (tf2k_dataset.py) instead of dataset.py")
    parser.add_argument('--social', type = str, default="", help = 'Which dataset from a social to load (default: niet)') 
    # ── ResNet-specific fine-tuning flag (passed by launcher, ignored in CLIP-D) ──
    parser.add_argument("--r50unfreezeL4", action="store_true",
                        help="Unfreeze ResNet layer4 during fine-tuning (ResNet detectors only)")

    # ── MLP head options (for future CLIP-D variants) ─────────────────────
    parser.add_argument("--mlp",         action="store_true",
                        help="Add MLP head instead of linear fc (not used in LoRA variant)")
    parser.add_argument("--mlp_hidden",  type=int,   default=256,
                        help="Hidden dimension of MLP head")
    parser.add_argument("--mlp_dropout", type=float, default=0.3,
                        help="Dropout probability in MLP head")
    
    parser.add_argument('--resume', action='store_true', help='Resume training from the last saved per-epoch checkpoint in the run\'s checkpoint dir')

    return parser