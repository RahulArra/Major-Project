"""Shared helpers for read-only DeepLabV3+ diagnostics."""

from datetime import datetime
from pathlib import Path

import torch

from deeplabv3 import DeepLabV3Plus


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints_deeplab" / "best_deeplab.pth"


def new_output_dir(name):
    """Create a unique output directory so reruns preserve prior results."""
    base = PROJECT_ROOT / "outputs" / name
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base / stamp
    suffix = 1
    while candidate.exists():
        candidate = base / f"{stamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def load_deeplab(device):
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"DeepLab checkpoint not found: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model = DeepLabV3Plus(in_channels=2, num_classes=1, pretrained=False).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
