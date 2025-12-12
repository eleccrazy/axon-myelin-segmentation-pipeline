"""
File: utils.py
Description: Utility functions for model loading and device management.
Author: Gizachew Kassa
Date Created: 12/12/2025
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import torch

from src.models.unet import UNet
from src.models.unet_deep import UNetDeep
from src.models.unet_stain import UNetStain
from src.utils.paths import MODELS_ROOT, PROJECT_ROOT


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_filename_for_exp(exp: str) -> Tuple[str, str]:
    exp = exp.lower()
    if exp == "tb":
        return "tb", "unet_tb_best.pth"
    if exp == "ihc":
        return "ihc", "unet_ihc_best.pth"
    if exp == "mixed":
        return "mixed", "unet_mixed_stain_best.pth"
    raise ValueError(f"Unknown experiment: {exp}")


def load_model_for_exp(exp: str, device: torch.device | None = None) -> torch.nn.Module:
    """Load the trained model weights for a given experiment.

    Parameters
    ----------
    exp : {"tb", "ihc", "mixed"}
    device : torch.device, optional
        Target device. If None, CUDA is used when available.
    """
    if device is None:
        device = get_device()

    sub, filename = _model_filename_for_exp(exp)
    model_path = MODELS_ROOT / sub / filename

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    if exp in {"tb", "ihc"}:
        model = UNetDeep(in_channels=1, out_channels=1)
    elif exp == "mixed":
        model = UNetStain(in_channels=1, out_channels=1)
    else:  # pragma: no cover
        raise ValueError(f"Unknown experiment: {exp}")

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_calibrated_threshold(exp: str, default: float = 0.5) -> float:
    """Load the validation-calibrated threshold from metrics_<exp>.json.

    Expects files written by `src.scripts.test`, e.g.:
      outputs/metrics/<exp>/metrics_<exp>.json
    """
    exp = exp.lower()
    metrics_root = PROJECT_ROOT / "outputs" / "metrics"
    subdir = metrics_root / exp
    filename = f"metrics_{exp}.json"
    path = subdir / filename

    if not path.exists():
        return float(default)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    t = data.get("threshold", default)
    return float(t)
