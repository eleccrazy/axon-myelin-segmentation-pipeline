"""
File: overlays.py
Description: Functions to generate FP/FN/TP overlays for TB, IHC, or mixed-stain models.
Author: Gizachew Kassa
Date Created: 12/12/2025
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from src.datasets.ihc_dataset import IHCLMDataset
from src.datasets.mixed_dataset import MixedLMDataset
from src.datasets.tb_dataset import TBLMDataset
from src.utils.paths import OUTPUTS_ROOT
from src.visualization.utils import (
    get_device,
    load_calibrated_threshold,
    load_model_for_exp,
)


def _get_dataset(exp: str, split: str):
    exp = exp.lower()
    if exp == "tb":
        return TBLMDataset(split=split, augment=None)
    if exp == "ihc":
        return IHCLMDataset(split=split, augment=None)
    if exp == "mixed":
        return MixedLMDataset(split=split, augment=None)
    raise ValueError(f"Unknown experiment: {exp}")


def _get_image_path(exp: str, dataset, idx: int) -> Path:
    if exp in {"tb", "ihc"}:
        img_path, _ = dataset.pairs[idx]
    elif exp == "mixed":
        img_path, _, _ = dataset.samples[idx]
    else:
        raise ValueError(f"Unknown experiment: {exp}")
    return Path(img_path)


def _ensure_same_size(rgb: np.ndarray, mask_hw: Tuple[int, int]) -> np.ndarray:
    """Resize RGB image to match (H,W) if needed."""
    Hm, Wm = mask_hw
    Hr, Wr = rgb.shape[:2]
    if (Hr, Wr) == (Hm, Wm):
        return rgb
    pil = Image.fromarray(rgb).resize((Wm, Hm), Image.BILINEAR)
    return np.asarray(pil)


def _build_overlay(mask_np: np.ndarray, pred_np: np.ndarray) -> np.ndarray:
    """
    Overlay encoding (full image):
      FP: red
      FN: green
      TP: yellow
    Returns uint8 RGB (H,W,3).
    """
    h, w = mask_np.shape
    overlay = np.zeros((h, w, 3), dtype=np.float32)

    tp = (pred_np == 1) & (mask_np == 1)
    fp = (pred_np == 1) & (mask_np == 0)
    fn = (pred_np == 0) & (mask_np == 1)

    overlay[..., 0][fp] = 1.0
    overlay[..., 0][tp] = 1.0
    overlay[..., 1][fn] = 1.0
    overlay[..., 1][tp] = 1.0

    return (overlay * 255.0).astype(np.uint8)


def generate_overlays(
    exp: str,
    split: str = "test",
    threshold: Optional[float] = None,
    device: Optional[torch.device] = None,
    save_dir: Optional[str | Path] = None,
) -> None:
    exp = exp.lower()
    if device is None:
        device = get_device()

    if threshold is None:
        threshold = load_calibrated_threshold(exp, default=0.5)

    # Kept only for consistent directory structure; overlays are full-image.
    crop_size = 256

    if save_dir is None:
        save_dir = OUTPUTS_ROOT / "figures" / exp / "overlays"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = _get_dataset(exp, split=split)
    n = len(dataset)

    model = load_model_for_exp(exp, device=device)

    print(
        f"Generating {n} full-image overlays for exp='{exp}', split='{split}', "
        f"threshold={float(threshold):.3f}"
    )

    for i in range(n):
        sample = dataset[i]

        if exp == "mixed":
            img_t, mask_t, stain_t = sample
            stain_id = int(stain_t.item())
            stain_str = "tb" if stain_id == 0 else "ihc"
        else:
            img_t, mask_t = sample
            stain_str = None

        img_path = _get_image_path(exp, dataset, i)
        basename = img_path.stem

        # Group mixed outputs by stain (optional, but consistent with triplets)
        out_dir = save_dir / (f"stain_{stain_str}" if stain_str else "")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load original just to ensure alignment (some datasets resize internally)
        orig_rgb = np.asarray(Image.open(img_path).convert("RGB"))

        mask_np = mask_t.squeeze(0).cpu().numpy().astype(np.uint8)  # (H,W)
        orig_rgb = _ensure_same_size(orig_rgb, mask_np.shape)  # alignment safeguard

        x = img_t.unsqueeze(0).to(device)  # [1,1,H,W]
        with torch.no_grad():
            if exp == "mixed":
                logits = model(x, stain_t.view(1).to(device))
            else:
                logits = model(x)

        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred_np = (prob >= float(threshold)).astype(np.uint8)

        overlay_rgb = _build_overlay(mask_np, pred_np)

        if exp == "mixed" and stain_str is not None:
            out_name = f"{basename}_{stain_str}_idx{i:03d}_overlay.png"
        else:
            out_name = f"{basename}_idx{i:03d}_overlay.png"

        Image.fromarray(overlay_rgb).save(out_dir / out_name)
        print(f"[{i+1}/{n}] Saved overlay → {out_name}")

    print("Done generating overlays.")
