"""
File: threshold_calibration.py
Description: Validation-based threshold calibration for segmentation models.

Author: Gizachew Kassa
Date Created: 10/12/2025
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import dice_iou_over_loader


def find_best_threshold(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    grid: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Calibrate the probability threshold t* on a validation loader using Dice.

    This function reproduces the logic from tb.py and ihc.py:
    - collect sigmoid outputs over the whole validation set,
    - sweep t over a fixed grid (default 0.20..0.60 with 17 points),
    - compute the mean Dice for each t,
    - return the best (t*, Dice*).
    """
    if grid is None:
        grid = np.linspace(0.20, 0.60, 17)  # 0.20..0.60 step 0.025

    model.eval()
    all_probs, all_targs = [], []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                images, masks = batch
                stains = None
            else:
                images, masks, stains = batch

            images = images.to(device)

            if stains is None:
                logits = model(images)
            else:
                logits = model(images, stains)

            probs = torch.sigmoid(logits).cpu()
            all_probs.append(probs)
            all_targs.append(masks)

    if not all_probs:
        return 0.5, -1.0

    probs = torch.cat(all_probs, dim=0)  # [N,1,H,W]
    targs = torch.cat(all_targs, dim=0)  # [N,1,H,W]
    targs = targs.float()

    best_t, best_d = 0.5, -1.0
    for t in grid:
        pred = (probs >= t).float()
        inter = (pred * targs).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + targs.sum(dim=(1, 2, 3))
        dice = ((2.0 * inter + 1.0) / (union + 1.0)).mean().item()
        if dice > best_d:
            best_d, best_t = dice, float(t)

    return best_t, best_d


def sweep_thresholds_dice_iou(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    grid: Optional[Iterable[float]] = None,
) -> Tuple[float, float, np.ndarray]:
    """
    Sweep a range of thresholds and report the best Dice and its threshold.

    This function is analogous to sweep_thresholds in mixed_experiment2.py:
    - for each t in the grid, compute mean Dice and IoU over the loader,
    - track the t that maximises Dice,
    - return (best_t, best_dice, grid_array).

    It uses dice_iou_over_loader, so it works for both TB/IHC (model(x))
    and mixed-stain models (model(x, stains)).
    """
    if grid is None:
        grid = np.linspace(0.10, 0.90, 17)  # 0.10..0.90, 17 points

    grid_array = np.array(list(grid), dtype=float)

    best_t = 0.5
    best_d = -1.0

    for t in grid_array:
        d, _ = dice_iou_over_loader(model, loader, device=device, threshold=float(t))
        if d > best_d:
            best_d = d
            best_t = float(t)

    return best_t, best_d, grid_array
