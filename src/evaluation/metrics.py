"""
File: metrics.py
Description: Dice, IoU, precision and recall metrics for segmentation models.

Author: Gizachew Kassa
Date Created: 10/12/2025
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


def _binarize_logits(
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Apply sigmoid and threshold logits to obtain a binary prediction mask.
    """
    probs = torch.sigmoid(logits)
    return (probs >= threshold).float()


def dice_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> float:
    """
    Compute the foreground Dice coefficient from logits and binary targets.

    This matches the implementation used in tb.py and ihc.py, where Dice is
    computed on thresholded predictions with a smoothing constant of 1.0.
    """
    preds = _binarize_logits(logits, threshold=threshold)
    targets = targets.float()

    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + smooth) / (union + smooth)
    return dice.mean().item()


def iou_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    """
    Compute the Intersection-over-Union (IoU) from logits and binary targets.
    """
    preds = _binarize_logits(logits, threshold=threshold)
    targets = targets.float()

    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    inter = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - inter

    iou = (inter + eps) / (union + eps)
    return iou.mean().item()


def precision_recall_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> Tuple[float, float]:
    """
    Compute precision and recall for the foreground class from logits.
    """
    preds = _binarize_logits(logits, threshold=threshold)
    targets = targets.float()

    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    tp = (preds_flat * targets_flat).sum(dim=1)
    fp = (preds_flat * (1.0 - targets_flat)).sum(dim=1)
    fn = ((1.0 - preds_flat) * targets_flat).sum(dim=1)

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return precision.mean().item(), recall.mean().item()


def dice_iou_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> Tuple[float, float]:
    """
    Compute Dice and IoU from logits, as in mixed_experiment2.py.

    This function mirrors the per-batch logic of dice_iou_from_logits used in
    the mixed-stain experiment: predictions are thresholded, flattened, and
    Dice/IoU are computed per sample and then averaged.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    B = targets.size(0)
    dice_sum = 0.0
    iou_sum = 0.0

    for b in range(B):
        p = preds[b].view(-1)
        t = targets[b].view(-1)

        inter = (p * t).sum().item()
        union = p.sum().item() + t.sum().item() - inter

        dice = (2.0 * inter + eps) / (p.sum().item() + t.sum().item() + eps)
        iou = (inter + eps) / (union + eps)

        dice_sum += dice
        iou_sum += iou

    return dice_sum / max(B, 1), iou_sum / max(B, 1)


def dice_iou_over_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[float, float]:
    """
    Compute mean Dice and IoU over a DataLoader.

    The loader is expected to yield either:
        (images, masks)                for TB/IHC models
        (images, masks, stains)        for mixed-stain models

    In the mixed case, the model is called as model(images, stains); otherwise
    as model(images).

    IMPORTANT: stains are moved to the same device as the model to avoid
    CPU/CUDA mismatch errors.
    """
    model.eval()
    dice_sum = 0.0
    iou_sum = 0.0
    n = 0

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                images, masks = batch
                stains = None
            else:
                images, masks, stains = batch

            images = images.to(device)
            masks = masks.to(device)
            if stains is not None:
                stains = stains.to(device)

            if stains is None:
                logits = model(images)
            else:
                logits = model(images, stains)

            d, i = dice_iou_from_logits(logits, masks, threshold=threshold)
            dice_sum += d
            iou_sum += i
            n += 1

    return dice_sum / max(n, 1), iou_sum / max(n, 1)


def dice_from_numpy(pred_bin: np.ndarray, true_bin: np.ndarray) -> float:
    """
    Compute Dice between two binary NumPy masks.

    This matches compute_dice_np used in tb.py and ihc.py, where masks are
    thresholded to {0,1} before evaluation.
    """
    pred = pred_bin.astype(np.uint8).ravel()
    targ = true_bin.astype(np.uint8).ravel()
    inter = (pred & targ).sum()
    return (2.0 * inter) / (pred.sum() + targ.sum() + 1e-8)


def iou_from_numpy(pred_bin: np.ndarray, true_bin: np.ndarray) -> float:
    """
    Compute IoU between two binary NumPy masks.
    """
    pred = pred_bin.astype(np.uint8).ravel()
    targ = true_bin.astype(np.uint8).ravel()
    inter = (pred & targ).sum()
    union = pred.sum() + targ.sum() - inter
    return (inter + 1e-8) / (union + 1e-8)
