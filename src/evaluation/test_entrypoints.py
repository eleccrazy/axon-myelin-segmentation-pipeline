"""
File: test_entrypoints.py
Description: Evaluation entrypoints for TB, IHC and mixed-stain models
             with validation-based threshold calibration and test metrics.

Author: Gizachew Kassa
Date Created: 11/12/2025
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.configs.preprocessing_configs import (
    IHC_TRAIN_PREPROCESS_CONFIG,
    MIXED_IHC_PREPROCESS_CONFIG,
    MIXED_TB_PREPROCESS_CONFIG,
    TB_TRAIN_PREPROCESS_CONFIG,
)
from src.datasets.ihc_dataset import IHCLMDataset
from src.datasets.mixed_dataset import MixedLMDataset
from src.datasets.tb_dataset import TBLMDataset
from src.evaluation.metrics import dice_iou_over_loader
from src.models.unet import UNet
from src.models.unet_stain import UNetStain
from src.utils.paths import PROJECT_ROOT

# -------------------------------------------------------------------------
# Paths and basic setup
# -------------------------------------------------------------------------

MODELS_ROOT = PROJECT_ROOT / "models"
METRICS_ROOT = PROJECT_ROOT / "outputs" / "metrics"

TB_MODEL_PATH = MODELS_ROOT / "tb" / "unet_tb_best.pth"
IHC_MODEL_PATH = MODELS_ROOT / "ihc" / "unet_ihc_best.pth"
MIXED_MODEL_PATH = MODELS_ROOT / "mixed" / "unet_mixed_stain_best.pth"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# DataLoader helpers (val + test only, no augmentation)
# -------------------------------------------------------------------------


def _pin_memory(device: torch.device) -> bool:
    return device.type == "cuda"


def _make_tb_val_test_loaders(
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    val_ds = TBLMDataset(split="val", augment=None)
    test_ds = TBLMDataset(split="test", augment=None)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin_memory(device),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin_memory(device),
    )
    return val_loader, test_loader


def _make_ihc_val_test_loaders(
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    val_ds = IHCLMDataset(split="val", augment=None)
    test_ds = IHCLMDataset(split="test", augment=None)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin_memory(device),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin_memory(device),
    )
    return val_loader, test_loader


def _make_mixed_val_test_loaders(
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    val_ds = MixedLMDataset(split="val", augment=None)
    test_ds = MixedLMDataset(split="test", augment=None)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin_memory(device),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin_memory(device),
    )
    return val_loader, test_loader


# -------------------------------------------------------------------------
# Threshold calibration (binary foreground)
# -------------------------------------------------------------------------


def _calibrate_threshold_binary(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    thresholds: List[float],
) -> Tuple[float, List[Dict[str, float]]]:
    """
    Sweep thresholds on the validation set and pick the one with the best Dice.
    Returns the best threshold and a list of per-threshold scores.
    """
    model.eval()
    best_t = thresholds[0]
    best_dice = -1.0
    records: List[Dict[str, float]] = []

    for t in thresholds:
        dice, iou = dice_iou_over_loader(
            model=model,
            loader=val_loader,
            device=device,
            threshold=t,
        )
        records.append(
            {
                "threshold": float(t),
                "dice": float(dice),
                "iou": float(iou),
            }
        )
        if dice > best_dice:
            best_dice = dice
            best_t = t

    return float(best_t), records


# -------------------------------------------------------------------------
# Precision and recall (binary) over a DataLoader
# -------------------------------------------------------------------------


def _precision_recall_over_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> Tuple[float, float]:
    """
    Compute precision and recall over a DataLoader for a given threshold.
    """
    model.eval()
    eps = 1e-8

    tp = 0.0
    fp = 0.0
    fn = 0.0

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                images, masks = batch
            else:
                images, masks, _stains = batch  # mixed dataset case

            images = images.to(device)
            masks = masks.to(device)

            logits = (
                model(images) if len(batch) == 2 else model(images, _stains.to(device))
            )
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            # Flatten for counting
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)

            tp += float(((preds_flat == 1) & (masks_flat == 1)).sum())
            fp += float(((preds_flat == 1) & (masks_flat == 0)).sum())
            fn += float(((preds_flat == 0) & (masks_flat == 1)).sum())

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return precision, recall


# -------------------------------------------------------------------------
# Mixed-stain: per-stain Dice on a loader (for test)
# -------------------------------------------------------------------------


def _mixed_per_stain_dice(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> Dict[str, float]:
    """
    Compute mean Dice separately for TB and IHC samples on a given loader.
    """
    from src.evaluation.metrics import (
        dice_iou_from_logits,  # local import to avoid cycles
    )

    model.eval()
    dice_tb_sum, n_tb = 0.0, 0
    dice_ihc_sum, n_ihc = 0.0, 0

    with torch.no_grad():
        for images, masks, stains in loader:
            images = images.to(device)
            masks = masks.to(device)
            stains = stains.to(device)

            logits = model(images, stains)

            tb_mask = stains == 0
            ihc_mask = stains == 1

            if tb_mask.any():
                d_tb, _ = dice_iou_from_logits(
                    logits[tb_mask],
                    masks[tb_mask],
                    threshold=threshold,
                )
                dice_tb_sum += d_tb
                n_tb += 1

            if ihc_mask.any():
                d_ihc, _ = dice_iou_from_logits(
                    logits[ihc_mask],
                    masks[ihc_mask],
                    threshold=threshold,
                )
                dice_ihc_sum += d_ihc
                n_ihc += 1

    metrics: Dict[str, float] = {}
    if n_tb > 0:
        metrics["dice_tb"] = dice_tb_sum / n_tb
    if n_ihc > 0:
        metrics["dice_ihc"] = dice_ihc_sum / n_ihc
    return metrics


# -------------------------------------------------------------------------
# Mixed-stain: per-stain Precision and Recall on a loader (for test)
# -------------------------------------------------------------------------
def _mixed_per_stain_precision_recall(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> Dict[str, float]:
    """
    Compute precision and recall separately for TB (stain_id=0)
    and IHC (stain_id=1) over a mixed-stain DataLoader.
    """
    model.eval()
    eps = 1e-8

    # Counters for TB
    tp_tb = fp_tb = fn_tb = 0.0
    # Counters for IHC
    tp_ihc = fp_ihc = fn_ihc = 0.0

    with torch.no_grad():
        for images, masks, stains in loader:
            images = images.to(device)
            masks = masks.to(device)
            stains = stains.to(device)

            logits = model(images, stains)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            # Loop over samples in the batch and accumulate per stain
            batch_size = stains.shape[0]
            for i in range(batch_size):
                stain_id = int(stains[i].item())
                pred_i = preds[i].view(-1)
                mask_i = masks[i].view(-1)

                tp = float(((pred_i == 1) & (mask_i == 1)).sum())
                fp = float(((pred_i == 1) & (mask_i == 0)).sum())
                fn = float(((pred_i == 0) & (mask_i == 1)).sum())

                if stain_id == 0:
                    tp_tb += tp
                    fp_tb += fp
                    fn_tb += fn
                elif stain_id == 1:
                    tp_ihc += tp
                    fp_ihc += fp
                    fn_ihc += fn

    precision_tb = tp_tb / (tp_tb + fp_tb + eps) if tp_tb + fp_tb > 0 else 0.0
    recall_tb = tp_tb / (tp_tb + fn_tb + eps) if tp_tb + fn_tb > 0 else 0.0

    precision_ihc = tp_ihc / (tp_ihc + fp_ihc + eps) if tp_ihc + fp_ihc > 0 else 0.0
    recall_ihc = tp_ihc / (tp_ihc + fn_ihc + eps) if tp_ihc + fn_ihc > 0 else 0.0

    return {
        "precision_tb": precision_tb,
        "recall_tb": recall_tb,
        "precision_ihc": precision_ihc,
        "recall_ihc": recall_ihc,
    }


# -------------------------------------------------------------------------
# TB evaluation
# -------------------------------------------------------------------------


def run_tb_test(
    batch_size: int = 1,
    num_workers: int = 0,
) -> Dict[str, object]:
    """
    Evaluate the TB model: calibrate threshold on val, then score test set.
    Returns a dict with threshold, metrics and calibration sweep records.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nRunning TB evaluation...")

    val_loader, test_loader = _make_tb_val_test_loaders(
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
    )

    # Load model
    model = UNet(in_channels=1, out_channels=1).to(device)
    state = torch.load(TB_MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Threshold sweep, 0.20–0.60 with 0.025 steps
    thresholds = [float(t) for t in np.linspace(0.20, 0.60, 17)]
    t_star, sweep = _calibrate_threshold_binary(model, val_loader, device, thresholds)
    print(f"  Calibrated threshold t*: {t_star:.3f}")

    # Test metrics
    dice, iou = dice_iou_over_loader(model, test_loader, device, threshold=t_star)
    precision, recall = _precision_recall_over_loader(
        model, test_loader, device, threshold=t_star
    )

    result: Dict[str, object] = {
        "experiment": "tb",
        "threshold": float(t_star),
        "metrics_test": {
            "dice": float(dice),
            "iou": float(iou),
            "precision": float(precision),
            "recall": float(recall),
        },
        "calibration": sweep,
    }
    return result


# -------------------------------------------------------------------------
# IHC evaluation
# -------------------------------------------------------------------------


def run_ihc_test(
    batch_size: int = 1,
    num_workers: int = 0,
) -> Dict[str, object]:
    """
    Evaluate the IHC model: calibrate threshold on val, then score test set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nRunning IHC evaluation...")

    val_loader, test_loader = _make_ihc_val_test_loaders(
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
    )

    model = UNet(in_channels=1, out_channels=1).to(device)
    state = torch.load(IHC_MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    thresholds = [float(t) for t in np.linspace(0.20, 0.60, 17)]
    t_star, sweep = _calibrate_threshold_binary(model, val_loader, device, thresholds)
    print(f"  Calibrated threshold t*: {t_star:.3f}")

    dice, iou = dice_iou_over_loader(model, test_loader, device, threshold=t_star)
    precision, recall = _precision_recall_over_loader(
        model, test_loader, device, threshold=t_star
    )

    result: Dict[str, object] = {
        "experiment": "ihc",
        "threshold": float(t_star),
        "metrics_test": {
            "dice": float(dice),
            "iou": float(iou),
            "precision": float(precision),
            "recall": float(recall),
        },
        "calibration": sweep,
    }
    return result


# -------------------------------------------------------------------------
# Mixed-stain evaluation
# -------------------------------------------------------------------------


def run_mixed_test(
    batch_size: int = 1,
    num_workers: int = 0,
) -> Dict[str, object]:
    """
    Evaluate the mixed-stain model with threshold calibration and
    per-stain precision/recall on the test set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nRunning mixed-stain evaluation...")

    val_loader, test_loader = _make_mixed_val_test_loaders(
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
    )

    model = UNetStain(in_channels=1, out_channels=1).to(device)
    state = torch.load(MIXED_MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Threshold sweep: 0.10–0.90 with 0.05 steps
    thresholds = [float(t) for t in np.linspace(0.10, 0.90, 17)]
    t_star, sweep = _calibrate_threshold_binary(model, val_loader, device, thresholds)
    print(f"  Calibrated threshold t*: {t_star:.3f}")

    # Overall test metrics at t*
    dice, iou = dice_iou_over_loader(model, test_loader, device, threshold=t_star)
    precision, recall = _precision_recall_over_loader(
        model, test_loader, device, threshold=t_star
    )

    # Per-stain precision/recall at t*
    per_stain_pr = _mixed_per_stain_precision_recall(
        model, test_loader, device, threshold=t_star
    )

    result: Dict[str, object] = {
        "experiment": "mixed",
        "threshold": float(t_star),
        "metrics_test": {
            "dice": float(dice),
            "iou": float(iou),
            "precision": float(precision),
            "recall": float(recall),
            "precision_tb": float(per_stain_pr["precision_tb"]),
            "recall_tb": float(per_stain_pr["recall_tb"]),
            "precision_ihc": float(per_stain_pr["precision_ihc"]),
            "recall_ihc": float(per_stain_pr["recall_ihc"]),
        },
        "calibration": sweep,
    }
    return result
