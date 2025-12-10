"""
File: entrypoints.py
Description: High-level training entrypoints for TB, IHC and mixed-stain models.

Author: Gizachew Kassa
Date Created: 10/12/2025
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.optim as optim

from src.configs.loss_configs import IHC_LOSS_CONFIG, MIXED_LOSS_CONFIG, TB_LOSS_CONFIG
from src.datasets.dataloaders import (
    make_ihc_dataloaders,
    make_mixed_dataloaders,
    make_tb_dataloaders,
)
from src.evaluation.metrics import dice_iou_from_logits, dice_iou_over_loader
from src.models.unet import UNet
from src.models.unet_stain import UNetStain
from src.training.callbacks import EarlyStopping
from src.training.loops import run_training
from src.training.losses import BCEDiceBoundary, DiceBCELoss

# -----------------------
# Global hyperparameters
# -----------------------

TB_BATCH_SIZE = 2
IHC_BATCH_SIZE = 2
MIXED_BATCH_SIZE = 2

TB_EPOCHS = 60
IHC_EPOCHS = 60
MIXED_EPOCHS = 70

LR = 1e-4  # used in all three experiments
PATIENCE = 7
MIN_DELTA = 1e-4


def _make_pos_weight(config, device: torch.device):
    """
    Build a positive-class weight tensor from a loss config, if enabled.
    """
    if getattr(config, "use_pos_weight", False) and config.pos_weight_value is not None:
        return torch.tensor([config.pos_weight_value], device=device)
    return None


# -----------------------
# Validation metrics
# -----------------------


def tb_val_metrics(model, val_loader, device) -> Dict[str, float]:
    """
    Toluidine blue: mean Dice at threshold 0.5 on validation set.
    """
    dice, _ = dice_iou_over_loader(
        model=model,
        loader=val_loader,
        device=device,
        threshold=0.5,
    )
    return {"val_dice": dice}


def ihc_val_metrics(model, val_loader, device) -> Dict[str, float]:
    """
    IHC: mean Dice at threshold 0.5 on validation set.
    """
    dice, _ = dice_iou_over_loader(
        model=model,
        loader=val_loader,
        device=device,
        threshold=0.5,
    )
    return {"val_dice": dice}


def mixed_val_metrics(model, val_loader, device, threshold: float = 0.5) -> Dict[str, float]:
    """
    Mixed-stain: overall Dice and IoU on validation set, plus per-stain Dice.
    Assumes val_loader yields (images, masks, stains) and that stains encode
    TB as 0 and IHC as 1.
    """
    model.eval()

    dice_all_sum, iou_all_sum, n_batches = 0.0, 0.0, 0
    dice_tb_sum, n_tb = 0.0, 0
    dice_ihc_sum, n_ihc = 0.0, 0

    with torch.no_grad():
        for images, masks, stains in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            stains = stains.to(device)

            logits = model(images, stains)

            # overall Dice & IoU
            d_batch, i_batch = dice_iou_from_logits(
                logits, masks, threshold=threshold
            )
            dice_all_sum += d_batch
            iou_all_sum += i_batch
            n_batches += 1

            # per-stain Dice
            tb_mask = (stains == 0)
            ihc_mask = (stains == 1)

            if tb_mask.any():
                d_tb, _ = dice_iou_from_logits(
                    logits[tb_mask], masks[tb_mask], threshold=threshold
                )
                dice_tb_sum += d_tb
                n_tb += 1

            if ihc_mask.any():
                d_ihc, _ = dice_iou_from_logits(
                    logits[ihc_mask], masks[ihc_mask], threshold=threshold
                )
                dice_ihc_sum += d_ihc
                n_ihc += 1

    metrics: Dict[str, float] = {
        "val_dice": dice_all_sum / max(n_batches, 1),
        "val_iou": iou_all_sum / max(n_batches, 1),
    }
    if n_tb > 0:
        metrics["val_dice_tb"] = dice_tb_sum / n_tb
    if n_ihc > 0:
        metrics["val_dice_ihc"] = dice_ihc_sum / n_ihc

    return metrics


# -----------------------
# Training entrypoints
# -----------------------


def run_tb_training():
    """
    Train the stain-specific toluidine blue model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = make_tb_dataloaders(
        batch_size=TB_BATCH_SIZE,
        device=device,
    )

    model = UNet(in_channels=1, out_channels=1).to(device)

    pos_weight = _make_pos_weight(TB_LOSS_CONFIG, device)
    criterion = DiceBCELoss(
        alpha=TB_LOSS_CONFIG.alpha,
        pos_weight=pos_weight,
    )

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = None  # no scheduler in original tb.py

    early_stopping = EarlyStopping(
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        mode="min",
    )

    model, history = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=TB_EPOCHS,
        early_stopping=early_stopping,
        scheduler=scheduler,
        scheduler_mode=None,
        val_metrics_fn=tb_val_metrics,
    )

    return model, history


def run_ihc_training():
    """
    Train the stain-specific IHC (DAB) model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = make_ihc_dataloaders(
        batch_size=IHC_BATCH_SIZE,
        device=device,
    )

    model = UNet(in_channels=1, out_channels=1).to(device)

    criterion = DiceBCELoss(
        alpha=IHC_LOSS_CONFIG.alpha,
        pos_weight=None,  # no pos_weight for IHC
    )

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ReduceLROnPlateau on validation loss, as in ihc.py
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    early_stopping = EarlyStopping(
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        mode="min",
    )

    # small wrapper to pass val_loss into scheduler
    def ihc_run_training(*args, **kwargs):
        return run_training(
            *args,
            **kwargs,
            scheduler_mode="plateau",
        )

    model, history = ihc_run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=IHC_EPOCHS,
        early_stopping=early_stopping,
        scheduler=scheduler,
        val_metrics_fn=ihc_val_metrics,
    )

    return model, history


class WarmupCosineLR:
    """
    Lightweight warmup + cosine LR scheduler.

    This mirrors the WarmupCosineLR used in mixed_experiment2.py. It adjusts
    the optimiser learning rate in-place and returns the current LR from step().
    """

    def __init__(self, optimizer, base_lr: float, warmup_epochs: int, total_epochs: int):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

    def step(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / (self.warmup_epochs + 1e-8)
        else:
            progress = (epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            lr = 0.5 * self.base_lr * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)))
            lr = float(lr)

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


def run_mixed_training():
    """
    Train the mixed-stain model with stain embedding and boundary-aware loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = make_mixed_dataloaders(
        batch_size=MIXED_BATCH_SIZE,
        device=device,
    )

    model = UNetStain(
        in_channels=1,
        out_channels=1,
        # other UNetStain args should match mixed_experiment2.py
    ).to(device)

    criterion = BCEDiceBoundary(
        alpha=MIXED_LOSS_CONFIG.alpha,
        boundary_weight=MIXED_LOSS_CONFIG.boundary_weight,
        sigma=MIXED_LOSS_CONFIG.sigma,
        pos_weight=None,
    )

    optimizer = optim.Adam(model.parameters(), lr=LR)

    scheduler = WarmupCosineLR(
        optimizer=optimizer,
        base_lr=LR,
        warmup_epochs=3,
        total_epochs=MIXED_EPOCHS,
    )

    early_stopping = EarlyStopping(
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        mode="min",
    )

    model, history = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=MIXED_EPOCHS,
        early_stopping=early_stopping,
        scheduler=scheduler,
        scheduler_mode="epoch",
        val_metrics_fn=lambda m, dl, dev: mixed_val_metrics(m, dl, dev, threshold=0.5),
    )

    return model, history
