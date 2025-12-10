"""
File: loops.py
Description: Shared training and validation loops for TB, IHC and mixed-stain models.

Author: Gizachew Kassa
Date Created: 10/12/2025
"""

from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .callbacks import EarlyStopping


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Run one training epoch and return the mean training loss.
    """
    model.train()
    running = 0.0

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        if len(batch) == 2:
            imgs, masks = batch
            stains = None
        else:
            imgs, masks, stains = batch

        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        if stains is None:
            logits = model(imgs)
        else:
            logits = model(imgs, stains)

        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running += loss.item()
        pbar.set_postfix(loss=f"{running / max(1, pbar.n):.4f}")

    return running / max(1, len(loader))


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Run one validation epoch and return the mean validation loss.
    """
    model.eval()
    running = 0.0

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                imgs, masks = batch
                stains = None
            else:
                imgs, masks, stains = batch

            imgs = imgs.to(device)
            masks = masks.to(device)

            if stains is None:
                logits = model(imgs)
            else:
                logits = model(imgs, stains)

            loss = criterion(logits, masks)
            running += loss.item()

    return running / max(1, len(loader))


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    early_stopping: Optional[EarlyStopping] = None,
    scheduler: Optional[object] = None,
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Full training loop with optional early stopping and LR scheduler.

    This function mirrors the structure of the original training loops in
    tb.py, ihc.py and mixed_experiment2.py:

    - For each epoch, run a training phase and a validation phase.
    - Track the best validation loss and restore the corresponding weights.
    - Optionally apply early stopping based on validation loss.
    - Optionally update a scheduler after each epoch.
    """
    model.to(device)

    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    if early_stopping is None:
        early_stopping = EarlyStopping(patience=10, min_delta=0.0, mode="min")

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # --------- TRAIN ---------
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        # --------- VALIDATE ---------
        val_loss = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # --------- LR SCHEDULE ---------
        if scheduler is not None:
            # WarmupCosineLR and most PyTorch schedulers update via step(epoch)
            try:
                scheduler.step(epoch)
            except TypeError:
                # Fallback for schedulers that do not take epoch as argument
                scheduler.step()
        # Read current LR from the first param group (assumes single LR)
        current_lr = optimizer.param_groups[0].get("lr", None)
        history["lr"].append(current_lr)

        print(
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"LR: {current_lr:.6f}"
            if current_lr is not None
            else f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}"
        )

        # --------- EARLY STOPPING ---------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            print("  → New best model (val loss improved).")

        stop = early_stopping.step(val_loss)
        if stop:
            print("  → Early stopping triggered.")
            break

    # Restore best weights
    model.load_state_dict(best_state)
    print("\nBest model weights restored based on validation loss.")

    return model, history
