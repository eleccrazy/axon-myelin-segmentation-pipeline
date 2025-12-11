"""
File: dataloaders.py
Description: Helper functions to build DataLoaders for TB, IHC and mixed datasets.

Author: Gizachew Kassa
Date Created: 10/12/2025
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader

from src.datasets.ihc_dataset import IHCLMDataset
from src.datasets.mixed_dataset import MixedLMDataset
from src.datasets.tb_dataset import TBLMDataset


def _pin_memory(device: torch.device) -> bool:
    return device.type == "cuda"


def make_tb_dataloaders(
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders for the TB experiment.
    """
    train_ds = TBLMDataset(
        split="train", augment=None
    )  # plug augmentation here if needed
    val_ds = TBLMDataset(split="val", augment=None)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=_pin_memory(device),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin_memory(device),
    )

    return train_loader, val_loader


def make_ihc_dataloaders(
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders for the IHC experiment.
    """
    train_ds = IHCLMDataset(split="train", augment=None)
    val_ds = IHCLMDataset(split="val", augment=None)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=_pin_memory(device),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin_memory(device),
    )

    return train_loader, val_loader


def make_mixed_dataloaders(
    batch_size: int,
    device: torch.device,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders for the mixed-stain experiment.
    """
    train_ds = MixedLMDataset(split="train", augment=None)
    val_ds = MixedLMDataset(split="val", augment=None)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=_pin_memory(device),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin_memory(device),
    )

    return train_loader, val_loader
