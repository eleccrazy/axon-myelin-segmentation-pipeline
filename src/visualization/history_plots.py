"""
File: history_plots.py
Description: Utilities for plotting training history curves (loss and Dice).
            This reads the JSON histories written by src.scripts.train and produces
            a 2-panel figure with:
                - train / val loss
                - validation Dice curves

            Supported experiments:
                - "tb"
                - "ihc"
                - "mixed" (overall Dice + per-stain Dice)
Author: Gizachew Kassa
Date Created: 12/12/2025
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def _moving_average(x: List[float], window: int) -> np.ndarray:
    """Simple centered moving average (no padding)."""
    if window <= 1 or len(x) < window:
        return np.asarray(x, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x_arr, kernel, mode="valid")


def _smooth_if_needed(x: List[float], window: int) -> np.ndarray:
    if window is None or window <= 1:
        return np.asarray(x, dtype=float)
    return _moving_average(x, window)


def load_history(
    exp: str,
    history_path: Optional[str] = None,
) -> Dict:
    """
    Load a training history JSON for a given experiment.

    Parameters
    ----------
    exp:
        One of "tb", "ihc", "mixed".
    history_path:
        Optional explicit path. If None, uses the default
        outputs/training_logs/<exp>/history_<exp>.json

    Returns
    -------
    history : dict
        Parsed JSON with keys like "train_loss", "val_loss", "val_dice", etc.
    """
    exp = exp.lower()
    if history_path is None:
        history_path = os.path.join(
            "outputs", "training_logs", exp, f"history_{exp}.json"
        )

    if not os.path.exists(history_path):
        raise FileNotFoundError(f"History file not found: {history_path}")

    with open(history_path, "r") as f:
        history = json.load(f)

    return history


def plot_history(
    exp: str,
    history: Dict,
    save_path: Optional[str] = None,
    smooth: int = 0,
    show: bool = False,
) -> None:
    """
    Plot loss + Dice curves for a single experiment.

    Parameters
    ----------
    exp:
        "tb", "ihc" or "mixed".
    history:
        Parsed JSON dict from load_history().
    save_path:
        Where to save the figure. If None, a default under
        outputs/figures/<exp>/curves/ is used.
    smooth:
        Optional moving-average window for smoothing the curves (>= 2).
    show:
        If True, also display the figure (useful locally).
    """
    exp = exp.lower()

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    n_epochs = len(train_loss)

    if save_path is None:
        save_dir = os.path.join("outputs", "figures", exp, "curves")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"training_curves_{exp}.png")

    # x-axis: epochs 1..N (if smoothing, adjust length)
    if smooth and smooth > 1:
        train_loss_s = _smooth_if_needed(train_loss, smooth)
        val_loss_s = _smooth_if_needed(val_loss, smooth)
        # After convolution(mode="valid"), length shrinks
        offset = len(train_loss) - len(train_loss_s)
        epochs = np.arange(1 + offset // 2, 1 + offset // 2 + len(train_loss_s))
    else:
        train_loss_s = np.asarray(train_loss, dtype=float)
        val_loss_s = np.asarray(val_loss, dtype=float)
        epochs = np.arange(1, n_epochs + 1)

    # --- Figure ---
    fig, (ax_loss, ax_dice) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Loss subplot ----
    ax_loss.plot(epochs, train_loss_s, label="Train")
    ax_loss.plot(epochs, val_loss_s, label="Val")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # ---- Dice subplot ----
    # Common overall Dice
    val_dice = history.get("val_dice", [])
    if val_dice:
        val_dice_s = _smooth_if_needed(val_dice, smooth)
        if len(val_dice_s) == len(epochs):
            ax_dice.plot(epochs, val_dice_s, label="Dice")
        else:
            # Fallback: align to first len(val_dice_s) epochs
            ax_dice.plot(
                np.arange(1, 1 + len(val_dice_s)),
                val_dice_s,
                label="Dice",
            )

    if exp == "mixed":
        # Get per-stain Dice curves
        val_dice_tb = history.get("val_dice_tb")
        val_dice_ihc = history.get("val_dice_ihc")

        if val_dice_tb:
            val_dice_tb_s = _smooth_if_needed(val_dice_tb, smooth)
            ax_dice.plot(
                np.arange(1, 1 + len(val_dice_tb_s)),
                val_dice_tb_s,
                label="Dice TB",
            )

        if val_dice_ihc:
            val_dice_ihc_s = _smooth_if_needed(val_dice_ihc, smooth)
            ax_dice.plot(
                np.arange(1, 1 + len(val_dice_ihc_s)),
                val_dice_ihc_s,
                label="Dice IHC",
            )

    ax_dice.set_title("Dice Curves")
    ax_dice.set_xlabel("Epoch")
    ax_dice.set_ylabel("Dice")
    ax_dice.set_ylim(0.0, 1.0)
    ax_dice.legend()
    ax_dice.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_history_from_file(
    exp: str,
    history_path: Optional[str] = None,
    save_path: Optional[str] = None,
    smooth: int = 0,
    show: bool = False,
) -> None:
    """
    Convenience wrapper: load history from disk and plot.
    """
    history = load_history(exp, history_path)
    plot_history(exp, history, save_path=save_path, smooth=smooth, show=show)
