"""
File: common.py
Description: Shared preprocessing helpers for stain-specific pipelines.

Author: Gizachew Kassa
Date Created: 10/12/2025
"""

from __future__ import annotations

import numpy as np


def percentile_clip(x: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """
    Clip an array to the [low, high] percentiles and rescale to [0, 1].

    This is used to suppress extreme intensity values before normalisation.
    """
    p_low, p_high = np.percentile(x, (low, high))
    x = np.clip(x, p_low, p_high)
    return (x - p_low) / (p_high - p_low + 1e-8)


def zscore(x: np.ndarray) -> np.ndarray:
    """
    Apply per-image z-score normalisation.

    Returns an array with approximately zero mean and unit variance.
    """
    mean = float(x.mean())
    std = float(x.std() + 1e-8)
    return (x - mean) / std
