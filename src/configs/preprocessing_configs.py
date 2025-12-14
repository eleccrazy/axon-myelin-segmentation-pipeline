"""
File: preprocessing_configs.py
Description: Preprocessing parameter presets for TB, IHC and mixed-stain experiments.

Author: Gizachew Kassa
Date Created: 10/12/2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class TBPreprocessConfig:
    """Configuration for toluidine blue preprocessing."""

    clahe_clip: float
    clahe_tile: Tuple[int, int]
    illum_sigma: float
    denoise_sigma: float
    use_percentile_clip: bool


@dataclass(frozen=True)
class DABPreprocessConfig:
    """Configuration for IHC (DAB) preprocessing."""

    clahe_clip: float
    denoise_sigma: float
    use_percentile_clip: bool


# -------------------------------------------------------------------------
# Toluidine blue (TB) presets
# -------------------------------------------------------------------------

# Stain-specific TB model
TB_TRAIN_PREPROCESS_CONFIG = TBPreprocessConfig(
    clahe_clip=2.0,
    clahe_tile=(8, 8),
    illum_sigma=31.0,
    denoise_sigma=0.5,
    use_percentile_clip=False,
)

# Mixed-stain TB branch
MIXED_TB_PREPROCESS_CONFIG = TBPreprocessConfig(
    clahe_clip=2.0,
    clahe_tile=(8, 8),
    illum_sigma=30.0,
    denoise_sigma=0.5,
    use_percentile_clip=True,
)


# -------------------------------------------------------------------------
# IHC (DAB) presets
# -------------------------------------------------------------------------

# Stain-specific IHC model
IHC_TRAIN_PREPROCESS_CONFIG = DABPreprocessConfig(
    clahe_clip=0.02,
    denoise_sigma=0.5,
    use_percentile_clip=False,
)

# Mixed-stain IHC branch
MIXED_IHC_PREPROCESS_CONFIG = DABPreprocessConfig(
    clahe_clip=0.02,
    denoise_sigma=0.5,
    use_percentile_clip=True,
)
