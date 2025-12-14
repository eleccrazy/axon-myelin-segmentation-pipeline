"""
File: loss_configs.py
Description: Loss function presets for TB, IHC and mixed-stain experiments.

Author: Gizachew Kassa
Date Created: 10/12/2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DiceBCELossConfig:
    """Configuration for the standard Dice + BCEWithLogits composite loss."""

    alpha: float
    use_pos_weight: bool
    pos_weight_value: Optional[float] = None  # scalar weight for the positive class


@dataclass(frozen=True)
class BCEDiceBoundaryConfig:
    """Configuration for the BCE + boundary-aware Dice loss."""

    alpha: float
    boundary_weight: float
    sigma: float
    use_pos_weight: bool
    pos_weight_value: Optional[float] = None


# -------------------------------------------------------------------------
# TB / IHC stain-specific models (Dice + BCE)
# -------------------------------------------------------------------------

# Both TB and IHC use alpha = 0.5.
TB_LOSS_CONFIG = DiceBCELossConfig(
    alpha=0.5,
    use_pos_weight=True,
    pos_weight_value=None,  # filled at runtime if you compute it from masks
)

IHC_LOSS_CONFIG = DiceBCELossConfig(
    alpha=0.5,
    use_pos_weight=False,
    pos_weight_value=None,
)


# -------------------------------------------------------------------------
# Mixed-stain model (BCE + boundary-aware Dice)
# -------------------------------------------------------------------------

MIXED_LOSS_CONFIG = BCEDiceBoundaryConfig(
    alpha=0.4,
    boundary_weight=3.0,
    sigma=2.0,
    use_pos_weight=False,
    pos_weight_value=None,
)
