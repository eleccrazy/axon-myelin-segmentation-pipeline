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

# Matches ALPHA_BCE = 0.4, boundary_weight = 3.0, sigma = 2 from mixed_experiment2.
MIXED_LOSS_CONFIG = BCEDiceBoundaryConfig(
    alpha=0.4,
    boundary_weight=3.0,
    sigma=2.0,
    use_pos_weight=False,
    pos_weight_value=None,
)


"""
Example usages:

# TB training
from src.training.losses import DiceBCELoss
from src.configs.loss_configs import TB_LOSS_CONFIG

pos_w = None
if TB_LOSS_CONFIG.use_pos_weight and TB_LOSS_CONFIG.pos_weight_value is not None:
    pos_w = torch.tensor([TB_LOSS_CONFIG.pos_weight_value], device=device)

criterion = DiceBCELoss(
    alpha=TB_LOSS_CONFIG.alpha,
    pos_weight=pos_w,
)

# IHC training
from src.training.losses import DiceBCELoss
from src.configs.loss_configs import IHC_LOSS_CONFIG

criterion = DiceBCELoss(
    alpha=IHC_LOSS_CONFIG.alpha,
    pos_weight=None,  # no pos weight for IHC
)

# Mixed-stain training
from src.training.losses import BCEDiceBoundary
from src.configs.loss_configs import MIXED_LOSS_CONFIG

pos_w = None
if MIXED_LOSS_CONFIG.use_pos_weight and MIXED_LOSS_CONFIG.pos_weight_value is not None:
    pos_w = torch.tensor([MIXED_LOSS_CONFIG.pos_weight_value], device=device)

criterion = BCEDiceBoundary(
    alpha=MIXED_LOSS_CONFIG.alpha,
    boundary_weight=MIXED_LOSS_CONFIG.boundary_weight,
    sigma=MIXED_LOSS_CONFIG.sigma,
    pos_weight=pos_w,
)
"""
