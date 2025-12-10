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

# Stain-specific TB model (matches BluePreprocess in tb.py)
TB_TRAIN_PREPROCESS_CONFIG = TBPreprocessConfig(
    clahe_clip=2.0,
    clahe_tile=(8, 8),
    illum_sigma=31.0,
    denoise_sigma=0.5,
    use_percentile_clip=False,
)

# Mixed-stain TB branch (matches LABLPreprocess in mixed_experiment2.py)
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

# Stain-specific IHC model (matches DABPreprocess in ihc.py)
IHC_TRAIN_PREPROCESS_CONFIG = DABPreprocessConfig(
    clahe_clip=0.02,
    denoise_sigma=0.5,
    use_percentile_clip=False,
)

# Mixed-stain IHC branch (matches DABPreprocess in mixed_experiment2.py)
MIXED_IHC_PREPROCESS_CONFIG = DABPreprocessConfig(
    clahe_clip=0.02,
    denoise_sigma=0.5,
    use_percentile_clip=True,
)

"""
Example Usage:

# TB only
from src.preprocessing.tb_preprocessing import TBPreprocess
from src.configs.preprocessing_configs import TB_TRAIN_PREPROCESS_CONFIG

tb_pre = TBPreprocess(
    clahe_clip=TB_TRAIN_PREPROCESS_CONFIG.clahe_clip,
    clahe_tile=TB_TRAIN_PREPROCESS_CONFIG.clahe_tile,
    illum_sigma=TB_TRAIN_PREPROCESS_CONFIG.illum_sigma,
    denoise_sigma=TB_TRAIN_PREPROCESS_CONFIG.denoise_sigma,
    use_percentile_clip=TB_TRAIN_PREPROCESS_CONFIG.use_percentile_clip,
)

# TB in mixed-stain
from src.preprocessing.tb_preprocessing import TBPreprocess
from src.configs.preprocessing_configs import MIXED_TB_PREPROCESS_CONFIG

tb_pre_mixed = TBPreprocess(
    clahe_clip=MIXED_TB_PREPROCESS_CONFIG.clahe_clip,
    clahe_tile=MIXED_TB_PREPROCESS_CONFIG.clahe_tile,
    illum_sigma=MIXED_TB_PREPROCESS_CONFIG.illum_sigma,
    denoise_sigma=MIXED_TB_PREPROCESS_CONFIG.denoise_sigma,
    use_percentile_clip=MIXED_TB_PREPROCESS_CONFIG.use_percentile_clip,
)

# IHC only
from src.preprocessing.ihc_preprocessing import DABPreprocess
from src.configs.preprocessing_configs import IHC_TRAIN_PREPROCESS_CONFIG

ihc_pre = DABPreprocess(
    clahe_clip=IHC_TRAIN_PREPROCESS_CONFIG.clahe_clip,
    denoise_sigma=IHC_TRAIN_PREPROCESS_CONFIG.denoise_sigma,
    use_percentile_clip=IHC_TRAIN_PREPROCESS_CONFIG.use_percentile_clip,
)

# IHC in mixed-stain
from src.configs.preprocessing_configs import MIXED_IHC_PREPROCESS_CONFIG

ihc_pre_mixed = DABPreprocess(
    clahe_clip=MIXED_IHC_PREPROCESS_CONFIG.clahe_clip,
    denoise_sigma=MIXED_IHC_PREPROCESS_CONFIG.denoise_sigma,
    use_percentile_clip=MIXED_IHC_PREPROCESS_CONFIG.use_percentile_clip,
)
"""
