"""
File: paths.py
Description: This module defines and manages key directory paths used throughout the project.
Author: Gizachew Kassa
Date Created: 9/12/2025
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # project_root
DATA_ROOT = PROJECT_ROOT / "data"
ORIGIONAL_DATA_DIR = DATA_ROOT / "origional"

TB_DIR = ORIGIONAL_DATA_DIR / "data2"
IHC_DIR = ORIGIONAL_DATA_DIR / "data2ihc"

MODELS_ROOT = PROJECT_ROOT / "models"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"


SPLIT_ROOT = DATA_ROOT / "splitted"

TB_SPLIT_DIR = SPLIT_ROOT / "tb"
IHC_SPLIT_DIR = SPLIT_ROOT / "ihc"
