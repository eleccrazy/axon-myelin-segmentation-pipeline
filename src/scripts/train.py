"""
File: train.py
Description: Command-line entrypoint for training TB, IHC and mixed-stain models.

Author: Gizachew Kassa
Date Created: 10/12/2025
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

from src.training.entrypoints import (
    run_ihc_training,
    run_mixed_training,
    run_tb_training,
)
from src.utils.paths import PROJECT_ROOT

# Argument parsing and interactive experiment selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train U-Net models for TB, IHC, or mixed-stain LM segmentation."
    )
    parser.add_argument(
        "--exp",
        "--experiment",
        dest="experiment",
        choices=["tb", "ihc", "mixed"],
        help="Experiment to run: 'tb', 'ihc', or 'mixed'. "
        "If not provided, you will be prompted interactively.",
    )
    return parser.parse_args()


def choose_experiment_interactively() -> str:
    print("Select experiment to train:")
    print("[1] Toluidine blue (TB)")
    print("[2] IHC (DAB)")
    print("[3] Mixed-stain model")

    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "1":
        return "tb"
    if choice == "2":
        return "ihc"
    if choice == "3":
        return "mixed"

    print("Invalid choice. Please run again and select 1, 2 or 3.")
    sys.exit(1)


# Saving utilities


def _get_save_paths(exp: str) -> tuple[Path, Path]:
    """
    Return (model_path, history_path) for a given experiment key.

    The convention is:
      - models/<exp>/<filename>.pth
      - outputs/training_logs/<exp>/<filename>.json
    """
    models_root = PROJECT_ROOT / "models"
    logs_root = PROJECT_ROOT / "outputs" / "training_logs"

    # Create base directories
    models_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    # Per-experiment subfolders
    if exp == "tb":
        exp_name = "tb"
        model_filename = "unet_tb_best.pth"
        history_filename = "history_tb.json"
    elif exp == "ihc":
        exp_name = "ihc"
        model_filename = "unet_ihc_best.pth"
        history_filename = "history_ihc.json"
    elif exp == "mixed":
        exp_name = "mixed"
        model_filename = "unet_mixed_stain_best.pth"
        history_filename = "history_mixed.json"
    else:
        raise ValueError(f"Unknown experiment key: {exp}")

    model_dir = models_root / exp_name
    log_dir = logs_root / exp_name
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / model_filename
    history_path = log_dir / history_filename
    return model_path, history_path


def _save_model_and_history(
    exp: str,
    model: torch.nn.Module,
    history: dict,
) -> None:
    """
    Save model weights and training history for a given experiment.
    """
    model_path, history_path = _get_save_paths(exp)

    # Save model weights (best checkpoint already loaded by run_training)
    torch.save(model.state_dict(), model_path)

    # Attach a timestamp to the history for traceability
    history = dict(history)  # shallow copy to avoid side-effects
    history["_saved_at"] = datetime.now().isoformat()

    # Save history as JSON (float-serialisation via default=float)
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=float)

    print(f"\nSaved best model to:   {model_path}")
    print(f"Saved training history to: {history_path}")


# Main entrypoint


def main() -> None:
    args = parse_args()

    if args.experiment is None:
        exp = choose_experiment_interactively()
    else:
        exp = args.experiment

    try:
        if exp == "tb":
            print("\n→ Training toluidine blue (TB) model...")
            model, history = run_tb_training()

        elif exp == "ihc":
            print("\n→ Training IHC (DAB) model...")
            model, history = run_ihc_training()

        elif exp == "mixed":
            print("\n→ Training mixed-stain model...")
            model, history = run_mixed_training()

        else:
            # Should never happen because of choices/interactive guard
            print(f"Unknown experiment: {exp}")
            sys.exit(1)

        _save_model_and_history(exp, model, history)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
