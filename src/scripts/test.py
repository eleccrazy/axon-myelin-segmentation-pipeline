"""
File: test.py
Description: Command-line entrypoint for evaluating TB, IHC and mixed-stain models.

Author: Gizachew Kassa
Date Created: 11/12/2025
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from src.evaluation.test_entrypoints import run_ihc_test, run_mixed_test, run_tb_test
from src.utils.paths import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TB, IHC or mixed-stain models on validation+test sets."
    )
    parser.add_argument(
        "--exp",
        "--experiment",
        dest="experiment",
        choices=["tb", "ihc", "mixed"],
        help="Experiment to evaluate: 'tb', 'ihc', or 'mixed'. "
        "If not provided, you will be prompted interactively.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for evaluation (default: 2).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (default: 0).",
    )
    return parser.parse_args()


def choose_experiment_interactively() -> str:
    print("Select experiment to evaluate:")
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


def _get_metrics_save_path(exp: str) -> Path:
    metrics_root = PROJECT_ROOT / "outputs" / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)

    if exp == "tb":
        sub = "tb"
        filename = "metrics_tb.json"
    elif exp == "ihc":
        sub = "ihc"
        filename = "metrics_ihc.json"
    elif exp == "mixed":
        sub = "mixed"
        filename = "metrics_mixed.json"
    else:
        raise ValueError(f"Unknown experiment key: {exp}")

    exp_dir = metrics_root / sub
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir / filename


def main() -> None:
    args = parse_args()

    if args.experiment is None:
        exp = choose_experiment_interactively()
    else:
        exp = args.experiment

    if exp == "tb":
        print("\n→ Evaluating toluidine blue (TB) model...")
        result = run_tb_test(batch_size=args.batch_size, num_workers=args.num_workers)
    elif exp == "ihc":
        print("\n→ Evaluating IHC (DAB) model...")
        result = run_ihc_test(batch_size=args.batch_size, num_workers=args.num_workers)
    elif exp == "mixed":
        print("\n→ Evaluating mixed-stain model...")
        result = run_mixed_test(
            batch_size=args.batch_size, num_workers=args.num_workers
        )
    else:
        print(f"Unknown experiment: {exp}")
        sys.exit(1)

    # Attach timestamp and save JSON
    result = dict(result)
    result["_evaluated_at"] = datetime.now().isoformat()

    out_path = _get_metrics_save_path(exp)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved evaluation metrics to: {out_path}")


if __name__ == "__main__":
    main()
