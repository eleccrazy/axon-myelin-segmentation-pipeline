"""
File: plot_history.py
Description: Script to plot training history (loss and Dice curves) for experiments.
Author: Gizachew Kassa
Date Created: 12/12/2025
"""

from __future__ import annotations

import argparse

from src.visualization.history_plots import plot_history_from_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training history (loss + Dice curves) for an experiment."
    )
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        choices=["tb", "ihc", "mixed"],
        help="Experiment to plot (tb, ihc, mixed).",
    )
    parser.add_argument(
        "--history-path",
        type=str,
        default=None,
        help=(
            "Optional explicit path to history JSON. "
            "If omitted, uses outputs/training_logs/<exp>/history_<exp>.json"
        ),
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help=(
            "Optional explicit path to save the figure. "
            "If omitted, uses outputs/figures/<exp>/curves/training_curves_<exp>.png"
        ),
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=0,
        help="Optional moving-average window for smoothing curves (0 or 1 = no smoothing).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="If set, also display the figure (useful when running locally).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_history_from_file(
        exp=args.exp,
        history_path=args.history_path,
        save_path=args.save_path,
        smooth=args.smooth,
        show=args.show,
    )


if __name__ == "__main__":
    main()
