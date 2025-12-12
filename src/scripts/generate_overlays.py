"""
File: generate_overlays.py
Description: Script to generate FP/FN/TP overlays for TB, IHC, or mixed-stain models.
Author: Gizachew Kassa
Date Created: 12/12/2025
"""

from __future__ import annotations

import argparse

from src.visualization.overlays import generate_overlays


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate FP/FN/TP overlays for all test images (full image)."
    )
    p.add_argument(
        "--exp",
        type=str,
        required=True,
        choices=["tb", "ihc", "mixed"],
        help="Experiment (tb, ihc, mixed).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    generate_overlays(exp=args.exp)


if __name__ == "__main__":
    main()
