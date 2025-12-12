"""
File: generate_triplets.py
Description: Script to generate input/GT/prediction triplet crops for TB, IHC, or mixed-stain models.
Author: Gizachew Kassa
Date Created: 12/12/2025
"""

from __future__ import annotations

import argparse

from src.visualization.triplets import generate_triplets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate input/GT/prediction triplet crops "
            "for TB, IHC or mixed-stain models."
        )
    )
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        choices=["tb", "ihc", "mixed"],
        help="Experiment to visualise (tb, ihc, mixed).",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=256,
        help="Square crop size in pixels (e.g. 256, 512, 1024).",
    )

    # NEW preferred flag name
    parser.add_argument(
        "--num-fragments",
        type=int,
        default=2,
        help=(
            "Number of fragments/crops to generate per image. "
            "Total outputs = num_test_images × num_fragments."
        ),
    )

    # Backwards-compatible alias (optional, but helpful)
    parser.add_argument(
        "--num-triplets",
        type=int,
        default=None,
        help="(Deprecated) Same as --num-fragments.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (default: test).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Optional override for probability threshold. "
            "If omitted, uses the calibrated threshold from metrics_<exp>.json."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=(
            "Optional custom output directory. "
            "If omitted, uses OUTPUTS_ROOT/figures/<exp>/triplets/<crop_size>/."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for fragment centre selection (default: 42).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    crops_per_image = args.num_fragments

    generate_triplets(
        exp=args.exp,
        crop_size=args.crop_size,
        crops_per_image=crops_per_image,
        split=args.split,
        threshold=args.threshold,
        save_dir=args.out_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
