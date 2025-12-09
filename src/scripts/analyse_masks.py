"""
File: analyse_masks.py
Description: Analyze binary mask images and summarise pixel-class distributions.
Author: Gizachew Kassa
Date Created: 09/12/2025
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
from PIL import Image

from src.utils.paths import IHC_DIR, OUTPUTS_ROOT, TB_DIR


def analyze_mask(mask_path: Path) -> Dict[str, Any]:
    """
    Compute basic statistics for a single mask image.

    The mask is assumed to be a single-channel image with integer labels.
    Returns per-label pixel proportions (in percent) and basic metadata.
    """
    mask_array = np.array(Image.open(mask_path))
    unique_vals, counts = np.unique(mask_array, return_counts=True)
    total = mask_array.size

    proportions = {
        int(v): (int(c) / total) * 100.0 for v, c in zip(unique_vals, counts)
    }

    return {
        "file": mask_path.name,
        "shape": mask_array.shape,
        "dtype": str(mask_array.dtype),
        "unique_values": unique_vals.tolist(),
        "proportions": proportions,
    }


def analyze_folder(
    mask_dir: Path,
    extensions: Iterable[str] = ("_mask.tif", "_mask.tiff", "_mask.png"),
) -> Dict[str, Any]:
    """
    Aggregate mask statistics over all mask files in a folder.

    Only files whose names end with one of the given suffixes are included.
    Returns the average per-label pixel proportions across all masks.
    """
    mask_dir = mask_dir.expanduser().resolve()

    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")

    mask_files = [
        f
        for f in os.listdir(mask_dir)
        if f.lower().endswith(tuple(ext.lower() for ext in extensions))
    ]

    if not mask_files:
        return {"folder": str(mask_dir), "avg_distribution": {}}

    results = [analyze_mask(mask_dir / f) for f in mask_files]

    # Collect all labels that appear in any mask
    all_labels = set()
    for r in results:
        all_labels.update(r["proportions"].keys())

    # Compute average proportion per label across masks
    avg_props: Dict[int, float] = {
        label: float(np.mean([r["proportions"].get(label, 0.0) for r in results]))
        for label in all_labels
    }

    return {
        "folder": str(mask_dir),
        "avg_distribution": avg_props,
    }


def infer_default_mask_dirs() -> List[Path]:
    """
    Return the default stain folders inferred from the project layout.

    Uses the TB_DIR and IHC_DIR constants defined in src.utils.paths.
    """
    return [TB_DIR, IHC_DIR]


def compute_overall_average(results: Iterable[Mapping[str, Any]]) -> Dict[int, float]:
    """
    Compute the overall average label distribution across multiple folders.

    Each element in `results` is expected to come from `analyze_folder`.
    """
    overall_labels = set()
    folder_distributions: List[Mapping[int, float]] = []

    for res in results:
        dist = res.get("avg_distribution", {})
        folder_distributions.append(dist)
        overall_labels.update(dist.keys())

    if not folder_distributions:
        return {}

    overall_avg: Dict[int, float] = {
        label: float(np.mean([dist.get(label, 0.0) for dist in folder_distributions]))
        for label in overall_labels
    }

    return overall_avg


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the mask analysis script.
    """
    parser = argparse.ArgumentParser(
        description="Analyze binary mask images and summarise class distributions."
    )
    parser.add_argument(
        "--mask-dirs",
        type=Path,
        nargs="+",
        help=(
            "One or more directories containing mask images. "
            "If omitted, default stain folders (TB_DIR, IHC_DIR) are used."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """
    Run mask analysis for the selected folders and save results to JSON.

    By default, both toluidine blue and IHC mask folders are analysed,
    and a JSON summary is written under OUTPUTS_ROOT / 'analysis'.
    """
    print("Starting mask analysis...")

    args = parse_args()

    if args.mask_dirs is not None:
        stain_folders = [d.expanduser().resolve() for d in args.mask_dirs]
    else:
        stain_folders = infer_default_mask_dirs()

    all_results = [analyze_folder(folder) for folder in stain_folders]
    overall_avg = compute_overall_average(all_results)

    # Save the results to a JSON file
    output_dir = OUTPUTS_ROOT / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mask_stats.json"

    json_payload = {
        "folders": [
            {
                "folder": res["folder"],
                "avg_distribution": {
                    str(k): float(v) for k, v in res["avg_distribution"].items()
                },
            }
            for res in all_results
        ],
        "overall_average": {str(k): float(v) for k, v in overall_avg.items()},
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    print(f"Saved mask statistics to: {output_path}")


if __name__ == "__main__":
    main()
