"""
File: analyse_masks.py
Description:
    Analyze binary mask images for one or more staining domains and
    summarise the distribution of pixel values (class proportions).

    This script is intended to be run once on the full dataset to check
    that masks are well-formed (values, dtypes, shapes) and to compute
    average foreground/background percentages per staining domain and
    overall.

Usage:
    # Use the default folders inferred from the repository layout
    python analyse_masks.py

    # Or explicitly specify one or more mask directories
    python analyse_masks.py --mask-dirs /path/to/tb_masks /path/to/ihc_masks
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
from PIL import Image

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import IHC_DIR, TB_DIR


def analyze_mask(mask_path: Path) -> Dict[str, Any]:
    """
    Analyze a single binary mask image.

    Parameters
    ----------
    mask_path : Path
        Path to the mask image file. The mask is assumed to be a
        single-channel image with integer labels (e.g. 0 for background,
        255 for foreground).

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
            - "file": filename (str)
            - "shape": image shape (tuple[int, int] or tuple[int, int, int])
            - "dtype": numpy dtype as string
            - "unique_values": sorted list of unique label values
            - "proportions": mapping label -> percentage of pixels (0–100)
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
    Analyze all mask files in a folder and compute the average class distribution.

    Parameters
    ----------
    mask_dir : Path
        Directory containing mask image files.
    extensions : Iterable[str], optional
        Filename suffixes that identify mask images. Only files whose
        names end with any of these suffixes (case-insensitive) will
        be included.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
            - "folder": path to the analyzed folder as a string
            - "avg_distribution": mapping label -> average percentage
              of pixels across all masks in the folder
    """
    mask_dir = mask_dir.expanduser().resolve()

    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")

    mask_files = [
        f
        for f in os.listdir(mask_dir)
        if f.lower().endswith(tuple(ext.lower() for ext in extensions))
    ]

    print(f"\nFound {len(mask_files)} mask files in '{mask_dir}'")

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

    shapes = sorted({r["shape"] for r in results})
    dtypes = sorted({r["dtype"] for r in results})
    unique_vals = sorted({val for r in results for val in r["unique_values"]})

    print(f"   Image shapes: {shapes}")
    print(f"   Dtypes: {dtypes}")
    print(f"   Unique values: {unique_vals}")
    print(
        "   Average class distribution (%): "
        f"{ {int(k): round(v, 2) for k, v in avg_props.items()} }"
    )

    return {
        "folder": str(mask_dir),
        "avg_distribution": avg_props,
    }


def infer_default_mask_dirs() -> List[Path]:
    """
    Infer default mask directories based on the repository layout.

    This assumes the following structure:

        project_root/
            data/
                origional/
                    data2/      # toluidine blue masks
                    data2ihc/   # IHC masks
            scripts/
                analyse_masks.py

    Returns
    -------
    List[Path]
        List of default mask directory paths.
    """
    # scripts/ is the parent of this file; project_root is one level above that

    default_dirs = [TB_DIR, IHC_DIR]

    return default_dirs


def compute_overall_average(results: Iterable[Mapping[str, Any]]) -> Dict[int, float]:
    """
    Compute the overall average class distribution across multiple folders.

    Parameters
    ----------
    results : Iterable[Mapping[str, Any]]
        Iterable of folder-level results as returned by `analyze_folder`.

    Returns
    -------
    Dict[int, float]
        Mapping from label value to overall average percentage across
        all provided folders.
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
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attribute `mask_dirs`, a list of Path objects.
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
            "If not provided, default folders under data/origional/ "
            "will be used."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point for the script.

    - Determines which mask folders to analyze (CLI or default).
    - Runs analysis for each folder.
    - Computes and prints an overall summary across all folders.
    """
    args = parse_args()

    if args.mask_dirs is not None:
        stain_folders = [d.expanduser().resolve() for d in args.mask_dirs]
    else:
        stain_folders = infer_default_mask_dirs()

    print("Mask analysis")
    print("=============")
    print("Folders to be analyzed:")
    for folder in stain_folders:
        print(f" - {folder}")

    all_results = [analyze_folder(folder) for folder in stain_folders]

    overall_avg = compute_overall_average(all_results)

    print("\nGeneralized Summary Across All Stainings")
    print("=======================================")
    print(f"   Combined folders: {[str(f) for f in stain_folders]}")
    print(
        "   Overall average class distribution (%): "
        f"{ {int(k): round(v, 2) for k, v in overall_avg.items()} }"
    )


if __name__ == "__main__":
    main()
