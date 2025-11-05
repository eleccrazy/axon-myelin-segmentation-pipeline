import os

import numpy as np
from PIL import Image


def analyze_mask(mask_path):
    """Analyze a single mask image for unique values and pixel distribution."""
    mask = np.array(Image.open(mask_path))
    unique_vals, counts = np.unique(mask, return_counts=True)
    total = mask.size

    proportions = {int(v): (c / total) * 100 for v, c in zip(unique_vals, counts)}

    return {
        "file": os.path.basename(mask_path),
        "shape": mask.shape,
        "dtype": str(mask.dtype),
        "unique_values": unique_vals.tolist(),
        "proportions": proportions,
    }


def analyze_folder(mask_dir, extensions=("_mask.tif", "_mask.tiff", "_mask.png")):
    """Analyze all mask files in a directory and compute average distribution."""
    mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(extensions)]
    print(f"\n Found {len(mask_files)} mask files in '{mask_dir}'")

    results = [analyze_mask(os.path.join(mask_dir, f)) for f in mask_files]

    # Collect averages
    all_labels = set()
    for r in results:
        all_labels.update(r["proportions"].keys())

    avg_props = {
        label: np.mean([r["proportions"].get(label, 0) for r in results])
        for label in all_labels
    }

    shapes = list(set([r["shape"] for r in results]))
    dtypes = list(set([r["dtype"] for r in results]))
    unique_vals = sorted(list(set(sum([r["unique_values"] for r in results], []))))

    print(f"   Image shapes: {shapes}")
    print(f"   Dtypes: {dtypes}")
    print(f"   Unique values: {unique_vals}")
    print(
        f"   Average class distribution (%): { {int(k): round(v, 2) for k, v in avg_props.items()} }"
    )

    return {
        "folder": mask_dir,
        "avg_distribution": avg_props,
    }


if __name__ == "__main__":
    # Define your two staining folders
    stain_folders = ["../data/data2", "../data/data2ihc"]

    all_results = [analyze_folder(folder) for folder in stain_folders]

    # Compute generalized averages across all stainings
    overall_labels = set()
    for res in all_results:
        overall_labels.update(res["avg_distribution"].keys())

    overall_avg = {
        label: np.mean([res["avg_distribution"].get(label, 0) for res in all_results])
        for label in overall_labels
    }

    print("\nGeneralized Summary Across All Stainings")
    print("=======================================")
    print(f"   Combined folders: {stain_folders}")
    print(
        f"   Overall average class distribution (%): { {int(k): round(v, 2) for k, v in overall_avg.items()} }"
    )
