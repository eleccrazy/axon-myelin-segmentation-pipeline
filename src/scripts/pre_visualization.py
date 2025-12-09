"""
File: pre_visualization.py
Description: This script processes microscopy image data by visualizing binary masks and copying the corresponding original images to a specified output directory.
Author: Gizachew Kassa
Date Created: 06/11/2025
"""

import os
import shutil

import numpy as np
from PIL import Image

input_dir = "data/real_lm_axon/combined"
output_dir = "data/real_lm_axon/visualize"


def main():
    """Visualize binary masks and copy original images to the output directory."""
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith("_mask.tiff"):
            continue

        # Construct full paths
        mask_path = os.path.join(input_dir, filename)
        base_name = filename.replace("_mask.tiff", ".tiff")
        image_path = os.path.join(input_dir, base_name)

        # Output paths
        mask_out_path = os.path.join(output_dir, filename)
        image_out_path = os.path.join(output_dir, base_name)

        # Load and binarize the mask for visualization
        mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask)
        mask_vis = (mask_np > 0).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask_vis)

        # Save visualized mask
        mask_img.save(mask_out_path)

        # Copy original image
        if os.path.exists(image_path):
            shutil.copy(image_path, image_out_path)
        else:
            print(f"⚠️ Image not found for mask: {filename}")


if __name__ == "__main__":
    main()
