"""
File: split_dataset.py
Description: This script splits a dataset of microscopy images and their corresponding masks into training, validation, and test sets.
Author: Gizachew Kassa
Date Created: 06/11/2025
"""

import os
import random
import shutil

random.seed(42)  # for reproducibility

input_dir = "data/real_lm_axon/data2"
output_base = "data/real_lm_axon/sourceA"


def main():
    """Split dataset into train, val, and test sets."""

    os.makedirs(output_base, exist_ok=True)

    # Get all image-mask pairs
    all_images = [
        f
        for f in os.listdir(input_dir)
        if f.endswith(".tiff") and not f.endswith("_mask.tiff")
    ]

    # Validate mask existence
    image_mask_pairs = []
    for img in all_images:
        mask = img.replace(".tiff", "_mask.tiff")
        if os.path.exists(os.path.join(input_dir, mask)):
            image_mask_pairs.append((img, mask))
        else:
            print(f"⚠️ No mask found for {img}")

    # Shuffle and split
    random.shuffle(image_mask_pairs)
    n_total = len(image_mask_pairs)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)

    train = image_mask_pairs[:n_train]
    val = image_mask_pairs[n_train : n_train + n_val]
    test = image_mask_pairs[n_train + n_val :]

    splits = {"train": train, "val": val, "test": test}

    # Copy files into folders
    for split_name, pairs in splits.items():
        split_dir = os.path.join(output_base, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for img, mask in pairs:
            shutil.copy(os.path.join(input_dir, img), os.path.join(split_dir, img))
            shutil.copy(os.path.join(input_dir, mask), os.path.join(split_dir, mask))

    print("✅ Dataset successfully split:")
    print(f" - Train: {len(train)} pairs")
    print(f" - Val:   {len(val)} pairs")
    print(f" - Test:  {len(test)} pairs")


if __name__ == "__main__":
    """Entry point of the script."""
    main()
