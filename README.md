# Axon–Myelin Segmentation Pipeline

A deep learning–based segmentation pipeline for myelin sheath extraction in light microscopy (LM) images of peripheral nerve cross-sections.

The project implements and evaluates U-Net–style models for binary myelin vs non-myelin segmentation in bright-field images from two staining domains:

- Toluidine blue (TB)
- immunohistochemistry (IHC)

Three training configurations are supported:

- TB-only model
- IHC-only model
- Mixed-stain model with a small stain embedding branch

The code is organised as a reproducible pipeline that goes from raw image–mask pairs, through data splitting and training, to validation-based threshold calibration and quantitative test-time evaluation.

---

## Repository structure

At a high level, the repository is organised as follows:

```text
project_root/
  README.md
  src/  
    configs/
      preprocessing_configs.py
      loss_configs.py
    datasets/
      base.py
      tb_dataset.py
      ihc_dataset.py
      mixed_dataset.py
      dataloaders.py
    models/
      unet.py
      unet_stain.py
    preprocessing/
      tb_preprocessing.py
      ihc_preprocessing.py
    scripts/
      analyse_masks.py
      generate_overlays.py
      generate_triplets.py
      plot_history.py
      pre_visualization.py
      previsualize_preprocessing.py
      split_dataset.py
      train.py
      test.py
    training/
      loops.py
      callbacks.py
      losses.py
      entrypoints.py
    evaluation/
      metrics.py
      test_entrypoints.py
    utils/
      paths.py
    visualization/
      history_plots.py
      overlays.py
      triplets.py
      utils.py
  data/
    origional/
      data2/      # raw TB images and masks
      data2ihc/   # raw IHC images and masks
    splitted/
      tb/
        train/
        val/
        test/
      ihc/
        train/
        val/
        test/
  models/
    tb/
      unet_tb_best.pth
    ihc/
      unet_ihc_best.pth
    mixed/
      unet_mixed_stain_best.pth
  outputs/
    mask_analysis/
    pre_visualization/
    training_logs/
      tb/
      ihc/
      mixed/
    metrics/
      tb/
      ihc/
      mixed/
    figures/
      # optional plots and visualisations
```

### Top-level components

- `scripts/`  
  Standalone scripts that you run from the command line for data analysis, splitting, training, and testing.

- `src/`  
  Library-style Python modules (datasets, models, training loops, evaluation utilities, configuration, and path handling). The scripts in `scripts/` delegate most logic to this package.

- `data/`  
  Input data directory. Raw images and masks live under `data/origional/...`, while the train/val/test splits are written to `data/splitted/...`.

- `models/`  
  Saved model checkpoints for the three experiments (TB-only, IHC-only, mixed).

- `outputs/`  
  All generated outputs apart from model weights:
  - mask statistics
  - pre-visualisation image–mask previews
  - training histories
  - evaluation metrics
  - plots and figures

---

## Data organisation

The pipeline expects LM image–mask pairs in the following layout:

```text
data/
  origional/
    data2/       # TB images and masks
    data2ihc/    # IHC images and masks
```

Each staining directory should contain:

- Images: `*.tiff` (or similar extension)
- Binary masks: same stem with `_mask` suffix, e.g.  
  - `sample_001.tiff`  
  - `sample_001_mask.tiff`

The dataset splitting script writes split subsets into:

```text
data/
  splitted/
    tb/
      train/
      val/
      test/
    ihc/
      train/
      val/
      test/
```

Each split folder again contains image–mask pairs with the same naming convention.

The mixed-stain model uses these split TB and IHC folders jointly for each split.

---

## Installation and setup

1. Clone the repository:

   ```bash
   git clone https://github.com/eleccrazy/axon-myelin-segmentation-pipeline.git
   cd axon-myelin-segmentation-pipeline
   ```

2. Create and activate a virtual environment (example with `venv`):

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Linux/macOS
   # .venv\Scripts\activate           # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Prepare the data directories:

   - Place raw TB image–mask pairs under: `data/origional/data2/`
   - Place raw IHC image–mask pairs under: `data/origional/data2ihc/`

   Make sure the naming convention for masks is consistent (`*_mask.tiff`).

---

## Scripts and their purpose

### `scripts/analyse_masks.py`

Analyzes binary mask images across one or more folders:

- computes unique label values and average class proportions,
- aggregates statistics per folder and across folders,
- writes a JSON report under `outputs/mask_analysis/`.

This is useful to verify that the masks are binary, correctly encoded, and reasonably balanced.

Example usage:

```bash
python3 -m src.scripts.analyse_masks
```

(The script reads base paths from `src/utils/paths.py` so you do not need to pass them on the command line.)

---

### `scripts/pre_visualization.py`

Generates a small set of preview image–mask pairs from the raw dataset:

- selects a few representative image–mask pairs per stain,
- optionally binarizes masks for visual clarity,
- writes them to `outputs/pre_visualization/` for manual inspection.

This is intended for quick sanity checks before running the full pipeline.

Example usage:

```bash
python3 -m src.scripts.previsualize_preprocessing
```

---

### `scripts/split_dataset.py`

Splits the raw TB and IHC datasets into train/val/test sets:

- reads raw image–mask pairs from `data/origional/data2/` (TB) and `data/origional/data2ihc/` (IHC),
- verifies that each image has a corresponding mask,
- shuffles pairs with a fixed random seed,
- splits into 70% train, 15% validation, 15% test,
- copies image–mask pairs into:

  ```text
  data/splitted/tb/{train,val,test}/
  data/splitted/ihc/{train,val,test}/
  ```

Run once after preparing the raw data:

```bash
python3 -m src.scripts.split_dataset
```

All training and testing code assumes that these split folders exist.

---

### `scripts/previsualize_preprocessing.py`

Generates a small set of qualitative examples to verify the stain-specific preprocessing for TB and IHC:

- selects 1 sample image from each stain (TB + IHC),
- extracts 2 random crops per image (crop size: 256),
- applies the corresponding preprocessing pipeline,
- saves the original and preprocessed crops as separate PNG files with `_org` and `_pre` postfixes.

Outputs are written to:

```text
outputs/pre_visualization/orginal_vs_preprocessed/
```
Example usage:

```bash
python3 -m src.scripts.previsualize_preprocessing
```

### `src/datasets/*` and `src/datasets/dataloaders.py`

The dataset modules provide PyTorch `Dataset` and `DataLoader` objects for the three configurations:

- `base.py`  
  Generic `PairedImageMaskDataset` reading image–mask pairs from a single folder.

- `tb_dataset.py`  
  `TBLMDataset` that wraps the TB split folders and applies TB-specific preprocessing.

- `ihc_dataset.py`  
  `IHCLMDataset` that wraps the IHC split folders and applies IHC-specific preprocessing.

- `mixed_dataset.py`  
  `MixedLMDataset` that combines TB and IHC samples for a given split and returns `(image, mask, stain_id)` pairs for the mixed-stain model.

- `dataloaders.py`  
  Helper functions `make_tb_dataloaders`, `make_ihc_dataloaders`, `make_mixed_dataloaders` that build train and validation `DataLoader`s with appropriate batch sizes and options.

You normally do not call these modules directly; they are used by the training and testing entrypoints.

---

### `src/models/unet.py` and `src/models/unet_stain.py`

Model definitions:


- `unet_stain.py`  
  A U-Net variant with an additional stain embedding at the bottleneck. It takes `(image, stain_id)` as input and is used for the mixed-stain experiment.

- unet_deep.py
  A deeper U-Net variant with more feature channels and additional downsampling/upsampling layers. (used for ihc and tb experiments)

- `unet.py`  
  (not used in main experiments) A lightweight U-Net. Used for tb and ihc initial experiments.
---

### `src/preprocessing/*`

Stain-specific preprocessing pipelines:

- `tb_preprocessing.py`  
  Functions/classes for converting TB RGB images into single-channel contrast-enhanced inputs (e.g. LAB L-channel extraction, shading correction, CLAHE, denoising).

- `ihc_preprocessing.py`  
  Functions/classes for converting IHC RGB images into single-channel inputs (e.g. colour deconvolution to extract the DAB channel, contrast normalisation, denoising).

The exact parameters are configured in `src/configs/preprocessing_configs.py`.

---

### `src/training/loops.py`, `src/training/losses.py`, `src/training/entrypoints.py`

Training utilities:

- `loops.py`  
  Contains the shared training loop:
  - `train_one_epoch`
  - `validate_one_epoch`
  - `run_training` with early stopping, optional learning-rate scheduling, and optional validation metrics callback.

- `losses.py`  
  Implements the loss functions used in the experiments (e.g. Dice+BCE, boundary-aware loss) and their configuration.

- `entrypoints.py`  
  High-level functions:
  - `run_tb_training()`
  - `run_ihc_training()`
  - `run_mixed_training()`

  Each function:
  - builds the appropriate dataloaders,
  - instantiates the correct model and loss,
  - configures optimiser and scheduler,
  - runs `run_training`,
  - returns the trained model and training history.

---

### `scripts/train.py`

Command-line entrypoint for training:

- prompts you to select an experiment interactively, or
- accepts `--exp tb`, `--exp ihc`, or `--exp mixed`.

It calls the corresponding training entrypoint and then saves:

- the best model weights to `models/<exp>/...`,
- the training history (losses and metrics per epoch) to `outputs/training_logs/<exp>/history_*.json`.

Example usage:

```bash
# interactive choice
python3 -m src.scripts.train --exp <exp type>

# direct selection
python3 -m src.scripts.train --exp tb
python3 -m src.scripts.train --exp ihc
python3 -m src.scripts.train --exp mixed
```

---

### `src/evaluation/metrics.py` and `src/evaluation/test_entrypoints.py`

Evaluation utilities:

- `metrics.py`  
  Functions for computing segmentation metrics such as Dice and IoU, both per image and aggregated over a `DataLoader`.

- `test_entrypoints.py`  
  Implements test-time evaluation with validation-based threshold calibration:

  - For TB and IHC:
    - Builds validation and test dataloaders.
    - Loads the corresponding U-Net model from `models/tb/` or `models/ihc/`.
    - Sweeps thresholds on the validation set in the range `[0.20, 0.60]` to select a best threshold based on validation Dice.
    - Evaluates the test set at the calibrated threshold, computing Dice, IoU, precision, and recall.

  - For the mixed-stain model:
    - Builds mixed validation and test dataloaders.
    - Loads the mixed U-Net with stain embedding from `models/mixed/`.
    - Sweeps thresholds on the validation set (e.g. `[0.10, 0.90]`).
    - Evaluates the test set at the calibrated threshold, computing overall Dice, IoU, precision, and recall, and per-stain precision and recall (TB, IHC).

Each evaluation entrypoint returns a dictionary with the calibrated threshold, test metrics, and the threshold sweep records.

---

### `scripts/test.py`

Command-line entrypoint for evaluation:

- prompts for the experiment or accepts `--exp tb|ihc|mixed`,
- calls the corresponding evaluation entrypoint from `src/evaluation/test_entrypoints.py`,
- attaches a timestamp,
- writes the result to:

  ```text
  outputs/metrics/tb/metrics_tb.json
  outputs/metrics/ihc/metrics_ihc.json
  outputs/metrics/mixed/metrics_mixed.json
  ```

Example usage:

```bash
python3 -m src.scripts.test --exp tb
python3 -m src.scripts.test --exp ihc
python3 -m src.scripts.test --exp mixed
```

You can adjust `--batch-size` and `--num-workers` if needed.

## Visualisation (training curves, triplets, and overlays)

After running src/scripts/test.py, you can generate training curves from the saved history files and qualitative figures using the calibrated threshold stored in outputs/metrics/<exp>/metrics_<exp>.json
### Training curves
Generate training curves for all experiments:

```bash
python3 -m src.scripts.plot_history --exp tb
python3 -m src.scripts.plot_history --exp ihc
python3 -m src.scripts.plot_history --exp mixed
```
Outputs are saved to:

```text
outputs/figures/<exp>/curves/
```

Note: it assumes that the training history files are present in outputs/training_logs/<exp>/ from previous training runs.

### Triplet fragments (input / ground truth / prediction)

Generate triplet fragments for all test images:

```bash
python3 -m src.scripts.generate_triplets --exp tb
python3 -m src.scripts.generate_triplets --exp ihc
python3 -m src.scripts.generate_triplets --exp mixed
```

Control how many fragments are generated **per test image**:

```bash
python3 -m src.scripts.generate_triplets --exp ihc --num-fragments 2
```
Control the fragment crop size (default: 256):

```bash
python3 -m src.scripts.generate_triplets --exp ihc --crop-size 512
```

Outputs are saved to:

```text
outputs/figures/<exp>/triplets/<crop_size>/
```

For the mixed-stain model, results are additionally grouped under:

```text
outputs/figures/mixed/triplets/<crop_size>/stain_tb/
outputs/figures/mixed/triplets/<crop_size>/stain_ihc/
```

### Full-image error overlays (FP/FN/TP)

Generate full-image overlays for all test images:

```bash
python3 -m src.scripts.generate_overlays --exp tb
python3 -m src.scripts.generate_overlays --exp ihc
python3 -m src.scripts.generate_overlays --exp mixed
```

Overlay encoding:
- **FP** = red
- **FN** = green
- **TP** = yellow

Outputs are saved to:

```text
outputs/figures/<exp>/overlays/
```

For the mixed-stain model, results are additionally grouped under:

```text
outputs/figures/mixed/overlays/stain_tb/
outputs/figures/mixed/overlays/stain_ihc/
```

---

## Running the full pipeline

A typical end-to-end workflow is:

1. **Prepare data**

   - Place raw TB and IHC image–mask pairs under `data/origional/data2/` and `data/origional/data2ihc/`.

2. **Optional: analyze masks**

   ```bash
   python3 -m src.scripts.analyse_masks
   ```

   Inspect the JSON report in `outputs/mask_analysis/` to verify mask encoding and foreground proportions.

3. **Split into train/val/test**

   ```bash
   python3 -m src.scripts.split_dataset
   ```

   This creates `data/splitted/tb/{train,val,test}/` and `data/splitted/ihc/{train,val,test}/`.

4. **Optional: pre-visualisation and view preprocessed images**

   ```bash
   python3 -m src.scripts.pre_visualization
   python3 -m src.scripts.previsualize_preprocessing
   ```

   Check `outputs/pre_visualization/` to visually confirm image–mask alignment and staining quality.

5. **Train models**

   ```bash
   # TB-only model
   python3 -m src.scripts.train --exp tb

   # IHC-only model
   python3 -m src.scripts.train --exp ihc

   # Mixed-stain model
   python3 -m src.scripts.train --exp mixed
   ```

   Each run saves the best checkpoint to `models/<exp>/...` and training history to `outputs/training_logs/<exp>/`.

6. **Evaluate models**

   ```bash
   python3 -m src.scripts.test --exp tb
   python3 -m src.scripts.test --exp ihc
   python3 -m src.scripts.test --exp mixed
   ```

   The calibrated thresholds and test metrics are written to `outputs/metrics/<exp>/metrics_*.json`.
7. **Generate visualisations**

   ```bash
   # Training curves
   python3 -m src.scripts.plot_history --exp tb
   python3 -m src.scripts.plot_history --exp ihc
   python3 -m src.scripts.plot_history --exp mixed

   # Triplet fragments
   python3 -m src.scripts.generate_triplets --exp tb
   python3 -m src.scripts.generate_triplets --exp ihc
   python3 -m src.scripts.generate_triplets --exp mixed

   # Full-image overlays
   python3 -m src.scripts.generate_overlays --exp tb
   python3 -m src.scripts.generate_overlays --exp ihc
   python3 -m src.scripts.generate_overlays --exp mixed
   ```
---

This structure is intended to make it straightforward to:

- rerun experiments from raw data,
- inspect training dynamics and evaluation metrics,
- and extend the pipeline with additional models, datasets, or downstream morphometric analysis.