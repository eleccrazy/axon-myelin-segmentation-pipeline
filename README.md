# Axon–Myelin Segmentation Pipeline

A deep learning–based segmentation pipeline for the combined axon+myelin foreground in light microscopy (LM) images of peripheral nerve cross-sections.

The project implements and evaluates U-Net–style models for binary foreground segmentation (axon+myelin vs background) in bright-field images from two staining domains:

- Toluidine blue (TB)
- DAB-based immunohistochemistry (IHC)

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
  scripts/
    analyse_masks.py
    pre_visualization.py
    split_dataset.py
    train.py
    test.py
    # (visualisation scripts for overlays/triplets can be added here later)
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
   git clone <this-repo-url>.git
   cd <this-repo-name>
   ```

2. Create and activate a virtual environment (example with `venv`):

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Linux/macOS
   # .venv\Scripts\activate           # Windows
   ```

3. Install dependencies (adjust as needed, depending on your environment):

   ```bash
   pip install torch torchvision torchaudio  # choose CPU/GPU build as appropriate
   pip install numpy scipy scikit-image pillow tqdm matplotlib
   ```

   If you maintain a `requirements.txt`, you can instead use:

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
python scripts/analyse_masks.py
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
python scripts/pre_visualization.py
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
python scripts/split_dataset.py
```

All training and testing code assumes that these split folders exist.

---

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

- `unet.py`  
  A lightweight U-Net used for the TB-only and IHC-only experiments. It takes single-channel preprocessed images and outputs a single-channel foreground probability map.

- `unet_stain.py`  
  A U-Net variant with an additional stain embedding at the bottleneck. It takes `(image, stain_id)` as input and is used for the mixed-stain experiment.

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
python scripts/train.py

# direct selection
python scripts/train.py --exp tb
python scripts/train.py --exp ihc
python scripts/train.py --exp mixed
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
python scripts/test.py --exp tb
python scripts/test.py --exp ihc
python scripts/test.py --exp mixed
```

You can adjust `--batch-size` and `--num-workers` if needed.

---

## Running the full pipeline

A typical end-to-end workflow is:

1. **Prepare data**

   - Place raw TB and IHC image–mask pairs under `data/origional/data2/` and `data/origional/data2ihc/`.

2. **Optional: analyze masks**

   ```bash
   python scripts/analyse_masks.py
   ```

   Inspect the JSON report in `outputs/mask_analysis/` to verify mask encoding and foreground proportions.

3. **Split into train/val/test**

   ```bash
   python scripts/split_dataset.py
   ```

   This creates `data/splitted/tb/{train,val,test}/` and `data/splitted/ihc/{train,val,test}/`.

4. **Optional: pre-visualisation**

   ```bash
   python scripts/pre_visualization.py
   ```

   Check `outputs/pre_visualization/` to visually confirm image–mask alignment and staining quality.

5. **Train models**

   ```bash
   # TB-only model
   python scripts/train.py --exp tb

   # IHC-only model
   python scripts/train.py --exp ihc

   # Mixed-stain model
   python scripts/train.py --exp mixed
   ```

   Each run saves the best checkpoint to `models/<exp>/...` and training history to `outputs/training_logs/<exp>/`.

6. **Evaluate models**

   ```bash
   python scripts/test.py --exp tb
   python scripts/test.py --exp ihc
   python scripts/test.py --exp mixed
   ```

   The calibrated thresholds and test metrics are written to `outputs/metrics/<exp>/metrics_*.json`.

7. **Visualisation (to be added)**

   Once visualisation scripts are in place (for overlays, triplet panels, and training curves), they will live under:

   - `scripts/` for the command-line entrypoints,
   - `models/<exp>/...` for model weights,
   - `outputs/training_logs/<exp>/history_*.json` for training curves,
   - `data/splitted/...` for input images and masks,
   - and will write figures to `outputs/figures/<exp>/`.

---

This structure is intended to make it straightforward to:

- rerun experiments from raw data,
- inspect training dynamics and evaluation metrics,
- and extend the pipeline with additional models, datasets, or downstream morphometric analysis.