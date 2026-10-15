# Codex Task: Audit and Improve the Sen1Floods11 DeepLabV3+ Pipeline

## Project context

This is a major project on satellite-based flood segmentation using the Sen1Floods11 dataset. The model uses two Sentinel-1 SAR channels (VV/VH) and binary segmentation labels.

Current environment:
- GPU: NVIDIA RTX 3050 Laptop GPU, 4 GB VRAM
- Input chips: 512 × 512
- Training batch size: 1
- Training samples: approximately 252
- Validation samples: approximately 89
- Labels: `-1 = ignore`, `0 = non-flood`, `1 = flood`
- DeepLabV3+-style model: ResNet50 encoder, ASPP, low-level decoder
- Current encoder initialization: `pretrained=False`
- Normalization: primarily BatchNorm, with GroupNorm in the ASPP global-pooling branch
- Mixed precision is enabled when CUDA is available.

Reported results from existing experiments:
- Original DeepLabV3+ baseline: best validation IoU approximately 0.6200; validation Dice approximately 0.7655.
- Learning-rate experiments: best validation IoU approximately 0.6118 and 0.6123.
- A BCE + Dice experiment exists in `src/train_deeplab_bcedice.py`; inspect its current state before assuming it has completed.
- A separate U-Net experiment has completed held-out test evaluation. Do not modify its code, checkpoints, metrics, or results.

These DeepLabV3+ results are validation results, not held-out test results. Do not conflate the two.

## Overall task

Inspect the existing repository, make necessary correctness fixes, and implement diagnostic scripts. Edit files directly and test the changes. Do not merely provide recommendations.

Prioritize reliable evaluation, reproducibility, and small, justified changes. Do not make speculative model changes or launch full training automatically.

## 1. Inspect the existing code first

Inspect these files and related dependencies:
- `src/sen1_dataset.py`
- `src/deeplabv3.py`
- `src/losses.py`
- `src/train_deeplab.py`
- `src/train_deeplab_bcedice.py`
- `src/find_best_threshold.py`, if present
- Any directly related model, metric, and test files.

Verify:
1. Ignored pixels are excluded from every relevant loss term.
2. Ignored pixels are excluded from all reported evaluation metrics.
3. The model and checkpoint used by each script match.
4. Threshold search uses validation data only.
5. Fully ignored batches are handled safely.
6. Checkpoint metadata matches the actual experiment and loss.
7. Existing histories and checkpoints will not be overwritten or mixed.
8. The training and validation flood-pixel distributions can be measured.
9. The implementation is compatible with the existing Python/PyTorch environment.

Do not assume a bug exists without evidence. Clearly distinguish confirmed defects from potential risks.

## 2. Add a DeepLabV3+ validation threshold-search script

Create `src/find_best_threshold_deeplab.py`.

Requirements:
- Load the architecture from `src/deeplabv3.py`.
- Load `checkpoints_deeplab/best_deeplab.pth`.
- Use only `data/raw/sen1floods11/splits/flood_valid_data.csv`.
- Use `model.eval()` and `torch.inference_mode()`.
- Do not train or modify weights.
- Use a small batch size (1 or 2) suitable for a 4 GB GPU.
- Support CUDA and CPU.
- Apply sigmoid to logits.
- Exclude ignored pixels from all metrics.
- Search thresholds from 0.10 through 0.90 in steps of 0.01.
- Report IoU, Dice, precision, recall, false-positive count, and false-negative count for each threshold.
- Select the threshold with the highest validation IoU.
- Handle empty valid-pixel collections safely.
- Save the complete threshold table to CSV and the selected threshold/checkpoint information to JSON.
- Save outputs to a new, dedicated directory. Do not overwrite checkpoints or unrelated experiment outputs.
- Make it clear that the selected threshold is a validation-selected operating point and must not be chosen using test data.

## 3. Add dataset-distribution diagnostics

Create `src/analyze_sen1_distribution.py`.

For the training and validation splits, calculate:
- Number of images.
- Total pixels.
- Valid pixels.
- Ignored pixels.
- Flood pixels.
- Non-flood pixels.
- Flood percentage among valid pixels.
- Ignore percentage among all pixels.
- Per-image flood percentage statistics: minimum, median, mean, and maximum.

Ignored pixels must not be counted as non-flood pixels. Do not modify any images or labels. Print a readable summary and save results to CSV in a new output directory.

## 4. Add validation-prediction visualization

Create `src/visualize_deeplab_predictions.py`.

Requirements:
- Load the existing best DeepLabV3+ checkpoint.
- Use the validation split only.
- Use `model.eval()` and `torch.inference_mode()`.
- Visualize approximately 10–15 validation samples.
- For each sample, show:
  1. SAR input visualization.
  2. Ground-truth mask.
  3. Predicted probability map.
  4. Binary prediction at threshold 0.50.
  5. Binary prediction at the threshold selected by the threshold-search script.
- Clearly distinguish ignored ground-truth pixels from background.
- Read the selected threshold from the generated JSON file. If it is absent, explicitly warn and fall back to 0.50.
- Use sensible display scaling for SAR. Do not change the model's actual input normalization.
- Save figures to a new output directory without overwriting checkpoints or existing results.
- The visualizations should be readable and useful for research presentation.

## 5. Audit and test the loss functions

Inspect `src/losses.py`, particularly `TverskyDiceLoss` and `BCEDiceLoss`.

Verify that:
- Ignored pixels are excluded from all relevant loss calculations.
- Expected shapes are handled correctly: logits `[B, 1, H, W]`, targets `[B, H, W]`, ignore masks `[B, H, W]`.
- Losses and gradients remain finite.
- Completely ignored batches do not produce NaN or invalid gradients.
- Numerical safeguards are appropriate.

Make only necessary corrections. Add or update focused tests for:
- A normal labelled batch.
- A partially ignored batch.
- A completely ignored batch.
- Finite forward loss.
- Finite backward gradients.
- Expected tensor shapes.

Run the tests and report actual results. Do not change loss weights merely to force better metrics.

## 6. Audit the BCE + Dice training script

Inspect `src/train_deeplab_bcedice.py`.

Correct only confirmed issues, including:
- Stale Tversky-specific metadata in BCE + Dice checkpoints.
- Recording the actual loss name and parameters.
- Skipping fully ignored training and validation batches.
- Excluding ignored pixels from losses and metrics.
- Incorrect loss-related error messages.
- Consistency of checkpoint selection, CSV fields, scheduler, and early stopping.
- Risk of appending to an incompatible CSV or overwriting/mixing experiment artifacts.

Preserve the experiment's existing hyperparameters unless a correctness issue requires a change. Use separate output locations for separate experiments. Do not start or restart training automatically.

## 7. Investigate normalization behavior without overclaiming

Inspect normalization layers in `src/deeplabv3.py` and report:
- The types and locations of BatchNorm and GroupNorm layers.
- Which parts of the model are pretrained or randomly initialized.
- Why batch size 1 may be a concern, and where spatial dimensions still provide observations for BatchNorm.

Do not claim batch size 1 is necessarily the cause of the performance plateau.

If a frozen-BatchNorm experiment is justified, optionally create `src/train_deeplab_frozen_bn.py`:
- Preserve the original training scripts.
- Use the same train/validation splits and evaluation protocol.
- Use a distinct checkpoint directory and history CSV.
- Document exactly how normalization behavior differs.
- Do not launch training automatically.

Do not replace all BatchNorm layers automatically.

## 8. Review pretrained two-channel initialization

Inspect the current two-channel stem initialization in `src/deeplabv3.py`, including the logic that initializes two SAR channels from the mean of pretrained RGB kernels.

Assess whether the implementation is internally correct. Do not assume ImageNet weights will improve SAR segmentation.

Do not download weights or launch training automatically. If a pretrained experiment is recommended, describe it separately and preserve the baseline.

## 9. Reproducibility and safety constraints

Mandatory:
- Do not delete project files.
- Do not alter completed U-Net results, checkpoints, or code.
- Do not overwrite the original DeepLabV3+ checkpoint.
- Do not mix training histories from different experiments.
- Do not change dataset splits.
- Never use test data for threshold tuning or model selection.
- Do not launch full training automatically.
- Do not install or upgrade packages unless necessary; inspect the current environment first.
- Avoid unnecessary refactoring.
- Keep compatibility with the existing environment and 4 GB GPU.
- Use separate output directories for diagnostic outputs.
- Preserve existing artifacts before editing any file that might affect them.

## 10. Verification

After editing:
1. Run Python syntax checks for all created or modified scripts.
2. Test loading at least one dataset sample.
3. Verify image, target, and ignore-mask shapes and label semantics.
4. Run the loss tests.
5. Run a small DeepLabV3+ forward-pass test if feasible.
6. Confirm the threshold-search script loads the correct model and checkpoint.
7. Confirm the distribution and visualization scripts do not train the model.
8. Check paths and output directories.
9. Confirm test data is not used for model selection.
10. Do not claim a test passed unless it was actually run. If a test cannot be run, explain why.

## Final report

When finished, report:
A. Files inspected.
B. Files created.
C. Files modified.
D. Confirmed bugs versus potential risks.
E. Exact changes made.
F. Tests run and actual results.
G. Exact commands to run diagnostics, in the correct order.
H. Remaining issues or tests that need manual verification.
I. Explicit confirmation that existing checkpoints and completed U-Net results were preserved.

Do the work directly in the repository. Favor correctness and reproducibility over speculative architecture changes.
