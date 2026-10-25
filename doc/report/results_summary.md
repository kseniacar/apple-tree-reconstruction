# Results summary (runs of 2026-10-14 and 2026-10-25)

All runs: 541 summer images with ground truth (row R04 excluded), split by tree into 379 train / 81 validation / 81 test. Metrics are means over the 81 test images. Dice and IoU: higher is better. Chamfer distance (pixels, between skeletons): lower is better. Times measured on Apple-silicon MacBooks (CPU). The 2026-10-25 re-run of `src/train.py` (Apple M2, 8 GB) used the same split, seed and settings and reproduced every accuracy number of the 2026-10-14 run exactly.

## 1. Main comparison (`python src/train.py`, working size 128 x 128, RGB-D)

| Method | IoU | Dice | Precision | Recall | Occluded recall | Chamfer (px) | Fit time (s) | Time per image (s) | Threshold (from validation) |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (ExG + depth thresholding) | 0.081 | 0.149 | 0.083 | 0.741 | 0 (by construction) | 20.10 | 0 | 0.025 | n/a |
| Shape prior (mean training mask, no image input) | 0.237 | 0.377 | 0.281 | 0.574 | 0.492 | 10.18 | 0 | 0 | 0.20 |
| Linear regression | 0.123 | 0.216 | 0.140 | 0.541 | 0.504 | 22.71 | 0.1 | 0.002 | 0.55 |
| Decision tree regression | 0.223 | 0.358 | 0.264 | 0.565 | 0.478 | 11.99 | 6.0 | 0.002 | 0.75 |
| Support vector regression (3,000 px) | 0.159 | 0.271 | 0.186 | 0.526 | 0.505 | 16.02 | 0.2 | 1.092 | 0.80 |
| Neural network (MLP) regression | 0.230 | 0.369 | 0.261 | 0.636 | 0.552 | 10.05 | 55.4 | 0.003 | 0.70 |

Fit and per-image times are from the 2026-10-14 run; the re-run gave similar times (`res/ml_results.csv`).

The shape prior is the same mask for every test image: the mean of the training masks, thresholded on the validation trees. It uses no information from the image, so it is the score a method gets from knowing only where trees usually are in the frame. Source: `res/ml_results.csv`, `res/ml_validation.csv`.

## 1a. 5-fold cross-validation by tree (`python src/crossval.py`)

The 460 non-test images (379 train + 81 validation) were split into 5 folds by tree; the 81 test images were not used. Each tree has exactly one summer image with GT, so splitting by tree and splitting by image give the same folds here; the tree-wise code matters only if more images per tree are added. Each model was trained on 4 folds and its threshold chosen on the held-out fold, so these Dice values are slightly optimistic and are for ranking, not the final accuracy (`res/cv_folds.csv`, `res/cv_summary.csv`).

| Method | Mean Dice | Std. dev. (5 folds) | Per fold |
|---|---|---|---|
| Shape prior | **0.379** | 0.013 | 0.399, 0.372, 0.368, 0.370, 0.384 |
| MLP regression | 0.373 | 0.014 | 0.390, 0.371, 0.359, 0.359, 0.385 |
| Decision tree regression | 0.366 | 0.007 | 0.372, 0.367, 0.357, 0.361, 0.373 |
| Support vector regression | 0.287 | 0.013 | 0.300, 0.277, 0.276, 0.281, 0.302 |
| Linear regression | 0.219 | 0.007 | 0.220, 0.216, 0.209, 0.223, 0.227 |

The ranking is stable across folds: prior ≥ MLP > tree > SVR > linear. The prior beats the tree in all 5 folds and the MLP in 4 of 5 (the fifth is level, +0.001). The fold-to-fold spread (about 0.01 Dice) is as large as the gap between the MLP and the tree, so their single-split difference on the test set (0.369 vs 0.358) is not meaningful. The tree is the better practical choice: about the same accuracy as the MLP, roughly 9x faster to train, and interpretable. Run time: 53 min on an Apple M2, most of it SVR prediction; peak resident memory 850 MB.

## 1b. Occluded-branch recall

Occluded recall is the recall computed only on ground-truth pixels the baseline's visible-bark detector misses (`metrics.occluded_recall`). These are about a quarter of the GT pixels (the baseline's recall is 0.741). They are a proxy for "hidden by leaves": they also include wood the detector misses for other reasons, such as shading or the GT offset (section 1c).

The learned models recover about half of these pixels (0.48 to 0.55), but so does the shape prior (0.49), which never looks at the image. Only the MLP is clearly above the prior (0.552 vs 0.492). So the models' ability to "see through" leaves comes mostly from knowing where trunks and main branches usually are, not from image evidence. Source: `res/ml_results.csv` (`occ_recall`).

## 1c. Ground-truth alignment check

- **Manual check** (`python tools/gt_offset.py`, 24 random summer images, `res/gt_offset_sheet_*.png`, readings in `res/gt_offset_manual.csv`). Rule: at row 230 of 256, offset = GT trunk centre column minus visible trunk centre column. 22 images had a visible trunk. Median |offset| is 13.5 px, range -25 to +30 px; half are 15 px or more. At 256 x 256 that is about 25 px in the 480 x 480 crop, consistent with the 15-45 px reported on Ed. The offsets go **both ways**, so no single correction fixes them. Readings are approximate (about ±6 px).
- **Automatic check** (`alignment.estimate_offset`, FFT cross-correlation of each GT with the baseline mask, run by `src/train.py`). On the same 22 images it agrees with the manual readings: correlation 0.81, mean difference 9.1 px, same direction in 82% of images. On the 81 test images the median |horizontal shift| is 17 px and the median |vertical shift| is 3 px (`res/gt_offsets.csv`).
- **Scores against the shifted GT** (`res/ml_results_aligned.csv`):

| Method | Dice, original GT | Dice, GT shifted onto visible wood |
|---|---|---|
| Baseline | 0.149 | 0.168 |
| Shape prior | 0.377 | 0.303 |
| Decision tree | 0.358 | 0.307 |
| MLP | 0.369 | 0.301 |

Moving the GT onto the visible wood helps the baseline and lowers the prior and the learned models by about 0.05 to 0.07 Dice. The models were trained on the offset masks, so they reproduce where the annotators drew the branches, offset included. The shift is estimated against the baseline's own mask, so the aligned scores favour the baseline; they show how much the offset matters, not the true error. The offset also limits how small the Chamfer distance can meaningfully get: a trunk offset of 13-17 px on its own gives a skeleton distance of that order, so much of the 10-12 px reached by the best methods may be annotation offset rather than model error.

## 2. Sensitivity: working resolution (decision tree and MLP, RGB-D)

| Working size | Model | Dice | Chamfer (px) | Fit time (s) | Time per image (s) |
|---|---|---|---|---|---|
| 64 x 64 | Tree | 0.359 | 11.22 | 4.7 | 0.001 |
| 64 x 64 | MLP | 0.368 | 11.38 | 32.5 | 0.002 |
| 128 x 128 | Tree | 0.358 | 11.99 | 6.0 | 0.002 |
| 128 x 128 | MLP | 0.369 | 10.05 | 55.4 | 0.003 |
| 256 x 256 | Tree | 0.359 | 11.47 | 6.7 | 0.006 |
| 256 x 256 | MLP | 0.374 | 9.40 | 18.6 | 0.015 |

The 64 x 64 run had fewer training pixels (307,344) because some images have fewer than 500 branch pixels at that size. MLP fit time depends on when early stopping triggers, so it is not a clean measure of cost.

## 3. Sensitivity: does depth help? (128 x 128)

| Input | Model | Dice | Chamfer (px) |
|---|---|---|---|
| RGB-D | Tree | 0.358 | 11.99 |
| RGB only | Tree | 0.361 | 11.51 |
| RGB-D | MLP | 0.369 | 10.05 |
| RGB only | MLP | 0.368 | 10.55 |
| RGB-D | Baseline | 0.149 | 20.10 |
| RGB only | Baseline | 0.138 | 20.38 |

## 3b. Sensitivity: without the row/column position features (`--no-position`, 128 x 128, RGB-D)

| Features | Model | Dice | Chamfer (px) |
|---|---|---|---|
| With position | Tree | 0.358 | 11.99 |
| Without position | Tree | 0.207 | 19.63 |
| With position | MLP | 0.369 | 10.05 |
| Without position | MLP | 0.232 | 18.97 |
| (no image input) | Shape prior | 0.377 | 10.18 |

Source: `res/ml_results_nopos.csv`. With position features, the decision tree gives 88% of its importance to `col` and `row` (`res/tree_feature_importance.csv`); without them, the smoothed green and depth channels dominate (`res/tree_feature_importance_nopos.csv`).

## 3c. Sensitivity: baseline thresholds (`python tools/baseline_sensitivity.py`)

The excess-green threshold (-0.10 to 0.15) and the lower edge of the depth band (1 to 140) were swept on the 81 validation images (`res/baseline_sensitivity.csv`). The best setting (ExG < -0.05, depth ≥ 120) gives test Dice 0.169, against 0.149 for the default (ExG < 0.05, depth ≥ 1). Recall falls from 0.741 to 0.260. Tuning the baseline cannot close the gap to the learned models or the shape prior (Dice 0.36 to 0.38).

## 4. Profiling the baseline (`python tools/profile_baseline.py`, 20 synthetic 640 x 480 images)

| Function | Cumulative time (s) | Share |
|---|---|---|
| Total | 0.412 | 100% |
| `mask_to_skeleton` (skimage Zhang skeletonisation) | 0.233 | 57% |
| `predict_mask` | 0.179 | 43% |
| of which `excess_green` | 0.079 | 19% |
| of which `binary_closing` | 0.047 | 11% |

## 4b. Memory (`python tools/profile_memory.py`, synthetic 640 x 480 RGB-D, run 2026-10-25)

| Step | Peak memory |
|---|---|
| Baseline `predict_mask` + skeleton, one image | 19.7 MB |
| Per-pixel features, one image at 64 x 64 / 128 x 128 / 256 x 256 | 1.2 / 4.9 / 18.4 MB |

`src/train.py` keeps every image's feature matrix in memory: 17 float32 features per pixel is 0.28 / 1.11 / 4.46 MB per image at 64 / 128 / 256, so 0.15 / 0.60 / 2.41 GB for all 541 images, plus 0.57 GB for the (4, 256, 256) inputs. The sampled training set itself is small (379 images x 1,000 pixels x 17 features = 25.8 MB). Memory, not compute, is what limits running at 256 x 256 on a free Colab instance; 64 x 64 needs about 16x less with the same Dice (section 2). These are `tracemalloc` peaks of Python/NumPy allocations, not whole-process memory. The full `python src/train.py` run at 128 x 128 (all 541 images, all models) peaked at **760 MB resident memory** (`/usr/bin/time -l`) and took 289 s on an Apple M2 (8 GB).

## 4c. U-Net cost (`python tools/unet_cost.py`, input 1 x 4 x 256 x 256, run 2026-10-25)

| Base channels | Levels | Parameters | GFLOPs per image | Weights (MB) | Layer outputs (MB) | FLOP / byte |
|---|---|---|---|---|---|---|
| 8 | 4 | 486,625 | 1.54 | 1.95 | 20.2 | 69.6 |
| 16 | 3 | 482,881 | 4.66 | 1.93 | 38.0 | 116.6 |
| **16** | **4** | **1,942,721** | **6.08** | **7.77** | **40.1** | **127.1** (configured) |
| 32 | 4 | 7,763,329 | 24.18 | 31.05 | 80.0 | 217.8 |

FLOPs count only the convolution layers (2 x multiply-accumulates; batch norm, ReLU and pooling add a few percent). FLOP / byte assumes each weight and layer output is moved once; at over 100 FLOP/byte the network is compute-bound on a CPU or GPU, so runtime should scale with GFLOPs. Doubling the width roughly quadruples the FLOPs and parameters. For comparison, the decision tree predicts an image in 0.002 s on a laptop CPU (section 1). The U-Net has not been trained, so these are costs only, with no accuracy attached.

## 5. Tests

- Run of 2026-10-14 (author's MacBook, `requirements.txt` only): 15 passed, **1 skipped**. The skipped test is `test_unet_output_shape` in `tests/test_train.py`, which calls `pytest.importorskip("torch")` and so is skipped whenever PyTorch is not installed.
- Re-run of 2026-10-25 (Python 3.10, pinned `requirements-train.txt`, which includes `torch==2.11.0`): 16 passed, 0 skipped.
- After the `main.py` depth-pairing fix of 2026-10-25, which added `test_main_runs_on_treeid_style_files`: 17 passed, 0 skipped.
- After adding the occluded recall, alignment check, cross-validation and U-Net cost code (2026-10-25, `tests/test_eval.py`, including end-to-end runs of `train.py` and `crossval.py` on a small synthetic dataset): 24 passed, 0 skipped. This is the run in the committed `doc/test_report.html`.
- CI installs only `requirements.txt`, so on GitHub Actions the two PyTorch tests are skipped (22 passed, 2 skipped). The test only checks the network's output shape on random input; the U-Net has not been trained or evaluated on the dataset, and no U-Net results are reported here.

## 6. Limitations of these results

- **One test split.** The test-set numbers come from one split with `seed = 0` (`config/default.toml`). The 5-fold cross-validation (section 1a) gives the spread over validation folds (about 0.01 Dice), but the held-out test set was evaluated once, with no confidence interval. Differences smaller than about 0.01-0.02 Dice (for example MLP vs tree, or RGB-D vs RGB only) should not be read as real.

- **Occluded recall is a proxy.** "Hidden" means "missed by the baseline's visible-bark detector", not a labelled leaf-occlusion mask. The baseline scores 0 by construction, and the metric also counts wood missed for other reasons (shade, GT offset).
- **Ground-truth offset.** The summer GT masks are offset from the visible trunk by a median of about 13-17 px at 256 x 256, in both directions (section 1c). All accuracy numbers are against the GT as supplied. No correction is applied, because the true offset cannot be known at test time and the teaching team has not confirmed how the masks were made.
- **Not done, out of scope for this submission:**
  - **Robustness to leaf cover:** the dataset has summer and winter images only (no autumn), and only summer has GT masks.
  - **3D skeleton:** the skeleton is not lifted to 3D with depth, and the winter depth-consistency check was not run. There is no 3D ground truth to validate against, and the 8-bit depth has unknown scale.
  - **Junction-count error:** not computed for the skeletons.
  - **Duplicate-depth check:** row R04 is excluded by name in `config/default.toml` instead of a hash check.
- **U-Net: future work.** `src/train_unet.py` and `src/unet.py` are implemented and tested for output shape, and their cost is in section 4c, but the network has not been trained or evaluated. No U-Net accuracy is claimed. Section 1 suggests any future method should be compared with the shape prior, not only the baseline.
- **U-Net: future work.** `src/train_unet.py` and `src/unet.py` are implemented and tested for output shape, and their cost is in section 4c, but the network has not been trained or evaluated. No U-Net accuracy is claimed.

## Points to interpret in the report

- **The shape prior scores as well as the best learned model** (Dice 0.377 vs MLP 0.369 and tree 0.358, Chamfer 10.2 vs 10.1 and 12.0 px), and without position features the learned models fall well below it (Dice 0.21 to 0.23). So most of what the regression models gain over the baseline comes from learning where trunks usually are in the frame, not from seeing branches in the image. This needs discussing in the report, and the prior is the fairer comparison for any future method (for example the U-Net).
- Both learned models more than double the baseline's Dice and halve its chamfer distance. The baseline has high recall but very low precision: it marks almost every non-green pixel (soil, sky, posts) as wood.
- The MLP and decision tree are almost tied (0.369 vs 0.358 Dice); the tree trains about 9x faster and is easier to interpret.
- Working resolution barely changes accuracy (Dice 0.36 to 0.37 across 64 to 256), so 64 x 64 is the cheapest adequate choice. Bilinear upsampling of the prediction loses little.
- Depth adds almost nothing to the learned models. Possible reasons: the depth band already isolates the target row, position features capture the same "tree is in the centre" prior, or the 8-bit depth is too coarse.
- Check `res/tree_feature_importance.csv`: on a small test, row/column position dominated. That means the models partly learn where trunks usually are, which is a limitation (would not transfer to differently framed images).
- SVR is limited by its training-set size (cost grows roughly with n^2) and is about 400x slower per image than the tree.
- In the baseline, skeletonisation is the bottleneck (57% of time).
