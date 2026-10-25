# Results summary (runs of 2026-10-14)

All runs: 541 summer images with ground truth (row R04 excluded), split by tree into 379 train / 81 validation / 81 test. Metrics are means over the 81 test images. Dice and IoU: higher is better. Chamfer distance (pixels, between skeletons): lower is better. Times measured on the author's MacBook (Apple silicon, CPU).

## 1. Main comparison (`python src/train.py`, working size 128 x 128, RGB-D)

| Method | IoU | Dice | Precision | Recall | Chamfer (px) | Fit time (s) | Time per image (s) | Threshold (from validation) |
|---|---|---|---|---|---|---|---|---|
| Baseline (ExG + depth thresholding) | 0.081 | 0.149 | 0.083 | 0.741 | 20.10 | 0 | 0.025 | n/a |
| Shape prior (mean training mask, no image input) | 0.237 | 0.377 | 0.281 | 0.574 | 10.18 | 0 | 0 | 0.20 |
| Linear regression | 0.123 | 0.216 | 0.140 | 0.541 | 22.71 | 0.1 | 0.002 | 0.55 |
| Decision tree regression | 0.223 | 0.358 | 0.264 | 0.565 | 11.99 | 6.0 | 0.002 | 0.75 |
| Support vector regression (3,000 px) | 0.159 | 0.271 | 0.186 | 0.526 | 16.02 | 0.2 | 1.092 | 0.80 |
| Neural network (MLP) regression | 0.230 | 0.369 | 0.261 | 0.636 | 10.05 | 55.4 | 0.003 | 0.70 |

The shape prior is the same mask for every test image: the mean of the training masks, thresholded on the validation trees. It uses no information from the image, so it is the score a method gets from knowing only where trees usually are in the frame. Source: `res/ml_results.csv`, `res/ml_validation.csv`.

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

## 4. Profiling the baseline (`python tools/profile_baseline.py`, 20 synthetic 640 x 480 images)

| Function | Cumulative time (s) | Share |
|---|---|---|
| Total | 0.412 | 100% |
| `mask_to_skeleton` (skimage Zhang skeletonisation) | 0.233 | 57% |
| `predict_mask` | 0.179 | 43% |
| of which `excess_green` | 0.079 | 19% |
| of which `binary_closing` | 0.047 | 11% |

## 5. Tests

- Run of 2026-10-14 (author's MacBook, `requirements.txt` only): 15 passed, **1 skipped**. The skipped test is `test_unet_output_shape` in `tests/test_train.py`, which calls `pytest.importorskip("torch")` and so is skipped whenever PyTorch is not installed.
- Re-run of 2026-10-25 (Python 3.10, pinned `requirements-train.txt`, which includes `torch==2.11.0`): 16 passed, 0 skipped.
- After the `main.py` depth-pairing fix of 2026-10-25, which added `test_main_runs_on_treeid_style_files`: 17 passed, 0 skipped. This is the run in the committed `doc/test_report.html`.
- CI installs only `requirements.txt`, so on GitHub Actions the U-Net test is still skipped (16 passed, 1 skipped). The test only checks the network's output shape on random input; the U-Net has not been trained or evaluated on the dataset, and no U-Net results are reported here.

## 6. Limitations of these results

- **Single seed.** Every number above comes from one run with `seed = 0` (`config/default.toml`), which fixes the train / validation / test split by tree, the pixel sampling and the model initialisation (`random_state` for the decision tree and MLP). No repeats with other seeds were run, so there are no standard deviations or confidence intervals. Differences smaller than a few hundredths of Dice (for example MLP 0.369 vs tree 0.358, or RGB-D vs RGB only) may not be real. To check, re-run `src/train.py` with several values of `seed` and report the mean and spread.

## Points to interpret in the report

- **The shape prior scores as well as the best learned model** (Dice 0.377 vs MLP 0.369 and tree 0.358, Chamfer 10.2 vs 10.1 and 12.0 px), and without position features the learned models fall well below it (Dice 0.21 to 0.23). So most of what the regression models gain over the baseline comes from learning where trunks usually are in the frame, not from seeing branches in the image. This needs discussing in the report, and the prior is the fairer comparison for any future method (for example the U-Net).
- Both learned models more than double the baseline's Dice and halve its chamfer distance. The baseline has high recall but very low precision: it marks almost every non-green pixel (soil, sky, posts) as wood.
- The MLP and decision tree are almost tied (0.369 vs 0.358 Dice); the tree trains about 9x faster and is easier to interpret.
- Working resolution barely changes accuracy (Dice 0.36 to 0.37 across 64 to 256), so 64 x 64 is the cheapest adequate choice. Bilinear upsampling of the prediction loses little.
- Depth adds almost nothing to the learned models. Possible reasons: the depth band already isolates the target row, position features capture the same "tree is in the centre" prior, or the 8-bit depth is too coarse.
- Check `res/tree_feature_importance.csv`: on a small test, row/column position dominated. That means the models partly learn where trunks usually are, which is a limitation (would not transfer to differently framed images).
- SVR is limited by its training-set size (cost grows roughly with n^2) and is about 400x slower per image than the tree.
- In the baseline, skeletonisation is the bottleneck (57% of time).
