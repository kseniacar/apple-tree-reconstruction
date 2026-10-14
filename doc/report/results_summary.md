# Results summary (runs of 2026-10-14)

All runs: 541 summer images with ground truth (row R04 excluded), split by tree into 379 train / 81 validation / 81 test. Metrics are means over the 81 test images. Dice and IoU: higher is better. Chamfer distance (pixels, between skeletons): lower is better. Times measured on the author's MacBook (Apple silicon, CPU).

## 1. Main comparison (`python src/train.py`, working size 128 x 128, RGB-D)

| Method | IoU | Dice | Precision | Recall | Chamfer (px) | Fit time (s) | Time per image (s) | Threshold (from validation) |
|---|---|---|---|---|---|---|---|---|
| Baseline (ExG + depth thresholding) | 0.081 | 0.149 | 0.083 | 0.741 | 20.10 | 0 | 0.025 | n/a |
| Linear regression | 0.123 | 0.216 | 0.140 | 0.541 | 22.71 | 0.1 | 0.002 | 0.55 |
| Decision tree regression | 0.223 | 0.358 | 0.264 | 0.565 | 11.99 | 6.0 | 0.002 | 0.75 |
| Support vector regression (3,000 px) | 0.159 | 0.271 | 0.186 | 0.526 | 16.02 | 0.2 | 1.092 | 0.80 |
| Neural network (MLP) regression | 0.230 | 0.369 | 0.261 | 0.636 | 10.05 | 55.4 | 0.003 | 0.70 |

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

## 4. Profiling the baseline (`python tools/profile_baseline.py`, 20 synthetic 640 x 480 images)

| Function | Cumulative time (s) | Share |
|---|---|---|
| Total | 0.412 | 100% |
| `mask_to_skeleton` (skimage Zhang skeletonisation) | 0.233 | 57% |
| `predict_mask` | 0.179 | 43% |
| of which `excess_green` | 0.079 | 19% |
| of which `binary_closing` | 0.047 | 11% |

## 5. Tests

`bash tools/run_tests.sh`: 15 passed, 1 skipped (U-Net test, PyTorch not installed). Report: `doc/test_report.html`.

## Points to interpret in the report

- Both learned models more than double the baseline's Dice and halve its chamfer distance. The baseline has high recall but very low precision: it marks almost every non-green pixel (soil, sky, posts) as wood.
- The MLP and decision tree are almost tied (0.369 vs 0.358 Dice); the tree trains about 9x faster and is easier to interpret.
- Working resolution barely changes accuracy (Dice 0.36 to 0.37 across 64 to 256), so 64 x 64 is the cheapest adequate choice. Bilinear upsampling of the prediction loses little.
- Depth adds almost nothing to the learned models. Possible reasons: the depth band already isolates the target row, position features capture the same "tree is in the centre" prior, or the 8-bit depth is too coarse.
- Check `res/tree_feature_importance.csv`: on a small test, row/column position dominated. That means the models partly learn where trunks usually are, which is a limitation (would not transfer to differently framed images).
- SVR is limited by its training-set size (cost grows roughly with n^2) and is about 400x slower per image than the tree.
- In the baseline, skeletonisation is the bottleneck (57% of time).
