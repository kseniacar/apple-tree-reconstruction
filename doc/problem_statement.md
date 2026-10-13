# Proposed problem statement: apple-tree branch reconstruction

*Draft for Ksenia, MMA3001 2026-S2 individual project. Section numbers refer to the Project Brief PDF.*

## 1. Engineering problem (Brief, Project Requirements §1)

### 1.1 Engineering context

Apple harvesting is still largely done by hand, and growers increasingly struggle to find seasonal labour at an acceptable cost. Automated harvesting is a mechanical engineering response to this problem. A harvesting robot is a mechatronic system: an RGB-D camera (colour plus per-pixel depth) senses the canopy, a perception module interprets the scene, a motion planner computes collision-free trajectories, and a multi-degree-of-freedom manipulator positions a gripper or vacuum end-effector to detach the fruit.

The mechanical design of such a machine depends on knowing where the tree's rigid structure is. Leaves and fruit are compliant and yield on contact, but the trunk and branches do not. A collision with wood can damage the tree, bruise fruit, overload joints or jam the end-effector, or trigger an emergency stop that costs cycle time. The branch geometry therefore defines the obstacle field the manipulator's reachable workspace must be planned around. It also drives design choices such as arm reach, the number of degrees of freedom, end-effector size and approach direction. Modern orchards grow trees in narrow, near-planar rows on trellis wires (as in the TreeID dataset, Chen et al., 2020) specifically to make this geometry more accessible to machines, but in summer the foliage still hides a large share of the wood from the camera.

The same problem appears in aerospace engineering. Aerial platforms are being developed for orchard work, including UAV-based fruit picking, canopy inspection and yield mapping. These aircraft must hover and approach close to the canopy. A UAV that strikes a hidden branch risks propeller damage, loss of control and a crash, so reliable detection of thin, partly occluded obstacles is a flight-safety requirement. More broadly, estimating occluded structure from onboard RGB-D sensing, under tight runtime and computing limits, is the same perception task that autonomous aircraft face in obstacle avoidance and vision-based navigation in cluttered environments.

This project therefore addresses the perception step that both ground-based manipulators and aerial platforms rely on: recovering the rigid branch structure of a tree, including the parts hidden by leaves, from a single onboard RGB-D observation.

### 1.2 Problem being addressed

The brief defines the apple-tree reconstruction task as:

- **Input:** RGB-D observations of apple trees collected over three seasons.
- **Output:** estimated three-dimensional tree skeletons.
- **Validation:** comparison with ground-truth skeletons that may be partially obscured by leaves.

This project adopts that definition and makes it precise enough to implement and test:

> **Given RGB-D observations of a trellised apple tree from three seasons, estimate a three-dimensional skeleton of its trunk and branches, *including branches hidden by leaves in the leafy seasons*, and quantify how accurately it matches the ground truth.**

**Inputs.** Each image is a 640 × 480 colour image plus an aligned 8-bit depth image. Each season plays a different role:

| Season | Foliage | Role in this project |
|---|---|---|
| Summer | Dense leaves (the harvesting condition) | The main test case. The skeleton must be recovered *through* the foliage. Summer is the season with ground-truth masks. |
| Autumn | Partial leaf cover | An intermediate occlusion level, used to test how accuracy changes as leaf cover changes (robustness). |
| Winter | Leaf-free, wood fully visible | A reference showing the true structure of the same tree. It is used to check depth along the branches and, where trees match across seasons, as extra training or validation evidence. |

**Output.** A 3D tree skeleton is a graph:

- **Nodes** are the trunk base, branch junctions and branch tips, each with a position (u, v, d): image column, image row and depth.
- **Edges** are the branch segments between nodes, each stored as the chain of 3D points along the branch centreline.

The skeleton is built in three steps:

1. Segment the wood, including occluded parts, into a 256 × 256 branch mask.
2. Thin the mask to a one-pixel centreline.
3. Lift each centreline point to 3D using the depth image. Where leaves cover a branch, the depth sensor sees the leaf rather than the wood, so depth along hidden sections is interpolated from the visible wood on either side.

The 2D mask is an intermediate output that can be checked directly against the ground truth.

**Ground truth and evaluation.** The supplied ground truth consists of 256 × 256 branch masks (`_GT`), which are the image projection of the true skeleton. Validation is therefore done in two parts:

- **Quantitative (2D):** the estimated skeleton is projected back onto the image and compared with the `_GT` masks using IoU, Dice, precision, recall and the mean distance between the estimated and true centrelines (section 5).
- **Depth consistency:** the estimated depth along each branch is checked against the leaf-free winter depth of the same tree, where tree IDs can be matched between seasons.

Depth is 8-bit with unknown scaling, so 3D positions are in relative depth units, not millimetres, unless calibration data is released.

**The core difficulty** is occlusion. A method that only labels visible bark scores well on precision but misses the hidden wood, so its skeleton breaks into disconnected pieces. The project compares a simple baseline that can only see visible bark against a learned method that can infer hidden branches from the surrounding context. It also measures how each method degrades from winter to autumn to summer as leaf cover increases.

### 1.3 Why the problem matters

- **Safety and reliability of the machine.** The branch mask defines the obstacles the planner has to avoid. A **missed branch** (false negative) is a potential collision. A **false branch** (false positive) makes the planner treat a free path as blocked, so reachable fruit is skipped. The two errors cost different things, which is why precision and recall are reported separately rather than as a single accuracy figure.
- **Throughput.** Harvesters have to work in seconds per fruit, so a method that is accurate but very slow is not usable on the machine. Runtime per image is therefore an evaluation criterion (Brief §6), alongside accuracy.
- **Sensing cost.** If branches can be inferred from one RGB-D frame, the robot does not need extra viewpoints, extra sensors or a prior leaf-free (winter) scan of every tree. The comparison between RGB-only and RGB-D input also measures whether the depth sensor actually pays for itself.
- **Transfer to other tasks.** The same branch map supports pruning, yield estimation and orchard digital twins.

### 1.4 Intended use of the computational solution

The deliverable is an **offline perception component and evaluation toolkit**, not a robot controller. Concretely, the code in the repository:

1. **reads** a summer RGB image, its depth image and the matching ground-truth mask (`src/utils.py`);
2. **predicts** a branch mask. The current baseline (`src/baseline.py`) marks pixels as wood when their excess-green index is low (not leaf-coloured) and their depth lies in a set band. It then centre-crops to 480 × 480 to match the masks' assumed geometry, resizes to 256 × 256 and applies a morphological closing. A learned model (U-Net) is planned as the alternative method and will share the same interface;
3. **skeletonises** the mask into a centreline (`skimage.morphology.skeletonize`). The step that lifts the centreline to a 3D skeleton using depth is planned but **not yet implemented**;
4. **scores** the prediction against ground truth (`src/metrics.py`: IoU, Dice, precision, recall, Chamfer skeleton distance) and reports averages and runtime over the dataset (`src/main.py`).

The intended user is an engineer choosing or tuning a perception method for a harvesting or pruning robot. The output answers questions such as "which method misses fewer branches?", "how much accuracy does depth add?" and "what does that accuracy cost in runtime?". Downstream, the mask or skeleton would be passed to a motion planner as an obstacle map. That integration is outside this project.

### 1.5 Limitations of the selected scope (and why they are accepted)

| Limitation | Justification |
|---|---|
| **3D accuracy can only be partly validated.** The output is a 3D skeleton, but the ground truth is 2D masks and depth is relative. | The skeleton's image projection is validated quantitatively against `_GT`, and its depth is checked against the winter capture. Absolute 3D error in millimetres cannot be established without metric depth, camera intrinsics or 3D ground truth (see section 7). |
| **Depth is relative, not metric.** The depth PNGs are 8-bit with unknown scaling, and no camera intrinsics are provided. | Depth is used only as a foreground/background cue and as an extra input channel. No distances in millimetres are claimed. This changes if the teaching team releases the raw 16-bit depth and calibration. |
| **Single viewpoint, single frame.** | A real robot could move the camera, but the dataset has one view per tree per season. Single-frame inference is the harder case and the cheapest one to deploy. |
| **One orchard, one camera, one year.** | Results do not establish generalisation to other cultivars, training systems, lighting or sensors. The tree-wise train/test split limits leakage within the dataset but cannot fix this. |
| **Ground-truth quality is unconfirmed.** Summer masks appear offset by 15–45 px from the visible trunk (Ed report), and the mask-to-image transform is assumed. | Results will be reported with and without an alignment correction until the teaching team confirms how the masks were made (section 7). The offset bounds how small the skeleton distance can meaningfully get. |
| **Row 4 excluded.** | Its winter depth images are duplicates of another tree's (Ed report), so its data is not trustworthy. |
| **No robot integration.** Motion planning and arm control are out of scope. | The project assesses the perception step that those systems depend on. The engineering consequences of errors are discussed, not simulated. |
| **The current tests prove the code is correct, not that the method is accurate.** | The 10 pytest tests use small synthetic arrays with known answers. They check that the metrics compute the right values (identical masks score 1, disjoint masks score 0, a known half-overlap gives IoU = 1/3, a 2-pixel line shift gives a 2-pixel Chamfer distance, mismatched shapes raise an error). They also check that the baseline separates a brown stripe from a green background, that cropping produces 480 × 480, and that `main` exits cleanly when the data folder is missing. They say **nothing** about performance on real trees. That comes only from running `src/main.py` on the dataset and the held-out evaluation in section 5. The RGB/depth/GT file pairing in `main.py` is also still a placeholder until the naming is confirmed. |

## 2. Inputs and outputs (Brief §2)

| | Item | Format | Notes |
|---|---|---|---|
| **Input** | RGB images, summer / autumn / winter | PNG, 640 × 480, 8-bit, 3 channels | Summer = main test case; autumn = partial leaf cover; winter = leaf-free reference |
| **Input** | Depth images, same three seasons | PNG, 640 × 480, 8-bit, 1 channel | Relative depth only; units and scaling unknown, 0 assumed to mean "no reading" |
| Output (intermediate) | Branch mask | Binary array, 256 × 256 (same grid as GT) | 1 = trunk or branch, including occluded |
| Output (intermediate) | 2D centreline | One-pixel-wide skeleton in image coordinates | From skeletonising the mask, before lifting to 3D |
| **Output (main)** | 3D skeleton | Graph: nodes (u, v, d) at base, junctions and tips; edges as 3D point chains | Relative depth unless calibration is provided |
| **Ground truth** | `_GT` masks | PNG, 256 × 256, binary | Only for evaluation and training, never as model input |

**Invalid inputs.** Missing depth file → RGB-only mode with a warning. Depth image that is a duplicate of another tree (Row 4 issue) → flagged by a hash check and excluded. Size mismatch → rejected with a clear error message.

## 3. Computational solution options (Brief §3)

1. **Classical baseline (required comparison).** Colour-index thresholding (for example excess-green to remove leaves, plus a bark-colour range in HSV), combined with a depth band to remove background rows, then morphological closing and skeletonisation (`skimage.morphology.skeletonize`). It is fast, interpretable and needs no training. It is expected to miss occluded branches.
2. **Per-pixel classical ML.** Random forest or gradient boosting on hand-built features (RGB, HSV, depth, local gradients, texture filters). This is a middle ground in complexity and shows feature importance.
3. **Deep segmentation (likely final method).** A small U-Net taking 4-channel RGB-D input and trained on the `_GT` masks with Dice + BCE loss. This is the approach family used in the dataset paper (Chen et al., 2020, arXiv:2010.06879, which used image-to-image translation to infer occluded branches). Train on a few hundred images on Colab GPU, with augmentation (flips, colour jitter).

The plan is to build option 1 first, because it sets the baseline and exercises the data pipeline. Then option 3, with option 2 if time allows.

## 4. Baseline and alternatives (Brief §4)

The comparison will cover accuracy (metrics below), runtime per image, memory and training cost, and interpretability. A further alternative is "RGB only" versus "RGB-D", which shows whether the depth channel actually helps.

## 5. Validation (Brief §5)

- **Data split by tree, not by image**, so the same tree never appears in both training and test sets. The test set is held out until the end. Model selection uses k-fold cross-validation on the training trees.
- **Mask metrics against `_GT`:** IoU, Dice/F1, precision and recall. Precision and recall matter separately because missed branches and false branches have different engineering costs.
- **Occluded-region metric:** recall computed only on GT pixels that are hidden by leaves in the summer image. Hidden pixels can be approximated as GT pixels that the baseline's visible-bark detector does not fire on, or taken from winter-image visibility if trees match across seasons. This is the metric that tests the actual problem.
- **Skeleton metrics:** mean centreline distance (symmetric Chamfer distance between predicted and GT skeletons, in pixels) and the error in junction count.
- **Alignment check before any of this:** the Ed report says summer GT masks are offset by 15–45 px from the visible trunk. The plan is to measure that offset per image (cross-correlation between GT and a trunk detection) and report results both with and without correction until the teaching team confirms how the masks were made.
- **What validation cannot show:** true 3D accuracy (2D GT only), or generalisation to other orchards, cameras or seasons.

## 6. Optimisation and performance (Brief §6)

Profile the pipeline with `cProfile` and timing, using the >30-minute runtimes reported on Ed as a reason to do this early. Run sensitivity studies on input resolution (128 vs 256), threshold values for the baseline, and network depth and width for the U-Net. Report FLOPs and parameter count for the U-Net.

## 7. Open data questions for the teaching team

The Ed thread is still open on these. Answers change the scope above.

1. **Summer GT alignment.** Were the `_GT` masks drawn on these exact summer images or on another capture, such as the paper's autumn set? They appear shifted 15–45 px left of the visible trunk, by a different amount for each tree.
2. **Mask-to-image transform.** What crop or resize maps the 640 × 480 RGB to the 256 × 256 masks? Is it a 480 × 480 centre crop followed by a resize?
3. **Winter masks.** Are there GT masks for the winter images, or only for summer? If winter masks exist, were they produced independently?
4. **Do masks include occluded branches?** Do the white pixels mean trunk and all branches including hidden ones, or only visible wood?
5. **Tree ID correspondence.** Is R01N01_Summer the same physical tree as R01N01_Winter? Why does `R04U01_Winter_RGB-R04N03_RGB_12-17-14.png` carry two IDs, and is there a correspondence table? (Keenan confirmed R = row, N = number, U = unmatched.)
6. **Capture timing and pruning.** Which season was captured first, how far apart were the captures, and were trees pruned in between?
7. **Depth encoding.** How do the 8-bit depth values map to distance? Is each image scaled independently? Does 0 mean no reading? Are the raw 16-bit depth and the camera intrinsics available?
8. **Row 4 depth duplicates.** Every Row 4 winter depth image in "Dataset (With Summer GT)" is a copy of R03N34. Is there a corrected version?
9. **Scope check.** The brief asks for 3D skeletons but the GT is 2D masks. Is validating the 3D skeleton by its 2D projection against GT, with a depth check against winter, acceptable? Is there an autumn capture, and is any 3D ground truth available?
