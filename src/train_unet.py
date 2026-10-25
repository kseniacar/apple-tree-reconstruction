"""Future work: train a U-Net (convolutional network) and compare it with the baseline.

Status: implemented and unit-tested (output shape, FLOP count) but NOT yet
trained or evaluated on the dataset, so no U-Net accuracy is reported anywhere
in this repository. Its cost is reported by tools/unet_cost.py.

The main method is in train.py (regression models taught in the unit). This
script is an extra comparison and needs PyTorch (requirements-train.txt).

Usage:
    python src/train_unet.py                       # settings from config/default.toml
    python src/train_unet.py --epochs 2 --limit 40 # quick check that everything runs
    python src/train_unet.py --no-depth            # RGB-only model for the RGB vs RGB-D comparison

Outputs (tag is "rgbd", or "rgb" with --no-depth):
    outputs/unet_<tag>_best.pt      best weights (by validation Dice), not committed
    res/train_log_<tag>.csv         loss and validation Dice per epoch
    res/training_curve_<tag>.png    plot of the above
    res/test_results_<tag>.csv      test-set metrics, baseline vs U-Net
    res/test_examples_<tag>.png     RGB / ground truth / baseline / U-Net on test images
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from baseline import predict_mask  # noqa: E402
from dataset import find_pairs, load_sample, split_by_tree  # noqa: E402
from metrics import dice  # noqa: E402
from train import mean_scores, score  # noqa: E402
from unet import UNet, count_parameters  # noqa: E402
from utils import load_config, read_depth, read_image, read_mask  # noqa: E402


def pick_device(name="auto"):
    """Choose the device to train on.

    Args:
        name: "auto" (GPU if there is one: CUDA on Colab, MPS on Apple
            silicon; otherwise CPU), or an explicit "cpu", "mps" or "cuda".

    Returns:
        ``torch.device``.
    """
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_split(pairs, size, use_depth):
    """Load samples into memory for training.

    Stored as float16 to halve memory; batches are converted to float32 on use.

    Args:
        pairs: List of ``(rgb_path, depth_path, gt_path)``.
        size: Side length the samples are resized to.
        use_depth: If False, only the three RGB channels are kept.

    Returns:
        (x, y): (N, C, size, size) inputs and (N, size, size) masks, float16.
    """
    xs, ys = [], []
    for i, p in enumerate(pairs, 1):
        x, y = load_sample(*p, size=size)
        xs.append(x if use_depth else x[:3])
        ys.append(y)
        if i % 100 == 0:
            print(f"  loaded {i}/{len(pairs)}", flush=True)
    return np.stack(xs).astype(np.float16), np.stack(ys).astype(np.float16)


def augment(x, y):
    """Random horizontal flip and brightness change, applied per batch.

    Trees are roughly left-right symmetric and lighting varies between
    captures, so both keep the labels valid. Depth is not scaled.

    Args:
        x: (B, C, H, W) input batch in [0, 1].
        y: (B, 1, H, W) target batch.

    Returns:
        (x, y) augmented copies.
    """
    if torch.rand(1).item() < 0.5:
        x, y = x.flip(-1), y.flip(-1)
    gain = 0.8 + 0.4 * torch.rand(x.shape[0], 1, 1, 1, device=x.device)
    x = x.clone()
    x[:, :3] = (x[:, :3] * gain).clamp(0, 1)
    return x, y


def loss_fn(logits, y):
    """Binary cross-entropy plus soft Dice loss.

    Branch pixels are only a few percent of the image; BCE alone lets the
    network predict "no branch" everywhere, while the Dice term rewards
    overlap on the thin branches.

    Args:
        logits: (B, 1, H, W) raw network outputs.
        y: (B, 1, H, W) targets in {0, 1}.

    Returns:
        Scalar loss tensor.
    """
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3))
    soft_dice = (2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
    return bce + (1 - soft_dice.mean())


@torch.no_grad()
def predict(model, x, device, batch_size=8):
    """Predict boolean masks, thresholding the sigmoid output at 0.5.

    Args:
        model: Trained ``UNet``.
        x: (N, C, H, W) inputs.
        device: ``torch.device`` to run on.
        batch_size: Images per forward pass.

    Returns:
        (N, H, W) boolean array.
    """
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i:i + batch_size].astype(np.float32)).to(device)
        out.append((torch.sigmoid(model(xb)) > 0.5).cpu().numpy()[:, 0])
    return np.concatenate(out)


def main(argv=None):
    """Train the U-Net, keep the best checkpoint by validation Dice and compare it with the baseline.

    Args:
        argv: Command-line arguments (None reads ``sys.argv``).

    Returns:
        0 on success, 1 if the data folder is missing or has fewer than 3
        images with ground truth.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/default.toml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no-depth", action="store_true", help="train on RGB only")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N images (quick test)")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    parser.add_argument("--out", default="outputs", help="folder for model weights")
    parser.add_argument("--res", default="res", help="folder for logs, results and figures")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    ucfg = cfg.get("unet", {})
    epochs = args.epochs or ucfg.get("epochs", 40)
    batch_size = args.batch_size or ucfg.get("batch_size", 8)
    lr = args.lr or ucfg.get("lr", 1e-3)
    size = ucfg.get("image_size", 256)
    use_depth = not args.no_depth
    tag = "rgbd" if use_depth else "rgb"
    torch.manual_seed(ucfg.get("seed", 0))

    root = Path(args.data_root or cfg["data_root"])
    if not root.exists():
        print(f"Data folder not found: {root}. See README 'Data'.", file=sys.stderr)
        return 1
    pairs = find_pairs(root, cfg["season"], cfg.get("exclude_rows", []))[:args.limit]
    if len(pairs) < 3:
        print(f"Only {len(pairs)} images with ground truth found under {root}.", file=sys.stderr)
        return 1
    split = split_by_tree(pairs, ucfg.get("val_frac", 0.15), ucfg.get("test_frac", 0.15), ucfg.get("seed", 0))
    print(f"{len(pairs)} images: {len(split['train'])} train, {len(split['val'])} val, "
          f"{len(split['test'])} test (split by tree)")

    print("Loading images...")
    x_tr, y_tr = load_split(split["train"], size, use_depth)
    x_va, y_va = load_split(split["val"], size, use_depth)
    x_te, y_te = load_split(split["test"], size, use_depth)

    device = pick_device(args.device)
    model = UNet(in_channels=x_tr.shape[1], base=ucfg.get("base_channels", 16)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"U-Net ({tag}), {count_parameters(model):,} parameters, training on {device} for {epochs} epochs")

    out_dir, res_dir = Path(args.out), Path(args.res)
    out_dir.mkdir(exist_ok=True)
    res_dir.mkdir(exist_ok=True)
    best_path = out_dir / f"unet_{tag}_best.pt"
    log, best = [], -1.0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order, losses = np.random.permutation(len(x_tr)), []
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            xb = torch.from_numpy(x_tr[idx].astype(np.float32)).to(device)
            yb = torch.from_numpy(y_tr[idx].astype(np.float32)).to(device)[:, None]
            xb, yb = augment(xb, yb)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        val_pred = predict(model, x_va, device)
        val_dice = float(np.mean([dice(p, g > 0.5) for p, g in zip(val_pred, y_va)]))
        log.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_dice": val_dice})
        saved = ""
        if val_dice > best:
            best = val_dice
            torch.save(model.state_dict(), best_path)
            saved = "  (best, saved)"
        print(f"epoch {epoch:3d}/{epochs}  loss {log[-1]['train_loss']:.3f}  val Dice {val_dice:.3f}"
              f"  [{time.perf_counter() - start:.0f} s]{saved}", flush=True)
    train_time = time.perf_counter() - start

    with open(res_dir / f"train_log_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log[0].keys())
        w.writeheader()
        w.writerows(log)
    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot([r["epoch"] for r in log], [r["train_loss"] for r in log], label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss")
    ax2 = ax1.twinx()
    ax2.plot([r["epoch"] for r in log], [r["val_dice"] for r in log], color="tab:orange", label="val Dice")
    ax2.set_ylabel("validation Dice")
    fig.legend(loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(res_dir / f"training_curve_{tag}.png", dpi=150)
    plt.close(fig)

    # Final comparison on the held-out test trees, using the best checkpoint.
    model.load_state_dict(torch.load(best_path, map_location=device))
    t0 = time.perf_counter()
    unet_pred = predict(model, x_te, device)
    unet_time = (time.perf_counter() - t0) / len(x_te)
    base_rows, unet_rows, base_preds = [], [], []
    t0 = time.perf_counter()
    for (rgb_p, d_p, gt_p), up in zip(split["test"], unet_pred):
        depth = read_depth(d_p) if d_p is not None else None
        bp = predict_mask(read_image(rgb_p), depth if use_depth else None, **cfg["baseline"])
        gt = read_mask(gt_p)
        base_preds.append(bp)
        base_rows.append(score(bp, gt))
        unet_rows.append(score(up, gt))
    base_time = (time.perf_counter() - t0) / len(x_te)
    results = {"baseline": mean_scores(base_rows), f"unet_{tag}": mean_scores(unet_rows)}
    results["baseline"]["sec_per_image"] = base_time
    results[f"unet_{tag}"]["sec_per_image"] = unet_time

    print(f"\nTest set ({len(x_te)} images), training took {train_time / 60:.1f} min")
    keys = list(results["baseline"])
    print(f"{'':>14}" + "".join(f"{k:>15}" for k in keys))
    for name, r in results.items():
        print(f"{name:>14}" + "".join(f"{r[k]:>15.3f}" for k in keys))
    with open(res_dir / f"test_results_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + keys)
        for name, r in results.items():
            w.writerow([name] + [f"{r[k]:.4f}" for k in keys])

    n = min(4, len(x_te))
    fig, axes = plt.subplots(n, 4, figsize=(10, 2.6 * n), squeeze=False)
    for i in range(n):
        for ax, img, title in zip(axes[i], [x_te[i][:3].astype(np.float32).transpose(1, 2, 0), y_te[i],
                                            base_preds[i], unet_pred[i]],
                                  ["RGB", "ground truth", "baseline", "U-Net"]):
            ax.imshow(img, cmap=None if img.ndim == 3 else "gray")
            ax.set_title(title if i == 0 else "")
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(res_dir / f"test_examples_{tag}.png", dpi=120)
    plt.close(fig)
    print(f"Saved weights to {best_path} and results/figures to {res_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
