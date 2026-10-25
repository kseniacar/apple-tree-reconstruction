"""Parameter count, FLOPs and arithmetic intensity of the U-Net (Brief §6).

Needs PyTorch but no dataset and no training. Prints a table for the
configured network and for narrower/wider and shallower/deeper variants, so
the cost of each design choice can be compared before training anything.

Usage: python tools/unet_cost.py [--size 256] [--channels 4]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from unet import UNet, count_flops, count_parameters  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--size", type=int, default=256, help="input side length in pixels")
parser.add_argument("--channels", type=int, default=4, help="4 for RGB-D, 3 for RGB")
args = parser.parse_args()

print(f"Input 1 x {args.channels} x {args.size} x {args.size} (float32)")
print(f"{'base':>5} {'levels':>6} {'params':>10} {'GFLOPs':>8} {'MB weights':>10} {'MB acts':>8} {'FLOP/byte':>9}")
for base, levels in [(8, 4), (16, 3), (16, 4), (32, 4)]:
    model = UNet(in_channels=args.channels, base=base, levels=levels)
    params = count_parameters(model)
    flops, acts = count_flops(model, (1, args.channels, args.size, args.size))
    # Rough intensity: each weight and each layer output read or written once, 4 bytes each.
    intensity = flops / (4 * (params + acts))
    note = "  <- config/default.toml" if (base, levels) == (16, 4) else ""
    print(f"{base:>5} {levels:>6} {params:>10,} {flops / 1e9:>8.2f} {4 * params / 1e6:>10.2f} "
          f"{4 * acts / 1e6:>8.1f} {intensity:>9.1f}{note}")
