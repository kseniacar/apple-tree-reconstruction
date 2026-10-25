"""Small U-Net for binary branch segmentation from RGB-D input."""

import torch
from torch import nn


def _conv_block(c_in, c_out):
    """Two 3x3 convolutions, each followed by batch norm and ReLU.

    Args:
        c_in: Input channels.
        c_out: Output channels.

    Returns:
        ``nn.Sequential`` block; the spatial size is unchanged (padding 1).
    """
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """U-Net (Ronneberger et al., 2015) returning one logit per pixel.

    Args:
        in_channels: 4 for RGB-D, 3 for RGB only.
        base: channels in the first level; doubles at each level down.
        levels: number of down-sampling steps (input size must divide by 2**levels).
    """

    def __init__(self, in_channels=4, base=16, levels=4):
        """Build the encoder, decoder and 1x1 output layer (arguments as for the class)."""
        super().__init__()
        chs = [base * 2 ** i for i in range(levels + 1)]
        self.down = nn.ModuleList([_conv_block(in_channels, chs[0])] +
                                  [_conv_block(chs[i - 1], chs[i]) for i in range(1, levels + 1)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(levels))])
        self.dec = nn.ModuleList([_conv_block(2 * chs[i], chs[i]) for i in reversed(range(levels))])
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x):
        """Run the network.

        Args:
            x: (B, in_channels, H, W) tensor, H and W divisible by 2**levels.

        Returns:
            (B, 1, H, W) tensor of logits (apply a sigmoid for probabilities).
        """
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else self.pool(x))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], dim=1))
        return self.head(x)


def count_parameters(model):
    """Number of trainable parameters.

    Args:
        model: A ``torch.nn.Module``.

    Returns:
        int.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, input_shape):
    """Estimate the floating-point operations of one forward pass.

    Counts the multiply-accumulates (MACs) of every ``Conv2d`` and
    ``ConvTranspose2d`` layer with forward hooks and reports FLOPs = 2 x MACs.
    Batch norm, ReLU, pooling and concatenation are left out; they are a few
    percent of the total for this network. The total size of the layer outputs
    is also returned, for a rough arithmetic-intensity estimate.

    Args:
        model: A ``torch.nn.Module`` such as ``UNet``.
        input_shape: Shape of one input batch, e.g. ``(1, 4, 256, 256)``.

    Returns:
        Tuple ``(flops, activations)``: FLOPs of the convolution layers and the
        number of values in their outputs, for the whole batch.
    """
    totals = {"macs": 0, "acts": 0}

    def hook(layer, inputs, output):
        """Add one convolution layer's MACs and output size to the totals."""
        k = layer.kernel_size[0] * layer.kernel_size[1] * layer.in_channels // layer.groups
        if isinstance(layer, nn.ConvTranspose2d):
            # every input value is multiplied by every output channel's kernel
            totals["macs"] += inputs[0].numel() * layer.out_channels * k // layer.in_channels
        else:
            totals["macs"] += output.numel() * k
        totals["acts"] += output.numel()

    handles = [m.register_forward_hook(hook) for m in model.modules()
               if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(input_shape))
    for h in handles:
        h.remove()
    return 2 * totals["macs"], totals["acts"]
