"""Small U-Net for binary branch segmentation from RGB-D input."""

import torch
from torch import nn


def _conv_block(c_in, c_out):
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
        super().__init__()
        chs = [base * 2 ** i for i in range(levels + 1)]
        self.down = nn.ModuleList([_conv_block(in_channels, chs[0])] +
                                  [_conv_block(chs[i - 1], chs[i]) for i in range(1, levels + 1)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(levels))])
        self.dec = nn.ModuleList([_conv_block(2 * chs[i], chs[i]) for i in reversed(range(levels))])
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else self.pool(x))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], dim=1))
        return self.head(x)


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
