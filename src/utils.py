"""Small shared helpers: config loading and image I/O."""

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Python 3.10: same API from the tomli package
    import tomli as tomllib

import numpy as np
from PIL import Image


def load_config(path):
    """Load a TOML config file.

    Args:
        path: Path to a ``.toml`` file.

    Returns:
        dict of settings.

    Raises:
        FileNotFoundError: If the file does not exist.
        tomllib.TOMLDecodeError: If the file is not valid TOML.
    """
    with open(path, "rb") as f:
        return tomllib.load(f)


def read_image(path):
    """Read an image file into a numpy array.

    Args:
        path: Path to a PNG (or any format Pillow reads).

    Returns:
        uint8 array, (H, W) for greyscale or (H, W, C) for colour.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    return np.asarray(Image.open(path))


def read_depth(path):
    """Read an 8-bit depth image as a single-channel array.

    Some TreeID depth PNGs store the grey value in three identical channels;
    only the first is kept.

    Args:
        path: Path to the depth PNG.

    Returns:
        (H, W) uint8 array; 0 means no depth reading.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    depth = read_image(path)
    return depth[..., 0] if depth.ndim == 3 else depth


def read_mask(path):
    """Read a ground-truth mask as a boolean array (white = branch).

    Args:
        path: Path to a ``_GT`` mask PNG.

    Returns:
        (H, W) boolean array, True where the pixel value is above 127.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    img = read_image(path)
    if img.ndim == 3:
        img = img[..., 0]
    return img > 127
