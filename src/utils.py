"""Small shared helpers: config loading and image I/O."""

from pathlib import Path
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
    """
    with open(path, "rb") as f:
        return tomllib.load(f)


def read_image(path):
    """Read an image file into a numpy array (uint8)."""
    return np.asarray(Image.open(path))


def read_mask(path):
    """Read a ground-truth mask and return it as a boolean array (white = branch)."""
    img = read_image(path)
    if img.ndim == 3:
        img = img[..., 0]
    return img > 127


def find_samples(data_root, season="Summer", exclude_rows=()):
    """List image files for one season, skipping excluded rows.

    The exact file naming is still being confirmed with the teaching team,
    so this matches on the season name and tree-ID prefix only.

    Args:
        data_root: Folder containing the dataset.
        season: "Summer" or "Winter".
        exclude_rows: Row prefixes such as ``"R04"`` to skip.

    Returns:
        Sorted list of ``Path`` objects.
    """
    root = Path(data_root)
    files = sorted(root.rglob(f"*{season}*.png"))
    return [f for f in files if not any(f.name.startswith(r) for r in exclude_rows)]
