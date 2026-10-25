#!/usr/bin/env bash
# Build HTML API documentation from docstrings into doc/api/
set -euo pipefail
cd "$(dirname "$0")/.."
modules=(src/main.py src/baseline.py src/metrics.py src/utils.py src/dataset.py src/features.py src/train.py src/crossval.py src/alignment.py)
# The U-Net modules import PyTorch (requirements-train.txt). Document them when it is installed;
# otherwise skip them with a warning so CI, which only installs requirements.txt, still builds.
if python -c "import torch" 2>/dev/null; then
    modules+=(src/unet.py src/train_unet.py)
else
    echo "warning: PyTorch not installed, skipping src/unet.py and src/train_unet.py" >&2
fi
rm -rf doc/api
PYTHONPATH=src python -m pdoc "${modules[@]}" -o doc/api
