#!/usr/bin/env bash
# Build HTML API documentation from docstrings into doc/api/
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m pdoc src/main.py src/baseline.py src/metrics.py src/utils.py -o doc/api
