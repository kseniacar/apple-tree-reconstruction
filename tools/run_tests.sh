#!/usr/bin/env bash
# Run the test suite and write an HTML test report to doc/test_report.html
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest --html=doc/test_report.html --self-contained-html "$@"
