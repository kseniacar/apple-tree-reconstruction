# Apple-tree branch reconstruction (MMA3001 individual project)

Estimate the trunk and branch structure of apple trees, including branches hidden by leaves, from summer RGB-D images. Results are validated against ground-truth branch masks.

## Repository layout

```
.github/workflows/ci.yml   CI: runs tests and builds the docs on every push
config/                    Settings (data paths, image sizes, method parameters)
data/                      Local data only, NOT committed (see "Data" below)
doc/                       Project report sources and generated HTML docs
res/                       Figures and images used in the README and report
src/                       Source code (importable modules, no notebook code)
tests/                     pytest unit tests
tools/                     Helper scripts (build docs, run tests, profiling)
```

## Data

The TreeID dataset is too large for GitHub and is **not stored in this repository**.

1. Download it from: `https://drive.google.com/drive/folders/16qylQXmTf9OiKgKCMliudNYEUYxqlNEr?usp=sharing`
2. Unzip it into `data/` so that you have `data/TreeID/...`, or point `data_root` in `config/default.toml` at wherever you keep it (for example a mounted Google Drive folder in Colab).

Dataset reference: Chen et al. (2020), arXiv:2010.06879.

## Quick start

Needs Python 3.10 or newer (Colab is fine).

```bash
pip install -r requirements.txt
python src/main.py --config config/default.toml   # run the baseline on the dataset
pytest                                            # run the tests
```

In Colab:

```python
!git clone https://github.com/kseniacar/apple-tree-reconstruction.git
%cd <repo>
!pip install -r requirements.txt
from google.colab import drive; drive.mount('/content/drive')
!python src/main.py --data-root /content/drive/MyDrive/TreeID
```

## Documentation

- HTML API docs: `doc/api/index.html`, rebuilt with `bash tools/build_docs.sh` (pdoc)
- Test report: `doc/test_report.html`, rebuilt with `bash tools/run_tests.sh` (pytest-html)

Both are committed so they are included in a clone of the repository. Rebuild and commit them after changing code.
- Project report: `doc/report/`

## AI use

See `doc/ai_use.md`.

## Licence

MIT, see `LICENSE`.
