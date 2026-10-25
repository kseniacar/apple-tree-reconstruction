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

Needs Python 3.11 or newer (Colab is fine). Dependency versions are pinned in `requirements.txt` and `requirements-train.txt`.

```bash
pip install -r requirements.txt
python src/main.py --config config/default.toml   # run the baseline on the dataset
pytest                                            # run the tests
```

In Colab:

```python
!git clone https://github.com/kseniacar/apple-tree-reconstruction.git
%cd apple-tree-reconstruction
!pip install -r requirements.txt
from google.colab import drive; drive.mount('/content/drive')
!python src/main.py --data-root /content/drive/MyDrive/TreeID
```

## Training the models

The main method (`src/train.py`) uses regression models taught in the unit. Each pixel gets colour, depth, neighbourhood and position features, and a model predicts how likely it is to be branch. Models: linear regression, decision tree, support vector regression and neural network (MLP) regression.

```bash
python src/train.py --limit 40          # quick check
python src/train.py                     # all models, settings in config/default.toml [ml]
python src/train.py --work-size 64      # resolution sensitivity study
python src/train.py --no-depth          # does depth help?
python src/train.py --no-position       # without row/column features: do models use appearance?
```

Images are split by tree into train / validation / test. The validation trees set each model's threshold; the test trees are only used for the final comparison with the baseline. Trained models go to `outputs/` (not committed); results, example images and decision-tree feature importance go to `res/`.

Optional extension: `src/train_unet.py` trains a U-Net (needs `pip install -r requirements-train.txt`).

## Documentation

- HTML API docs: `doc/api/index.html`, rebuilt with `bash tools/build_docs.sh` (pdoc)
- Test report: `doc/test_report.html`, rebuilt with `bash tools/run_tests.sh` (pytest-html)

Both are committed so they are included in a clone of the repository. Rebuild and commit them after changing code.
- Project report: `doc/report/`

## AI use

See `doc/ai_use.md`.

## Licence

MIT, see `LICENSE`.
