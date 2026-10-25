# AI use log

| Date | Tool | What it was used for | How the output was checked |
|---|---|---|---|
| 2026-10-12 | Claude | Drafted problem statement and repo scaffold | Reviewed against project brief |
| 2026-10-14 | Claude | Wrote training code: per-pixel regression models (src/train.py, src/features.py), optional U-Net (src/train_unet.py, src/unet.py), data split (src/dataset.py); fixed file pairing in src/main.py | Ran pytest; trained all models on a 16-image subset and checked they beat the baseline |
| 2026-10-25 | Claude | Repo clean-up for the rubric: pinned dependency versions, API docs for the U-Net modules, Colab `%cd` line fixed, skipped-test and single-seed limitations noted in the results summary | Re-ran pytest (16 passed with PyTorch, 15 + 1 skipped without) and rebuilt the docs; checked the diffs |
| 2026-10-25 | Claude | Rubric check of the code: found that `main.py` never loaded depth (looked for `_Depth-` files; TreeID uses `_D-`), two module docstrings had been overwritten by function docstrings, and the shape-prior result was missing from the results summary; fixed these and completed the docstrings | Wrote a test that fails on the old `main.py` and passes on the fix; 17 tests pass; checked the added table rows against `res/*.csv` |
