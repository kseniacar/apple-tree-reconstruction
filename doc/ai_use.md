# AI use log

| Date | Tool | What it was used for | How the output was checked |
|---|---|---|---|
| 2026-10-12 | Claude | Drafted problem statement and repo scaffold | Reviewed against project brief |
| 2026-10-14 | Claude | Wrote training code: per-pixel regression models (src/train.py, src/features.py), optional U-Net (src/train_unet.py, src/unet.py), data split (src/dataset.py); fixed file pairing in src/main.py | Ran pytest; trained all models on a 16-image subset and checked they beat the baseline |
| 2026-10-25 | Claude | Repo clean-up for the rubric: pinned dependency versions, API docs for the U-Net modules, Colab `%cd` line fixed, skipped-test and single-seed limitations noted in the results summary | Re-ran pytest (16 passed with PyTorch, 15 + 1 skipped without) and rebuilt the docs; checked the diffs |
| 2026-10-25 | Claude | Rubric check of the code: found that `main.py` never loaded depth (looked for `_Depth-` files; TreeID uses `_D-`), two module docstrings had been overwritten by function docstrings, and the shape-prior result was missing from the results summary; fixed these and completed the docstrings | Wrote a test that fails on the old `main.py` and passes on the fix; 17 tests pass; checked the added table rows against `res/*.csv` |
| 2026-10-25 | Claude | Added the evaluation items from the project plan: occluded-branch recall (src/metrics.py), GT alignment check (src/alignment.py), k-fold cross-validation by tree (src/crossval.py), memory profiling (tools/profile_memory.py) and U-Net FLOPs/parameter count (src/unet.py, tools/unet_cost.py); listed what still needs a dataset run and what is out of scope in the results summary | 24 pytest tests including hand-checked FLOP counts, a known-shift alignment case and end-to-end runs on synthetic data; ran the memory and U-Net cost scripts. Cross-validation, occluded recall and alignment results still need a run on the real dataset |
| 2026-10-25 | Claude | Ran the new evaluation on the real dataset: re-ran `train.py` (occluded recall, automatic GT offset), 5-fold cross-validation with the shape prior added, a new baseline threshold sweep (`tools/baseline_sensitivity.py`) and a manual GT offset check (`tools/gt_offset.py`); wrote the results into the results summary | The re-run reproduced every 2026-10-14 accuracy number exactly; the 22 manual offset readings were compared with the automatic estimate (correlation 0.81); 26 tests pass. The manual readings were made by Claude from the overlay sheets and should be spot-checked |
| 2026-10-25 | Claude | Pass against the brief's "Excellent" column: input checks (missing, duplicate and mismatched depth), remaining docstrings and Google-style pdoc, failure-case analysis (`tools/failure_cases.py`), selected-method comparison in the results summary, README workflow table, "as implemented" inputs/outputs in the problem statement; fixed `--work-size` runs overwriting the main results | New tests for each input case and for the `--work-size` file names (30 pass); per-image scores checked to average exactly to `res/ml_results.csv`; ast script to confirm every function has Args/Returns/Raises; ran `src/main.py` and a 1-epoch `train_unet.py` smoke test (output not kept) |

## Errors and weak suggestions from AI that were found and corrected

A factual record, from the commit history, to draw on in the reflection below.

| Found | Problem | How it was found | Fix |
|---|---|---|---|
| 2026-10-25 | `src/main.py` (AI-written 2026-10-14) looked for `_Depth-` files; TreeID names them `_D-`, so the baseline in `main.py` silently ran without depth on every image | Code review against the file names | Shared `find_pairs`; a test that fails on the old code |
| 2026-10-25 | Module docstrings of `metrics.py` and `train.py` had been replaced by a function's docstring (`train.py --help` showed the wrong text) | Reading the generated HTML docs | Restored and completed |
| 2026-10-25 | The results summary left out the shape-prior row, which scores as well as the best model | Comparing the summary with `res/ml_results.csv` | Row and interpretation added |
| 2026-10-25 | An AI-written automatic GT offset detector (trunk from colour/depth) gave noise on real images | Checking it visually on overlay images | Replaced by FFT cross-correlation, checked against 22 manual readings |
| 2026-10-25 | The AI wrote "77 trees" for the cross-validation pool; each tree has one image, so it is 460 | Counting trees in the split | Corrected before commit |
| 2026-10-25 | An AI-prepared patch series arrived without the commit that adds `crossval.py`/`alignment.py` to the docs build | The rebuilt `doc/api` lacked those pages | Added to `tools/build_docs.sh` |
| 2026-10-25 | Python 3.11-only pins (numpy 2.4) were proposed although the project machine runs Python 3.10 | `pip install` could not run on the Mac | Re-pinned to versions that work on 3.10 to 3.13 |
| 2026-10-25 | `train.py --work-size` runs overwrote the main `res/ml_results.csv` | Writing the README reproduction steps | Separate `_ws<size>` file names |

## Reflection (to be written by the student, in her own words)

The brief (§7) asks the reflection to cover each point below.

1. **Which AI tools were used.**
2. **What they were used for.**
3. **Approximate level of AI contribution**, for example per part: problem framing, code, evaluation design, results interpretation, documentation.
4. **Why AI was used for those tasks.**
5. **How AI-generated material was checked**: tests, re-runs, reading the code, comparing with `res/` files, manual inspection.
6. **Which important decisions remained the student's**: for example the choice of problem, the methods taught in the unit, the tree-wise split, which results to trust, what to report as limitations.
7. **Errors, limitations or unhelpful suggestions from AI**: see the table above.
8. **How AI use affected the student's understanding or workflow.**
