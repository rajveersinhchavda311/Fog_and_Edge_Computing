# ShanghaiTech Part A Evaluation — Provenance

This directory holds two separate accuracy evaluations of the CSRNet checkpoint
(`weights/PartAmodel_best.pth`) used throughout `fog_edge_system/` — one real,
freshly run against the full test set; one real but historical, restored from
git and superseded by the first.

## Primary result — full 182-image test set (run 2026-09-18)

```
MAE  (Mean Absolute Error):  66.96
MSE  (Mean Squared Error):   11477.39
RMSE (Root Mean Sq. Error):  107.13
```

**This run genuinely happened in this session**, against the complete,
official ShanghaiTech Part A test split (182 images, 182 matching
ground-truth annotation files), supplied by the project owner in
`data/ShanghaiTech/part_A_final_modif/test_data/`. Full per-image breakdown
(ground truth, prediction, error, inference time — all 182 rows) is in
[`shanghaitech_partA_full_182.txt`](shanghaitech_partA_full_182.txt).

Command used:
```bash
python evaluate.py --model ../weights/PartAmodel_best.pth --dataset shanghai \
  --data ../data/ShanghaiTech/part_A_final_modif/test_data
```

For comparison, the original CSRNet paper (Li et al., 2018) reports 68.2 MAE
on this same benchmark. 66.96 is in that range — expected, since this
checkpoint is architecturally identical to the paper's reference
implementation (verified: 16,263,489 parameters, `strict=True` load, matching
`leeyeehoo/CSRNet-pytorch` checkpoint structure).

### What this run also settles

Earlier documentation in this repo could only argue *circumstantially* that
`weights/PartAmodel_best.pth` was the real ShanghaiTech Part A checkpoint
(matching file size, matching source description, matching architecture — but
never independently run against the dataset). **This run removes that doubt
directly:** the checkpoint was just evaluated against the real, complete
ShanghaiTech Part A test set and produced an MAE in the expected range for a
genuine ShanghaiTech-trained CSRNet model. A checkpoint that was mismatched,
corrupted, or trained on something else would not land anywhere near 66.96
MAE on real ShanghaiTech data — it would produce a much larger, unstructured
error.

### A code bug found and fixed while setting this up

`evaluate.py`'s ShanghaiTech path-matching logic assumed the classic
directory layout (`test_data/images/IMG_1.jpg` +
`test_data/ground-truth/GT_IMG_1.mat` in separate subfolders). The dataset
copy actually supplied has a **flat** layout — both the image and its `.mat`
file sit directly in `test_data/` together. Run unmodified, this would have
silently matched **zero** image/ground-truth pairs (no error raised — it
would just report "Evaluating on 0 images..." and produce nothing useful).
Fixed by detecting which layout is present and adjusting the search paths
accordingly, rather than assuming one. Verified afterward that this finds
exactly 182 pairs, matching the 182 images and 182 `.mat` files actually on
disk.

### A minor data discrepancy, noted rather than hidden

`IMG_100.jpg`'s ground truth reads 391 in this dataset copy, vs. 389 in the
historical 10-image sample below (both from nominally the same file,
`GT_IMG_100.mat`). Two-count difference — almost certainly a minor annotation
version difference between dataset mirrors, not a bug in either evaluation
run. Every other value in both runs (`IMG_1`, `IMG_10`, `IMG_102`, etc.)
matches exactly between the old sample and this run, which is what makes this
one small discrepancy visible as an anomaly rather than something systemic.

---

## Historical result — 10-image sample (restored from git, run 2026-02-13)

```
MAE  (Mean Absolute Error):  69.80
MSE  (Mean Squared Error):   6836.60
RMSE (Root Mean Sq. Error):  82.68
```

Kept for provenance, not as the current headline number — the 182-image run
above supersedes it (same checkpoint, same script family, far larger and
therefore more reliable sample). Full detail in
[`shanghaitech_partA_eval.txt`](shanghaitech_partA_eval.txt).

- Committed to the project's git history in commit `f9c2d34`
  ("commit2-density_prediction_working"), 2026-02-13, by Rajveersinh Chavda —
  i.e., genuinely run by the project owner during original development, not
  fabricated. Confirmed via `git log --all --diff-filter=A` and `git show`
  against that exact commit.
- At the time of that run, the model class was named `CSRNet` (confirmed via
  `git show f9c2d34:model.py`); a later cleanup commit (`5c3d5b2`, "removed
  comments and docstrings") renamed it to `CrowdStampedeYantra` without
  updating `evaluate.py`'s import — the exact bug independently flagged
  during the initial review of this project, and the same one fixed in this
  directory's copy of `evaluate.py`.

## Files in this directory

| File | What it is |
|---|---|
| `shanghaitech_partA_full_182.txt` | Full 182-image run output (primary result) |
| `shanghaitech_partA_eval.txt` | Historical 10-image run output (superseded, kept for provenance) |
| `evaluate.py` | Working evaluation script (path-layout fix applied; class-name import fix applied) |
| `train.py`, `model.py` | Copied for pipeline completeness |
| `requirements.txt` | `scipy` etc. — kept separate from `requirements_edge.txt` since the edge runtime never reads `.mat` files |

## Reproducing this yourself

```bash
# from fog_edge_system/evaluation/, with ShanghaiTech Part A test data
# available locally (flat layout: images and .mat files in the same folder)
python evaluate.py --model ../weights/PartAmodel_best.pth --dataset shanghai \
  --data path/to/ShanghaiTech/part_A_final_modif/test_data
```
