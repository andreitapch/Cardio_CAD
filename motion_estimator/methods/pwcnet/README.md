# PWC-Net: step by step

A PyTorch rewrite of the old TF1 project (`PWC/pwcNet_t1/pwcNet`). The model is pretrained on synthetic data (5 folds by patient). Each in vivo fold then tests three versions against MITT: synthetic-only, fine-tuned unsupervised, and fine-tuned on MITT.

Run every command from the **repo root** (`D:\PHD\Development\Python\Cardio_CAD`) in PowerShell.

---

## Step 0: open the environment (every new terminal)

```powershell
cd D:\PHD\Development\Python\Cardio_CAD
conda activate cardioCad
$env:MKL_THREADING_LAYER = "TBB"   # without this torch fails to load (WinError 127)
```

The F: drive must be mounted, since the in vivo data and MITT files are read from there.

## Step 1: remove the test run

A quick test run is left over from setup. Delete it so the notebook doesn't pick it up:

```powershell
Remove-Item -Recurse -Force artifacts\runs\2026-09-25_1344_pwcnet-cv-quick
```

## Step 2: check the caches (usually nothing to do)

- The synthetic cache is `motion_estimator/artifacts/simulated-cache/` and already holds all 798 clips.
- The in vivo cache is `motion_estimator/artifacts/invivo-cache/Retest_dataset/`. It already holds 116 clips.

Only if the in vivo folder is empty or missing, run:
```powershell
python -m motion_estimator.methods.pwcnet.data --build-invivo-cache
```

## Step 3: optional smoke test (about 10 min)

This checks that everything runs with tiny step counts. The results are meaningless.
```powershell
python -m motion_estimator.methods.pwcnet.run_cv --folds 0 --quick
```
Delete that `*_pwcnet-cv-quick` folder afterwards, as in Step 1.

## Step 4: train and evaluate fold 0 first (about 5 h)

```powershell
python -m motion_estimator.methods.pwcnet.run_cv --folds 0
```
- This creates `artifacts/runs/<date>_<time>_pwcnet-cv/`. **Copy that path**, because every later step needs it (`<cv>` below).
- It runs, in order:
  1. synthetic pretraining (30k steps, about 3.3 h);
  2. unsupervised in vivo fine-tuning (3k steps, about 25 min);
  3. MITT-supervised in vivo fine-tuning (about 25 min);
  4. evaluation and plots.
- For a faster first pass, add `--sim-steps 15000`.
- If it stops, rerun the same command with `--run-dir <cv>`. Stages that already finished are skipped.

## Step 5: look at fold 0 before running the rest

Open `motion_estimator/notebooks/pwcnet_results.ipynb`, leave `RUN_DIR = None` (it picks the newest run), and run all cells. Check:

- **Section 2, training curves:** the synthetic validation EPE should fall and flatten.
- **Section 5, contours over the images:** the colored contours should follow the white dashed MITT contour, and the strain curves should look cardiac (negative in systole).
- **Section 6, flow fields:** the unsupervised flow should be smooth but not flat. If it is nearly zero everywhere, the smoothness weight is too high. If it is noisy, it is too low. Retrain only that stage:
  ```powershell
  Remove-Item -Recurse -Force <cv>\invivo_fold0_unsup
  python -m motion_estimator.methods.pwcnet.train --run-dir <cv> --fold 0 --cohort in_vivo --mode unsup --smooth-weight 2
  python -m motion_estimator.methods.pwcnet.evaluate --run-dir <cv> --folds 0
  ```
  Use the same `--smooth-weight` for all folds afterwards.

## Step 6: run the remaining folds (about 17 h, fine overnight)

```powershell
python -m motion_estimator.methods.pwcnet.run_cv --run-dir <cv> --folds 1 2 3 4
```
If you changed the smoothness weight in Step 5, run the stages by hand instead (see "Commands one stage at a time" below).

## Step 7: final results

```powershell
python -m motion_estimator.methods.pwcnet.evaluate --run-dir <cv> --summarise-only
```
Then rerun the notebook. The main outputs:

| where | what |
|---|---|
| `<cv>/eval/summary_in_vivo.csv` | per variant: mean (std) across folds of distance to MITT (at ES, median, p90) and GLS difference against MITT and GE |
| `<cv>/eval/summary_simulated.csv` | true errors and gain on held-out synthetic patients |
| `<cv>/eval/fold<k>/plots/*.png` | per clip: contours vs MITT at start and ES, distance per frame, strain curves |
| `<cv>/eval/*.png` | box plots per fold, GLS agreement plots |

---

## Commands one stage at a time

These are equivalent to `run_cv`, for fold `k`:
```powershell
python -m motion_estimator.methods.pwcnet.data --make-folds --run-dir <cv>          # once; writes folds.json
python -m motion_estimator.methods.pwcnet.train --run-dir <cv> --fold k --cohort simulated
python -m motion_estimator.methods.pwcnet.train --run-dir <cv> --fold k --cohort in_vivo --mode unsup
python -m motion_estimator.methods.pwcnet.train --run-dir <cv> --fold k --cohort in_vivo --mode mitt
python -m motion_estimator.methods.pwcnet.evaluate --run-dir <cv> --folds k
```
Useful options:
- `--steps`, `--lr`, `--batch-size`, `--smooth-weight` (train).
- `--pairings consecutive strided` (evaluate; the default is `consecutive`).
- `--no-plots` (evaluate).

Tracking with the standard evaluator also works:
`python -m motion_estimator.scripts.evaluate_tracking --in-vivo --methods pwcnet --checkpoint-method pwcnet --checkpoint <cv>\sim_fold0\best.pt`

## Folder layout of a run

```
<cv>/folds.json                  patients in train / val / test for each fold, both cohorts
<cv>/sim_fold<k>/                best.pt, last.pt, history.json, config.json
<cv>/invivo_fold<k>_unsup/       fine-tuned from sim_fold<k>/best.pt
<cv>/invivo_fold<k>_mitt/
<cv>/eval/fold<k>/               in_vivo_rows.json, simulated_rows.json, tracks/, plots/
```

## Files in this folder

| file | what |
|---|---|
| `net.py` | the whole model (6-level pyramid, warp, cost volume, dense decoder, context net) |
| `losses.py` | sparse point loss; photometric + smoothness loss |
| `data.py` | folds by patient, in vivo cache, in vivo pair dataset |
| `train.py` | one training (synthetic or in vivo) for one fold |
| `estimator.py` | wrapper for tracking (registry name `pwcnet`) |
| `evaluate.py` | tracking of held-out patients, metrics, strain, plots |
| `plots.py` | figures used by `evaluate.py` and the notebook |
| `run_cv.py` | all of the above, resumable |

## Good to know

- **Folds are by patient**, 5 on each cohort. Synthetic fold *k* is the starting point for in vivo fold *k*.
- **Only Retest is used.** `Test_dataset` is never touched, so it stays available for a final held-out result.
- **`best.pt` is chosen on validation patients** inside each fold's training part, never on its test patients. The unsupervised model is selected on photometric loss, so MITT never influences it.
- **MITT is not ground truth.** "Distance to MITT" is a disagreement. Only the synthetic results are true errors.
- **Out of GPU memory?** Use `--batch-size 4`.
- **Too slow?** Use `--sim-steps 15000`.
