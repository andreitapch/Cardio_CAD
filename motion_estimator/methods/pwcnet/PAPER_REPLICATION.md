# PWC-Net paper replication with the new synthetic data

Replicates the training part of Pulido et al., *Impact of training data
composition on deep learning-based cardiac motion estimation* (OJ-UFFC),
Table 1. The ten tests are the same; the old synthetic set (240 sequences) is
replaced by the new simulated dataset `realistic_dataset_v1` (798 clips, 46
patients).

Everything a run produces ends up on the external drive:
**`F:\PHD\Experiments\PWC_Replication_2026\`**. It is written first to the
local disk (`motion_estimator/artifacts/paper-runs/`, same layout) and copied
to F: after every validation (every 1,000 steps) and at the end of each run.
The first trial wrote straight to F: and died at step 500 with
`PermissionError(13)` on a plain log write; a failed copy now just retries and
never stops training.

## What the paper's numbers mean (checked on disk)

- **Table 1 counts consecutive frame pairs, not frames.** The paper's in vivo
  training tree (`F:\PHD\Experiments\Journal_EXP_2022\Datasets\2D_GE_intervendor_in_vivo_pxDisp\inVivo\train`)
  holds 236 recordings and 14,169 frames, and 14,169 - 236 = 13,933 = IV100.
- **The in vivo pool is Retest + Test**: 118 patients, both views, 19-97 pairs
  per recording. The paper text says "116 recordings"; the training tree has 236.
- **Old synthetic pool**: 240 sequences x 59 pairs = 14,160 pairs. MS47 =
  10,620 = 180 sequences exactly. MS100 = 22,405 is larger than that pool and
  was set to match MS47_IV85's total size.
- **The original 6-fold patient lists are lost**, so folds are re-drawn here.

## Decisions for this replication

| | Choice | Why |
|---|---|---|
| In vivo pool | Retest + Test, the paper's own files and MITT dense fields | Reproduces the paper's IV counts exactly. No untouched hold-out remains. |
| Synthetic amount | Same **fraction** of the new pool: `ms / 22,405` | Chosen by Andrea. MS30_IV50 is therefore not balanced, and MS100_IV0 no longer equals MS47_IV85 in size; `ms_pairs=` on a test pins an absolute count. |
| Synthetic sampling | Pairs, from one fixed permutation (nested subsets) | The paper sampled at pair level. |
| In vivo sampling | Whole patients, fixed order | Labels are subject percentages. |
| Folds | 6 by in vivo patient (`<dataset>/<IV##>`), seed 0, shared by every CV test | Paired comparison across tests, as in the paper's t-tests. |
| Validation | 10% of patients, both sides, never trained on | Picks `best.pt`. In vivo train + val per fold is 11.4-11.7k pairs, close to the paper's 11,785. |
| Synthetic validation | 5 fixed patients (2,984 pairs) | So "100%" synthetic = the other 36,740 pairs. |

## The tests

Defined in `experiments.py` (one line each, the paper's numbers as printed).
Exact counts per fold are in `artifacts/splits/paper_replication/summary.csv`.

| Test | Paper MS | New MS pairs | Paper IV | IV train pairs (per fold) | Protocol |
|---|---|---|---|---|---|
| MS0_IV85 | 0 | 0 | 11,785 | 10,279-10,543 | 6-fold CV |
| MS12_IV85 | 2,645 | 4,337 | 11,785 | same | 6-fold CV |
| MS23_IV85 | 5,292 | 8,678 | 11,785 | same | 6-fold CV |
| MS35_IV85 | 7,937 | 13,015 | 11,785 | same | 6-fold CV |
| MS47_IV85 | 10,620 | 17,415 | 11,785 | same | 6-fold CV |
| MS30_IV50 | 6,612 | 10,842 | 6,612 | 6,562-6,652 | 6-fold CV |
| MS47_IV0 | 10,620 | 17,415 | 0 | 0 | one model, tested on all 118 in vivo patients |
| MS100_IV0 | 22,405 | 36,740 | 0 | 0 | one model, tested on all 118 in vivo patients |
| MS0_IV100 | 0 | 0 | 13,933 | 12,509 | one model, tested on SUS |
| MS60_IV100 | 13,225 | 21,687 | 13,933 | 12,509 | one model, tested on SUS |

40 training runs in total (6 CV tests x 6 folds + 4 single runs).

## Training setup

| | |
|---|---|
| Network | PWC-Net **from scratch**, as the paper (`net.py`, the rewrite of the TF1 architecture: 6 levels, search range 4, dense decoder, context network), greyscale input. Chosen by Andrea; see "From scratch vs pretrained" below. |
| Pairs | Consecutive frames only (t -> t+1), synthetic and in vivo mixed in one pool, sampled uniformly |
| Synthetic target | Dense flow from the scatterers: bilinear splat + Gaussian (sigma 1 px), mask = myocardium (splat weight > 0.1). Matches Delaunay-linear interpolation to 0.004 px median. |
| In vivo target | MITT dense motion fields from the paper's tree (`disp/frame<k+1>.flo` for pair k -> k+1, measured: won 60 of 72 pairs on a warp test over 3 clips); mask = where MITT's field is non-zero (~77% of a crop). Crops centred on the 500 x 500 frame, +-48 px jitter. |
| Loss | Multiscale masked end-point error, level weights 6..2 = 0.03, 0.06, 0.12, 0.25, 0.5 (the old echo runs'); coarse cells weighted by myocardium coverage |
| Crops | 384 x 384, centred on the myocardium with +-48 px jitter; random horizontal flip |
| Optimiser | AdamW, lr 1e-4, weight decay 4e-4, gradient clip 5, fp16 autocast. lr halved at steps 100k, 150k, 200k, 250k, 300k -- the old TF1 synthetic schedule. It depends on the step only, so a run can be extended later with a larger `--steps` and continues exactly. |
| Steps | First budget 100,000 at batch 8 (~15 h per run at 0.58 s/step); extended per run if validation is still improving. The TF1 runs went up to 600k with best checkpoints at 50-90k. |
| Selection | `best.pt` = lowest validation EPE (mean over sources), checked every 1,000 steps |

## From scratch vs pretrained (measured 2026-10-22)

| | synthetic val EPE | in vivo val EPE |
|---|---|---|
| predicting zero motion | ~1.0-1.13 px | 1.32 px |
| from scratch, 5,000 steps (the first MS47_IV0 run) | 0.977 px -- flat since step 1,000 | -- |
| from scratch, pairs 1-8 frames apart, 1,500 steps | ratio to zero flow 0.996 | -- |
| FlyingChairs PWC-Net (IRR weights on D:), no echo training | 0.52 px | 0.98 px |
| same, fine-tuned 1,000 steps on MS47_IV0 | 0.26 px | -- |

The data was checked: warping frame t+1 with the target flow matches frame t
better than zero flow in 12/12 pairs, and better than the sign-flipped or
axis-swapped flow. A single batch can be overfitted (1.15 -> 0.12 px). So the
from-scratch network is not broken; it sits on the zero-flow plateau, which
the TF1 runs left only after tens of thousands of steps.

Andrea chose **from scratch** to stay with the paper, and asked for
**pretrained twins** of the synthetic-only tests and of the paper's best test
(MS47_IV85) for comparison. Twins are tagged `pretrained` (their own run folder,
`<test>_pretrained/`), use the same splits, folds, validation patients and loss,
and differ only in the start: FlyingChairs weights, 30k steps, lr 1e-4 halved at
15k, 20k, 25k. The first MS47_IV0 run was stopped at step 5,000 for this
diagnosis and resumed from there (its first 5,000 steps match the new schedule).

## The paper's in vivo metric (Table 2), ported and verified

Table 2's "EPE (mm)" is **not** a flow-field EPE. It is the endocardial
contour (72 points) propagated from frame 1 through the predicted flow and
compared with MITT's tracked points, in mm, averaged over frames 2..n-1 and
then over sequences, +- the SD across sequences
(`cardiacMotionEstimation/matlab`: `main.m` -> `loadPredPoints.m` ->
`calculateEPE.m`). Ported in `paper_metrics.py`; `paper_evaluate.py` runs it
for a trained run; `scripts/verify_paper_epe.py` checks the port against the
paper's own predicted contours on F:.

| Old run (F:\...\Results\InVivo) | Table 2 | Port, paper-exact |
|---|---|---|
| Test_3 = MS12_IV85 | 1.30 +- 0.44 | 1.297 +- 0.465 |
| Test_4 = MS23_IV85 | 1.17 +- 0.40 | 1.174 +- 0.401 |
| Test_5 = MS35_IV85 | 1.19 +- 0.36 | 1.197 +- 0.390 |
| Test_1 / Test_2 / Test_6 | 1.39 / 4.71 / 1.10 | 1.54 / 2.72 / 0.90 -- not the final models |

Fixed (2026-10-27), in the port by default and in `loadPredPoints.m`
(original kept as `loadPredPoints.m.before-fix`):

- **Swapped pixel spacing**: x (columns) was scaled by the row spacing and y by
  the column spacing (GE pixels are e.g. 0.366 x 0.301 mm). `paper_exact=True`
  reproduces the old behaviour for comparison with the paper.
- **Folder mean**: overwritten per patient (it printed the last patient); the
  in vivo loop also scored each patient once per sub-folder. Now: mean +- SD
  across sequences per data folder and overall.
- **Views**: the paper scored 2ch only; this replication scores 2ch and 4ch
  (per view and together).

Training logs this metric on the fold's validation clips every 5,000 steps
(`val_paper_epe_mm` in `history.json`, from MS47_IV85 pretrained fold 0 on);
`best.pt` is still chosen on the dense validation EPE.

## Fair protocol (from 2026-10-27)

Every run, from scratch or pretrained, any test: **150k steps, lr 1e-4 halved
at 100k** (the TF1 schedule), `best.pt` on validation, and snapshots
`step30000.pt`, `step60000.pt`, `step100000.pt`, `step150000.pt` for equal-step
comparisons. Only the starting weights differ.

- The from-scratch runs trained to 100k under the same schedule, so they simply
  extend to 150k (their `step100000.pt` was copied from `last.pt`; no 30k/60k
  snapshots exist for those two).
- Pretrained runs are retrained as `<test>_pretrained150k`. The earlier
  30k-step pretrained runs (`<test>_pretrained`, lr halved at 15/20/25k) are
  kept as a "short fine-tune" reference, not used for the main comparison.
- Why: the first pretrained runs had a decaying lr and the scratch runs did
  not, and MS47_IV85 fold 0's best checkpoint was its last step (still
  improving), so neither budget nor schedule was comparable.

Status any time: `python -m motion_estimator.methods.pwcnet.queue_status`
(columns: best dense val, best paper val on validation patients, paper test on
the held-out test patients).

**Test scoring is automatic** (from 2026-10-28): when a run finishes,
`best.pt` and every snapshot are scored with the paper metric on the run's
test patients (`eval_invivo.json`, `eval_invivo_step<N>.json`). Runs started
before that (MS47_IV0_pretrained150k) are scored by hand with
`python -m motion_estimator.methods.pwcnet.paper_evaluate <run> --checkpoint <file>`.

## In vivo results (paper metric, contour EPE vs MITT, mm, best.pt)

Synthetic-only runs: 230 clips (115 2ch + 115 4ch; 6 skipped, MITT length
differs from the cycle). Cross-validated runs: only that fold's test
patients. Mean +- SD across sequences.

| Run | Clips | All, corrected | All, paper-exact | 2ch paper-exact | Paper Table 2 |
|---|---|---|---|---|---|
| MS47_IV0 scratch, best of 150k (110k) | 230 | 1.93 +- 0.56 | 1.77 +- 0.48 | 1.75 +- 0.48 | 4.71 +- 0.44 |
| MS47_IV0 pretrained, best of 30k (28k) | 230 | **1.66 +- 0.49** | 1.58 +- 0.44 | 1.60 +- 0.48 | -- |
| MS47_IV85 pretrained fold 0, best of 30k (30k) | 40 | 1.14 +- 0.60 | 1.12 +- 0.53 | 1.18 +- 0.72 | 1.10 (all folds) |

- MS47_IV0 scratch was 1.97 +- 0.57 at its 100k best.pt. The extra 50k steps
  gained 0.04 mm.
- Pretrained beat scratch (100k) on 194/230 clips, by 0.31 mm on average.
- Both synthetic-only models do far better in vivo than the paper's
  synthetic-only model (4.71 mm), so the new synthetic data transfers much
  better.
- MS47_IV85 fold 0 is a single fold (20 patients x 2 views). It is not yet
  comparable to the paper's 6-fold mean.

### Equal-step snapshots (overfitting check, 2026-10-28/29)

In vivo test, all clips, corrected spacing, mm (synthetic val px in brackets):

| Run | 30k | 60k | 100k | 150k | best.pt |
|---|---|---|---|---|---|
| MS47_IV0 scratch | -- | -- | 1.92 (0.172) | 1.93 (0.160) | 1.93 (110k, 0.157) |
| MS47_IV0 pretrained150k | 1.73 (0.159) | **1.72** (0.147) | 1.96 (0.145) | 1.84 (0.145) | 1.81 (135k, 0.142) |
| MS47_IV0 pretrained, 30k, decayed lr | **1.66** (28k, 0.150) | | | | |

Paired over the same 230 clips (difference, 95% CI):

- **Scratch, 100k -> 150k:** +0.003 mm (-0.020..+0.027). Flat.
- **Pretrained150k, 30k -> 60k:** -0.009 mm (-0.029..+0.011). Flat.
- **Pretrained150k, 60k -> 100k:** +0.240 mm (+0.199..+0.280), worse on
  188/230 clips.
- **Pretrained150k, 60k -> 150k:** +0.113 mm (+0.089..+0.137), worse on
  173/230 clips.
- **Pretrained150k, 60k -> best.pt (135k):** +0.090 mm (+0.065..+0.115).
- **Pretrained 30k decayed lr -> pretrained150k at 150k:** +0.178 mm
  (+0.146..+0.210).
- **Scratch 150k -> pretrained150k 150k:** -0.091 mm (-0.135..-0.047). At
  equal steps, pretrained is still better than scratch.
- **Same 30k steps, decayed lr vs constant lr:** +0.074 mm (+0.049..+0.099)
  for constant lr.

**Finding.** With synthetic-only training, the pretrained network fits the
synthetic domain better and better after ~60k steps (synthetic val 0.147 ->
0.142 px) while its in vivo error rises (1.72 -> 1.81-1.96 mm). Synthetic
validation is therefore not a reliable proxy for in vivo accuracy beyond that
point, and `best.pt` selected on it is 0.09 mm worse in vivo than the 60k
snapshot. Halving the lr at 100k recovers part of the loss (1.96 -> 1.84 mm)
but not all. The from-scratch network shows no such drift between 100k and
150k (no earlier snapshots were scored). Runs that train with in vivo data
validate on in vivo patients too, so they are not exposed to this.

### MS47_IV85 fold 0: equal-step snapshots (2026-10-30)

In vivo test, fold 0's 40 test clips, corrected spacing, mm:

| Checkpoint | Scratch | Pretrained150k | Pretrained, 30k, lr halved at 15/20/25k |
|---|---|---|---|
| 30k | -- | 1.60 | **1.14** |
| 60k | -- | 1.52 | |
| 100k | 1.46 | 1.32 | |
| 150k | 1.30 | 1.17 | |
| best.pt | 1.18 (127k) | **1.10** (146k) | |

Paired over the same 40 clips (difference, 95% CI):

- **Scratch, 100k -> best.pt:** -0.274 mm (-0.341..-0.207), better on 37/40.
  100k -> 150k: -0.155 mm (-0.229..-0.082).
- **Pretrained150k, 60k -> 100k -> 150k:** -0.204 then -0.144 mm, both
  significant. 30k -> best.pt: -0.500 mm.
- **Pretrained150k best.pt vs the 30k decayed-lr run:** -0.042 mm
  (-0.086..+0.002), not significant; 150k snapshot vs it: +0.029 mm.
- **Pretrained vs scratch, best.pt:** -0.082 mm (-0.133..-0.032).

**Finding.** With in vivo data in training, both starts keep improving up to
150k, unlike the synthetic-only runs. For the pretrained start, the 150k run
only gets back to what the 30k fast-decay run reached in 5 h (23.6 h): the
gain comes from lowering the lr, not from the extra steps. Dense in vivo
validation still falls at 150k but only by ~0.005 px over the last 20k steps.
Checkpoint-to-checkpoint scatter is large: the validation contour error
jumps between 0.95 and 1.35 mm from one 5k checkpoint to the next, and
scratch best.pt (127k) and 150k differ by 0.12 mm, so differences under
~0.1 mm on one fold are not reliable.

## Results so far (synthetic validation EPE, best.pt)

| Run | Start | Steps | Best val EPE | Notes |
|---|---|---|---|---|
| MS47_IV0 | scratch | 150k (12.1 h + 7.2 h) | **0.157 px** at 110k | Zero-flow plateau (~0.977) until ~11k, then 0.735 at 12k, 0.349 at 14k, 0.214 at 30k, 0.164 at 94k. Zero flow scores 1.08 px. |
| MS47_IV0_pretrained | FlyingChairs | 30k | 0.150 px at 28k | short fine-tune reference |
| MS47_IV85_pretrained fold 0 | FlyingChairs | 30k | 0.307 px at 30k (mixed val) | still improving at the end |

The PC slept from 2026-10-23 10:00 to 10-26 ~17:00 with MS47_IV0_pretrained
paused at step 8,400; it resumed by itself. The runner now holds off idle sleep
(see `run_queue.py`).

## Files

Code, in `motion_estimator/methods/pwcnet/`:

| File | What |
|---|---|
| `experiments.py` | the test registry (Table 1) |
| `build_splits.py` | test -> exact pair lists per fold |
| `paper_cache.py` | memory-mappable copies of both pools (`artifacts/paper-cache/ms` 23 GB, `iv` ~18 GB, on D:) |
| `paper_data.py` | pairs + dense targets |
| `train_paper.py` | one run (test, fold); resumable |
| `run_queue.py` | runs the queue one job after another, unattended |
| `start_queue.cmd` | starts `run_queue.py` detached with the `cardioCad` env (see below) |
| `net_pretrained.py` | the FlyingChairs-pretrained alternative (`--net pretrained`), vendored from IRR |
| `paper_progress.py` | progress figure for a run |
| `PAPER_REPLICATION.md` | this file (copied to F: as `README.md`) |

On the external drive, `F:\PHD\Experiments\PWC_Replication_2026\`:

```
queue.json            the jobs, in order (re-read before every job)
queue.log             START / DONE / FAILED / RETRY lines
queue_state.json      attempts per job
splits/               copy of the split files the runs used
<test>[_pretrained]/single/   or .../fold0..fold5/
    config.json       every setting + the split's counts
    train.log         full output
    history.json      validation after every 1,000 steps
    status.json       running / done / failed, step, best
    best.pt           best validation model
    last.pt           model + optimiser state, for resuming
    progress.png      curves + predicted vs true flow
_trial/               the trial run (crashed at step 500 writing to F:, see above)
```

The queue runner is started detached so it keeps running when the terminal or
Claude Code closes; `runner.lock` holds its PID. Since 2026-10-29 the launcher is
`start_queue.cmd` (sets the `cardioCad` env, appends output to `runner.out`),
started hidden: `Start-Process -FilePath motion_estimator\methods\pwcnet\start_queue.cmd -WindowStyle Hidden`.
Earlier launches used WMI `Win32_Process.Create`; two of those died ~7 min after
starting on 2026-10-26 for no logged reason, Andrea's own terminal never did.
To stop a run cleanly: Ctrl+C in the runner's terminal, or kill the `train_paper`
process right after a validation (every 1,000 steps); the next runner start
resumes it from `last.pt`.

## Viewer (part 1: in vivo dataset)

`python -m motion_estimator.viewer.app`, then http://127.0.0.1:8050.

- Pick models A and B (any trained run; CV tests use, per patient, the fold
  that never trained on it), choose a patient from the sortable table (paper
  EPE for A and B), and play the sequence with MITT (white dashed), A (blue)
  and B (orange) contours, optional flow overlay, per-frame EPE and strain
  curves (with GE GLS and the ES frame).
- "Predict selected clip" / "Run A & B on all clips" start the GPU worker
  (`motion_estimator.viewer.worker`) in its own process; results are cached in
  `artifacts/viewer-cache/<model>/` (tracks, EPE, strain; flow only for clips
  predicted one by one).
- Part 2 will open new recordings (data.mat/.srd, DICOM, HDF5 after scan
  conversion) once there is a way to place the starting contour; part 3 feeds
  strain to the remodeling classifiers.

## Commands

From the repo root, in the `cardioCad` env, with `MKL_THREADING_LAYER=TBB`:

```
python -m motion_estimator.methods.pwcnet.build_splits                         # once
python -m motion_estimator.methods.pwcnet.paper_cache --ms --iv                # once
python -m motion_estimator.methods.pwcnet.run_queue --add MS47_IV0 MS100_IV0   # queue jobs
python -m motion_estimator.methods.pwcnet.run_queue                            # run the queue
python -m motion_estimator.methods.pwcnet.paper_progress F:\PHD\Experiments\PWC_Replication_2026\MS47_IV0\single
```

## Status

- [x] Splits for all 10 tests
- [x] Synthetic cache, trainer, queue, progress figure
- [x] Trial run on MS47_IV0: 400 steps, still at the zero-flow level (val EPE 1.13 px vs 1.01 px for predicting no motion), as expected that early
- [ ] Queue (18 runs; 4 done on 2026-10-29, ends ~2026-11-11): MS47_IV0 scratch/pretrained150k (done) -> MS47_IV85 scratch/pretrained150k alternating fold 0..5 -> MS100_IV0 scratch/pretrained150k
- [x] Which `disp/frame<k>.flo` belongs to pair k: `frame<k+1>.flo` (warp test); in vivo loading checked against the raw `.flo`
- [x] In vivo cache (236 clips, 17 GB)
- [x] In vivo EPE with the paper metric (Table 2 EPE), automatic for every finished run and snapshot
- [ ] Not queued yet: MS0_IV85, MS12_IV85, MS23_IV85, MS35_IV85, MS30_IV50 (6 folds each, 30 runs, ~25 h each) and MS0_IV100, MS60_IV100 (SUS, 2 runs)
- [ ] End-systolic peak strain on in vivo (Table 2 strain columns): the viewer computes it per clip, but it is not yet scored per run against GE
- [ ] SUS evaluation (Tables 3-4): no pipeline yet
- [ ] Code is not committed (`motion_estimator/` is untracked); the MATLAB fix in `loadPredPoints.m` is not committed either

Never use scipy in a process that has imported torch: their OpenMP runtimes
abort the process (`OMP: Error #15`).

Editing `paper_data.py` while a run trains is live: Windows starts DataLoader
workers by re-importing the module, and the validation workers restart at
every validation. Keep edits backward compatible, or wait for the run to end.
