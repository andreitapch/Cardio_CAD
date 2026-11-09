# `realistic_dataset_v1` — simulated echo with exact ground-truth motion

Everything established by opening the files, plus the plan that depends on it.
Where this disagrees with the dataset's own `readme.txt`, the disagreement is
called out — several of its claims do not match the tree.

**Why this dataset matters here.** The in-vivo cohort has no ground-truth
motion: MITT is a tracker with its own error, GE ships strain values and no
contours, and BEAS gives one contour per clip. This dataset stores the position
of every myocardial scattering site **for every frame**, so correspondence
between any two frames is exact and dense, with no tracker in the loop. It is
the only supervision available for fine-tuning a matcher.

Root defaults to `F:\realistic_dataset_v1`, overridable with
`$MOTION_ESTIMATOR_SIM_ROOT`.

## What it is

CAMUS left-ventricular geometries, deformed by CircAdapt strain patterns, imaged
through the COLE convolution ultrasound simulator (Burman et al., KU Leuven,
CC-BY-NC-SA-4.0). Cite as the `readme.txt` specifies.

## Verified facts

- **MATLAB v7.3, so HDF5.** `scipy.io.loadmat` raises `NotImplementedError`;
  `data/simulated.py` uses `h5py`. Cell arrays (`scat_x`, `scat_y`,
  `myocardial_scat_ind`) are object arrays of HDF5 references and need a second
  dereference per frame.
- **1.5 TB, ~2.0–2.4 GB per file, and most of it is `RF_image`** — which motion
  estimation does not use. `read_simulated` never touches it; frames plus
  scatterers are ~5% of a file.
- **HDF5 reverses the axis order.** `cart_image` reads as
  `(n_frames, Nx, Ny)`; MATLAB's own `(Ny, Nx, n_frames)` comes back by
  transposing each frame.
- **Outside the sector is `NaN`, not zero.** The sector mask is therefore free
  (`np.isfinite`), and every normalisation has to be NaN-aware.
- **`myocardial_scat_ind` is `1..79644`** (or up to 126,562) — the myocardial
  scatterers are the *leading block* of the scatter map, the same set in every
  frame, stored 1-based. The other ~1.48 M sites are background clutter whose
  motion is not a meaningful target.
- **Pixel conversion, from the bundled script, and it is not symmetric:**

  ```
  x_px = scat_x / dx + 1
  y_px = scat_y / dy + Ny/2 + 1
  ```

  Verified visually — the converted points sit exactly on the myocardium.
- **The simulated sector points left**, apex at low column; the clinical view has
  the apex at the top. The bundled MATLAB script hides this at plot time with
  `view([90 90])` and `XDir reverse`. `orientation="clinical"` (the default)
  rotates frames and scatterers together so the geometry matches the in-vivo
  pipeline. Verified visually.

## What the tree actually holds

`python -m motion_estimator.scripts.inventory_simulated`

| | |
|---|---|
| recordings | 798 |
| patients | 46 |
| views | 399 `2ch`, 399 `4ch` |

| simulation type | recordings | patients |
|---|---|---|
| `global_simulations` | 594 | 30/46 |
| `regional_simulations` | 204 | 37/46 |

21 patients hold both, 16 only `regional_simulations`, 9 only
`global_simulations`.

From a 40-file probe (`--probe`):

| | |
|---|---|
| frames per clip | 35–45 (median 38) |
| myocardial scatterers | 79,400–126,562 |
| frame size | 1213 × 1000 @ ~0.195 mm isotropic |
| ejection fraction | 38.5–83.4, median 63.6 |
| file size | ~2.0 GB mean |

### Three corrections to the dataset's `readme.txt`

1. **`global_simulations1` and `global_simulations2` do not exist on disk.** The
   readme (and the bundled script) list four simulation types; the tree has two.
2. **46 patient folders, not the 50 or 56 the readme gives.**
3. **The cohort is not reduced-EF.** The readme says "50 patients with reduced
   ejection fraction (< 40%) were selected". Measured EF over the probed files
   is **38.5–83.4, median 63.6** — a normal-to-high range with one clip near the
   stated cutoff. This matters: a training set's strain range bounds what the
   model can be expected to generalise to, and the readme's claim would have
   suggested the opposite bias to the one actually present.

## Code

```
data/simulated.py                     index, read, clinical orientation, GT tracks
scripts/inventory_simulated.py        what is on disk (metadata only, no pixels)
scripts/build_simulated_cache.py      the one-pass trainable cache
```

```python
from motion_estimator.data.simulated import index_simulated, read_simulated

refs = index_simulated()                       # 798 recordings
clip = read_simulated(refs[0].path, max_scatterers=20000)
clip.frames        # (n_frames, rows, cols), NaN outside the sector
clip.scat_x        # (n_frames, n_scatterers), 1-based pixels
clip.displacement(0, 12)   # (n_scatterers, 2) exact, no tracker involved
clip.sector_mask           # free, from the NaNs
refs[0].group_id           # "Pat_108" -- what folds must be grouped on
```

### The cache

```
python -m motion_estimator.scripts.build_simulated_cache
python -m motion_estimator.scripts.build_simulated_cache --pixel-mm 0.26 --max-scatterers 20000
```

One `.npz` per recording, so an interrupted pass resumes and a bad file costs
only itself; `meta.json` fingerprints the settings and a mismatched rebuild is
refused rather than silently mixed — the same contract as
`echoclip/embed_cache.py`. Written through a `.partial` handle and renamed, so
a kill never leaves a half-written `.npz`.

Two settings that are choices rather than conventions, both of which should be
measured rather than assumed:

- **`--pixel-mm`** (default 0.26) resamples to a common pixel size. The
  simulations are ~0.195 mm and the GE in-vivo clips 0.26 × 0.32 mm, so a model
  trained at the simulated scale sees speckle at the wrong size — a domain gap
  that costs nothing to remove.
- **`--max-scatterers`** (default 20,000) subsamples evenly, not randomly, so
  repeated builds agree and the subset stays spread over the wall. 20 k is
  already several times LoFTR's coarse-cell count at these resolutions.

Frames are stored `uint8`, quantised over the **clip's own** intensity range
rather than per frame — per-frame normalisation would rescale the image between
frames, which is a change in appearance a matcher can see and would learn to
chase. Value `0` is reserved for "outside the sector"; data starts at 1.

## Training code

```
training/sim_dataset.py         frame pairs + tracked points from the cache
training/supervision.py         those tracks -> LoFTR's coarse/fine targets
training/losses.py              focal coarse loss, inverse-variance fine loss
scripts/finetune_loftr.py       the training loop
scripts/evaluate_tracking.py    baseline and after, on simulated and in-vivo
```

Three things worth knowing before running any of it:

- **kornia 0.8.3's LoFTR cannot train as shipped.**
  `CoarseMatching.get_coarse_match` unpacks a four-way `zip` as two values and
  raises `ValueError: too many values to unpack (expected 2)` whenever
  `self.training` is True. The padding it does there -- topping predicted
  matches up with ground-truth ones so the fine level always has something to
  refine -- is therefore unavailable. `finetune_loftr.py` runs the two matcher
  submodules in eval mode instead. `conf_matrix` is produced *before* that code
  and is still differentiable, so the coarse loss (the main signal) is
  unaffected; the fine level only ever sees cells the model already matches,
  which is weaker supervision than upstream LoFTR gets.
- **Supervision is myocardium-only.** A coarse cell with no scatterer in it gets
  no target and is dropped from the loss, rather than trained as a negative.
  The blood pool genuinely has no ground truth, and training it as "must not
  match" is not what the data says.
- **Pair spacing is sampled, not fixed at 1.** Training only on consecutive
  frames teaches small-baseline matching and leaves `pairing="reference"`
  failing at end-systole -- the failure already seen in the notebook.

Measured on an RTX 2060: 320-pixel crops, batch 1, backbone frozen, fp16 -->
**0.98 GB**, so there is room for larger crops or a larger batch.

### Run directories

One directory per experiment, under `artifacts/runs/<date>_<time>_<name>/`,
holding `config.json`, `history.json`, `best.pt`, `last.pt` and the
`eval_*.json` results. The first fine-tune wrote straight into
`artifacts/loftr-finetuned/`, so a second run would have silently replaced the
checkpoint you needed to compare against, and its evaluation JSONs landed in the
repo root under hand-chosen names -- which checkpoint produced which number was
recorded nowhere but the filename.

```
python -m motion_estimator.scripts.finetune_loftr --run-name spacing-floor
python -m motion_estimator.scripts.evaluate_tracking --simulated --in-vivo --run-dir latest
```

`--run-dir latest` picks the newest run and defaults `--checkpoint` to its
`best.pt`, so the results land beside the weights that produced them. Omit
`--checkpoint` on a run directory to measure the pretrained baseline into the
same folder (it is tagged `eval_pretrained.json`).

### First fine-tune: what went wrong

Kept as a record, because the failure is instructive and the numbers are the
only baseline-versus-trained comparison on this cohort so far.

2000 steps, 320 crops, frozen backbone. Validation loss fell monotonically,
0.448 -> 0.3866, with no sign of overfitting. The model was **worse than the
pretrained weights on held-out simulated data and in vivo**:

| | pretrained | fine-tuned |
|---|---|---|
| simulated, `consecutive` (Pat_109 sim1 / sim3) | 1.78 / 2.78 px | 3.58 / 7.05 px |
| in-vivo, `consecutive` (5 shared clips, median) | 5.03 px | 7.79 px |
| in-vivo, cycle closure | 6.2-10.7 px | 1.8-5.0 px |
| in-vivo, `reference` points lost | 299/360 | 357/360 |

Cycle closure improving sharply while accuracy degraded is the signature of a
model **under-predicting displacement**, and the cause was in the pair sampler:
spacing was drawn with weight `1/n`, putting 32% of pairs at spacing 1 and
**77% of all pairs below one coarse cell**. Motion runs at ~1 px per frame step
and the coarse grid is 8 px, so for most of the training data the correct coarse
target was "same cell" -- the model was taught to predict no motion. A focal
loss on the assignment matrix is minimised nicely that way, which is why the
validation curve looked healthy throughout.

Two fixes followed, and both matter:

- **`min_spacing` floor** (default 8, ceiling 20), so the coarse level trains on
  motion it can represent. `SimPairDataset._spacing` carries the measurements.
- **Ground-truth padding of the coarse matches** before the fine level, done in
  `finetune_loftr._pad_with_ground_truth` rather than by patching kornia. At
  ~1 px per frame, consecutive-frame motion is *inherently* sub-cell, so the
  fine level does all the work for `consecutive` tracking -- and it was the
  under-supervised half, because kornia's broken training path meant it only
  ever saw cells the model already matched.

The coarse supervision itself was checked and is correct: decoding the target
cell indices back to displacements agrees with the true scatterer motion to
0.3-1.7 px, inside the 4 px half-cell quantisation limit.

### Baseline before any training

Pretrained `outdoor` weights, one held-out simulated clip, 120 tracked points:

| pairing | median endpoint error | final frame | points lost |
|---|---|---|---|
| `consecutive` | 1.91 px | 2.44 px | 0 |
| `reference` | 0.75 px | 0.18 px | 15 |

Re-measure this over the full held-out set before and after a fine-tune; a
single clip is a smoke test, not a baseline.

## The fine-tuning plan

Phase 1 (inventory and cache) is the code above. What follows depends on it.

**Phase 2 — domain matching.** Resample to a common pixel size (done by the
cache) and match log-compression / dynamic range to the GE clips. This decides
whether the transfer works at all; a model trained on differently-compressed
speckle will not carry over.

**Phase 3 — ground truth to LoFTR supervision.**
- Coarse level: for each 1/8-resolution cell, derive its target cell by
  interpolating the scatterer displacements. **Mask cells with no myocardial
  scatterer support** — the ground truth is valid in the myocardium and nowhere
  else.
- Fine level: L2 on the subpixel offsets at scatterer locations.
- Pairs: ~37 consecutive per clip ≈ 29 k across the cohort, plus frame-0→k
  wide-baseline pairs, which is exactly where the `reference` pairing is weakest.

**Phase 4 — train.** Freeze the ResNet backbone, fine-tune the coarse
transformer first. On 6 GB (RTX 2060): 256–320 crops on the LV, batch 1–2, fp16
(**not** bf16 — Turing raises inside cuDNN, see `docs/environment-notes.md`),
gradient checkpointing. Infer at full resolution. Split by patient — one CAMUS
geometry feeds every simulation derived from it, so `SimClipRef.group_id` is the
only safe grouping.

**Phase 5 — evaluate at two levels.** On held-out simulated clips: endpoint
error against exact ground truth — the clean number. On the GE in-vivo cohort:
against MITT, with cycle-closure error reported separately. The second is the
one that matters; sim-to-real transfer is the real risk, not training
convergence.

## Open risks

- **Sim-to-real transfer is unmeasured.** COLE speckle is not GE speckle, and
  nothing here has been tested across that gap yet.
- **Ground truth covers myocardium only.** Nothing supervises the blood pool —
  which is precisely where LoFTR currently produces confident-but-wrong matches.
  Consider masking it at inference as well as in the loss.
- **EF range.** See the correction above: measure the strain range in this
  dataset against the in-vivo cohort before assuming coverage.
- **Frame rate differs.** Simulated `dt` is ~0.017–0.025 s over 35–45 frames per
  cycle; the in-vivo clips run 44–66 frames per cycle. Inter-frame motion
  magnitude therefore differs between the two, which matters for a
  consecutive-pairing tracker.
