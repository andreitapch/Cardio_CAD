# motion_estimator

Strain estimation from 2D echocardiography using EchoCLIP embeddings, evaluated
against the GE vendor ground truth in the 2D GE intervendor in-vivo dataset.

Milestone 1 (the data readers) is in place. The model side — `echoclip/` — is
next and not written yet.

## Verified facts about the data

Worth not re-deriving:

- **`.srd` is a MATLAB MAT-file with a different extension.** No bespoke reader
  is needed. `Acquisition/Processing/pack_SRD_HDPULSE.m` in the MATLAB tree ends
  with `save(write_file_name, 'USData', 'USInfo', '-mat')`, switching to
  `-v7.3` only above 1 GB. Confirmed against
  `speqle3d/Speqle3D_source/DemoFiles/Clinical2Dt_2ch.srd`, whose first bytes
  read `MATLAB 5.0 MAT-file`.
- **Two payload layouts.** `USData.tissue3` is scan-converted Cartesian data
  with geometry in `USInfo.gridsize` / `USInfo.voxelsize` — this is what the
  intervendor dataset uses. `USData.tissue` is the HDPulse acquisition layout,
  still polar (`samples`/`beams`/`r0`/`dr`); scan conversion for it is not
  implemented, and `prepare_clip` refuses it rather than distorting the image.
- **`tissue3` may be 4-D.** Speqle stores volumetric data in the same field, so
  a clip can come back `(x, y, z, t)`; two of the bundled Speqle demos do. For
  3-D data axis 2 is taken to be the frame axis, which is right for 2D+t clips
  and for everything `loadIntervendorData.m` handled. Nothing in `USInfo`
  distinguishes a 2D+t clip from a single static volume.
- **`USInfo.frames` is unreliable** — it reads `1` on a 267-frame demo clip.
  Frame counts come from the array.
- **The three exports in an `IV##` directory are one acquisition at three
  different geometries.** Frame counts agree; nothing else does.

  | export | shape | geometry |
  |---|---|---|
  | `<view>/data.mat` | `(500, 500, n_frames)` | scan-converted, fixed 500×500 grid, ~0.26–0.37 mm |
  | `IV##_<view>.srd` | `(~214–256, 200, n_frames)` | scan-converted, coarser 0.75 mm; often absent |
  | `IV##_<view>.h5` | `Tissue (n_frames, n_beams, n_samples)` | polar, un-scan-converted |

  So `data.mat` is the clip source, and the pipeline does **not** fall back to
  the `.srd`: substituting a 0.75 mm export for a 0.3 mm one would hand the
  model two different resolutions depending on which files a patient happened
  to ship. `index_dataset` reports a missing `data.mat` as a missing clip;
  `clip_source="any"` opts into the fallback if you ever want it. This costs
  nothing here — all 236 2ch/4ch clips have a `data.mat`, whereas Retest ships
  no 4ch `.srd` at all (0/58). `data.mat` is also what
  `loadIntervendorData.m:15` reads. `read_clip` refuses
  `.h5` because it is polar data with the frame axis first — no transpose maps
  it onto the MAT files, and turning it into an image needs scan conversion.
  `read_h5_tissue` returns the polar array plus the scanner metadata the MAT
  exports drop (`PatientID`, `ManufacturerModelName` = Vivid E9, `ProbeName`,
  `RecordingTime`), alongside `Ecg`/`Resp`/`Aux2` traces. `ViewName` is present
  but blank throughout the cohort, so the view has to come from the filename.
- **The GE ground truth is strain values, not tracks.**
  `GT_STRAIN/GE_mat/GE_strain<ID>.mat` holds one `strain` struct with, per view,
  three 6-element segmental vectors (peak / end / post-systolic) plus
  `GLS_<view>_ES`. The peak and end-systolic vectors are nearly identical —
  IV02's 2ch pair differs in segment 4 alone — because
  `valPeakS = min(strainCurve(1:ED))` and `valES = strainCurve(ED)` coincide
  wherever a segment has not begun recovering. Easy to confuse; the reader keeps
  all three so the choice stays explicit.
- **There is no end-systolic frame index for the in-vivo data.**
  `scar_struct.mat` gives `cycle_frame_limits` only, and `gls/ESframe.m`
  hardcodes ES per *vendor* for the simulated Martinos set. This is why the
  model aggregates over the cycle instead of sampling a known ES frame.
  `Data/IV##/<view>/srd/analysis.mat` (~75 kB) has not been inspected and may
  carry the timing.
- **`scar_struct` contours are loaded but their coordinate convention is
  unverified.** The MATLAB pipeline never consumed them (it used MITT-tracked
  contours), so nothing pins down whether they index the raw `tissue3` frame or
  the clinical view, nor whether they are 1-based. Check visually before using
  them quantitatively. The strain targets do not depend on them.
- **`Retest_dataset` and `Test_dataset` are independent cohorts that reuse the
  same `IV##` numbering.** `Retest_dataset/IV02` and `Test_dataset/IV02` are
  different people, and 58 ids appear in both folders. Grouping
  cross-validation folds on the bare patient id would fuse them and leak across
  the train/test boundary — use `ClipRef.group_id`, which is
  `<dataset>/<patient>`.
- **Only 2ch and 4ch are usable.** `Test_dataset` also ships 3ch clips, but
  `scar_struct.mat` never defines an `info_3ch`, so those clips have no cardiac
  cycle window — and the GE ground truth is 2ch/4ch only regardless. Hence
  `DEFAULT_VIEWS = ("2ch", "4ch")`; pass `views=None` to see everything on
  disk, and `verify_views=True` to have a view that `scar_struct` lacks flagged
  at index time instead of crashing at load time.

## The per-clip `Tracking/` and `srd/analysis.mat` artifacts

Opened for the first time; `data/tracking_artifacts.py` reads them. Facts worth
not re-deriving, and two corrections to what the tree looks like from outside:

- **There are no GE contours anywhere.** `GT_STRAIN/GE_mat/` is strain values
  only, confirmed. Any per-point or per-segment comparison with GE is therefore
  impossible -- its segment 3 is not our segment 3 -- and GE GLS can only be a
  scalar sanity check.
- **`scar_struct`'s contour convention is now verified**, closing an open
  question above. `endo_sample` is `(2, n)` with x in row 0, indexing the
  **clinical** view directly, 0-based: plotted over `frames_first(...)[0]` it
  lands on the endocardium. It agrees with the MITT contour.
- **MITT's stored axes are transposed relative to the clinical image.**
  `points_y` indexes clinical columns and `points_x` clinical rows, and
  `pixel_spacing_mm` is ordered `(row, col)` to match, so the conversion is
  `x = points_y / spacing[1] + 1`, `y = points_x / spacing[0] + 1`. Getting
  this backwards puts the contour 90 degrees off the myocardium, which is
  exactly what it looks like.
- **`Tracking/data<k>.raw` headers lie.** `DimSize = 500 500` at `MET_DOUBLE`
  is 2 MB; the files are 4 MB, holding the frame written twice. Each half
  reproduces `tissue3` frame-for-frame (correlation 1.0000), so they are in the
  **raw** orientation and contain no pixel data `data.mat` does not.
- **`Tracking/Param*/` is a dense-registration tracking baseline already
  computed** -- one ITK cubic B-spline per consecutive frame pair, 14x14
  control grid at spacing 5 in the `.mhd` physical frame (0.1 per pixel).
  116/120 clips have the full set. `read_itk_bspline` parses them and
  `ItkBSpline.transform_points` evaluates one.

  **Not yet pinned:** direction and axis assignment. Chaining all 43 transforms
  moves IV02's 4ch contour 5.2 px on average, against MITT's 4.75 px of net
  cycle motion -- the right order -- but ends 8.6 px from MITT's final
  positions, which is too far to call it validated. elastix conventionally
  stores the *inverse* map (fixed to moving), so the chain may be running
  backwards. Settle that before using these as a baseline.
- **`Tracking/Results1e2/`** holds that pipeline's own output: `strain_mat.mat`
  (`strain_curves` is `(6, n_frames)`, plus `ps_strain`, `pss_strain`,
  `ES_strains`, `PSI`), `GL_average_segs.mat`, and three `exclude*.mat`
  per-segment flag arrays. **The flag polarity is unknown** -- IV02 4ch reads
  `[1 1 1 1 1 1]`, and "every segment excluded" is not a plausible reading of a
  file present for every clip, so 1 most likely means keep. `TrackingResults`
  hands back the raw arrays rather than a helper that picks a side.
- **`srd/analysis.mat` gives one contour, not one per frame.** `BEASData` is
  shaped `(1, n_frames)`, which invites the opposite conclusion, but only the
  **first** entry carries a segmentation -- 1/1 on all 40 clips sampled. So it
  is a single 48-point reference contour plus a 61-point `MITTx`/`MITTy` and
  the 6 manual `userPoints` landmarks, not a tracked sequence.

  **Its coordinate frame is unresolved.** BEAS ran on `srd/Beas_iso.srd`, an
  isotropic 0.26052 mm resampling -- hence `xSize = 609 = 500 * 0.3172 /
  0.26052`. Resampling alone does not place it: every combination of axis swap,
  flip and rescale leaves a mean nearest-point distance of ~34 px from the
  verified `scar_struct` contour, and the `Heart2World` rotation, `cx`/`cy`
  centre and `PadData = 50` do not close it either. `read_beas_contours`
  returns the stored values with `frame_resolved = False` and refuses to guess.
  Settling it means tracing the 2D surface construction in
  `speqle3d/.../BEAS_toolbox`; the 3D `BEAS_UpdateSurface.m` formula does not
  reproduce these arrays.

## The cohort

As reported by `report_cohort` over the mounted drive:

| | patients | clips | views |
|---|---|---|---|
| Retest_dataset | 58 | 116 | 2ch, 4ch |
| Test_dataset | 60 | 120 | 2ch, 4ch |

118 patients, 236 clips, all with `scar_struct` and GE ground truth present,
and all 236 load — 500×500 frames, 44–66 frames per cardiac cycle.

Retest is the development set and Test is held out; see the split protocol
before touching Test.

## Simulated data for fine-tuning

`realistic_dataset_v1` (Burman et al., KU Leuven) is 798 simulated recordings
whose myocardial scatterer positions are stored **for every frame** -- exact,
dense correspondence with no tracker in the loop. It is the only supervision
available for fine-tuning a matcher, since the in-vivo cohort has none.

See **[SIMULATED_DATASET.md](SIMULATED_DATASET.md)** for the verified file
facts, the inventory, three corrections to the dataset's own readme, and the
fine-tuning plan. Code is `data/simulated.py`,
`scripts/inventory_simulated.py` and `scripts/build_simulated_cache.py`.

## Layout

```
data/
  _matio.py         uniform access to v5 and v7.3 MAT files
  srd_reader.py     .srd / .mat clip loading -> UltrasoundClip
  image_prep.py     the clinical-view transform (loadIntervendorData.m:32-40)
  scar_struct.py    cycle limits, contours, the 6-segment split
  gt_strain.py      GT_STRAIN -> segmental strain + GLS, and the target vector
  dataset_index.py  cohort discovery and the (dataset, patient, view) key
  mitt.py           MITT tracking results: points, esFrame, segments, curves
tracking/
  matcher.py        LoFTR on a frame pair -> correspondences + confidences
  point_tracking.py that field interpolated at given points, and chained
notebooks/
  loftr_point_tracking.ipynb  the LoFTR test bench
strain/
  arclength.py      computeArclength.m / arclength.m
  curve.py          computeStrain.m, driftCompensation.m, mainComputeStrain.m
  propagate.py      interpPoints from loadPredictedfloInvivo.m
echoclip/
  config.py         model id + the settings the cache is fingerprinted on
  encoder.py        frame selection, preprocessing, image/text encoding
  embed_cache.py    build/load the on-disk embedding cache
scripts/
  report_cohort.py    what the tree actually contains
  build_embeddings.py one GPU pass over the cohort
  validate_strain.py  check the strain port against every MITT result
tests/
```

## Strain reconstruction

`strain/` turns tracked contour points into a strain curve. It is a
transcription of `cardiacMotionEstimation/matlab/gls`, written in torch so the
chain stays differentiable, and it is **validated rather than assumed**: MITT
stores its tracked points and the curves it derived from them side by side, so
a correct port must turn one into the other.

```
python -m motion_estimator.scripts.validate_strain
  reproduced exactly   236/236
  worst max|diff|      1.386e-13
```

Two conventions the validation pinned down empirically, both of which change
the answer by percentage points if guessed wrong:

- **`drift_comp=False`** reproduces the stored curves, matching the `drift_comp`
  flag recorded in each MITT file. The in-vivo pipeline passes `1` when it
  recomputes, so the two are not interchangeable.
- **Open contour.** The LV longitudinal contour runs base-to-apex-to-base and is
  not closed; closing it adds a spurious base-to-base chord and shifts GLS by
  several percent.

Also established: MITT's stored `timings{1}.peakval` is the **end-systolic**
value, not the systolic minimum — they differ whenever a segment keeps
shortening after end-systole. `compute_strain` returns both, separately.

Coordinates follow MATLAB throughout: **1-based**, `x` indexing columns and `y`
rows, with points that leave the image returning NaN rather than a clamped edge
value.

`image_prep` is a transcription of MATLAB, not a reimplementation:

```matlab
ImageData = USData.tissue3(:,:,cycle_frame_limits(1):cycle_frame_limits(2));
ImageData = permute(ImageData(end:-1:1,:,:),[2 1 3]);
pix_spacing = USInfo.voxelsize([2 1 3]) .* 1000;
```

If a number here disagrees with MATLAB, this package is wrong.

## EchoCLIP

The trunk is **frozen** — with 118 patients there is no honest way to train
150M parameters, so EchoCLIP only runs under `torch.no_grad` and the learning
happens in a small head on its embeddings. Everything verified by loading the
checkpoint rather than read off the paper:

- `hf-hub:mkaichristensen/echo-clip`, MIT, **not gated** — no `HF_TOKEN`. (The
  upstream `zero_shot_example.py` comment saying to `huggingface-cli login`
  does not apply. Its "unauthenticated requests" warning is harmless.)
- embedding dim **512**, text context 77, CLIP BPE via
  `open_clip.get_tokenizer(...)` — not the module-level `open_clip.tokenize`
  the upstream examples use, which ignores the checkpoint's tokenizer config.
- `preprocess_val` = `Resize(224) → CenterCrop(224) → RGB → ToTensor →
  Normalize`. Our frames are square, so the centre crop discards nothing.
- **`precision="fp16"`**. On Turing bf16 does not merely run slowly, it raises
  `RuntimeError: GET was unable to find an engine to execute this
  computation` — and `torch.cuda.is_bf16_supported()` returns `True` anyway
  because it counts emulation, so `EncoderConfig` checks the compute
  capability directly. See `docs/environment-notes.md`.

### Frame sampling

Default is `uniform`: `n_frames` spread evenly across the cycle. Our clips are
cropped to exactly one cardiac cycle, so the cycle is the unit of meaning and
all of it should reach the model. The `upstream` strategy reproduces
echo_CLIP's `frames[0:min(40, len):2]` verbatim — written for arbitrary-length
clinical loops, it truncates late diastole on our 44–66 frame cycles (a
66-frame cycle stops at frame 38). Both are available so the choice can be
measured rather than assumed.

### The cache

Embeddings are a deterministic function of (clip, `EncoderConfig`), so one GPU
pass writes them and every training run afterwards reads ~10 MB instead of
3.5 GB of `data.mat`. One `.npy` per clip, so an interrupted pass resumes; a
`meta.json` fingerprint means a cache built with different settings is refused
rather than silently mixed. Targets are **not** cached — they are cheap and
depend on the strain-phase choice, which should stay changeable without
discarding a GPU pass.

```
python -m motion_estimator.scripts.build_embeddings
python -m motion_estimator.scripts.build_embeddings --sampling upstream \
    --cache-dir motion_estimator/artifacts/embeddings-upstream
```

```python
from motion_estimator.data.dataset_index import index_dataset
from motion_estimator.echoclip.embed_cache import load_embeddings

data = load_embeddings(index_dataset())
data.embeddings   # (n_clips, n_frames, 512)
data.targets      # (n_clips, 7)  6 segmental end-systolic values, then GLS
data.groups       # "<dataset>/<patient>" -- what folds must be grouped on
```

**Open question, not yet settled:** the 500×500 frames have black corners
outside the ultrasound sector, whereas EchoCLIP was trained on cropped echo
videos. `<view>/track/IV##_<view>_mask.raw` looks like a usable sector mask.
This should be measured both ways on the development set, not guessed.

## Tracking with LoFTR

`tracking/` tracks a **given** set of contour points with LoFTR. The thing to
know first: LoFTR is *detector-free*, so it has no input for "track these
points" -- it picks its own correspondence locations on a coarse 1/8-resolution
grid and refines them to subpixel. Two steps bridge that:

```
tracking/
  matcher.py         LoFTR on a frame pair -> (xy0, xy1, confidence)
  point_tracking.py  interpolate that field at your points, and chain it
notebooks/
  loftr_point_tracking.ipynb   the test bench
```

```python
from motion_estimator.tracking import LoFTRMatcher, track_points

matcher = LoFTRMatcher(scale=1.0, mask=sector)          # sector mask optional
result = track_points(frames, start_x, start_y, matcher, pairing="reference")
# pixel_spacing_mm is (row, col); to_mm takes (x, y) -- hence the reversal
strain = compute_strain(*result.to_mm(prepared.pixel_spacing_mm[1::-1]), es_frame)
```

The interpolant is a **confidence-weighted local affine fit** with outlier
rejection, not a nearest match and not inverse-distance weighting. A strain
measurement is a displacement gradient, and averaging displacements over a
neighbourhood cannot represent one; nearest-match quantises to LoFTR's 8 px
grid and discards the subpixel refinement that moves the strain number.
`tests/test_point_tracking.py` pins this against motions with closed-form
answers -- a uniform 5% expansion must read as exactly +5% strain.

Two chaining modes, and the choice matters more than the interpolant:
`pairing="consecutive"` matches frame *k* to *k+1* and accumulates (what the
MATLAB in-vivo pipeline does, and what `propagate_points` assumes);
`pairing="reference"` matches frame 0 to frame *k* and displaces the original
points, so there is no drift to compensate but the baseline at end-systole is
wide. On the notebook's synthetic clip -- a speckle frame warped by a known 6%
contraction, so the true GLS is -5.99% -- `reference` reads -5.78% and
`consecutive` -4.82%, the accumulation loss being visible and in the expected
direction. That is a controlled deformation, not echo, and says nothing yet
about the cohort.

### Arc length amplifies one bad point

Established the hard way, and the reason the defaults look conservative. Arc
length is a sum of local distances over a contour sampled ~1.9 px apart, so a
single point with a wrong displacement does not perturb the strain curve, it
destroys that frame: one contour point with three supporting matches was handed
a 40 px displacement on a clip whose true motion was 4 px, and that frame read
**+21% strain** instead of -6%. Smoothing the tracks along the contour does not
help -- the error is not jitter, it is one confidently wrong point.

Three guards in `displacement_at`, each for a distinct failure:

| guard | default | failure it catches |
|---|---|---|
| `min_neighbours` | 6 | too little support to fit *or* to check a fit against -> NaN |
| conditioning | `MAX_CONDITION` | collinear support cannot fix a gradient -> weighted mean |
| `deviation_tolerance` | 3.0 | the fit extrapolating past its support -> weighted mean |

So NaN in a track is the tracker reporting failure, and a gap in a strain curve
is `strain_from_arc_length` propagating that rather than filling it in.
`TrackResult.lost` says which points, `TrackResult.support` says whether the
cause was missing matches or rejected ones -- read those before reading a
curve.

### Not settled

- **The domain gap.** These are MegaDepth (or ScanNet) weights on ultrasound
  speckle. The pretrained model is a floor, not a ceiling, and nothing here has
  been measured on the cohort yet -- the notebook exists to do that.
- **Whether GE's GLS is the right target.** It is GE's own tracking on GE's own
  contour, so agreeing with it is not the same as being correct. MITT is the
  better first comparison, because it tracked the same points and stores them.
- **Cost.** ~0.26 s/pair at 500x500 on an RTX 2060, so roughly an hour per
  pairing over 236 clips. `Matches` is cheap to keep and re-interpolate, which
  is what the notebook's parameter sweep relies on.

## Usage

```python
from motion_estimator.data.dataset_index import index_dataset
from motion_estimator.data.gt_strain import read_gt_strain
from motion_estimator.data.image_prep import frames_first, prepare_clip
from motion_estimator.data.scar_struct import read_view_analysis
from motion_estimator.data.srd_reader import read_clip

for ref in index_dataset():
    if not ref.is_complete:
        print(ref.key, "missing", ref.missing)
        continue

    analysis = read_view_analysis(ref.scar_struct_path, ref.key.view)
    prepared = prepare_clip(
        read_clip(ref.clip_path), cycle_frame_limits=analysis.cycle_frame_limits
    )
    frames = frames_first(prepared.images)              # (n_frames, rows, cols)
    target = read_gt_strain(ref.gt_strain_path).target_vector(ref.key.view)
    #  -> (7,) = 6 end-systolic segmental values, then GLS
```

The dataset root defaults to `F:\PHD\DataSets\2D_GE_intervendor_in_vivo` and is
overridable with `$MOTION_ESTIMATOR_DATA_ROOT`, since it lives on an external
drive whose letter moves.

## Running

```
python -m pytest motion_estimator/tests -q
python -m pytest motion_estimator/tests -q -m "not slow"   # skips the LoFTR checkpoint
python -m motion_estimator.scripts.report_cohort
python -m motion_estimator.scripts.report_cohort --probe          # opens every clip
python -m motion_estimator.scripts.report_cohort --views all      # includes 3ch
python -m motion_estimator.scripts.report_cohort --csv cohort.csv
```

Most tests build synthetic MAT files with `scipy.io.savemat` and run with the
dataset drive unplugged; the rest assert pinned values against the real IV02
files and skip when the root is unmounted. Unplugged the suite runs 251 tests
with 9 skips; with the drive mounted there are no skips. The `tracking/` tests
need neither the drive nor a GPU — the interpolation is checked against
closed-form motions on synthetic matches, and the LoFTR tests run the real
checkpoint on 128x128 speckle on CPU (`-m "not slow"` to skip those).
