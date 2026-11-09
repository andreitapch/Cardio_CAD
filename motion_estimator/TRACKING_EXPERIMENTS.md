# Tracking experiments -- LoFTR on the in-vivo and simulated cohorts

A lab notebook of what was tried, what failed, and what the numbers say. Two
conclusions here contradict earlier ones in this same document; where that
happens the correction is stated rather than the earlier claim quietly removed,
because the wrong diagnosis cost two training runs and is worth not repeating.

All numbers are measured, on held-out data. Simulated errors are against exact
scatterer positions; in-vivo errors are agreement with MITT, which is itself a
tracker and not ground truth.

## The pipeline

LoFTR is **detector-free**: it chooses its own correspondence locations on a
coarse 1/8-resolution grid and has no input for "track these points". So
tracking a given contour takes two stages:

```
tracking/matcher.py         LoFTR on a frame pair -> (xy0, xy1, confidence)
tracking/point_tracking.py  that field interpolated at the contour points
```

`displacement_at` is a confidence-weighted local affine fit with outlier
rejection. Four chaining strategies sit on top of it:

| strategy | how it pairs frames |
|---|---|
| `consecutive` | frame *k* to *k+1*, advecting the current estimate |
| `reference` | frame 0 to frame *k*, displacing the original points |
| `strided` | anchors every *stride* frames, each other frame one hop from a far enough anchor |
| `bidirectional` | `consecutive` forwards and backwards, blended by frame index |

## Coordinate bugs found and fixed

These were real, and each one moved the contour off the myocardium entirely.

- **MITT's stored axes are transposed relative to the clinical image**, and
  `pixel_spacing_mm` is ordered `(row, col)` rather than `(x, y)`. The correct
  conversion is

  ```python
  x = points_y / spacing[1] + 1
  y = points_x / spacing[0] + 1
  ```

  Getting it backwards puts the contour 90 degrees off the myocardium, which is
  exactly what it looks like. Verified visually -- the corrected contour lands
  on the endocardium.
- **`TrackResult.to_mm()` takes `(x, y)`**, so `pixel_spacing_mm` has to be
  passed reversed. Row and column spacing differ across this cohort
  (~0.26-0.37 mm), so confusing the two rescales strain rather than merely
  rounding it.
- **`scar_struct`'s convention is verified correct**, closing an open question
  in the package README: `endo_sample` is `(2, n)` with x in row 0, indexing the
  clinical view directly, 0-based. Plotted over `frames_first(...)[0]` it lands
  on the endocardium and agrees with the MITT contour.

## Fine-tuning: three runs, all worse than pretrained

The simulated cohort supplies exact dense correspondence, so fine-tuning LoFTR
on it looked like the obvious lever. It was not. Every run degraded the model
on **held-out simulated data as well as in vivo**, which rules out a
sim-to-real gap as the explanation.

### Baselines (pretrained `outdoor` weights, held-out)

| | `consecutive` | `reference` |
|---|---|---|
| simulated endpoint error | 1.89 px | 1.01 px |
| in-vivo vs MITT | 5.06 px | 5.47 px |

### Run 1

Validation loss fell monotonically, 0.448 -> 0.3866, with no sign of
overfitting. The model was worse anyway:

| | pretrained | run 1 |
|---|---|---|
| simulated, `consecutive` | 1.89 px | 5.28 px |
| in-vivo, `consecutive` | 5.06 px | 7.71 px |
| in-vivo, cycle closure | 7.24 px | 3.16 px |

Closure improving sharply *while* accuracy degraded is the signature of a model
under-predicting displacement: less motion predicted closes the cycle better and
tracks worse.

**Diagnosis at the time:** the pair sampler drew frame spacing with weight
`1/n`, putting 32% of pairs at spacing 1 and **77% of all pairs below one coarse
cell (8 px)**. Motion runs at ~1 px per frame step, so for most training data
the correct coarse target was "same cell" -- the model was being taught to
predict no motion, and a focal loss on the assignment matrix is minimised nicely
that way.

### Run 2 -- the fix that did not fix it

Spacing floor (min 8, max 20) plus ground-truth padding of the coarse matches.
Validation loss 0.8354 -> 0.7053, higher than run 1 as expected for a harder
target.

| | pretrained | run 1 | run 2 |
|---|---|---|---|
| simulated, `consecutive` | 1.89 px | 5.28 px | 5.19 px |
| simulated, `reference` | 1.01 px | 2.13 px | 2.05 px |
| in-vivo, `consecutive` | 5.06 px | 7.71 px | 8.08 px |

**Essentially unchanged.** The sub-cell sampling was a real defect, but it was
not what made the model worse. The diagnosis was wrong.

### What was eliminated by experiment

- **Resolution mismatch.** Training used 320 px crops, evaluation full frames,
  so the coarse token grid differs (40x40 vs 113x113) and the positional
  encoding could have specialised. Measured on raw matches at several crop
  sizes:

  | crop size | pretrained | fine-tuned |
  |---|---|---|
  | 320 | 1.43 px | 2.68 px |
  | 448 | 1.56 px | 2.52 px |
  | 640 | 1.47 px | 2.53 px |
  | full | 1.47 px | 2.63 px |

  Worse at **every** size, and flat across them. Not resolution. Match counts
  were comparable (461 vs 491 at full size), so matching did not collapse -- it
  became less accurate.

- **Checkpoint save/load.** Saving the pretrained weights through the same code
  path and reloading them via `LoFTRMatcher(checkpoint=...)` gave a bit-identical
  round trip (max weight difference 0.00e+00) and an identical 1.472 px. The
  checkpoint path is clean.

- **Learning rate.** 200-step probes, raw match error at full resolution:

  | lr | fine-tuned | pretrained |
  |---|---|---|
  | 1e-4 | 2.31 px | 1.47 px |
  | 2e-5 | 1.97 px | 1.47 px |
  | 5e-6 | 1.88 px | 1.47 px |

  Damage shrinks with the learning rate but never reverses. At lr -> 0 the best
  attainable outcome is "unchanged". **The gradient direction is wrong**, so no
  learning rate fixes it.

### Conclusion, and the unfixed suspect

This is a loss-formulation problem. The prime suspect: the coarse loss
supervises **only myocardial rows** of the assignment matrix and masks the rest,
but LoFTR's dual-softmax is competitive across the whole matrix. Raising
confidence on myocardial pairs necessarily suppresses everything else, and
nothing in the loss constrains that. Supervising unmatched cells as explicit
negatives, rather than masking them out, is the change to try.

Not fixed. Fine-tuning was stopped in favour of the finding below.

The coarse supervision itself was checked and is **correct**: decoding the
target cell indices back to displacements agrees with the true scatterer motion
to 0.3-1.7 px, inside the 4 px half-cell quantisation limit.

### kornia 0.8.3 cannot train LoFTR as shipped

`CoarseMatching.get_coarse_match` raises whenever `self.training` is true:

```
ValueError: too many values to unpack (expected 2)
```

It unpacks a four-way `zip` as two values. The ground-truth padding it would
have done -- topping predicted matches up so the fine level always has something
to refine -- is therefore unavailable. Worked around in
`finetune_loftr._pad_with_ground_truth`, applied after running the matcher
submodules in eval mode, rather than by patching site-packages. Patching
site-packages was the alternative and is worse: the torch reinstall in this
environment already demonstrated how long such an edit survives.

## The finding that actually mattered: gain

`diagnose_drift` measured **gain 0.855** -- the ratio of predicted to true
per-step displacement. The tracker reproduces 85% of the motion, so strain reads
about 15% low regardless of how well drift is handled. That is a scale error,
not drift, and `diagnose_gain` localised it.

Measured against exact scatterer ground truth, raw LoFTR matches versus the
interpolated field:

| frame gap | true motion | raw LoFTR gain | interp r=16 | interp r=32 |
|---|---|---|---|---|
| 1 | ~1 px | 0.798 | 0.754 | 0.743 |
| 2 | ~2 px | 0.859 | 0.843 | 0.831 |
| 6 | ~6 px | 0.979 | 0.977 | 0.958 |
| 12 | ~12 px | 0.987 | 0.984 | 0.968 |

**The loss is in the matcher, not the interpolation.** Interpolation costs about
2% at radius 32 and 0.2% at radius 16; the matcher itself is 20% short at 1 px
and 1.3% short at 12 px.

Why: LoFTR's fine level regresses its subpixel offset as the **spatial
expectation of a softmax heatmap** (DSNT). That estimator is biased toward the
centre of its window -- it shrinks small offsets toward zero. The bias only
bites when the true offset is small.

Our clips move ~1 px per frame. `consecutive` chaining therefore runs at a gain
near 0.8, and **this is the root cause of the flat, underestimated strain that
started the whole investigation**. Neither fine-tuning nor drift correction
would ever have fixed it, because neither addresses the magnitude of the
displacement being asked for.

It also explains why `reference` beat `consecutive` from the start (1.01 vs
1.89 px simulated): wider baselines, better gain.

## Drift decomposition

"Drift" covers two failures with different fixes, and the simulated cohort
separates them because the true per-step displacement is known.

| | |
|---|---|
| gain (predicted / true step) | 0.855 |
| bias per step | -0.162 px |
| noise per step | 0.238 px |
| closure observed | 2.47 px |
| closure predicted by bias | 6.98 px |
| closure predicted by random walk | 1.55 px |

Observed closure sits far closer to the random-walk prediction, so drift here is
**noise accumulating, not a coherent bias**. Bidirectional tracking cancels bias
but only buys about sqrt(2) against a random walk.

`track_bidirectional` was implemented and verified regardless -- on a
synthetically biased tracker, forward drifts 6.00 px, bidirectional closes to
0.00 px and the midpoint is exact -- but it is not the main lever the in-vivo
closure numbers had suggested.

**Caveat:** this decomposition was measured on simulated clips, where closure is
2.47 px. In-vivo closure is larger (4.78-5.08 px at radius 16, 7.24 px at radius
32), so the mechanism may differ there and the verdict should not be assumed to
transfer.

## What improved

### Simulated, held-out, 5 clips, radius 16

| method | endpoint error | cycle gain | points lost |
|---|---|---|---|
| `consecutive` | 1.74 px | 0.842 | 209 |
| `reference` | 0.71 px | 0.974 | 692 |
| `strided s=6` | 1.12 px | **0.987** | 689 |

`strided` wins on gain, which is what determines strain accuracy. `reference`
wins on position accuracy. They are not the same question.

### In-vivo, 8 clips, radius 16

| method | vs MITT | p90 | closure | points lost |
|---|---|---|---|---|
| `consecutive` | 3.78 px | 4.55 | 5.08 px | 327 |
| `reference` | 3.22 px | 3.78 | 4.78 px | 576 |
| `strided s=6` | **3.08 px** | 5.27 | nan | 576 |

`strided` wins on the median but has the **worst p90** (5.27 vs 3.78). It helps
most clips and hurts some; it is not uniformly better. Its `nan` closure is a
defect, not a result -- see open problems.

### The single biggest in-vivo win was the interpolation radius

Dropping `radius` from 32 to 16:

| method | radius 32 | radius 16 |
|---|---|---|
| `consecutive` | 5.06 px | 3.78 px |
| `reference` | 5.47 px | 3.22 px |

Larger than any change to the tracking strategy, and larger than anything
fine-tuning was ever going to deliver.

## Correction: strided is the best on simulated, the worst in vivo

The first in-vivo comparison put `strided` ahead on median (3.08 px against
`reference` 3.22 px). **That number was an artefact of a bug in `track_strided`
itself.** Anchors were chained, so a point the matcher lost at one anchor was
lost at every later anchor; those points became NaN and dropped out of the
median, which therefore scored only the points that never failed. On IV02/4ch
the last frame came back with 4 of 72 points, and cycle closure was undefined.

The fix is a **per-point fallback**: each target frame is filled from its
preferred source first, then from further anchors for whichever points are
still missing, up to `max_sources` attempts. Last-frame coverage on IV02/4ch
went from 4/72 to 13/72 and closure became finite.

Re-measured fairly, 8 clips, radius 16:

| method | vs MITT | p90 | closure | points lost |
|---|---|---|---|---|
| `consecutive` | 3.78 px | 4.55 | 5.08 px | 327 |
| **`reference`** | **3.22 px** | **3.78** | **4.78 px** | 576 |
| `strided` | 5.07 px | 6.64 | 9.79 px | 572 |

So the honest conclusion is the opposite of the first one:

- **On simulated data `strided` is right** -- gain 0.987 against 0.974, and gain
  is what determines strain accuracy.
- **In vivo `reference` wins** -- 3.22 px against 5.07 px, with less than half
  the closure error.

The reason is coverage. In vivo the matcher frequently fails near the contour,
and `strided` chains through anchors, so every failure compounds both error and
loss. `reference` re-derives each frame from frame 0 and accumulates neither.
Where matches are dense, chaining with wide gaps wins on gain; where they are
sparse, not accumulating wins by more.

**Current best in-vivo configuration: `reference` at `radius=16`**, 3.22 px
against MITT, down from 5.06 px at the start.

## Open problems

- **Point loss is now the dominant issue.** On IV02/4ch, **100% of NaN cells had
  `support == 0`** -- LoFTR produced no matches at all within 16 px of those
  points. The guards (`min_neighbours`, conditioning, extrapolation gate) are
  not rejecting anything; the matcher simply does not fire near much of the
  contour. Roughly 576 of 720 points are touched by NaN under the wide-baseline
  methods.
- **`track_strided` propagates NaN through the anchor chain.** Once an anchor
  loses a point, every later anchor inherits the loss. On IV02/4ch the last
  frame had **4/72 finite points** against 47/72 for `reference`, which is why
  its closure reads `nan`. It needs a per-point fallback to an earlier anchor.
  Not yet fixed.
- **The fine-tuning loss formulation is unfixed** -- see the dual-softmax
  suspect above.
- **`reference` loses ~80% of its in-vivo points even with pretrained weights**,
  so this is not something fine-tuning caused and not something it will fix.

## Preprocessing does not fix coverage

The remaining constraint was match coverage near the contour, so four image
variants were measured against each other on 4 clips with the `reference`
pairing (`scripts/diagnose_coverage.py`):

| variant | coverage | vs MITT | closure | matches |
|---|---|---|---|---|
| none | 32.1% | 3.22 px | 3.12 px | 641 |
| clahe | **37.1%** | 3.47 px | 4.37 px | 748 |
| gamma | 26.9% | **2.95 px** | 3.52 px | 606 |
| upsample 2x | 32.0% | 4.61 px | 4.48 px | 825 |

**None of them solves it**, and the baseline number is worse than the earlier
framing suggested: coverage is **32%**, so two thirds of all tracked positions
are NaN -- not merely "576 of 720 points touched by NaN".

- **CLAHE** buys 5 points of coverage and 107 more matches, and pays for it in
  both agreement and closure. The extra points are wrong ones.
- **Gamma** trades the other way: the fewest points, the best agreement.
- **Upsampling** produced the most matches and the worst agreement. The
  prediction that a denser coarse grid would help was simply wrong -- more
  matches, worse placed.

So the limit is not contrast, and not resolution. LoFTR returns 600-800 matches
over a 500x500 frame and they cluster in textured speckle. The endocardial
border is a low-contrast boundary between two dim regions, and that is exactly
where the MITT points sit. This is a structural mismatch between a sparse
matcher and a contour-tracking task, not a tuning problem.

Two directions that address it rather than working around it:

1. **Track mid-wall instead of the border.** Offset the contour a few mm into
   the myocardium where the speckle is strong, track there, and carry the
   deformation back out to the border. Uses LoFTR where it actually works, and
   costs an afternoon to test.
2. **Use a dense method.** Optical flow gives a displacement at every pixel, so
   coverage is 100% by construction. Not speculative here: the Farneback flow
   used earlier to verify scatterer alignment already correlated 0.93-0.99 with
   the ground-truth displacement field. A learned dense matcher (RoMa, DKM) is
   the same idea with better accuracy.

## Diagnostic scripts

| script | what it answers |
|---|---|
| `scripts/diagnose_resolution.py` | is a checkpoint specialised to its training crop size? Raw match error at several sizes, two checkpoints side by side. |
| `scripts/diagnose_drift.py` | is drift a systematic bias or a random walk? Decomposes per-step gain, bias and noise, and predicts closure from each. |
| `scripts/diagnose_gain.py` | where is the motion lost -- matcher or interpolation? Gain of raw matches versus `displacement_at`, swept over radius. |
| `scripts/diagnose_coverage.py` | can preprocessing get the matcher to fire near the contour? Coverage, agreement and closure per image variant. |
| `training/runs.py` | one directory per experiment, `artifacts/runs/<date>_<time>_<name>/`, holding `config.json`, `history.json`, `best.pt`, `last.pt` and `eval_*.json`. |

## Next steps, ranked

1. **Fix the NaN propagation in `track_strided`** -- per-point fallback to an
   earlier anchor when the nearest one has lost that point. Cheap, and its
   closure figure is unreadable until this is done.
2. **Attack match coverage near the contour.** This is the binding constraint
   now, and the obvious lever is untested: CLAHE or log-compression, a tighter
   sector mask, and cropping to the LV before matching. Hours of work, and the
   radius result suggests preprocessing-scale changes outweigh everything
   architectural here.
3. **Only then revisit fine-tuning**, with the coarse loss supervising negatives
   instead of masking them. Until the loss is fixed there is no reason to expect
   a different outcome from the three runs above.
