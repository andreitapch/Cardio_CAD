# Foundation-model backbone for motion estimation, without MITT

Status 2026-11-08: the backbone probe is done and points at DINOv3 ViT-S/16.
Nothing of the estimator itself is built yet. Decisions still open are listed
at the end; this file is the place to come back to.

## Why

The replication (`methods/pwcnet/PAPER_REPLICATION.md`) shows where the
in vivo accuracy comes from and what it costs:

| training data | contour EPE vs MITT, corrected spacing |
|---|---|
| synthetic + MITT dense fields in vivo (MS47_IV85 pretrained150k, 6 folds) | 1.10 / 0.92 / 1.05 / 0.89 / 0.98 / 1.03 mm, mean ~1.00 |
| synthetic only (MS47_IV0 pretrained 30k / pretrained150k, MS100_IV0 pretrained150k) | 1.66 / 1.81 / 1.70 mm |
| paper, Table 2, best | 1.10 mm |

Dropping MITT from training costs about 0.6 mm today. The aim is a model that
needs no MITT at all (the GE cohort has no other per-point reference) and
does at least as well, by borrowing features from a vision foundation model
and training the rest on signals that do not involve MITT.

Three traps the design has to respect, all measured earlier:

- motion is ~1 px per frame and pretrained flow nets shrink such
  displacements to ~0.8 of their size (`TRACKING_EXPERIMENTS.md`, gain);
- from-scratch training sits on the zero-flow plateau for ~11k steps, and
  unsupervised losses pull toward zero flow even harder;
- synthetic validation stops predicting in vivo accuracy after ~60k steps
  for the pretrained start.

And the metric is biased: distance to MITT rewards models trained on MITT.
MITT-independent references are the simulated true EPE and gain (exist) and
GE vendor GLS per clip (not scored per run yet).

## Hardware

RTX 2060, 6 GB, Turing (fp16 only, no bf16), 32 GB RAM. The EchoCLIP trunk
(ConvNeXt-Base, 88M parameters) already runs here in fp16. A frozen backbone
fits easily; end-to-end fine-tuning of a ViT-B does not and the 118 patients
would not support it anyway. The RAM ceiling (DataLoader workers hit it in
the queue) matters more than the GPU: cache features memory-mapped or run
the backbone on the fly.

## The backbone probe (done 2026-11-07)

`python -m motion_estimator.scripts.probe_backbones [--centre]`, code in
`backbone_probe/`, results and full tables in
`artifacts/backbone-probe/README.md`. Frozen features of two frames are
matched cell by cell and scored against exact scatterers (30 simulated
clips, paper-split validation patients) and against MITT (24 Retest clips).
Fraction of queries whose best cell over the whole frame is within one cell
of the truth, features centred:

| level | backbone | simulated, 8 / 16 frames apart | in vivo, 8 / 16 frames apart |
|---|---|---|---|
| 1/16 | **DINOv3 ViT-S/16, last block** | **0.97 / 0.94** | **0.92 / 0.86** |
| 1/16 | EchoCLIP ConvNeXt-B trunk | 0.90 / 0.82 | 0.88 / 0.81 |
| 1/16 | PWC-Net pyramid, current best (trained with MITT) | 0.75 / 0.61 | 0.60 / 0.44 |
| 1/16 | DINOv3 ConvNeXt-T | 0.81 / 0.73 | 0.66 / 0.58 |
| 1/16 | DINOv2 ViT-S/14, best block | 0.77 / 0.68 | 0.63 / 0.56 |
| 1/32 | DINOv3 ViT-S/16 pooled 2x2 | 1.00 / 1.00 | 0.99 / 0.98 |
| 1/32 | EchoCLIP | 1.00 / 0.99 | 1.00 / 0.99 |
| 1/32 | PWC-Net pyramid, current best | 0.89 / 0.81 | 0.81 / 0.67 |
| 1/64 | DINOv3 ViT-S/16 pooled 4x4 | 1.00 / 1.00 | 1.00 / 1.00 |

What it says:

- **DINOv3 ViT-S/16 (last block) is the best coarse matcher on both
  cohorts** with no echo training, and its lead grows with the frame gap.
  Block 9 is much worse than block 12. DINOv2 is far behind. DINOv3
  ConvNeXt-T is weak at 1/16, so the first guess (ConvNeXt-T) was wrong.
- **One frame apart, every coarse level scores ~1.0.** The coarse features
  pay off at wide baselines only: tracking from a reference frame across the
  cycle without accumulating drift, which `reference` pairing already showed
  to be the better strategy for LoFTR.
- **Fine levels (1/4, 1/8) cannot be matched by nearest neighbour** with any
  backbone, the trained PWC-Net pyramids included: the median error is above
  predicting zero motion. Sub-pixel precision has to come from a learned
  decoder plus a fine CNN, as in the current PWC-Net.
- **Centring** (subtracting each map's spatial mean) makes the foundation
  models' cosine margins positive at 1/32 and 1/64 and changes no ranking.
  Do it before the cost volume.
- **Cost**: ViT-S/16 at 512 px, 24 ms per pair in fp16, 355 MB peak, 41 MB
  of weights. EchoCLIP: 47 ms, 939 MB.

## Proposed design

- **Hybrid pyramid.** Frozen DINOv3 ViT-S/16, last block, centred, gives
  levels 1/16, 1/32 (2x2 pooled) and 1/64 (4x4 pooled). A small CNN trained
  from scratch gives 1/4 and 1/8. The PWC-Net cost volume, decoders and
  context network from `methods/pwcnet/net.py` stay as they are, with a
  linear projection from the ViT width to the channel counts the decoders
  expect. EchoCLIP's trunk is the ablation backbone.
- **Input** 512 crops (the ViT grid is 32x32), frames scaled by the clip's
  own range as today; RGB by repeating the grey channel and the backbone's
  own normalisation.
- **Three training signals, none from MITT:**
  1. supervised dense flow from the 798 simulated clips (the splat targets
     of `paper_data.py`), with pairs 1-16 frames apart, not only consecutive;
  2. unsupervised in vivo on raw frames: census or SSIM photometric term,
     forward-backward occlusion masking, sector mask, and a feature-metric
     term on the frozen backbone features (robust to speckle decorrelation);
  3. cycle closure: flows chained over one cardiac cycle must bring points
     back to their start.
- **Against collapse:** spaced pairs during training, initialise the decoder
  from the synthetic stage before unsupervised fine-tuning, low learning
  rate, and log gain on simulated validation at every checkpoint; a falling
  gain means collapse.
- **Method package** under `methods/` with a registry entry and the
  `MotionEstimator` interface, so `paper_evaluate`, the viewer and the
  chaining code work unchanged. Runs go through the queue runner.

## Evaluation

Same six folds and test patients as the replication so numbers pair with
MS47_IV85. Simulated held-out EPE and gain; the paper contour metric vs MITT
(comparable to 1.66 mm synthetic-only and ~1.00 mm MITT-trained); GE GLS
agreement per run, to be built; the equal-step snapshots as in the
replication.

## Phases

1. Backbone probe. Done.
2. Method package: backbone adapter (timm, frozen, fp16), hybrid pyramid,
   estimator wrapper, registry entry.
3. MITT-free losses and an in vivo pair dataset without MITT (the in vivo
   cache already holds frames and sector masks).
4. Runs: synthetic-only supervised against the 1.66 mm baseline; plus in
   vivo unsupervised; ablations on backbone (ViT-S/16 vs EchoCLIP), frozen
   vs last-block adaptation, with and without the cycle loss.
5. GE GLS scoring per run; results into the notebook and the lab deck.

## Open decisions

- **In vivo pool for unsupervised training:** Retest + Test as in the
  replication (no clean hold-out) or Retest only with Test kept for a final
  number. Flemengho-2D on F: could be an extra unlabeled pool if it reads
  cleanly.
- **Scope:** synthetic-supervised plus in vivo unsupervised (recommended),
  or purely unsupervised with no simulated supervision.
- **The parked scratch runs** of the replication (MS47_IV85 folds 2-5,
  MS100_IV0) still wait for the GPU; this work competes with them.
- **Licence:** DINOv3 weights are under Meta's DINOv3 licence; check its
  terms before publishing. DINOv2 is Apache 2.0 but lost the probe.
