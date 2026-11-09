# Archived notebooks

Kept as a record, not as working code. Their import paths pre-date the
`loaders/ methods/ tracking/ evaluation/` refactor and will not run unchanged.

- **`loftr_point_tracking_ARCHIVED.ipynb`** -- the LoFTR investigation, 46
  cells. Its findings are written up in `../../TRACKING_EXPERIMENTS.md`: the
  sub-pixel gain curve, the match-density arithmetic (~640 matches over a
  500x500 frame, ~14 px mean spacing, so a radius-16 neighbourhood holds
  about 4), and the three fine-tunes that all came out worse than pretrained.
  The section that still has standalone value is the support map, which showed
  the interpolation is starved across most of the sector rather than only at
  the endocardial border.
