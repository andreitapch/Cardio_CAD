"""Everything that reads data off disk.

Split by what the data *is*, because the two kinds have different failure
modes and different trust levels:

* :mod:`images` -- pixel sources. Clip files, the clinical-view transform, the
  simulated cohort, and the per-clip ``Tracking/`` artifacts.
* :mod:`annotations` -- contours and ground truth. MITT tracks, ``scar_struct``,
  the vendor strain values.
* :mod:`dataset_index` -- cohort discovery and the ``(dataset, patient, view)``
  key that folds must be grouped on.
* :mod:`train_loader` -- frame pairs with exact correspondence, for training.
* :mod:`test_loader` -- whole clips with their reference contours, for
  evaluation.

The train/test split lives here rather than in the training or evaluation code
so that both sides load data the same way, and so a leak has one place to be
introduced rather than two.
"""
