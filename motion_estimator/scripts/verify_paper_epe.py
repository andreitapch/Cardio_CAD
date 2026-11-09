"""Check the Python port of the paper's in vivo EPE against the paper's own predictions.

    python -m motion_estimator.scripts.verify_paper_epe
    python -m motion_estimator.scripts.verify_paper_epe --test Test_6 --variant unCorr_prop_uni

Reads the contours the MATLAB pipeline propagated for an old experiment
(``F:\\PHD\\Experiments\\Journal_MotionEstimation\\Results\\InVivo\\<test>\\crossVal``,
2ch, one ``predPoints_<variant>.mat`` per patient), scores them against MITT
with :mod:`motion_estimator.methods.pwcnet.paper_metrics`, and prints the
Table-2 numbers. Test_6 is MS47_IV85 (the paper reports 1.10 +- 0.37 mm).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..loaders.images import _matio
from ..methods.pwcnet.paper_metrics import contour_epe, summarise

RESULTS = Path(r"F:\PHD\Experiments\Journal_MotionEstimation\Results\InVivo")
CACHE = Path(__file__).resolve().parents[1] / "artifacts" / "invivo-cache"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test", default="Test_6")
    parser.add_argument("--variant", default="unCorr_prop_uni")
    parser.add_argument("--view", default="2ch")
    args = parser.parse_args(argv)

    roots = sorted((RESULTS / args.test / "crossVal").glob("*/*_dataset"))
    for exact in (True, False):
        sequences, missing = [], []
        for dataset_dir in roots:
            for patient in sorted(dataset_dir.glob("IV*")):
                pred_file = patient / f"predPoints_{args.variant}.mat"
                cache = CACHE / dataset_dir.name / f"{patient.name}_{args.view}.npz"
                if not (pred_file.is_file() and cache.is_file()):
                    missing.append(f"{dataset_dir.name}/{patient.name}")
                    continue
                result = _matio.field(_matio.load_mat(pred_file)["predResult"], "pointsX"), \
                    _matio.field(_matio.load_mat(pred_file)["predResult"], "pointsY")
                pred_x, pred_y = (np.asarray(a, float).T for a in result)        # -> (frames, points)
                with np.load(cache, allow_pickle=False) as data:
                    ref_x, ref_y = data["reference_x"], data["reference_y"]
                    spacing = data["spacing_xy"]
                if pred_x.shape != ref_x.shape:
                    missing.append(f"{dataset_dir.name}/{patient.name} shape {pred_x.shape} vs {ref_x.shape}")
                    continue
                sequences.append(contour_epe(pred_x, pred_y, ref_x, ref_y, spacing,
                                             f"{dataset_dir.name}/{patient.name}", paper_exact=exact))
        summary = summarise(sequences)
        label = "paper-exact (swapped spacing)" if exact else "corrected spacing"
        print(f"{args.test} {args.variant} {args.view}, {label}: "
              f"{summary['epe_mm_mean']:.3f} mm, SD pooled points {summary['epe_mm_sd_pooled_points']:.3f}, "
              f"SD across sequences {summary['epe_mm_sd']:.3f}, "
              f"n={summary['n_sequences']}, lost points {summary['lost_points']}")
        for dataset in sorted({s.key.split('/')[0] for s in sequences}):
            part = summarise([s for s in sequences if s.key.startswith(dataset)])
            print(f"    {dataset}: {part['epe_mm_mean']:.3f} mm (n={part['n_sequences']})")
    if missing:
        print(f"not scored ({len(missing)}): {', '.join(missing[:12])}{' ...' if len(missing) > 12 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
