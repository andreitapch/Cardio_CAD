"""Evaluate every fold's models on that fold's held-out patients.

    python -m motion_estimator.methods.pwcnet.evaluate --run-dir <cv>
    python -m motion_estimator.methods.pwcnet.evaluate --run-dir <cv> --folds 0 --pairings consecutive strided
    python -m motion_estimator.methods.pwcnet.evaluate --run-dir <cv> --summarise-only

**In vivo**, the three variants per fold, each tracked from MITT's contour at
its reference frame and compared with MITT on every frame:

* ``zero_shot`` -- ``sim_fold<k>/best.pt``, never saw in-vivo data;
* ``unsup``     -- ``invivo_fold<k>_unsup/best.pt``;
* ``mitt``      -- ``invivo_fold<k>_mitt/best.pt``.

A variant whose checkpoint does not exist yet is skipped, so this can run
after any stage. Distances to MITT are *disagreements*, not errors: MITT is a
tracker too. GLS is compared with both MITT's and GE's.

**Simulated**, ``sim_fold<k>/best.pt`` on the fold's held-out patients against
the exact scatterers -- the only place a tracking error really is an error,
and where ``gain`` is measured.

Writes, per fold::

    <cv>/eval/fold<k>/in_vivo_rows.json     one row per clip x variant x pairing
    <cv>/eval/fold<k>/simulated_rows.json
    <cv>/eval/fold<k>/tracks/<clip>.npz     every contour, for the notebook
    <cv>/eval/fold<k>/plots/<clip>_<pairing>.png   contours vs MITT, strain

and across folds ``<cv>/eval/summary_*.csv`` plus summary figures.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ...evaluation.metrics import evaluate
from ...tracking import chain
from .data import (SIM_CACHE, list_invivo_cached, load_folds, load_invivo_clip,
                   load_simulated_clip)
from .estimator import PWCNetEstimator, load_model

PAIRINGS = ("consecutive", "reference", "strided", "bidirectional")


def checkpoints(run_dir, fold):
    run_dir = Path(run_dir)
    found = {
        "zero_shot": run_dir / f"sim_fold{fold}" / "best.pt",
        "unsup": run_dir / f"invivo_fold{fold}_unsup" / "best.pt",
        "mitt": run_dir / f"invivo_fold{fold}_mitt" / "best.pt",
    }
    return {name: path for name, path in found.items() if path.is_file()}


def strain_of(x, y, clip):
    """``(curve, gls_es)`` from 1-based pixel tracks, NaN if it cannot be done."""
    import torch

    from ...strain.curve import compute_strain

    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return np.full(x.shape[0], np.nan), float("nan")
    spacing = np.asarray(clip.spacing_xy, dtype=float)
    try:
        result = compute_strain(
            torch.as_tensor((x - 1.0) * spacing[0]),
            torch.as_tensor((y - 1.0) * spacing[1]),
            int(np.clip(clip.es_frame, 1, x.shape[0])),
        )
    except Exception as error:
        print(f"  strain failed on {clip.key}: {error}", file=sys.stderr)
        return np.full(x.shape[0], np.nan), float("nan")
    return result.curve.numpy(), result.value_es


def _safe(key):
    return key.replace("/", "_")


def evaluate_in_vivo(run_dir, fold_id, folds, args, device):
    from .plots import plot_clip

    fold = folds["in_vivo"][fold_id]
    found = checkpoints(run_dir, fold_id)
    if not found:
        print(f"fold {fold_id}: no checkpoints yet, skipping in vivo")
        return []
    models = {name: load_model(path, device) for name, path in found.items()}
    test = [i for i in list_invivo_cached() if i["group_id"] in set(fold["test"])]
    out = Path(run_dir) / "eval" / f"fold{fold_id}"
    (out / "tracks").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    print(f"\nin vivo fold {fold_id}: {len(test)} clips, variants {list(models)}")

    rows = []
    for item in test:
        clip = load_invivo_clip(item["path"])
        n = min(clip.n_frames, clip.reference_x.shape[0])
        clip.frames = clip.frames[:n]
        clip.reference_x, clip.reference_y = clip.reference_x[:n], clip.reference_y[:n]
        mitt_curve, gls_mitt = strain_of(clip.reference_x, clip.reference_y, clip)
        saved = {"reference_x": clip.reference_x, "reference_y": clip.reference_y,
                 "mitt_strain": mitt_curve, "gls_ge": clip.gls_ge,
                 "start_frame": clip.start_frame, "es_frame": clip.es_frame,
                 "cache_path": str(item["path"]), "key": clip.key}
        per_pairing = {p: ({}, {}) for p in args.pairings}

        for variant, model in models.items():
            estimator = PWCNetEstimator(sector=clip.sector, model=model, device=device,
                                        consistency_px=args.consistency_px,
                                        name=f"pwcnet_{variant}")
            for pairing in args.pairings:
                result = chain(clip.frames, *clip.start(), estimator, pairing=pairing,
                               reference_frame=clip.start_frame, stride=args.stride)
                row = evaluate(result, clip, method=variant, pairing=pairing)
                curve, gls = strain_of(result.x, result.y, clip)
                row.update(fold=fold_id, variant=variant, group=f"{variant}:{pairing}",
                           patient=clip.group_id, view=clip.key.split("/")[-1],
                           gls_pred=gls, gls_mitt=gls_mitt, gls_ge=clip.gls_ge)
                rows.append(row)
                saved[f"x_{variant}_{pairing}"] = result.x
                saved[f"y_{variant}_{pairing}"] = result.y
                saved[f"strain_{variant}_{pairing}"] = curve
                tracks, strains = per_pairing[pairing]
                tracks[variant] = (result.x, result.y)
                strains[variant] = curve
                print(f"  {clip.key:32s} {variant:9s} {pairing:12s} "
                      f"ES {row['error_px_es']:6.2f} px  med {row['error_px_median']:5.2f}"
                      f"  GLS {gls:6.1f} (MITT {gls_mitt:6.1f}, GE {clip.gls_ge:6.1f})")

        np.savez_compressed(out / "tracks" / f"{_safe(clip.key)}.npz", **saved)
        if not args.no_plots:
            for pairing, (tracks, strains) in per_pairing.items():
                plot_clip(clip.frames, clip.reference_x, clip.reference_y, tracks,
                          clip.start_frame, clip.es_frame, strains, mitt_curve,
                          clip.gls_ge, title=f"{clip.key} -- fold {fold_id}, {pairing}",
                          path=out / "plots" / f"{_safe(clip.key)}_{pairing}.png")

    (out / "in_vivo_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def evaluate_simulated(run_dir, fold_id, folds, args, device):
    from ...loaders.train_loader import list_cached

    path = Path(run_dir) / f"sim_fold{fold_id}" / "best.pt"
    if not path.is_file():
        print(f"fold {fold_id}: no simulated checkpoint yet, skipping")
        return []
    model = load_model(path, device)
    fold = folds["simulated"][fold_id]
    test = [c for c in list_cached(args.sim_cache) if c.group_id in set(fold["test"])]
    if args.sim_limit and len(test) > args.sim_limit:
        test = [test[i] for i in np.linspace(0, len(test) - 1, args.sim_limit).astype(int)]
    out = Path(run_dir) / "eval" / f"fold{fold_id}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\nsimulated fold {fold_id}: {len(test)} clips")

    rows = []
    for cached in test:
        clip = load_simulated_clip(cached.path, args.n_points)
        estimator = PWCNetEstimator(sector=clip.sector, model=model, device=device,
                                    consistency_px=args.consistency_px)
        for pairing in args.pairings:
            result = chain(clip.frames, *clip.start(), estimator, pairing=pairing,
                           reference_frame=clip.start_frame, stride=args.stride)
            row = evaluate(result, clip, method="zero_shot", pairing=pairing)
            row.update(fold=fold_id, variant="zero_shot", group=f"zero_shot:{pairing}",
                       patient=clip.group_id)
            rows.append(row)
            print(f"  {clip.key[-30:]:30s} {pairing:12s} ES {row['error_px_es']:6.2f} px"
                  f"  med {row['error_px_median']:5.2f}  gain {row['gain']:.3f}")
    (out / "simulated_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def load_rows(run_dir, cohort):
    rows = []
    for path in sorted((Path(run_dir) / "eval").glob(f"fold*/{cohort}_rows.json")):
        rows += json.loads(path.read_text(encoding="utf-8"))
    return rows


def summary_table(rows, metrics):
    """Per group: mean and std over folds of each fold's median, plus pooled median."""
    import pandas as pd

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    if "gls_pred" in frame:
        frame["gls_abs_diff_mitt"] = (frame["gls_pred"] - frame["gls_mitt"]).abs()
        frame["gls_abs_diff_ge"] = (frame["gls_pred"] - frame["gls_ge"]).abs()
    metrics = [m for m in metrics if m in frame]
    per_fold = frame.groupby(["group", "fold"])[metrics].median()
    table = per_fold.groupby("group").agg(["mean", "std"])
    table.columns = [f"{m}_{s}" for m, s in table.columns]
    table["n_clips"] = frame.groupby("group").size()
    table["n_folds"] = per_fold.reset_index().groupby("group")["fold"].nunique()
    return table.reset_index()


IN_VIVO_METRICS = ["error_px_es", "coverage_es", "error_px_median", "error_px_p90",
                   "closure_px_mean", "gls_abs_diff_mitt", "gls_abs_diff_ge"]
SIM_METRICS = ["error_px_es", "error_px_median", "error_px_p90", "coverage",
               "closure_px_mean", "gain"]


def summarise(run_dir):
    from .plots import plot_fold_box, plot_gls_agreement

    out = Path(run_dir) / "eval"
    out.mkdir(parents=True, exist_ok=True)
    for cohort, metrics in (("in_vivo", IN_VIVO_METRICS), ("simulated", SIM_METRICS)):
        rows = load_rows(run_dir, cohort)
        if not rows:
            continue
        table = summary_table(rows, metrics)
        table.to_csv(out / f"summary_{cohort}.csv", index=False)
        print(f"\n=== {cohort}: mean (std) over folds of per-fold medians ===")
        for _, row in table.iterrows():
            cells = "  ".join(f"{m} {row[m + '_mean']:.2f} ({row[m + '_std']:.2f})"
                              for m in metrics if m + "_mean" in row)
            print(f"{row['group']:26s} {cells}  [{int(row['n_clips'])} clips]")
        ylabel = ("ES distance to MITT (px)" if cohort == "in_vivo"
                  else "ES error vs scatterers (px)")
        plot_fold_box(rows, "error_px_es", ylabel, path=out / f"box_{cohort}_es.png")
        if cohort == "in_vivo":
            plot_gls_agreement(rows, "gls_mitt", path=out / "gls_vs_mitt.png")
            plot_gls_agreement(rows, "gls_ge", path=out / "gls_vs_ge.png")
    print(f"\nwrote summaries to {out}")


def main(argv=None):
    import torch

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--cohorts", nargs="+", default=["in_vivo", "simulated"],
                        choices=("in_vivo", "simulated"))
    parser.add_argument("--pairings", nargs="+", default=["consecutive"], choices=PAIRINGS)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--consistency-px", type=float, default=None,
                        help="forward-backward tolerance; off by default so strain "
                             "can be computed on complete contours")
    parser.add_argument("--sim-limit", type=int, default=20,
                        help="held-out simulated clips per fold (0 = all)")
    parser.add_argument("--n-points", type=int, default=200)
    parser.add_argument("--sim-cache", default=str(SIM_CACHE))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--summarise-only", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    args = parser.parse_args(argv)

    if not args.summarise_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        folds = load_folds(args.run_dir)
        for fold_id in args.folds if args.folds is not None else range(folds["k"]):
            if "in_vivo" in args.cohorts:
                evaluate_in_vivo(args.run_dir, fold_id, folds, args, device)
            if "simulated" in args.cohorts:
                evaluate_simulated(args.run_dir, fold_id, folds, args, device)
    if not args.no_summary:
        summarise(args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
