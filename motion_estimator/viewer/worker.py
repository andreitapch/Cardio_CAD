"""Predict, track and compute strain for one model on in vivo clips; the viewer's GPU side.

    python -m motion_estimator.viewer.worker --model MS47_IV0_pretrained --all
    python -m motion_estimator.viewer.worker --model MS47_IV0 --clips Retest_dataset/IV02/2ch --flows
    python -m motion_estimator.viewer.worker --model MITT --all

Runs in its own process because it needs torch, which cannot share a process
with the app's scientific stack. Per clip it writes
``artifacts/viewer-cache/<model>/<dataset>/<IV##>_<view>.npz``:

* ``track_x`` / ``track_y`` -- the 72-point contour per frame, 1-based pixels,
  propagated from MITT's frame-1 contour through the model's flow
  (the paper's protocol; ``MITT`` stores MITT's own points);
* ``epe_per_frame`` / ``epe_mm`` -- the paper's contour EPE against MITT
  (corrected spacing; NaN where MITT's length differs from the cycle);
* ``strain`` -- longitudinal strain curve (%), endocardium, drift
  compensated (``mainComputeStrain.m``), with ``strain_es`` / ``strain_peak``;
* ``run`` and ``seen`` -- which run predicted it, and whether that run trained
  on this patient.

``--flows`` also keeps the predicted flow at half resolution, float16
(``.flow.npy``), for the viewer's overlay. Clips already cached are skipped
unless ``--overwrite``. Progress goes to ``--progress`` (JSON) for the app.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..methods.pwcnet.paper_evaluate import load_model, predict_flows
from ..methods.pwcnet.paper_metrics import contour_epe
from ..strain.curve import compute_strain
from ..strain.propagate import propagate_points
from . import catalog


def _strain(x, y, spacing, es_frame):
    x_mm = (np.asarray(x) - 1) * spacing[0]
    y_mm = (np.asarray(y) - 1) * spacing[1]
    result = compute_strain(x_mm, y_mm, es_frame=int(min(max(es_frame, 1), len(x_mm))))
    return result.curve.numpy(), result.value_es, result.value_peak_systolic


def _save(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(partial, **arrays)
    partial.replace(path)


def process_clip(model_id, model, key, device, loaded: dict, flows: bool) -> str:
    with np.load(catalog.clip_file(key), allow_pickle=False) as data:
        frames = data["frames"]
        ref_x, ref_y = data["reference_x"].astype(float), data["reference_y"].astype(float)
        spacing = data["spacing_xy"].astype(float)
        es_frame = int(data["es_frame"])
        gls_ge = float(data["gls_ge"])
    out = catalog.result_file(model_id, key)

    if model_id == catalog.MITT:
        curve, es, peak = _strain(ref_x, ref_y, spacing, es_frame)
        _save(out, track_x=ref_x, track_y=ref_y, epe_per_frame=np.zeros(len(ref_x)), epe_mm=0.0,
              strain=curve, strain_es=es, strain_peak=peak, es_frame=es_frame, gls_ge=gls_ge,
              run="MITT", seen=False)
        return "ok"

    run = model.run_for(key)
    if run is None:
        return "fold not trained yet"
    if run not in loaded:
        loaded[run] = load_model(run, "best.pt", device)
    flow = predict_flows(loaded[run], frames, device)                  # (n-1, H, W, 2)
    track_x, track_y = (t.numpy() for t in propagate_points(flow, ref_x[0], ref_y[0]))
    matched = len(ref_x) == len(frames)
    if matched:
        epe = contour_epe(track_x, track_y, ref_x, ref_y, spacing, key)
        per_frame, epe_mm = epe.per_frame_mm, epe.epe_mm
    else:
        per_frame, epe_mm = np.full(len(frames), np.nan), float("nan")
    curve, es, peak = _strain(track_x, track_y, spacing, es_frame)
    _save(out, track_x=track_x, track_y=track_y, epe_per_frame=per_frame, epe_mm=epe_mm,
          strain=curve, strain_es=es, strain_peak=peak, es_frame=es_frame, gls_ge=gls_ge,
          run=str(run), seen=model.seen(key))
    if flows:
        half = F.avg_pool2d(torch.from_numpy(flow).permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
        np.save(catalog.flow_file(model_id, key), half.numpy().astype(np.float16))
    return "ok" if matched else "ok (MITT length differs: no EPE)"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help=f"a run folder name, or {catalog.MITT}")
    parser.add_argument("--clips", nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--flows", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", type=Path, default=None)
    args = parser.parse_args(argv)

    keys = catalog.clip_keys() if args.all else (args.clips or [])
    model = None if args.model == catalog.MITT else catalog.models()[args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded, notes, started = {}, {}, time.time()

    def report(done, state="running", current=""):
        if args.progress:
            args.progress.parent.mkdir(parents=True, exist_ok=True)
            args.progress.write_text(json.dumps({"model": args.model, "done": done, "total": len(keys),
                                                 "current": current, "state": state, "notes": notes,
                                                 "seconds": round(time.time() - started)}), encoding="utf-8")

    for i, key in enumerate(keys):
        report(i, current=key)
        result = catalog.result_file(args.model, key)
        need_flow = args.flows and args.model != catalog.MITT and not catalog.flow_file(args.model, key).is_file()
        if result.is_file() and not args.overwrite and not need_flow:
            continue
        try:
            notes[key] = process_clip(args.model, model, key, device, loaded, args.flows)
        except Exception as error:  # one bad clip must not stop a batch
            notes[key] = f"failed: {error!r}"[:200]
        if not notes[key].startswith("ok"):
            print(f"  {key}: {notes[key]}", flush=True)
    report(len(keys), state="done")
    print(f"{args.model}: {len(keys)} clips, {time.time() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
