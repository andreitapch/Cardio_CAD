"""Probe frozen backbones for echo frame matching, before training anything.

    python -m motion_estimator.scripts.probe_backbones
    python -m motion_estimator.scripts.probe_backbones --backbones pixels dinov3_convnext_t --sim-clips 4

For every backbone in ``backbone_probe.backbones.BUILDERS`` and every pyramid
level it offers, features of two frames are matched cell by cell and scored
against the known motion (:mod:`backbone_probe.matching`): exact scatterers
on held-out simulated clips, MITT's contour on in vivo clips (a reference,
not a truth). The same pairs are used for every backbone. Also measured:
fp16 forward time per pair and peak GPU memory, the feasibility side.

Writes ``<out>/results.json`` (everything) and ``<out>/summary.md``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from ..backbone_probe import data, matching
from ..backbone_probe.backbones import ARTIFACTS, BUILDERS

MEDIANS = ("nn_epe", "soft_epe", "oracle", "zero")
COLUMNS = ("within1", "hit", "nn_epe", "soft_epe", "zero", "gain", "margin", "global1")


def collect(extractor, pairs, device, centre=False):
    acc, n, seconds = {}, 0, 0.0
    for pair in pairs:
        images = torch.from_numpy(np.stack([pair["image0"], pair["image1"]])).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        maps = extractor(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds += time.perf_counter() - started
        n += 1
        for key, fmap in maps.items():
            stride, tag = key if isinstance(key, tuple) else (key, "")
            result = matching.score(fmap[0], fmap[1], pair["points0"], pair["points1"], stride, centre)
            if not result:
                continue
            label = f"{float(stride):g}" + (f"@{tag}" if tag else "")
            slot = acc.setdefault((label, pair["spacing"]), {})
            for key, value in result.items():
                slot.setdefault(key, []).append(value.cpu())
    return acc, seconds / max(n, 1)


def summarise(acc):
    out = {}
    for (stride, spacing), metrics in acc.items():
        row = {"n": int(sum(v.numel() for v in metrics["hit"]))}
        for key, values in metrics.items():
            v = torch.cat(values)
            v = v[torch.isfinite(v)]
            row[key] = float(v.median() if key in MEDIANS else v.mean()) if v.numel() else float("nan")
        out.setdefault(stride, {})[str(spacing)] = row
    return out


def markdown(results, spacings):
    lines = ["# Backbone probe", "", f"generated {time.strftime('%Y-%m-%d %H:%M')}", "",
             "## Feasibility (fp16, one 512x512 pair)", "",
             "| backbone | params | ms/pair | peak MB | note |", "|---|---|---|---|---|"]
    for name, r in results["backbones"].items():
        if "error" in r:
            lines.append(f"| {name} | | | | FAILED: {r['error'][:80]} |")
            continue
        lines.append(f"| {name} | {r['params'] / 1e6:.1f}M | {r['ms_per_pair']:.0f} | "
                     f"{r['peak_mb']:.0f} | {r['note']} |")
    cohorts = (("simulated", "Simulated, exact scatterers"), ("in_vivo", "In vivo, vs MITT (not truth)"))
    for cohort, label in cohorts:
        lines += ["", f"## {label}", ""]
        for spacing in spacings:
            lines += [f"### frames {spacing} apart", "",
                      "| backbone | stride | n | " + " | ".join(COLUMNS) + " |",
                      "|---|---|---|" + "---|" * len(COLUMNS)]
            for name, r in results["backbones"].items():
                for stride, by_spacing in r.get(cohort, {}).items():
                    row = by_spacing.get(str(spacing))
                    if row is None:
                        continue
                    cells = [f"{row[c]:.3f}" if c in ("gain", "margin") else f"{row[c]:.2f}" for c in COLUMNS]
                    lines.append(f"| {name} | {stride} | {row['n']} | " + " | ".join(cells) + " |")
            lines.append("")
    lines += ["", "Columns: within1 / hit = fraction of queries whose best cell is within one cell of / "
              "exactly the true cell; nn_epe / soft_epe = median px error of the best cell and of the 3x3 "
              "soft-argmax; zero = median px error of predicting no motion; "
              "gain = soft estimate projected on the true motion (1 = full scale), queries "
              "moving > 1 px; margin = similarity at the true cell minus the best cell >= 2 away; "
              "global1 = best cell over the whole map within one cell of the truth."]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backbones", nargs="+", default=list(BUILDERS))
    parser.add_argument("--sim-clips", type=int, default=30)
    parser.add_argument("--invivo-clips", type=int, default=24)
    parser.add_argument("--starts", type=int, default=3, help="start frames per clip and spacing")
    parser.add_argument("--spacings", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--queries", type=int, default=512, help="scatterers per simulated pair")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--centre", action="store_true", help="subtract each map's spatial mean before matching")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.out or ARTIFACTS / "backbone-probe" / time.strftime("%Y-%m-%d_%H%M")
    out.mkdir(parents=True, exist_ok=True)
    sim_clips = data.simulated_clips(args.sim_clips, args.seed)
    iv_clips = data.invivo_clips(args.invivo_clips, seed=args.seed)
    results = {"args": {k: str(v) for k, v in vars(args).items()}, "device": str(device),
               "sim_clips": sim_clips, "invivo_clips": [p.name for p in iv_clips], "backbones": {}}
    if (out / "results.json").is_file():  # rerun some backbones into an existing probe
        results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    print(f"{len(sim_clips)} simulated clips, {len(iv_clips)} in vivo clips -> {out}", flush=True)

    for name in args.backbones:
        started = time.time()
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            extractor = BUILDERS[name](device)
            sim_pairs = data.simulated_pairs(sim_clips, args.spacings, args.starts, args.queries,
                                             args.size, args.seed)
            sim_acc, ms = collect(extractor, sim_pairs, device, args.centre)
            iv_pairs = data.invivo_pairs(iv_clips, args.spacings, args.starts, args.size, args.seed)
            iv_acc, _ = collect(extractor, iv_pairs, device, args.centre)
            peak = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else float("nan")
            results["backbones"][name] = {
                "params": extractor.params, "note": extractor.note, "ms_per_pair": ms * 1000,
                "peak_mb": peak, "simulated": summarise(sim_acc), "in_vivo": summarise(iv_acc)}
            sim = results["backbones"][name]["simulated"]
            level = sim.get("8") or sim.get("16") or next(iter(sim.values()), {})
            within = level.get("8", {}).get("within1", float("nan"))
            print(f"  {name:22s} {ms * 1000:6.0f} ms/pair  peak {peak:5.0f} MB  "
                  f"within1 (stride 8, 8 frames) {within:.2f}  {(time.time() - started) / 60:.1f} min",
                  flush=True)
            del extractor
        except Exception as error:  # one backbone failing must not cost the others
            results["backbones"][name] = {"error": f"{type(error).__name__}: {error}"}
            print(f"  {name}: FAILED {error!r}", file=sys.stderr, flush=True)
            traceback.print_exc()
        (out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        (out / "summary.md").write_text(markdown(results, args.spacings), encoding="utf-8")
    print(f"wrote {out / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
