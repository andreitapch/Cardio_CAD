"""SEA-RAFT behind the common interface.

Why this method, from what this project measured rather than from benchmark
rank:

* **Dense output fixes the binding constraint.** LoFTR answered for 32% of
  tracked positions in vivo and every failure was zero support -- about 640
  matches over a 500x500 frame is roughly 14 px mean spacing, so a radius-16
  neighbourhood holds ~4, below the neighbour floor. A dense field has no such
  problem by construction.
* **It is the only candidate with a learned per-pixel uncertainty.** The
  mixture-of-Laplace head emits ``info``, from which a confidence map is
  recovered. That replaces the sector mask plus forward-backward consistency
  check that a dense method otherwise needs to be able to say "no answer" --
  two heuristics of ours traded for one thing the model was trained to
  predict. The mask is still applied, because no amount of confidence makes a
  displacement outside the imaged sector meaningful.
* **It optimises the metric nearest our failure.** Published 1-pixel outlier
  rate 0.36 on Spring, the best of the family. Our problem is 0.5-2 px
  displacement, and 1px-outlier is the closest published proxy. It is only a
  proxy: no flow paper reports *gain* (systematic shrinkage of small
  displacements), which is what actually sets the strain scale, so
  ``scripts/diagnose_gain.py`` has to settle it here.

Two things to know before reading its numbers. The ``spring-M`` config sets
``scale: -1``, so the model halves the input before inference and rescales the
flow afterwards -- on 500x500 frames that is an effective 250x250, which is
worth remembering when comparing against a method that runs at full
resolution. And pretrained weights come from natural video, where sub-pixel
motion is rare; both LoFTR and plain RAFT shrank 1 px displacements to ~0.79,
so there is no reason to assume this one does not until measured.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ...tracking.sampling import sample_field

__all__ = ["SeaRaftEstimator", "VENDORED", "CONFIGS", "DEFAULT_WEIGHTS"]

VENDORED = Path(__file__).resolve().parents[3] / "third_party" / "sea_raft"
CONFIGS = VENDORED / "config" / "eval"
DEFAULT_WEIGHTS = "MemorySlices/Tartan-C-T-TSKH-spring540x960-M"
DEFAULT_CONFIG = "spring-M"


def _load_upstream():
    """Import the vendored ``RAFT``, warts and all.

    Upstream uses flat imports (``from corr import CorrBlock``) because it runs
    with ``sys.path.append('core')``. The vendored files are kept verbatim so a
    future update is a re-copy rather than a merge, which means reproducing
    that path insertion here -- once, contained, rather than rewriting the
    imports and owning the divergence.
    """
    core = str(VENDORED / "core")
    if core not in sys.path:
        sys.path.insert(0, core)
    if not (VENDORED / "core" / "raft.py").is_file():
        raise FileNotFoundError(
            f"SEA-RAFT is not vendored at {VENDORED}. See its VENDORED.md."
        )
    from raft import RAFT  # noqa: E402  -- only importable after the insert

    return RAFT


def _config(name: str) -> SimpleNamespace:
    path = CONFIGS / f"{name}.json"
    if not path.is_file():
        options = sorted(p.stem for p in CONFIGS.glob("*.json"))
        raise FileNotFoundError(f"no config {name!r}; have {options}")
    return SimpleNamespace(**json.loads(path.read_text(encoding="utf-8")))


def confidence_from_info(info, var_min, var_max):
    """The mixture-of-Laplace head's own confidence, as ``(rows, cols)``.

    Upstream's ``get_heatmap``: the two raw scale channels are clamped into
    their trained ranges and combined by the softmax weights over the mixture.
    The result is a log-scale, so **larger means less certain** -- the sign is
    flipped here so the returned map reads as confidence, which is what every
    caller wants.
    """
    raw = info[:, 2:]
    weight = info[:, :2].softmax(dim=1)
    log_b = torch.zeros_like(raw)
    log_b[:, 0] = raw[:, 0].clamp(min=0, max=var_max)
    log_b[:, 1] = raw[:, 1].clamp(min=var_min, max=0)
    return -(log_b * weight).sum(dim=1)


class SeaRaftEstimator:
    """Dense SEA-RAFT flow, sampled at the tracked points.

    *confidence_percentile* is the fraction of the **in-sector** confidence
    distribution to reject, per frame pair, rather than an absolute threshold.
    The head's output is a log scale with no calibrated units, and its spread
    differs between clips, so a fixed cut would mean different things on
    different images. ``None`` keeps everything the mask allows.

    *consistency_px* is kept as an optional second check even though the model
    predicts its own confidence, because the two catch different failures: the
    confidence is per pixel and learned on natural video, while a
    forward-backward round trip tests this pair of frames specifically.
    """

    def __init__(
        self,
        sector=None,
        checkpoint=None,
        config: str = DEFAULT_CONFIG,
        weights: str = DEFAULT_WEIGHTS,
        device=None,
        iterations: int | None = None,
        confidence_percentile: float | None = 20.0,
        consistency_px: float | None = None,
        pretrained: bool = True,
    ) -> None:
        raft = _load_upstream()
        self.args = _config(config)
        if iterations is not None:
            self.args.iters = iterations

        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if checkpoint is not None:
            model = raft(self.args)
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(state.get("model", state))
        elif pretrained:
            model = raft.from_pretrained(weights, args=self.args)
        else:
            model = raft(self.args)
        self.model = model.eval().to(self.device)

        self.config_name = config
        self.checkpoint = checkpoint
        self.mask = None if sector is None else np.asarray(sector, dtype=bool)
        self.confidence_percentile = confidence_percentile
        self.consistency_px = consistency_px
        self.name = "searaft" + ("_finetuned" if checkpoint else "")

    # -- the dense field ----------------------------------------------------

    def flow(self, image0, image1, vrange):
        """``(flow, confidence)`` at input resolution, x-displacement first."""
        import torch.nn.functional as functional

        low, high = vrange
        pair = np.stack([np.asarray(image0, np.float32), np.asarray(image1, np.float32)])
        pair = np.clip((pair - low) / max(high - low, 1e-9), 0.0, 1.0)
        pair = np.nan_to_num(pair, nan=0.0)
        # Upstream feeds 0-255 RGB, so match that rather than a [0, 1] range
        # the encoder has never seen; greyscale is repeated across channels.
        tensor = torch.from_numpy(pair)[:, None].to(self.device) * 255.0
        tensor = tensor.repeat(1, 3, 1, 1)

        scale = getattr(self.args, "scale", 0)
        if scale:
            tensor = functional.interpolate(
                tensor, scale_factor=2.0 ** scale, mode="bilinear",
                align_corners=False,
            )
        height, width = tensor.shape[2:]
        pad_h, pad_w = (-height) % 8, (-width) % 8
        if pad_h or pad_w:
            tensor = functional.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")

        with torch.inference_mode():
            output = self.model(
                tensor[:1], tensor[1:], iters=self.args.iters, test_mode=True
            )
        flow = output["flow"][-1][:, :, :height, :width]
        info = output["info"][-1][:, :, :height, :width]
        confidence = confidence_from_info(
            info, getattr(self.args, "var_min", 0), getattr(self.args, "var_max", 10)
        )[:, None]

        if scale:
            factor = 0.5 ** scale
            flow = functional.interpolate(
                flow, scale_factor=factor, mode="bilinear", align_corners=False
            ) * factor
            confidence = functional.interpolate(
                confidence, scale_factor=factor, mode="area"
            )

        return (
            flow[0].permute(1, 2, 0).float().cpu().numpy(),
            confidence[0, 0].float().cpu().numpy(),
        )

    # -- the interface ------------------------------------------------------

    def displacement(self, frames, source, target, x, y, vrange=None):
        frames = np.asarray(frames)
        if vrange is None:
            finite = np.isfinite(frames)
            low = float(frames[finite].min()) if finite.any() else 0.0
            high = float(frames[finite].max()) if finite.any() else 1.0
            vrange = (low, high if high > low else low + 1.0)

        field, confidence = self.flow(frames[source], frames[target], vrange)
        dx, dy, inside = sample_field(field, x, y, self.mask)
        support = np.where(inside, 1, 0)

        if self.confidence_percentile is not None:
            # Relative to this pair's own in-sector distribution: the head's
            # output is an uncalibrated log scale whose spread varies by clip,
            # so an absolute cut would mean different things on different
            # images.
            region = confidence if self.mask is None else confidence[self.mask]
            region = region[np.isfinite(region)]
            if region.size:
                floor = float(np.percentile(region, self.confidence_percentile))
                at_point, _, _ = sample_field(
                    np.stack([confidence, confidence], axis=-1), x, y, self.mask
                )
                confident = np.isfinite(at_point) & (at_point >= floor)
                inside &= confident
                support = np.where(confident, 2, support)

        if self.consistency_px is not None:
            back_field, _ = self.flow(frames[target], frames[source], vrange)
            back_dx, back_dy, back_inside = sample_field(
                back_field, np.asarray(x, float) + dx, np.asarray(y, float) + dy,
                self.mask,
            )
            round_trip = np.hypot(dx + back_dx, dy + back_dy)
            consistent = back_inside & (round_trip <= self.consistency_px)
            inside &= consistent
            support = np.where(consistent, support + 1, support)

        return np.where(inside, dx, np.nan), np.where(inside, dy, np.nan), support
