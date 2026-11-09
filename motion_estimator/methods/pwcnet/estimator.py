"""PWC-Net behind the common :class:`MotionEstimator` interface.

Dense flow, so like SEA-RAFT it needs the sector mask and an optional
forward-backward check to be able to say "no answer". Frames are scaled by the
clip's value range to ``[0, 1]`` -- what training saw -- and zero-padded to a
multiple of 64.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ...tracking.sampling import sample_field
from .net import PWCNet

__all__ = ["PWCNetEstimator", "load_model"]


def load_model(checkpoint, device):
    model = PWCNet()
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model", state))
    return model.eval().to(device)


class PWCNetEstimator:
    """PWC-Net flow sampled at the tracked points.

    *checkpoint* is a ``best.pt`` from ``train.py`` or ``finetune_invivo.py``;
    without one the network is untrained, which is only useful for tests.
    """

    def __init__(self, sector=None, checkpoint=None, device=None,
                 consistency_px: float | None = None, name: str | None = None,
                 model=None):
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model if model is not None else load_model(checkpoint, self.device)
        self.mask = None if sector is None else np.asarray(sector, dtype=bool)
        self.consistency_px = consistency_px
        self.name = name or "pwcnet"
        self._cache = {}

    @torch.inference_mode()
    def flow(self, image0, image1, vrange):
        """``(rows, cols, 2)`` flow in pixels, x-displacement first."""
        low, high = vrange
        pair = np.stack([np.asarray(image0, np.float32), np.asarray(image1, np.float32)])
        pair = np.nan_to_num(np.clip((pair - low) / max(high - low, 1e-9), 0, 1))
        tensor = torch.from_numpy(pair)[:, None].to(self.device)
        height, width = tensor.shape[2:]
        pad_h, pad_w = (-height) % 64, (-width) % 64
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
        with torch.autocast(self.device.type, dtype=torch.float16,
                            enabled=self.device.type == "cuda"):
            output = self.model(tensor[:1], tensor[1:])
        flow = output["flow"][0, :, :height, :width].float()
        return flow.permute(1, 2, 0).cpu().numpy()

    def _flow_cached(self, frames, source, target, vrange):
        # chain() asks for each pair once, but the consistency check asks for
        # the reverse too; keep the last few so neither is computed twice.
        key = (id(frames), source, target)
        if key not in self._cache:
            if len(self._cache) > 8:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = self.flow(frames[source], frames[target], vrange)
        return self._cache[key]

    def displacement(self, frames, source, target, x, y, vrange=None):
        frames = np.asarray(frames)
        if vrange is None:
            finite = frames[np.isfinite(frames)]
            low = float(finite.min()) if finite.size else 0.0
            high = float(finite.max()) if finite.size else 1.0
            vrange = (low, high if high > low else low + 1.0)

        field = self._flow_cached(frames, source, target, vrange)
        dx, dy, inside = sample_field(field, x, y, self.mask)
        support = np.where(inside, 1, 0)

        if self.consistency_px is not None:
            back = self._flow_cached(frames, target, source, vrange)
            back_dx, back_dy, back_inside = sample_field(
                back, np.asarray(x, float) + dx, np.asarray(y, float) + dy, self.mask
            )
            consistent = back_inside & (np.hypot(dx + back_dx, dy + back_dy)
                                        <= self.consistency_px)
            inside &= consistent
            support = np.where(consistent, 2, support)

        return np.where(inside, dx, np.nan), np.where(inside, dy, np.nan), support
