"""LoFTR on a frame pair -- semi-dense correspondences with confidences.

Wraps ``kornia.feature.LoFTR``. What the wrapper is for, beyond the two lines
of kornia API:

* **Sizes.** LoFTR's backbone strides by 8, so both images must have height and
  width divisible by 8. Our clinical-view frames are 500x500. Padding to 504
  keeps every pixel at its original coordinate (pad at the bottom/right only),
  which is why it is preferred here over resizing; matches landing in the pad
  are dropped rather than trusted.
* **Intensity range.** LoFTR wants greyscale in [0, 1]. Normalising each frame
  by its own min/max would rescale the image differently from frame to frame,
  which is a change in appearance the matcher can see. One range is therefore
  computed over the whole clip and reused -- pass ``vrange`` explicitly, or let
  :meth:`LoFTRMatcher.match_sequence` derive it.
* **Sector mask.** The 500x500 frames have black corners outside the
  ultrasound sector. Matches there are matches between two regions of constant
  zero; they can come back confident with an arbitrary displacement, so
  anything outside the mask is discarded.

The pretrained weights are ``outdoor``, trained on MegaDepth photographs. That
is a domain gap to echocardiographic speckle, and no claim is made here that it
is small -- this module exists so it can be measured. ``indoor`` (ScanNet) is
selectable for the same reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch

__all__ = ["Matches", "LoFTRMatcher", "value_range"]


@dataclass
class Matches:
    """Correspondences for one frame pair.

    ``xy0`` / ``xy1`` are ``(n_matches, 2)`` arrays of **0-based** ``(x, y)``
    pixel coordinates -- LoFTR's own convention, kept as-is so this class stays
    a faithful record of the matcher's output. The conversion to the package's
    1-based coordinates happens in :mod:`point_tracking`.
    """

    xy0: np.ndarray
    xy1: np.ndarray
    confidence: np.ndarray

    def __post_init__(self) -> None:
        self.xy0 = np.asarray(self.xy0, dtype=np.float64)
        self.xy1 = np.asarray(self.xy1, dtype=np.float64)
        self.confidence = np.asarray(self.confidence, dtype=np.float64)
        if self.xy0.shape != self.xy1.shape:
            raise ValueError(f"xy0 {self.xy0.shape} != xy1 {self.xy1.shape}")
        if self.xy0.ndim != 2 or self.xy0.shape[-1] != 2:
            raise ValueError(f"expected (n_matches, 2), got {self.xy0.shape}")
        if self.confidence.shape != (self.xy0.shape[0],):
            raise ValueError(
                f"confidence {self.confidence.shape} does not match "
                f"{self.xy0.shape[0]} matches"
            )

    @property
    def n_matches(self) -> int:
        return int(self.xy0.shape[0])

    @property
    def displacement(self) -> np.ndarray:
        """``xy1 - xy0``, ``(n_matches, 2)`` with the x-displacement first."""
        return self.xy1 - self.xy0

    def filter(self, keep) -> "Matches":
        """A new :class:`Matches` holding the subset selected by *keep*."""
        keep = np.asarray(keep)
        return Matches(self.xy0[keep], self.xy1[keep], self.confidence[keep])


def value_range(frames) -> tuple[float, float]:
    """The ``(min, max)`` intensity over a whole clip, for a stable scaling.

    NaNs are ignored. A constant clip returns a unit-width range so the
    normalisation stays finite.
    """
    array = np.asarray(frames, dtype=np.float64)
    low = float(np.nanmin(array)) if np.isfinite(array).any() else 0.0
    high = float(np.nanmax(array)) if np.isfinite(array).any() else 1.0
    if not np.isfinite(low):
        low = 0.0
    if not np.isfinite(high) or high <= low:
        high = low + 1.0
    return low, high


def _as_float_array(image) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] in (1, 3):
        # Colour frames are not expected, but tolerate them by luminance.
        weights = np.array([0.299, 0.587, 0.114])[: array.shape[-1]]
        array = (array.astype(np.float64) * weights).sum(-1) / weights.sum()
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D image, got {array.shape}")
    return array.astype(np.float32)


class LoFTRMatcher:
    """Match frame pairs with LoFTR.

    Parameters
    ----------
    weights:
        ``"outdoor"`` (MegaDepth) or ``"indoor"`` (ScanNet).
    device:
        Defaults to CUDA when available.
    min_confidence:
        Drop matches below this. LoFTR's own dual-softmax threshold already
        removes most of the tail; this is a second, adjustable cut.
    mask:
        Optional ``(rows, cols)`` boolean array, True inside the ultrasound
        sector. Matches whose endpoint falls outside it in *either* frame are
        dropped.
    scale:
        Downscale factor applied before matching, with keypoints mapped back to
        full-resolution coordinates. ``1.0`` (the default) matches at native
        resolution; ``0.5`` is ~4x cheaper and loses the finest speckle.
    half:
        Run in fp16. On this project's Turing card bf16 raises inside cuDNN --
        see ``docs/environment-notes.md`` -- so fp16 is the only reduced
        precision offered.
    """

    def __init__(
        self,
        weights: str = "outdoor",
        device: str | torch.device | None = None,
        min_confidence: float = 0.2,
        mask: np.ndarray | None = None,
        scale: float = 1.0,
        half: bool = False,
        checkpoint=None,
    ) -> None:
        if not 0.0 < scale <= 1.0:
            raise ValueError(f"scale must be in (0, 1], got {scale}")
        from kornia.feature import LoFTR  # imported late: heavy, and optional

        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = LoFTR(pretrained=weights)
        if checkpoint is not None:
            # The pretrained weights load first, so anything a fine-tune left
            # untouched keeps its upstream values instead of going random.
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.model.load_state_dict(state.get("model", state))
        self.checkpoint = checkpoint
        self.model = self.model.eval().to(self.device)
        if half:
            self.model = self.model.half()
        self.half = half
        self.weights = weights
        self.min_confidence = float(min_confidence)
        self.scale = float(scale)
        self.mask = None if mask is None else np.asarray(mask, dtype=bool)

    # -- tensor preparation -------------------------------------------------

    def _prepare(self, image, vrange) -> tuple[torch.Tensor, tuple[int, int]]:
        array = _as_float_array(image)
        rows, cols = array.shape
        low, high = vrange
        array = np.clip((array - low) / (high - low), 0.0, 1.0)
        array = np.nan_to_num(array, nan=0.0)

        tensor = torch.from_numpy(array)[None, None].to(self.device)
        if self.scale != 1.0:
            tensor = torch.nn.functional.interpolate(
                tensor,
                scale_factor=self.scale,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        # LoFTR strides by 8; pad bottom/right so pixel (0, 0) stays put and
        # every original coordinate keeps its value.
        _, _, height, width = tensor.shape
        pad_h = (-height) % 8
        pad_w = (-width) % 8
        if pad_h or pad_w:
            tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h))
        if self.half:
            tensor = tensor.half()
        return tensor, (rows, cols)

    # -- matching -----------------------------------------------------------

    def match(self, image0, image1, vrange=None) -> Matches:
        """Match one pair, returning full-resolution 0-based ``(x, y)``.

        *vrange* is the intensity range to normalise by; when omitted it is
        taken over the two images, which is fine for a one-off pair but not for
        a sequence -- use :meth:`match_sequence` there.
        """
        if vrange is None:
            vrange = value_range(
                np.stack([_as_float_array(image0), _as_float_array(image1)])
            )
        tensor0, shape0 = self._prepare(image0, vrange)
        tensor1, shape1 = self._prepare(image1, vrange)

        with torch.inference_mode():
            out = self.model({"image0": tensor0, "image1": tensor1})

        xy0 = out["keypoints0"].float().cpu().numpy()
        xy1 = out["keypoints1"].float().cpu().numpy()
        confidence = out["confidence"].float().cpu().numpy()
        if self.scale != 1.0:
            xy0 = xy0 / self.scale
            xy1 = xy1 / self.scale

        return self._clean(Matches(xy0, xy1, confidence), shape0, shape1)

    def match_sequence(
        self,
        frames,
        reference: int | None = None,
        vrange=None,
        progress: bool = False,
    ) -> list[Matches]:
        """Match every consecutive pair, or every frame against *reference*.

        With ``reference=None`` the result has ``n_frames - 1`` entries, entry
        *k* matching frame *k* to *k+1*. With ``reference=r`` it has
        ``n_frames`` entries, entry *k* matching frame *r* to frame *k* (entry
        *r* is an empty identity match, kept so the indexing stays
        frame-aligned).
        """
        frames = np.asarray(frames)
        if frames.ndim != 3:
            raise ValueError(f"expected (n_frames, rows, cols), got {frames.shape}")
        if vrange is None:
            vrange = value_range(frames)

        pairs: Sequence[tuple[int, int]]
        if reference is None:
            pairs = [(k, k + 1) for k in range(frames.shape[0] - 1)]
        else:
            if not 0 <= reference < frames.shape[0]:
                raise ValueError(
                    f"reference {reference} outside 0..{frames.shape[0] - 1}"
                )
            pairs = [(reference, k) for k in range(frames.shape[0])]

        iterator: Iterable = pairs
        if progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(pairs, desc="LoFTR", unit="pair")
            except ImportError:
                pass

        results = []
        for source, target in iterator:
            if source == target:
                empty = np.zeros((0, 2))
                results.append(Matches(empty, empty.copy(), np.zeros(0)))
                continue
            results.append(self.match(frames[source], frames[target], vrange))
        return results

    # -- filtering ----------------------------------------------------------

    def _clean(self, matches: Matches, shape0, shape1) -> Matches:
        keep = matches.confidence >= self.min_confidence
        keep &= _inside(matches.xy0, shape0)
        keep &= _inside(matches.xy1, shape1)
        if self.mask is not None:
            keep &= _in_mask(matches.xy0, self.mask)
            keep &= _in_mask(matches.xy1, self.mask)
        return matches.filter(keep)


def _inside(xy: np.ndarray, shape) -> np.ndarray:
    """True where a 0-based ``(x, y)`` lies within the unpadded image."""
    rows, cols = shape
    return (
        (xy[:, 0] >= 0)
        & (xy[:, 0] <= cols - 1)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] <= rows - 1)
    )


def _in_mask(xy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """True where the nearest pixel to a 0-based ``(x, y)`` is inside *mask*."""
    rows, cols = mask.shape
    col = np.rint(xy[:, 0]).astype(int).clip(0, cols - 1)
    row = np.rint(xy[:, 1]).astype(int).clip(0, rows - 1)
    return mask[row, col]
