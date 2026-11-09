"""Frames in, L2-normalised EchoCLIP embeddings out.

The trunk is frozen throughout -- see the package docstring. Everything here
runs under ``torch.no_grad``.

Preprocessing follows echonet/echo_CLIP's ``zero_shot_example.py``: each frame
becomes a PIL image, goes through the checkpoint's own ``preprocess_val``
(``Resize(224) -> CenterCrop(224) -> RGB -> ToTensor -> Normalize``), and the
resulting embedding is L2-normalised so a dot product is a cosine similarity.
Our frames are single-channel and square, so they are replicated to three
channels and the centre crop discards nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import EMBED_DIM, EncoderConfig

__all__ = ["EchoClipEncoder", "select_frame_indices"]

# Compute capability at which bf16 stops being emulated. Below this, the
# EchoCLIP convolutions raise rather than fall back.
_BF16_MIN_CAPABILITY = (8, 0)


def select_frame_indices(
    n_available: int, n_frames: int, sampling: str = "uniform"
) -> np.ndarray:
    """Which frames of a cycle to encode.

    ``uniform`` always returns exactly *n_frames* indices, repeating frames if
    the clip is shorter, so the cache stays rectangular and downstream code
    never has to handle ragged clips. ``upstream`` returns however many the
    original stride-2 rule yields, which is at most *n_frames*.
    """
    if n_available < 1:
        raise ValueError(f"clip has no frames (n_available={n_available})")

    if sampling == "uniform":
        return np.linspace(0, n_available - 1, n_frames).round().astype(int)
    if sampling == "upstream":
        return np.arange(0, min(2 * n_frames, n_available), 2)
    raise ValueError(f"unknown sampling strategy {sampling!r}")


@dataclass
class EchoClipEncoder:
    """A frozen EchoCLIP, loaded lazily so importing this module is cheap.

    Lazy loading matters: the cohort index, the tests and the CLI all import
    this, and none of them should pull 600 MB of weights onto the GPU to do it.
    """

    config: EncoderConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = EncoderConfig()
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    # ------------------------------------------------------------------ setup
    def load(self) -> "EchoClipEncoder":
        """Load weights onto the configured device. Idempotent."""
        if self._model is not None:
            return self

        import open_clip
        import torch

        device = self._resolve_device(torch)
        self._check_precision(torch, device)

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.config.model_id, precision=self.config.precision, device=device
        )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.config.model_id)
        self._device = device
        return self

    def _resolve_device(self, torch) -> str:
        if self.config.device != "cuda":
            return self.config.device
        if not torch.cuda.is_available():
            raise RuntimeError(
                "config.device is 'cuda' but torch.cuda.is_available() is False. "
                "Install a CUDA build of torch, or set device='cpu' "
                "(and precision='fp32')."
            )
        return "cuda"

    def _check_precision(self, torch, device: str) -> None:
        """Reject bf16 where it would raise deep inside the forward pass.

        torch.cuda.is_bf16_supported() counts emulation as support and returns
        True on Turing, so it cannot be used for this -- the capability is
        checked directly instead.
        """
        if self.config.precision != "bf16" or device != "cuda":
            return
        capability = torch.cuda.get_device_capability(0)
        if capability < _BF16_MIN_CAPABILITY:
            raise RuntimeError(
                f"precision='bf16' on {torch.cuda.get_device_name(0)} "
                f"(compute capability {capability[0]}.{capability[1]}): the "
                "convolutions have no bf16 kernel below "
                f"{_BF16_MIN_CAPABILITY[0]}.{_BF16_MIN_CAPABILITY[1]} and will "
                "raise mid-forward. Use precision='fp16'."
            )

    @property
    def embed_dim(self) -> int:
        return EMBED_DIM

    @property
    def device(self) -> str:
        self.load()
        return self._device

    # --------------------------------------------------------------- encoding
    def preprocess_frames(self, frames: np.ndarray):
        """``(n, h, w)`` uint8 -> ``(n, 3, 224, 224)`` on the model's device."""
        import torch
        import torchvision.transforms as transforms

        self.load()
        if frames.ndim != 3:
            raise ValueError(
                f"expected (n_frames, h, w), got shape {frames.shape}. Use "
                "image_prep.frames_first on a PreparedClip."
            )

        to_pil = transforms.ToPILImage()
        batch = torch.stack(
            [
                self._preprocess(to_pil(np.repeat(frame[:, :, None], 3, axis=2)))
                for frame in frames
            ]
        )
        return batch.to(self._device).to(self._torch_dtype(torch))

    def _torch_dtype(self, torch):
        return {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
            "amp": torch.float16,
        }.get(self.config.precision, torch.float32)

    def encode_clip(self, frames: np.ndarray) -> np.ndarray:
        """One cycle's frames -> ``(n_frames, 512)`` float32, L2-normalised.

        *frames* is the full cycle in ``(n_frames, h, w)`` order; this selects
        from it according to the configured sampling strategy.
        """
        import torch
        import torch.nn.functional as functional

        self.load()
        indices = select_frame_indices(
            len(frames), self.config.n_frames, self.config.sampling
        )
        with torch.no_grad():
            batch = self.preprocess_frames(frames[indices])
            embeddings = functional.normalize(self._model.encode_image(batch), dim=-1)
        return embeddings.float().cpu().numpy()

    def encode_text(self, prompts: Sequence[str]) -> np.ndarray:
        """Prompts -> ``(n_prompts, 512)`` float32, L2-normalised.

        Used by the zero-shot baselines. The tokenizer comes from
        ``open_clip.get_tokenizer(model_id)`` rather than the module-level
        ``open_clip.tokenize`` in the upstream examples, so the checkpoint's own
        tokenizer config is honoured.
        """
        import torch
        import torch.nn.functional as functional

        self.load()
        with torch.no_grad():
            tokens = self._tokenizer(list(prompts)).to(self._device)
            embeddings = functional.normalize(self._model.encode_text(tokens), dim=-1)
        return embeddings.float().cpu().numpy()
