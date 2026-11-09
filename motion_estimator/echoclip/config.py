"""Settings the embedding cache is keyed on.

Anything here that changes the numbers belongs in :class:`EncoderConfig`, so a
stale cache is detected rather than silently reused --
:func:`motion_estimator.echoclip.embed_cache.load_embeddings` compares the
stored fingerprint against the current one.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = [
    "MODEL_ID",
    "EMBED_DIM",
    "IMAGE_SIZE",
    "DEFAULT_CACHE_DIR",
    "SAMPLING_STRATEGIES",
    "EncoderConfig",
]

# EchoCLIP. MIT licensed and not gated, so no HF_TOKEN is needed; open_clip
# reads open_clip_config.json + open_clip_pytorch_model.bin from the hub.
MODEL_ID = "hf-hub:mkaichristensen/echo-clip"

# Verified by loading the checkpoint, not read off the paper.
EMBED_DIM = 512
IMAGE_SIZE = 224

DEFAULT_CACHE_DIR = Path("motion_estimator/artifacts/embeddings")

SAMPLING_STRATEGIES = ("uniform", "upstream")


@dataclass(frozen=True)
class EncoderConfig:
    """Everything that affects an embedding's value.

    ``precision`` must be ``fp16`` on Turing. bf16 is not merely slow there --
    the convolutions raise ``RuntimeError: GET was unable to find an engine to
    execute this computation`` -- and ``torch.cuda.is_bf16_supported()`` returns
    True anyway because it counts emulation, so it cannot be used to choose.
    :class:`~motion_estimator.echoclip.encoder.EchoClipEncoder` checks the
    compute capability directly.

    ``sampling`` picks how ``n_frames`` frames are drawn from a cardiac cycle:

    ``uniform`` (default)
        Spread evenly across the whole cycle. Our clips are cropped to exactly
        one cycle and the target is *end-systolic* strain, so the cycle is the
        unit of meaning and all of it should be represented.
    ``upstream``
        ``frames[0:min(2 * n_frames, len(frames)):2]``, verbatim from
        echonet/echo_CLIP's ``zero_shot_example.py``. That was written for
        arbitrary-length clinical loops, and on our 44-66 frame cycles it
        truncates late diastole. Kept so the two can be compared rather than
        assumed equivalent.
    """

    model_id: str = MODEL_ID
    precision: str = "fp16"
    device: str = "cuda"
    n_frames: int = 20
    sampling: str = "uniform"

    def __post_init__(self) -> None:
        if self.sampling not in SAMPLING_STRATEGIES:
            raise ValueError(
                f"sampling must be one of {list(SAMPLING_STRATEGIES)}, "
                f"got {self.sampling!r}"
            )
        if self.n_frames < 1:
            raise ValueError(f"n_frames must be >= 1, got {self.n_frames}")

    def fingerprint(self) -> dict:
        """The dict written beside a cache so a stale one can be spotted.

        ``device`` is deliberately excluded: CPU and GPU produce the same
        embeddings to within float tolerance, so it is not worth invalidating a
        cache over.
        """
        data = asdict(self)
        data.pop("device")
        return data
