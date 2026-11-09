"""Tests for frame selection and encoder configuration.

The parts that need weights are covered by test_echoclip_live.py; everything
here runs with no GPU, no network and no torch.

Run with:  python -m pytest motion_estimator/tests/test_encoder.py
"""
from __future__ import annotations

import numpy as np
import pytest

from motion_estimator.echoclip.config import EncoderConfig
from motion_estimator.echoclip.encoder import select_frame_indices


def test_uniform_spans_the_whole_cycle():
    """End-systole is the target, but the head still sees all of diastole."""
    indices = select_frame_indices(44, 20, "uniform")

    assert len(indices) == 20
    assert indices[0] == 0
    assert indices[-1] == 43


def test_uniform_is_always_exactly_n_frames():
    """A rectangular cache means downstream code never handles ragged clips."""
    for n_available in (1, 5, 20, 44, 66, 300):
        assert len(select_frame_indices(n_available, 20, "uniform")) == 20


def test_uniform_repeats_frames_for_a_short_clip():
    indices = select_frame_indices(5, 20, "uniform")

    assert len(indices) == 20
    assert indices.min() == 0
    assert indices.max() == 4
    assert len(np.unique(indices)) == 5


def test_uniform_is_monotonic():
    indices = select_frame_indices(66, 20, "uniform")

    assert (np.diff(indices) >= 0).all()


def test_upstream_matches_the_original_slice():
    """frames[0:min(2*n, len):2], verbatim from echo_CLIP."""
    for n_available in (10, 44, 66, 300):
        expected = np.arange(n_available)[0 : min(40, n_available) : 2]
        np.testing.assert_array_equal(
            select_frame_indices(n_available, 20, "upstream"), expected
        )


def test_upstream_truncates_a_long_cycle_but_uniform_does_not():
    """The reason uniform is the default: our clips are one cardiac cycle."""
    upstream = select_frame_indices(66, 20, "upstream")
    uniform = select_frame_indices(66, 20, "uniform")

    assert upstream.max() == 38  # late diastole never reaches the model
    assert uniform.max() == 65


def test_upstream_yields_fewer_frames_for_a_short_clip():
    assert len(select_frame_indices(10, 20, "upstream")) == 5


def test_empty_clip_is_rejected():
    with pytest.raises(ValueError, match="no frames"):
        select_frame_indices(0, 20, "uniform")


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown sampling"):
        select_frame_indices(44, 20, "every-other-heartbeat")


def test_config_rejects_an_unknown_strategy():
    with pytest.raises(ValueError, match="sampling must be"):
        EncoderConfig(sampling="nope")


def test_config_rejects_a_nonsense_frame_count():
    with pytest.raises(ValueError, match="n_frames"):
        EncoderConfig(n_frames=0)


def test_fingerprint_ignores_device_but_tracks_the_rest():
    """CPU and GPU give the same embeddings; sampling does not."""
    cuda = EncoderConfig(device="cuda").fingerprint()
    cpu = EncoderConfig(device="cpu").fingerprint()

    assert cuda == cpu
    assert "device" not in cuda
    assert cuda != EncoderConfig(sampling="upstream").fingerprint()
    assert cuda != EncoderConfig(n_frames=10).fingerprint()
    assert cuda != EncoderConfig(precision="fp32").fingerprint()


def test_encoder_defaults_are_constructed_lazily():
    """Importing and constructing must not touch torch or the network."""
    from motion_estimator.echoclip.encoder import EchoClipEncoder

    encoder = EchoClipEncoder()

    assert encoder.config == EncoderConfig()
    assert encoder.embed_dim == 512
    assert encoder._model is None
