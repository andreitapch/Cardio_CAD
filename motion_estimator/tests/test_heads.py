"""Tests for the trainable heads and the training loop.

Run with:  python -m pytest motion_estimator/tests/test_heads.py
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from motion_estimator.echoclip.heads import (  # noqa: E402
    AttnPoolHead,
    MeanPoolHead,
    TargetScaler,
    build_head,
    masked_mse,
)
from motion_estimator.echoclip.train import TrainConfig, train_cv  # noqa: E402
from .test_probe import make_dataset  # noqa: E402


# --------------------------------------------------------------- masked loss
def test_masked_mse_ignores_masked_entries():
    prediction = torch.zeros(2, 7)
    target = torch.full((2, 7), float("nan"))
    target[:, 0] = 2.0
    mask = torch.zeros(2, 7)
    mask[:, 0] = 1.0

    assert masked_mse(prediction, target, mask).item() == pytest.approx(4.0)


def test_masked_mse_does_not_let_nan_poison_the_gradient():
    prediction = torch.zeros(2, 7, requires_grad=True)
    target = torch.full((2, 7), float("nan"))
    target[:, 0] = 1.0
    mask = torch.zeros(2, 7)
    mask[:, 0] = 1.0

    masked_mse(prediction, target, mask).backward()

    assert torch.isfinite(prediction.grad).all()


def test_masked_mse_of_an_empty_mask_is_zero():
    loss = masked_mse(torch.ones(2, 7), torch.ones(2, 7), torch.zeros(2, 7))

    assert loss.item() == 0.0


def test_masked_mse_averages_over_the_mask_not_the_tensor():
    """Otherwise a clip with few valid targets would be quietly down-weighted."""
    prediction = torch.zeros(1, 7)
    target = torch.full((1, 7), float("nan"))
    target[0, :2] = 3.0
    mask = torch.zeros(1, 7)
    mask[0, :2] = 1.0

    assert masked_mse(prediction, target, mask).item() == pytest.approx(9.0)


# -------------------------------------------------------------------- scaler
def test_scaler_roundtrips():
    rng = np.random.default_rng(0)
    targets = rng.normal(size=(20, 7)) * 5 + 2
    scaler = TargetScaler().fit(targets)

    np.testing.assert_allclose(
        scaler.inverse_transform(scaler.transform(targets)), targets
    )


def test_scaler_is_nan_aware():
    targets = np.ones((10, 7))
    targets[:5, 0] = np.nan
    targets[5:, 0] = 4.0

    scaler = TargetScaler().fit(targets)

    assert scaler.mean[0] == pytest.approx(4.0)


def test_scaler_survives_a_constant_target():
    scaler = TargetScaler().fit(np.ones((5, 7)))

    assert np.isfinite(scaler.scale).all()
    assert (scaler.scale > 0).all()


def test_scaler_survives_an_all_nan_target():
    targets = np.ones((5, 7))
    targets[:, 3] = np.nan

    scaler = TargetScaler().fit(targets)

    assert np.isfinite(scaler.mean).all()
    assert np.isfinite(scaler.scale).all()


# --------------------------------------------------------------------- heads
def test_heads_produce_one_row_of_seven_targets():
    embeddings = torch.randn(4, 10, 16)
    views = torch.zeros(4, 2)

    for head in (MeanPoolHead(16), AttnPoolHead(16, hidden=8)):
        assert head(embeddings, views).shape == (4, 7)


def test_attention_weights_are_a_distribution_over_frames():
    head = AttnPoolHead(16, hidden=8)

    weights = head.attention(torch.randn(4, 10, 16))

    assert weights.shape == (4, 10)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(4))
    assert (weights >= 0).all()


def test_attention_pooling_is_not_just_the_mean():
    """Otherwise the extra parameters buy nothing by construction."""
    torch.manual_seed(0)
    head = AttnPoolHead(16, hidden=8)
    embeddings = torch.randn(4, 10, 16)

    assert not torch.allclose(
        head.pool(embeddings), embeddings.mean(dim=1), atol=1e-4
    )


def test_uniform_attention_would_reproduce_the_mean():
    """A sanity anchor on the einsum: equal weights must give mean pooling."""
    head = AttnPoolHead(16, hidden=8)
    embeddings = torch.randn(2, 5, 16)
    weights = torch.full((2, 5), 0.2)

    pooled = torch.einsum("bf,bfd->bd", weights, embeddings)

    torch.testing.assert_close(pooled, embeddings.mean(dim=1))


def test_build_head_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown head"):
        build_head("transformer", 16)


# ------------------------------------------------------------------ training
def _fast_config(**kwargs):
    return TrainConfig(epochs=20, patience=5, **kwargs)


def test_training_produces_an_out_of_fold_prediction_for_every_clip():
    data = make_dataset(n_patients=10)

    result = train_cv(data, _fast_config(), n_splits=3)

    assert np.isfinite(result.predictions).all()
    assert result.predictions.shape == data.targets.shape


def test_no_group_is_in_both_sides_of_a_training_split():
    data = make_dataset(n_patients=10)

    result = train_cv(data, _fast_config(), n_splits=3)

    for fold in result.folds:
        test_groups = set(data.groups[fold.indices])
        others = np.setdiff1d(np.arange(len(data)), fold.indices)
        assert not (test_groups & set(data.groups[others]))


def test_training_handles_partially_missing_targets():
    data = make_dataset(n_patients=10)
    data.targets[::3, 4] = np.nan

    result = train_cv(data, _fast_config(), n_splits=3)

    assert result.metrics[4].n < len(data)
    assert np.isfinite(result.predictions[:, 0]).all()


def test_a_never_observed_target_is_not_scored():
    data = make_dataset(n_patients=10)
    data.targets[:, 3] = np.nan

    result = train_cv(data, _fast_config(), n_splits=3)

    assert result.metrics[3].n == 0
    assert np.isnan(result.predictions[:, 3]).all()


def test_training_is_reproducible_for_a_fixed_seed():
    data = make_dataset(n_patients=10)

    first = train_cv(data, _fast_config(seed=7), n_splits=3)
    second = train_cv(data, _fast_config(seed=7), n_splits=3)

    np.testing.assert_allclose(first.predictions, second.predictions)


def test_both_head_types_train():
    data = make_dataset(n_patients=10)

    for name in ("mean", "attn"):
        result = train_cv(data, _fast_config(head=name), n_splits=3)
        assert result.name == f"{name}-head"
        assert np.isfinite(result.predictions).all()


def test_training_learns_something_on_learnable_data():
    """Not a claim about the cohort -- a check that the loop optimises at all."""
    data = make_dataset(n_patients=25)

    trained = train_cv(data, TrainConfig(epochs=300, patience=50), n_splits=4)
    from motion_estimator.echoclip.probe import run_cv

    baseline = run_cv(data, "mean", n_splits=4)

    assert trained.metrics[-1].mae < baseline.metrics[-1].mae
