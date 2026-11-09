"""Tests for the strain reconstruction port.

The decisive test is at the bottom: MITT stores tracked points and the strain
curves it derived from them, so a correct port turns one into the other. That
runs against real data when the drive is mounted. Everything above it pins the
arithmetic on cases whose answer can be worked out by hand.

Run with:  python -m pytest motion_estimator/tests/test_strain.py
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from motion_estimator.strain.arclength import (  # noqa: E402
    arc_length,
    contour_arc_length,
    layer_indices,
)
from motion_estimator.strain.curve import (  # noqa: E402
    compute_strain,
    drift_compensation,
    strain_from_arc_length,
)
from motion_estimator.strain.propagate import (  # noqa: E402
    advect_points,
    propagate_points,
    sample_bilinear,
)


# ----------------------------------------------------------------- arc length
def test_arc_length_of_a_straight_line():
    x = torch.tensor([[0.0, 3.0, 6.0]])
    y = torch.tensor([[0.0, 4.0, 8.0]])

    assert float(arc_length(x, y)) == pytest.approx(10.0)


def test_arc_length_is_summed_chords_not_endpoint_distance():
    """A zig-zag has a longer path than the distance between its ends."""
    x = torch.tensor([[0.0, 1.0, 2.0]])
    y = torch.tensor([[0.0, 1.0, 0.0]])

    assert float(arc_length(x, y)) == pytest.approx(2 * np.sqrt(2))


def test_arc_length_needs_two_points():
    with pytest.raises(ValueError, match="at least 2 points"):
        arc_length(torch.zeros(1, 1), torch.zeros(1, 1))


def test_open_contour_does_not_close_the_ring():
    """Three corners of a unit square: open is 2, closed adds the diagonal."""
    x = torch.tensor([[0.0, 1.0, 1.0]])
    y = torch.tensor([[0.0, 0.0, 1.0]])

    assert float(contour_arc_length(x, y, open_contour=True)) == pytest.approx(2.0)
    assert float(contour_arc_length(x, y, open_contour=False)) == pytest.approx(
        2.0 + np.sqrt(2)
    )


def test_arc_length_per_frame():
    x = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
    y = torch.zeros(3, 2)

    np.testing.assert_allclose(
        contour_arc_length(x, y).numpy(), [1.0, 2.0, 3.0]
    )


@pytest.mark.parametrize(
    "layer,n_layers,expected",
    [
        ("endo", 5, [0]),
        ("epi", 5, [4]),
        ("mid", 5, [2]),
        ("mid", 4, [1, 2]),  # even count averages the middle two
        ("all", 3, [0, 1, 2]),
    ],
)
def test_layer_selection_matches_the_matlab_switch(layer, n_layers, expected):
    assert layer_indices(layer, n_layers) == expected


def test_unknown_layer_is_rejected():
    with pytest.raises(ValueError, match="layer must be"):
        layer_indices("myocardium", 5)


def test_endo_is_the_first_layer():
    x = torch.zeros(1, 2, 3)
    y = torch.zeros(1, 2, 3)
    x[0, 1, 0] = 1.0  # layer 0 spans 1
    x[0, 1, 2] = 9.0  # layer 2 spans 9

    assert float(contour_arc_length(x, y, layer="endo")) == pytest.approx(1.0)
    assert float(contour_arc_length(x, y, layer="epi")) == pytest.approx(9.0)


def test_all_layers_are_averaged():
    x = torch.zeros(1, 2, 2)
    y = torch.zeros(1, 2, 2)
    x[0, 1, 0], x[0, 1, 1] = 2.0, 4.0

    assert float(contour_arc_length(x, y, layer="all")) == pytest.approx(3.0)


# ---------------------------------------------------------------- strain curve
def test_strain_is_relative_to_the_first_frame():
    curve = strain_from_arc_length(torch.tensor([10.0, 9.0, 11.0]), drift_comp=False)

    np.testing.assert_allclose(curve.numpy(), [0.0, -10.0, 10.0])


def test_shortening_is_negative():
    curve = strain_from_arc_length(torch.tensor([100.0, 80.0]), drift_comp=False)

    assert float(curve[1]) < 0


def test_drift_compensation_lands_the_last_frame_on_zero():
    curve = torch.tensor([0.0, -5.0, -8.0, -3.0])

    compensated = drift_compensation(curve)

    assert float(compensated[-1]) == pytest.approx(0.0)
    assert float(compensated[0]) == pytest.approx(0.0)


def test_drift_compensation_subtracts_a_linear_ramp():
    """y = x - x[-1] * i/(n-1), verbatim from driftCompensation.m."""
    curve = torch.tensor([0.0, 1.0, 2.0, 6.0])

    compensated = drift_compensation(curve).numpy()

    np.testing.assert_allclose(compensated, [0.0, 1.0 - 2.0, 2.0 - 4.0, 0.0])


def test_drift_compensation_is_a_no_op_on_a_closed_curve():
    curve = torch.tensor([0.0, -4.0, -2.0, 0.0])

    np.testing.assert_allclose(drift_compensation(curve).numpy(), curve.numpy())


def test_nan_frames_are_skipped_and_restored():
    lengths = torch.tensor([10.0, float("nan"), 8.0])

    curve = strain_from_arc_length(lengths, drift_comp=False).numpy()

    assert np.isnan(curve[1])
    np.testing.assert_allclose(curve[[0, 2]], [0.0, -20.0])


def test_all_nan_arc_length_gives_an_all_nan_curve():
    curve = strain_from_arc_length(torch.tensor([float("nan")] * 3))

    assert np.isnan(curve.numpy()).all()


def test_reference_is_the_first_valid_frame_not_frame_zero():
    """mainComputeStrain.m drops NaNs first, so a leading NaN shifts the reference."""
    curve = strain_from_arc_length(
        torch.tensor([float("nan"), 10.0, 5.0]), drift_comp=False
    ).numpy()

    assert curve[1] == pytest.approx(0.0)
    assert curve[2] == pytest.approx(-50.0)


# ------------------------------------------------------------- compute_strain
def _shrinking_contour(n_frames=5, es_frame=3):
    """A contour that shortens to end-systole then partly recovers."""
    lengths = [100.0, 95.0, 90.0, 88.0, 94.0]
    x = torch.zeros(n_frames, 2)
    x[:, 1] = torch.tensor(lengths[:n_frames])
    y = torch.zeros(n_frames, 2)
    return x, y, es_frame


def test_value_es_samples_the_curve_at_end_systole():
    x, y, es = _shrinking_contour()

    result = compute_strain(x, y, es_frame=es, drift_comp=False)

    assert result.value_es == pytest.approx(-10.0)


def test_peak_systolic_is_the_minimum_up_to_end_systole():
    """min(strainCurve(1:ED)) -- not the minimum of the whole curve."""
    x, y, es = _shrinking_contour()

    result = compute_strain(x, y, es_frame=es, drift_comp=False)

    # frame 4 is more negative (-12) but falls after ES, so it must not win.
    assert result.value_peak_systolic == pytest.approx(-10.0)
    assert float(result.curve.min()) == pytest.approx(-12.0)


def test_es_frame_is_one_based():
    x, y, _ = _shrinking_contour()

    first = compute_strain(x, y, es_frame=1, drift_comp=False)

    assert first.value_es == pytest.approx(0.0)


@pytest.mark.parametrize("es_frame", [0, 6, -1])
def test_out_of_range_es_frame_is_rejected(es_frame):
    x, y, _ = _shrinking_contour()

    with pytest.raises(ValueError, match="es_frame"):
        compute_strain(x, y, es_frame=es_frame)


def test_result_carries_the_arc_length_it_used():
    x, y, es = _shrinking_contour()

    result = compute_strain(x, y, es_frame=es, drift_comp=False)

    np.testing.assert_allclose(
        result.arc_length.numpy(), [100.0, 95.0, 90.0, 88.0, 94.0]
    )


def test_strain_is_differentiable_end_to_end():
    """The chain has to carry gradients for a training loss to reach a tracker."""
    x = torch.zeros(3, 2, dtype=torch.float64, requires_grad=True)
    offsets = torch.tensor([[0.0, 100.0], [0.0, 95.0], [0.0, 90.0]], dtype=torch.float64)
    points_x = x + offsets
    y = torch.zeros(3, 2, dtype=torch.float64)

    curve = strain_from_arc_length(
        contour_arc_length(points_x, y), drift_comp=False
    )
    curve[-1].backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


# ------------------------------------------------------------------ propagate
def test_bilinear_sample_hits_grid_values_exactly():
    field = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    # 1-based: (x=1, y=1) is the top-left entry.
    assert float(sample_bilinear(field, torch.tensor(1.0), torch.tensor(1.0))) == 1.0
    assert float(sample_bilinear(field, torch.tensor(2.0), torch.tensor(1.0))) == 2.0
    assert float(sample_bilinear(field, torch.tensor(1.0), torch.tensor(2.0))) == 3.0


def test_bilinear_sample_interpolates_the_centre():
    field = torch.tensor([[0.0, 0.0], [0.0, 4.0]])

    value = sample_bilinear(field, torch.tensor(1.5), torch.tensor(1.5))

    assert float(value) == pytest.approx(1.0)


def test_x_indexes_columns_and_y_indexes_rows():
    """interp2 with meshgrid(1:W, 1:H) -- getting this backwards mirrors the flow."""
    field = torch.tensor([[0.0, 5.0], [0.0, 0.0]])

    assert float(sample_bilinear(field, torch.tensor(2.0), torch.tensor(1.0))) == 5.0
    assert float(sample_bilinear(field, torch.tensor(1.0), torch.tensor(2.0))) == 0.0


def test_points_outside_the_image_are_nan_not_clamped():
    field = torch.ones(4, 4)

    values = sample_bilinear(
        field, torch.tensor([0.5, 2.0, 4.5]), torch.tensor([2.0, 2.0, 2.0])
    )

    assert np.isnan(float(values[0]))
    assert float(values[1]) == 1.0
    assert np.isnan(float(values[2]))


def test_advect_adds_the_sampled_displacement():
    flow = torch.zeros(4, 4, 2)
    flow[..., 0] = 1.5
    flow[..., 1] = -0.5

    x, y = advect_points(flow, torch.tensor([2.0]), torch.tensor([2.0]))

    assert float(x) == pytest.approx(3.5)
    assert float(y) == pytest.approx(1.5)


def test_propagate_accumulates_a_constant_flow():
    flows = torch.zeros(3, 5, 5, 2)
    flows[..., 0] = 1.0

    x, y = propagate_points(flows, torch.tensor([1.0]), torch.tensor([3.0]))

    np.testing.assert_allclose(x.numpy().ravel(), [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(y.numpy().ravel(), [3.0, 3.0, 3.0, 3.0])


def test_propagate_keeps_the_annotation_as_frame_zero():
    flows = torch.zeros(2, 4, 4, 2)

    x, y = propagate_points(flows, torch.tensor([2.0, 3.0]), torch.tensor([2.0, 2.0]))

    assert x.shape == (3, 2)
    np.testing.assert_allclose(x[0].numpy(), [2.0, 3.0])


def test_unpropagated_mode_restarts_from_the_reference():
    """opts.propagated == 0 isolates single-step error; it needs per-frame truth."""
    flows = torch.zeros(2, 5, 5, 2)
    flows[..., 0] = 1.0
    reference_x = torch.tensor([[1.0], [4.0]])
    reference_y = torch.tensor([[3.0], [3.0]])

    x, _ = propagate_points(
        flows,
        torch.tensor([1.0]),
        torch.tensor([3.0]),
        propagate_error=False,
        reference_x=reference_x,
        reference_y=reference_y,
    )

    np.testing.assert_allclose(x.numpy().ravel(), [1.0, 2.0, 5.0])


def test_unpropagated_mode_requires_a_reference():
    with pytest.raises(ValueError, match="reference_x"):
        propagate_points(
            torch.zeros(1, 4, 4, 2),
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            propagate_error=False,
        )


def test_propagation_survives_a_point_leaving_the_image():
    flows = torch.zeros(2, 4, 4, 2)
    flows[..., 0] = 10.0

    x, _ = propagate_points(flows, torch.tensor([2.0]), torch.tensor([2.0]))

    assert float(x[1]) == pytest.approx(12.0)
    assert np.isnan(float(x[2]))  # sampled outside on the next step


# -------------------------------------------------------- the real validation
def test_port_reproduces_mitt_strain_curves(dataset_root):
    """MITT's own points must yield MITT's own curves.

    The decisive check: an exact target on real data, independent of any
    network. Run across the full cohort with
    ``python -m motion_estimator.scripts.validate_strain``.
    """
    from motion_estimator.loaders.annotations.mitt import mitt_path, read_mitt

    checked = 0
    for dataset, patient, view in [
        ("Retest_dataset", "IV02", "2ch"),
        ("Retest_dataset", "IV02", "4ch"),
        ("Retest_dataset", "IV03", "2ch"),
    ]:
        path = mitt_path(patient, view, dataset)
        if not path.is_file():
            continue
        mitt = read_mitt(path)
        if mitt.strain_curve is None:
            continue

        result = compute_strain(
            mitt.points_x,
            mitt.points_y,
            mitt.es_frame,
            drift_comp=mitt.drift_compensated,
            open_contour=True,
            layer="endo",
        )

        np.testing.assert_allclose(
            result.curve.numpy(), mitt.strain_curve, atol=1e-6
        )
        checked += 1

    if not checked:
        pytest.skip("no MITT results available")
