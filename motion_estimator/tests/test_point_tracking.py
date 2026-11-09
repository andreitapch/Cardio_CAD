"""Tests for the correspondence-to-trajectory layer.

These build synthetic :class:`Matches` rather than running LoFTR, so they run
on CPU with no weights and no dataset drive, and they check the arithmetic
against motions whose answer is known in closed form: a translation, and an
affine expansion whose strain is exactly computable. The LoFTR wrapper itself
is tested separately in ``test_loftr_matcher.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from motion_estimator.methods.loftr.matcher import Matches
from motion_estimator.methods.loftr.tracking import (
    displacement_at,
    to_dense_flow,
    track_points,
)


def grid_matches(transform, spacing=8, size=200, confidence=0.9) -> Matches:
    """Matches on a LoFTR-like grid, displaced by *transform* (0-based xy)."""
    axis = np.arange(0, size, spacing, dtype=float)
    grid_x, grid_y = np.meshgrid(axis, axis)
    xy0 = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    xy1 = transform(xy0)
    return Matches(xy0, xy1, np.full(len(xy0), float(confidence)))


def translation(dx, dy):
    return lambda xy: xy + np.array([dx, dy])


def expansion(factor, centre=(100.0, 100.0)):
    centre = np.asarray(centre, dtype=float)
    return lambda xy: centre + (xy - centre) * factor


class TestDisplacementAt:
    def test_recovers_a_translation(self):
        matches = grid_matches(translation(2.5, -1.25))
        dx, dy, support = displacement_at(matches, [50.0, 100.0], [80.0, 120.0])
        assert np.allclose(dx, 2.5)
        assert np.allclose(dy, -1.25)
        assert (support > 0).all()

    def test_recovers_an_affine_expansion_off_grid(self):
        # 1.05x about (100, 100): the point at 0-based (140, 100) must move by
        # 0.05 * 40 = 2.0 px in x. Query at a 1-based half-pixel position, so
        # nothing lands on a match and the interpolation is actually exercised.
        matches = grid_matches(expansion(1.05))
        dx, dy, _ = displacement_at(matches, 141.5, 101.0, radius=40.0)
        assert dx == pytest.approx(0.05 * 40.5, abs=1e-6)
        assert dy == pytest.approx(0.05 * 0.0, abs=1e-6)

    def test_translation_model_is_biased_by_one_sided_support(self):
        """Why the default is affine.

        Averaging displacements is exact only when the neighbourhood is
        symmetric about the query. It is not, wherever the matches run out --
        the image border, the edge of the sector mask, a dropout in the
        speckle. There the mean is pulled towards the side that has support,
        while the affine fit extrapolates the gradient correctly.
        """
        # Matches only up to x = 152, so a query at x = 176 has support on one
        # side alone. 1.05x about 100 means the truth at 0-based x=176 is 3.8.
        matches = grid_matches(expansion(1.05), size=160)
        affine, _, _ = displacement_at(matches, 177.0, 101.0, radius=40.0)
        mean, _, _ = displacement_at(
            matches, 177.0, 101.0, radius=40.0, model="translation"
        )
        assert affine == pytest.approx(0.05 * 76.0, abs=1e-6)
        assert abs(mean) < abs(affine) - 0.5

    def test_one_based_input_is_shifted_before_interpolation(self):
        matches = grid_matches(translation(3.0, 0.0))
        one, _, _ = displacement_at(matches, 51.0, 51.0, one_based=True)
        zero, _, _ = displacement_at(matches, 50.0, 50.0, one_based=False)
        assert one == pytest.approx(zero)

    def test_outliers_are_rejected(self):
        matches = grid_matches(translation(1.0, 1.0))
        # One confident match that disagrees wildly, right next to the query.
        rogue = Matches(
            np.vstack([matches.xy0, [[100.0, 100.0]]]),
            np.vstack([matches.xy1, [[160.0, 40.0]]]),
            np.append(matches.confidence, 1.0),
        )
        clean, _, _ = displacement_at(rogue, 101.0, 101.0, robust_passes=1)
        naive, _, _ = displacement_at(rogue, 101.0, 101.0, robust_passes=0)
        assert clean == pytest.approx(1.0, abs=1e-6)
        assert abs(naive - 1.0) > 1.0

    def test_too_few_neighbours_gives_nan_rather_than_a_guess(self):
        """The observed failure: a 3-neighbour fit predicting 40 px of motion.

        Arc length is a sum of local distances, so one such point moved a
        strain curve by 20 percentage points. Refusing to answer is the
        documented convention and the only honest option -- with three
        matches there is nothing to check them against.
        """
        sparse = Matches(
            [[100.0, 100.0], [104.0, 100.0], [100.0, 104.0]],
            [[96.0, 60.0], [100.0, 60.0], [96.0, 64.0]],
            [0.9, 0.9, 0.9],
        )
        dx, dy, support = displacement_at(sparse, 101.0, 101.0, radius=32.0)
        assert np.isnan(dx) and np.isnan(dy)
        assert support == 0
        # Lowering the floor accepts it -- and shows what was being avoided.
        loose, _, _ = displacement_at(
            sparse, 101.0, 101.0, radius=32.0, min_neighbours=3
        )
        assert abs(loose) > 3.0

    def test_collinear_support_falls_back_from_the_affine_fit(self):
        """One row of LoFTR's grid cannot determine a gradient.

        The least-squares solution is then dominated by the direction with no
        spread, and extrapolating it off the line gives an arbitrary answer.
        """
        columns = np.arange(0.0, 200.0, 8.0)
        xy0 = np.column_stack([columns, np.full_like(columns, 100.0)])
        collinear = Matches(xy0, xy0 + np.array([1.0, 2.0]), np.full(len(xy0), 0.9))
        dx, dy, support = displacement_at(collinear, 101.0, 121.0, radius=32.0)
        # Falls back to the weighted mean, which is exactly the translation.
        assert dx == pytest.approx(1.0, abs=1e-6)
        assert dy == pytest.approx(2.0, abs=1e-6)
        assert support > 0

    def test_extrapolation_is_gated_to_the_neighbourhood(self):
        """An affine fit may not predict far outside what it was fitted on.

        Support confined to a narrow band determines the gradient only inside
        it. Extrapolated far enough, a modest gradient predicts a displacement
        an order of magnitude larger than any match it was fitted on -- the
        shape of the real failure, where a thinning contour neighbourhood
        produced 40 px on a clip whose true motion was 4.
        """
        band_x = np.arange(96.0, 120.0, 8.0)
        band_y = np.arange(0.0, 120.0, 8.0)
        grid_x, grid_y = np.meshgrid(band_x, band_y)
        xy0 = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        shear = 0.5 * (xy0[:, 0] - 104.0)
        xy1 = xy0 + np.column_stack([shear, np.zeros_like(shear)])
        band = Matches(xy0, xy1, np.full(len(xy0), 0.9))
        observed = np.abs(band.displacement[:, 0]).max()

        query = (201.0, 57.0)  # 1-based, so 0-based x = 200: far off the band
        gated, _, support = displacement_at(band, *query, radius=120.0)
        ungated, _, _ = displacement_at(
            band, *query, radius=120.0, deviation_tolerance=0.0
        )
        assert support > 0
        assert abs(ungated) > 10 * observed  # what the fit wants to predict
        assert abs(gated) <= observed  # what the gate allows

    def test_gating_leaves_an_honest_interpolation_alone(self):
        """The gate must not touch a fit that is interpolating, only one that
        is extrapolating -- otherwise it would flatten the gradient that makes
        the affine model worth having in the first place."""
        matches = grid_matches(expansion(1.05))
        gated, _, _ = displacement_at(matches, 141.0, 101.0, radius=40.0)
        ungated, _, _ = displacement_at(
            matches, 141.0, 101.0, radius=40.0, deviation_tolerance=0.0
        )
        assert gated == pytest.approx(2.0, abs=1e-6)
        assert gated == pytest.approx(ungated, abs=1e-9)

    def test_no_nearby_matches_gives_nan_not_a_guess(self):
        matches = grid_matches(translation(1.0, 1.0))
        dx, dy, support = displacement_at(matches, 900.0, 900.0, radius=16.0)
        assert np.isnan(dx) and np.isnan(dy)
        assert support == 0

    def test_empty_matches_gives_nan(self):
        empty = Matches(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0))
        dx, dy, support = displacement_at(empty, [10.0, 20.0], [10.0, 20.0])
        assert np.isnan(dx).all() and np.isnan(dy).all()
        assert (support == 0).all()

    def test_nan_query_stays_nan(self):
        matches = grid_matches(translation(1.0, 1.0))
        dx, _, _ = displacement_at(matches, [np.nan, 100.0], [100.0, 100.0])
        assert np.isnan(dx[0])
        assert np.isfinite(dx[1])

    def test_shape_is_preserved(self):
        matches = grid_matches(translation(1.0, 2.0))
        x = np.full((3, 4), 100.0)
        dx, dy, support = displacement_at(matches, x, x)
        assert dx.shape == dy.shape == support.shape == (3, 4)

    def test_rejects_an_unknown_model(self):
        matches = grid_matches(translation(1.0, 1.0))
        with pytest.raises(ValueError, match="affine"):
            displacement_at(matches, 100.0, 100.0, model="homography")

    def test_rejects_mismatched_shapes(self):
        matches = grid_matches(translation(1.0, 1.0))
        with pytest.raises(ValueError, match="shape mismatch"):
            displacement_at(matches, [1.0, 2.0], [1.0])


class TestTrackPoints:
    def test_consecutive_chaining_accumulates_the_motion(self):
        steps = [grid_matches(translation(1.0, 0.5)) for _ in range(4)]
        result = track_points(None, [100.0], [100.0], matches=steps)
        assert result.n_frames == 5
        assert result.x[:, 0] == pytest.approx([100.0, 101.0, 102.0, 103.0, 104.0])
        assert result.y[:, 0] == pytest.approx([100.0, 100.5, 101.0, 101.5, 102.0])

    def test_reference_pairing_displaces_the_original_points(self):
        # Entry k maps frame 0 -> frame k, so entry 0 is the identity.
        empty = Matches(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0))
        frames = [empty] + [grid_matches(translation(float(k), 0.0)) for k in (1, 2, 3)]
        result = track_points(None, [100.0], [100.0], matches=frames, pairing="reference")
        assert result.n_frames == 4
        assert result.x[:, 0] == pytest.approx([100.0, 101.0, 102.0, 103.0])

    def test_reference_pairing_does_not_accumulate_drift(self):
        """A biased matcher drifts under chaining but not against a reference."""
        bias = 0.3
        chained = track_points(
            None,
            [100.0],
            [100.0],
            matches=[grid_matches(translation(bias, 0.0)) for _ in range(10)],
        )
        empty = Matches(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0))
        anchored = track_points(
            None,
            [100.0],
            [100.0],
            matches=[empty] + [grid_matches(translation(bias, 0.0))] * 10,
            pairing="reference",
        )
        assert chained.x[-1, 0] == pytest.approx(100.0 + 10 * bias)
        assert anchored.x[-1, 0] == pytest.approx(100.0 + bias)

    def test_a_lost_point_stays_lost_under_chaining(self):
        far = grid_matches(translation(1.0, 0.0))
        result = track_points(
            None, [900.0, 100.0], [900.0, 100.0], matches=[far, far], radius=16.0
        )
        assert np.isnan(result.x[1:, 0]).all()
        assert np.isfinite(result.x[:, 1]).all()
        assert result.lost.tolist() == [1, -1]

    def test_support_records_the_matches_each_fit_used(self):
        steps = [grid_matches(translation(1.0, 0.0)) for _ in range(2)]
        result = track_points(None, [100.0], [100.0], matches=steps)
        assert result.support[0, 0] == 0  # frame 0 is the given annotation
        assert (result.support[1:, 0] > 0).all()

    def test_an_affine_expansion_gives_the_exact_strain(self):
        """End to end: uniform 5% expansion must read as +5% strain.

        This is the property that matters -- a contour scaled by 1.05 has 1.05x
        the arc length, so ``100 * (dl - dl0) / dl0`` is exactly 5. If the
        interpolation smoothed the deformation away, this is where it shows.
        """
        from motion_estimator.strain.curve import compute_strain

        angle = np.linspace(0.2, np.pi - 0.2, 40)
        start_x = 100.0 + 50.0 * np.cos(angle)
        start_y = 100.0 + 50.0 * np.sin(angle)
        factor = 1.05
        steps = [grid_matches(expansion(factor), size=260)]
        result = track_points(
            None, start_x, start_y, matches=steps, radius=40.0
        )
        assert np.isfinite(result.x).all()
        strain = compute_strain(
            result.x, result.y, es_frame=2, drift_comp=False, open_contour=True
        )
        assert strain.curve[1].item() == pytest.approx(100.0 * (factor - 1.0), abs=1e-6)

    def test_needs_a_matcher_or_matches(self):
        with pytest.raises(ValueError, match="matcher or precomputed matches"):
            track_points(np.zeros((3, 10, 10)), [1.0], [1.0])

    def test_rejects_an_unknown_pairing(self):
        with pytest.raises(ValueError, match="pairing"):
            track_points(None, [1.0], [1.0], matches=[], pairing="star")

    def test_consecutive_pairing_refuses_a_nonzero_reference(self):
        with pytest.raises(ValueError, match="chains from frame 0"):
            track_points(
                None, [1.0], [1.0], matches=[grid_matches(translation(1.0, 0.0))],
                reference_frame=3,
            )

    def test_to_mm_undoes_the_one_based_origin(self):
        steps = [grid_matches(translation(2.0, 0.0))]
        result = track_points(None, [11.0], [21.0], matches=steps)
        x_mm, y_mm = result.to_mm([0.5, 0.25])
        assert x_mm[0, 0] == pytest.approx(5.0)
        assert y_mm[0, 0] == pytest.approx(5.0)
        assert x_mm[1, 0] == pytest.approx(6.0)


class TestDenseFlow:
    def test_matches_the_direct_interpolation(self):
        """The dense route and the direct route must agree at a grid point."""
        from motion_estimator.strain.propagate import advect_points

        matches = grid_matches(expansion(1.05), size=200)
        flow = to_dense_flow(matches, (160, 160), radius=40.0)
        assert flow.shape == (160, 160, 2)

        direct_x, direct_y, _ = displacement_at(matches, 121.0, 101.0, radius=40.0)
        moved_x, moved_y = advect_points(flow, 121.0, 101.0)
        assert float(moved_x) == pytest.approx(121.0 + direct_x, abs=1e-9)
        assert float(moved_y) == pytest.approx(101.0 + direct_y, abs=1e-9)
