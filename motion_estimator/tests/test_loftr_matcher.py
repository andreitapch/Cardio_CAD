"""Tests for the LoFTR wrapper.

The matching tests need the pretrained checkpoint (~44 MB, cached under
``~/.cache/torch/hub``) and are skipped when kornia is absent. They run on
small synthetic speckle images so they stay feasible on CPU. The pure
bookkeeping -- coordinate conventions, masking, the padding contract -- is
tested without the model.
"""
from __future__ import annotations

import numpy as np
import pytest

from motion_estimator.methods.loftr.matcher import Matches, _in_mask, _inside, value_range

kornia = pytest.importorskip("kornia", reason="kornia is not installed")


def speckle(size=128, seed=0) -> np.ndarray:
    """A blob field with speckle-like second-order statistics."""
    from scipy import ndimage

    rng = np.random.default_rng(seed)
    field = ndimage.gaussian_filter(rng.random((size, size)).astype(np.float32), 2.0)
    return field / field.max()


class TestMatches:
    def test_displacement_is_target_minus_source(self):
        matches = Matches([[1.0, 2.0]], [[4.0, 6.0]], [0.9])
        assert matches.displacement.tolist() == [[3.0, 4.0]]
        assert matches.n_matches == 1

    def test_rejects_mismatched_arrays(self):
        with pytest.raises(ValueError, match="!="):
            Matches([[1.0, 2.0]], [[1.0, 2.0], [3.0, 4.0]], [0.5])

    def test_rejects_a_confidence_of_the_wrong_length(self):
        with pytest.raises(ValueError, match="confidence"):
            Matches([[1.0, 2.0]], [[1.0, 2.0]], [0.5, 0.5])

    def test_filter_keeps_the_selected_subset(self):
        matches = Matches([[0.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 2.0]], [0.1, 0.9])
        kept = matches.filter(matches.confidence > 0.5)
        assert kept.n_matches == 1
        assert kept.xy0.tolist() == [[1.0, 1.0]]


class TestBookkeeping:
    def test_inside_excludes_the_padded_region(self):
        # A 500x500 frame padded to 504: x = 501 is in the pad, not the image.
        xy = np.array([[499.0, 499.0], [501.0, 10.0], [10.0, 502.0], [-1.0, 5.0]])
        assert _inside(xy, (500, 500)).tolist() == [True, False, False, False]

    def test_in_mask_uses_the_nearest_pixel(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        xy = np.array([[5.4, 4.6], [3.0, 3.0]])
        assert _in_mask(xy, mask).tolist() == [True, False]

    def test_value_range_ignores_nan(self):
        assert value_range(np.array([np.nan, 1.0, 3.0])) == (1.0, 3.0)

    def test_value_range_of_a_constant_clip_stays_finite(self):
        low, high = value_range(np.full((2, 4, 4), 7.0))
        assert high > low


@pytest.fixture(scope="module")
def matcher():
    """One CPU matcher for the whole module -- loading the checkpoint is the
    expensive part, and nothing here mutates it."""
    from motion_estimator.methods.loftr.matcher import LoFTRMatcher

    return LoFTRMatcher(device="cpu", min_confidence=0.3)


@pytest.mark.slow
class TestLoFTRMatcher:
    def test_recovers_a_known_translation(self, matcher):
        from scipy import ndimage

        image0 = speckle()
        image1 = ndimage.shift(image0, (3.0, -2.0), order=3, mode="nearest")
        matches = matcher.match(image0, image1)

        assert matches.n_matches > 50
        median = np.median(matches.displacement, axis=0)
        # shift() takes (row, col); the matcher reports (x=col, y=row).
        assert median[0] == pytest.approx(-2.0, abs=0.5)
        assert median[1] == pytest.approx(3.0, abs=0.5)

    def test_coordinates_stay_inside_a_non_multiple_of_eight_frame(self, matcher):
        """Padding must not move a keypoint or leak matches into the pad."""
        image0 = speckle(size=125)
        image1 = speckle(size=125, seed=0)
        matches = matcher.match(image0, image1)
        assert matches.n_matches > 0
        assert matches.xy0.max() <= 124.0
        assert matches.xy1.max() <= 124.0

    def test_a_mask_drops_matches_outside_the_sector(self):
        from motion_estimator.methods.loftr.matcher import LoFTRMatcher

        mask = np.zeros((128, 128), dtype=bool)
        mask[32:96, 32:96] = True
        masked = LoFTRMatcher(device="cpu", min_confidence=0.3, mask=mask)
        image = speckle()
        matches = masked.match(image, image)
        assert matches.n_matches > 0
        assert (matches.xy0[:, 0] >= 31.5).all()
        assert (matches.xy0[:, 0] <= 95.5).all()

    def test_scale_maps_keypoints_back_to_full_resolution(self):
        from motion_estimator.methods.loftr.matcher import LoFTRMatcher

        halved = LoFTRMatcher(device="cpu", min_confidence=0.3, scale=0.5)
        image = speckle(size=256)
        matches = halved.match(image, image)
        assert matches.n_matches > 0
        # A keypoint found at 1/2 resolution must come back doubled, so the
        # spread has to reach past the downscaled image's own extent.
        assert matches.xy0[:, 0].max() > 130.0
        assert matches.xy0[:, 0].max() <= 255.0

    def test_match_sequence_consecutive_length(self, matcher):
        frames = np.stack([speckle(seed=k) for k in range(3)])
        pairs = matcher.match_sequence(frames)
        assert len(pairs) == 2

    def test_match_sequence_reference_is_frame_aligned(self, matcher):
        frames = np.stack([speckle(seed=k) for k in range(3)])
        pairs = matcher.match_sequence(frames, reference=0)
        assert len(pairs) == 3
        assert pairs[0].n_matches == 0  # the identity slot

    def test_rejects_an_upscaling_factor(self):
        from motion_estimator.methods.loftr.matcher import LoFTRMatcher

        with pytest.raises(ValueError, match="scale"):
            LoFTRMatcher(device="cpu", scale=2.0)
