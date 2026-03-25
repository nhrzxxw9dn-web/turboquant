"""Tests for Beta-optimal codebook computation."""

import numpy as np
import pytest

from turboquant.codebook import (
    beta_pdf,
    compute_codebook,
    dequantize_coordinates,
    get_codebook,
    quantize_coordinates,
)


class TestBetaPDF:
    def test_symmetric(self):
        """PDF should be symmetric around 0."""
        assert abs(beta_pdf(0.3, 768) - beta_pdf(-0.3, 768)) < 1e-10

    def test_zero_outside(self):
        """PDF is zero outside [-1, 1]."""
        assert beta_pdf(1.01, 768) == 0.0
        assert beta_pdf(-1.01, 768) == 0.0

    def test_peak_at_zero(self):
        """Peak should be at x=0 for d > 3."""
        assert beta_pdf(0.0, 768) > beta_pdf(0.1, 768)
        assert beta_pdf(0.0, 768) > beta_pdf(-0.1, 768)

    def test_concentration_increases_with_d(self):
        """Higher d → tighter concentration around 0."""
        # At x=0.1, higher d should give higher PDF (more concentrated)
        assert beta_pdf(0.0, 768) > beta_pdf(0.0, 64)


class TestCodebook:
    def test_centroids_sorted(self):
        """Centroids should be monotonically increasing."""
        centroids, boundaries = compute_codebook(768, 2)
        assert np.all(np.diff(centroids) > 0)

    def test_boundaries_between_centroids(self):
        """Each boundary should be between adjacent centroids."""
        centroids, boundaries = compute_codebook(768, 2)
        for i in range(len(boundaries)):
            assert centroids[i] < boundaries[i] < centroids[i + 1]

    def test_symmetric_codebook(self):
        """Codebook should be approximately symmetric (odd levels centered at 0)."""
        centroids, _ = compute_codebook(768, 2)
        # 4 centroids — should be roughly [-a, -b, b, a]
        np.testing.assert_allclose(centroids[0], -centroids[3], atol=1e-4)
        np.testing.assert_allclose(centroids[1], -centroids[2], atol=1e-4)

    def test_codebook_in_range(self):
        """All centroids should be in [-1, 1]."""
        for bits in [1, 2, 3, 4]:
            centroids, _ = compute_codebook(768, bits)
            assert np.all(centroids >= -1.0)
            assert np.all(centroids <= 1.0)

    def test_caching(self):
        """get_codebook should return same object on repeated calls."""
        a = get_codebook(768, 3)
        b = get_codebook(768, 3)
        assert a[0] is b[0]


class TestQuantization:
    def test_roundtrip_indices(self):
        """quantize → dequantize should recover centroid values."""
        centroids, boundaries = compute_codebook(768, 3)
        # Quantize the centroids themselves — should get exact indices
        values = centroids.copy()
        indices = quantize_coordinates(values, boundaries)
        recovered = dequantize_coordinates(indices, centroids)
        np.testing.assert_allclose(values, recovered, atol=1e-6)

    def test_batch_shape(self):
        """Should handle 2D arrays."""
        centroids, boundaries = compute_codebook(128, 2)
        rng = np.random.default_rng(0)
        values = rng.standard_normal((10, 128)).astype(np.float32) * 0.05
        indices = quantize_coordinates(values, boundaries)
        assert indices.shape == (10, 128)
        assert indices.dtype == np.uint8

    def test_indices_in_range(self):
        """Indices should be in [0, 2^bits - 1]."""
        centroids, boundaries = compute_codebook(768, 3)
        rng = np.random.default_rng(0)
        values = rng.standard_normal(768).astype(np.float32) * 0.04
        indices = quantize_coordinates(values, boundaries)
        assert np.all(indices >= 0)
        assert np.all(indices < 8)  # 2^3 = 8
