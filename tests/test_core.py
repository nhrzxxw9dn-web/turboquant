"""Tests for rotation matrix primitives."""

import numpy as np
import pytest

from turboquant.core import generate_rotation_matrix, normalize, rotate, unrotate


class TestRotationMatrix:
    def test_orthogonality(self):
        """Π @ Π.T should be identity."""
        pi = generate_rotation_matrix(64, seed=42)
        eye = pi @ pi.T
        np.testing.assert_allclose(eye, np.eye(64), atol=1e-5)

    def test_deterministic(self):
        """Same seed → same matrix."""
        a = generate_rotation_matrix(128, seed=99)
        b = generate_rotation_matrix(128, seed=99)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds(self):
        """Different seeds → different matrices."""
        a = generate_rotation_matrix(64, seed=1)
        b = generate_rotation_matrix(64, seed=2)
        assert not np.allclose(a, b)

    def test_preserves_norm(self):
        """Rotation should preserve L2 norm."""
        pi = generate_rotation_matrix(128, seed=42)
        rng = np.random.default_rng(0)
        v = rng.standard_normal(128).astype(np.float32)
        rotated = rotate(v, pi)
        np.testing.assert_allclose(
            np.linalg.norm(v), np.linalg.norm(rotated), rtol=1e-5
        )

    def test_roundtrip(self):
        """rotate then unrotate should recover original."""
        pi = generate_rotation_matrix(256, seed=42)
        rng = np.random.default_rng(0)
        v = rng.standard_normal((10, 256)).astype(np.float32)
        recovered = unrotate(rotate(v, pi), pi)
        np.testing.assert_allclose(v, recovered, atol=1e-5)

    def test_batch_and_single(self):
        """Works for both single vectors and batches."""
        pi = generate_rotation_matrix(64, seed=42)
        rng = np.random.default_rng(0)
        batch = rng.standard_normal((5, 64)).astype(np.float32)
        single = batch[0]

        batch_rotated = rotate(batch, pi)
        single_rotated = rotate(single, pi)
        np.testing.assert_allclose(batch_rotated[0], single_rotated, atol=1e-6)


class TestNormalize:
    def test_unit_norm(self):
        rng = np.random.default_rng(0)
        v = rng.standard_normal((5, 64)).astype(np.float32)
        normed, norms = normalize(v)
        np.testing.assert_allclose(
            np.linalg.norm(normed, axis=1), np.ones(5), atol=1e-6
        )

    def test_norms_correct(self):
        rng = np.random.default_rng(0)
        v = rng.standard_normal((5, 64)).astype(np.float32)
        normed, norms = normalize(v)
        np.testing.assert_allclose(norms, np.linalg.norm(v, axis=1), atol=1e-6)

    def test_reconstruct(self):
        rng = np.random.default_rng(0)
        v = rng.standard_normal((5, 64)).astype(np.float32)
        normed, norms = normalize(v)
        reconstructed = normed * norms[:, np.newaxis]
        np.testing.assert_allclose(v, reconstructed, atol=1e-5)
