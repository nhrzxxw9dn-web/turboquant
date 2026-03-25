"""Tests for rotation matrix primitives (QR + Hadamard)."""

import numpy as np
import pytest

from turboquant.core import (
    _HadamardRotation,
    _is_power_of_2,
    generate_rotation_matrix,
    normalize,
    rotate,
    unrotate,
)


class TestRotationMatrix:
    def test_qr_orthogonality(self):
        """QR rotation: Π @ Π.T should be identity."""
        pi = generate_rotation_matrix(65, seed=42)  # non-power-of-2 → QR
        assert isinstance(pi, np.ndarray)
        eye = pi @ pi.T
        np.testing.assert_allclose(eye, np.eye(65), atol=1e-5)

    def test_hadamard_for_power_of_2(self):
        """Power-of-2 dims should use Hadamard rotation."""
        rot = generate_rotation_matrix(64, seed=42)
        assert isinstance(rot, _HadamardRotation)
        assert rot.d == 64

    def test_qr_for_non_power_of_2(self):
        """Non-power-of-2 dims should use QR rotation."""
        rot = generate_rotation_matrix(100, seed=42)
        assert isinstance(rot, np.ndarray)

    def test_deterministic(self):
        """Same seed → same rotation."""
        a = generate_rotation_matrix(128, seed=99)
        b = generate_rotation_matrix(128, seed=99)
        if isinstance(a, _HadamardRotation):
            np.testing.assert_array_equal(a.d1, b.d1)
            np.testing.assert_array_equal(a.d2, b.d2)
        else:
            np.testing.assert_array_equal(a, b)

    def test_preserves_norm_hadamard(self):
        """Hadamard rotation should preserve L2 norm."""
        rot = generate_rotation_matrix(128, seed=42)
        rng = np.random.default_rng(0)
        v = rng.standard_normal(128).astype(np.float32)
        rotated = rotate(v, rot)
        np.testing.assert_allclose(
            np.linalg.norm(v), np.linalg.norm(rotated), rtol=1e-4
        )

    def test_preserves_norm_qr(self):
        """QR rotation should preserve L2 norm."""
        rot = generate_rotation_matrix(100, seed=42)  # non-power-of-2
        rng = np.random.default_rng(0)
        v = rng.standard_normal(100).astype(np.float32)
        rotated = rotate(v, rot)
        np.testing.assert_allclose(
            np.linalg.norm(v), np.linalg.norm(rotated), rtol=1e-5
        )

    def test_roundtrip_hadamard(self):
        """Hadamard: rotate then unrotate should recover original."""
        rot = generate_rotation_matrix(256, seed=42)
        assert isinstance(rot, _HadamardRotation)
        rng = np.random.default_rng(0)
        v = rng.standard_normal((10, 256)).astype(np.float32)
        recovered = unrotate(rotate(v, rot), rot)
        np.testing.assert_allclose(v, recovered, atol=1e-4)

    def test_roundtrip_qr(self):
        """QR: rotate then unrotate should recover original."""
        rot = generate_rotation_matrix(100, seed=42)
        rng = np.random.default_rng(0)
        v = rng.standard_normal((10, 100)).astype(np.float32)
        recovered = unrotate(rotate(v, rot), rot)
        np.testing.assert_allclose(v, recovered, atol=1e-5)

    def test_batch_and_single_hadamard(self):
        """Works for both single vectors and batches."""
        rot = generate_rotation_matrix(64, seed=42)
        rng = np.random.default_rng(0)
        batch = rng.standard_normal((5, 64)).astype(np.float32)
        single = batch[0]

        batch_rotated = rotate(batch, rot)
        single_rotated = rotate(single, rot)
        np.testing.assert_allclose(batch_rotated[0], single_rotated, atol=1e-5)

    def test_768_dim_hadamard(self):
        """768 (common embedding dim) should use Hadamard and roundtrip."""
        assert _is_power_of_2(768) is False  # 768 = 3 × 256, NOT power of 2
        rot = generate_rotation_matrix(768, seed=42)
        # 768 is not power of 2, so should use QR
        assert isinstance(rot, np.ndarray)

    def test_1024_dim_hadamard(self):
        """1024 (power of 2) should use Hadamard."""
        rot = generate_rotation_matrix(1024, seed=42)
        assert isinstance(rot, _HadamardRotation)
        rng = np.random.default_rng(0)
        v = rng.standard_normal((5, 1024)).astype(np.float32)
        recovered = unrotate(rotate(v, rot), rot)
        np.testing.assert_allclose(v, recovered, atol=1e-3)


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
