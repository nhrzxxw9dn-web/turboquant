"""Tests for QJL (Quantized Johnson-Lindenstrauss) module."""

import numpy as np
import pytest

from turboquant.qjl import (
    generate_jl_matrix,
    qjl_decode,
    qjl_encode,
    qjl_inner_product,
)


class TestJLMatrix:
    def test_shape(self):
        S = generate_jl_matrix(64, seed=42)
        assert S.shape == (64, 64)
        assert S.dtype == np.float32

    def test_deterministic(self):
        a = generate_jl_matrix(128, seed=7)
        b = generate_jl_matrix(128, seed=7)
        np.testing.assert_array_equal(a, b)


class TestQJLEncodeDecode:
    def test_sign_bits_shape(self):
        """Sign bits should be packed into ceil(d/8) bytes."""
        d = 64
        S = generate_jl_matrix(d, seed=42)
        rng = np.random.default_rng(0)
        residuals = rng.standard_normal((5, d)).astype(np.float32) * 0.01
        sign_bits, norms = qjl_encode(residuals, S)
        assert sign_bits.shape == (5, d // 8)
        assert norms.shape == (5,)

    def test_single_vector(self):
        """Should work with single vector input."""
        d = 64
        S = generate_jl_matrix(d, seed=42)
        rng = np.random.default_rng(0)
        residual = rng.standard_normal(d).astype(np.float32) * 0.01
        sign_bits, norm = qjl_encode(residual, S)
        assert sign_bits.shape == (d // 8,)
        assert isinstance(norm, (float, np.floating))

    def test_norms_positive(self):
        d = 128
        S = generate_jl_matrix(d, seed=42)
        rng = np.random.default_rng(0)
        residuals = rng.standard_normal((10, d)).astype(np.float32)
        _, norms = qjl_encode(residuals, S)
        assert np.all(norms >= 0)

    def test_decode_shape(self):
        d = 64
        S = generate_jl_matrix(d, seed=42)
        rng = np.random.default_rng(0)
        residuals = rng.standard_normal((5, d)).astype(np.float32) * 0.01
        sign_bits, norms = qjl_encode(residuals, S)
        decoded = qjl_decode(sign_bits, norms, S, d)
        assert decoded.shape == (5, d)


class TestQJLInnerProduct:
    def test_unbiased(self):
        """
        QJL inner product should be approximately unbiased.
        Over many random vectors, mean estimate ≈ true inner product.
        """
        d = 256
        S = generate_jl_matrix(d, seed=42)
        rng = np.random.default_rng(0)

        query = rng.standard_normal(d).astype(np.float32)
        query /= np.linalg.norm(query)

        n_trials = 1000
        true_ips = []
        estimated_ips = []

        for _ in range(n_trials):
            x = rng.standard_normal(d).astype(np.float32) * 0.01
            true_ip = float(query @ x)
            true_ips.append(true_ip)

            sign_bits, norms = qjl_encode(x, S)
            est_ip = qjl_inner_product(
                query, sign_bits[np.newaxis, :], np.array([norms]), S, d
            )
            estimated_ips.append(float(est_ip[0]))

        true_ips = np.array(true_ips)
        estimated_ips = np.array(estimated_ips)

        # Mean error should be near zero (unbiased)
        mean_error = np.mean(estimated_ips - true_ips)
        assert abs(mean_error) < 0.01, f"Bias too large: {mean_error}"
