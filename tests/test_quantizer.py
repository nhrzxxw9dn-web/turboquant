"""Tests for the TurboQuantizer main API."""

import numpy as np
import pytest

from turboquant import TurboQuantizer, CompressedVectors


def _random_vectors(n: int, d: int, seed: int = 0) -> np.ndarray:
    """Generate random unit-ish vectors for testing."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, d)).astype(np.float32)
    return v


class TestTurboQuantizerMSE:
    """Tests for MSE mode (Algorithm 1)."""

    def test_encode_decode_shape(self):
        tq = TurboQuantizer(dim=64, bits=3, mode="mse")
        vectors = _random_vectors(10, 64)
        compressed = tq.encode(vectors)
        decoded = tq.decode(compressed)
        assert decoded.shape == (10, 64)

    def test_compression_ratio(self):
        tq = TurboQuantizer(dim=768, bits=3, mode="mse")
        vectors = _random_vectors(100, 768)
        compressed = tq.encode(vectors)
        assert compressed.compression_ratio > 5.0

    def test_mse_within_bounds(self):
        tq = TurboQuantizer(dim=128, bits=3, mode="mse")
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((500, 128)).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        unit_vectors = vectors / norms

        compressed = tq.encode(unit_vectors)
        mse = tq.mse(unit_vectors, compressed)
        assert mse < 0.15, f"MSE too high: {mse}"

    def test_preserves_cosine_similarity(self):
        tq = TurboQuantizer(dim=128, bits=3, mode="mse")
        vectors = _random_vectors(50, 128)
        compressed = tq.encode(vectors)
        decoded = tq.decode(compressed)

        for i in range(50):
            cos_sim = (
                np.dot(vectors[i], decoded[i])
                / (np.linalg.norm(vectors[i]) * np.linalg.norm(decoded[i]) + 1e-10)
            )
            assert cos_sim > 0.8, f"Vector {i}: cosine sim = {cos_sim}"

    def test_4bit_better_than_3bit(self):
        vectors = _random_vectors(100, 128)

        tq3 = TurboQuantizer(dim=128, bits=3, mode="mse")
        tq4 = TurboQuantizer(dim=128, bits=4, mode="mse")

        c3 = tq3.encode(vectors)
        c4 = tq4.encode(vectors)

        mse3 = tq3.mse(vectors, c3)
        mse4 = tq4.mse(vectors, c4)

        assert mse4 < mse3, f"4-bit MSE ({mse4}) should be < 3-bit MSE ({mse3})"


class TestTurboQuantizerIP:
    """Tests for inner_product mode (Algorithm 2)."""

    def test_encode_decode_shape(self):
        tq = TurboQuantizer(dim=64, bits=3, mode="inner_product")
        vectors = _random_vectors(10, 64)
        compressed = tq.encode(vectors)
        decoded = tq.decode(compressed)
        assert decoded.shape == (10, 64)
        assert compressed.qjl_sign_bits is not None
        assert compressed.qjl_norms is not None

    def test_inner_product_reasonable(self):
        d = 128
        tq = TurboQuantizer(dim=d, bits=3, mode="inner_product")
        rng = np.random.default_rng(0)

        vectors = rng.standard_normal((50, d)).astype(np.float32)
        query = rng.standard_normal(d).astype(np.float32)

        compressed = tq.encode(vectors)
        approx_ips = tq.inner_product(query, compressed)
        true_ips = vectors @ query

        corr = np.corrcoef(true_ips, approx_ips)[0, 1]
        assert corr > 0.85, f"IP correlation too low: {corr}"

    def test_cosine_similarity(self):
        d = 64
        tq = TurboQuantizer(dim=d, bits=3, mode="mse")
        rng = np.random.default_rng(0)

        vectors = rng.standard_normal((20, d)).astype(np.float32)
        query = rng.standard_normal(d).astype(np.float32)

        compressed = tq.encode(vectors)
        sims = tq.cosine_similarity(query, compressed)

        assert sims.shape == (20,)
        assert np.all(sims >= -1.1)
        assert np.all(sims <= 1.1)



class TestAdaptiveCodebook:
    """Tests for adaptive codebook fitting."""

    def test_fit_improves_mse(self):
        """Fitted codebook should have equal or better MSE than theoretical."""
        d = 128
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((500, d)).astype(np.float32)

        tq_default = TurboQuantizer(dim=d, bits=3)
        tq_fitted = TurboQuantizer(dim=d, bits=3)
        tq_fitted.fit(vectors)

        c_default = tq_default.encode(vectors)
        c_fitted = tq_fitted.encode(vectors)

        mse_default = tq_default.mse(vectors, c_default)
        mse_fitted = tq_fitted.mse(vectors, c_fitted)

        # Fitted should be no worse (allow small tolerance for random variation)
        assert mse_fitted <= mse_default * 1.05, (
            f"Fitted MSE ({mse_fitted}) should not be much worse than "
            f"default ({mse_default})"
        )

    def test_fit_returns_self(self):
        """fit() should return self for chaining."""
        tq = TurboQuantizer(dim=64, bits=3)
        result = tq.fit(_random_vectors(50, 64))
        assert result is tq
        assert tq.is_fitted

    def test_fit_dim_mismatch_raises(self):
        tq = TurboQuantizer(dim=64, bits=3)
        with pytest.raises(ValueError, match="Expected dim=64"):
            tq.fit(_random_vectors(10, 128))


class TestSerialization:
    def test_roundtrip_bytes(self):
        tq = TurboQuantizer(dim=64, bits=3, mode="mse")
        vectors = _random_vectors(5, 64)
        compressed = tq.encode(vectors)

        blob = compressed.to_bytes()
        restored = CompressedVectors.from_bytes(blob)

        assert restored.n == compressed.n
        assert restored.d == compressed.d
        assert restored.bits == compressed.bits
        np.testing.assert_allclose(restored.norms, compressed.norms, atol=1e-6)

        decoded1 = tq.decode(compressed)
        decoded2 = tq.decode(restored)
        np.testing.assert_allclose(decoded1, decoded2, atol=1e-6)

    def test_roundtrip_bytes_ip_mode(self):
        tq = TurboQuantizer(dim=64, bits=3, mode="inner_product")
        vectors = _random_vectors(5, 64)
        compressed = tq.encode(vectors)

        blob = compressed.to_bytes()
        restored = CompressedVectors.from_bytes(blob)

        assert restored.qjl_sign_bits is not None
        assert restored.qjl_norms is not None

        decoded1 = tq.decode(compressed)
        decoded2 = tq.decode(restored)
        np.testing.assert_allclose(decoded1, decoded2, atol=1e-6)


class TestEdgeCases:
    def test_zero_vector(self):
        tq = TurboQuantizer(dim=64, bits=3, mode="mse")
        vectors = np.zeros((1, 64), dtype=np.float32)
        compressed = tq.encode(vectors)
        decoded = tq.decode(compressed)
        np.testing.assert_allclose(decoded, 0.0, atol=1e-5)

    def test_single_vector(self):
        tq = TurboQuantizer(dim=64, bits=3, mode="mse")
        vectors = _random_vectors(1, 64)
        compressed = tq.encode(vectors)
        decoded = tq.decode(compressed)
        assert decoded.shape == (1, 64)

    def test_dimension_mismatch_raises(self):
        tq = TurboQuantizer(dim=64, bits=3)
        vectors = _random_vectors(5, 128)
        with pytest.raises(ValueError, match="Expected dim=64"):
            tq.encode(vectors)
