"""
TurboQuantizer — the main API for TurboQuant vector compression.

Usage:
    tq = TurboQuantizer(dim=768, bits=3)
    compressed = tq.encode(vectors)       # (n, 768) → CompressedVectors
    decoded = tq.decode(compressed)       # CompressedVectors → (n, 768)
    sims = tq.cosine_similarity(query, compressed)

Adaptive codebooks:
    tq = TurboQuantizer(dim=768, bits=3)
    tq.fit(training_vectors)              # Learn optimal codebook from data
    compressed = tq.encode(new_vectors)   # Uses fitted codebook
"""

import numpy as np

from turboquant.codebook import (
    dequantize_coordinates,
    fit_codebook,
    get_codebook,
    quantize_coordinates,
)
from turboquant.core import (
    generate_rotation_matrix,
    normalize,
    rotate,
    unrotate,
)
from turboquant.qjl import (
    generate_jl_matrix,
    qjl_decode,
    qjl_encode,
    qjl_inner_product,
)
from turboquant.storage import CompressedVectors


class TurboQuantizer:
    """
    Near-optimal vector quantizer at 3-4 bits per dimension.

    Implements both MSE-optimal (Algorithm 1) and inner-product optimal
    (Algorithm 2) variants from the TurboQuant paper.

    Args:
        dim: Vector dimension. Must match the vectors you'll encode.
        bits: Bits per coordinate (1-8). Default 3 gives MSE ≤ 0.03.
        mode: "mse" for MSE-optimal or "inner_product" for unbiased IP estimation.
        seed: Random seed for rotation matrix. Must be the same for encode/decode.
        qjl_seed: Random seed for QJL matrix (inner_product mode only).
    """

    def __init__(
        self,
        dim: int = 768,
        bits: int = 3,
        mode: str = "mse",
        seed: int = 42,
        qjl_seed: int = 7,
    ):
        if bits < 1 or bits > 8:
            raise ValueError(f"bits must be 1-8, got {bits}")
        if mode not in ("mse", "inner_product"):
            raise ValueError(f"mode must be 'mse' or 'inner_product', got {mode}")
        if mode == "inner_product" and bits < 2:
            raise ValueError("inner_product mode requires bits >= 2 (1 for MSE + 1 for QJL)")

        self.dim = dim
        self.bits = bits
        self.mode = mode
        self.seed = seed
        self.qjl_seed = qjl_seed

        # Effective MSE bits (inner_product mode reserves 1 bit for QJL)
        self._mse_bits = bits - 1 if mode == "inner_product" else bits

        # Lazy-init
        self._rotation = None
        self._codebook: tuple[np.ndarray, np.ndarray] | None = None
        self._jl_matrix: np.ndarray | None = None
        self._fitted = False

    @property
    def rotation(self):
        if self._rotation is None:
            self._rotation = generate_rotation_matrix(self.dim, self.seed)
        return self._rotation

    @property
    def codebook(self) -> tuple[np.ndarray, np.ndarray]:
        if self._codebook is None:
            self._codebook = get_codebook(self.dim, self._mse_bits)
        return self._codebook

    @property
    def jl_matrix(self) -> np.ndarray:
        if self._jl_matrix is None:
            self._jl_matrix = generate_jl_matrix(self.dim, self.qjl_seed)
        return self._jl_matrix

    def fit(self, vectors: np.ndarray, max_iter: int = 100) -> "TurboQuantizer":
        """
        Fit an adaptive codebook from training data.

        Learns optimal quantization centroids from the empirical distribution
        of rotated coordinates, instead of using the theoretical Beta PDF.
        This can reduce MSE by 10-30% for specific embedding models.

        Args:
            vectors: Training vectors, shape (n, dim). 1000+ vectors
                recommended for stable codebook estimation.
            max_iter: Maximum Lloyd iterations for codebook fitting.

        Returns:
            self (for chaining: tq.fit(data).encode(data))
        """
        if vectors.shape[-1] != self.dim:
            raise ValueError(
                f"Expected dim={self.dim}, got vectors with dim={vectors.shape[-1]}"
            )

        vectors = vectors.astype(np.float32)
        normalized, _ = normalize(vectors)
        rotated = rotate(normalized, self.rotation)

        self._codebook = fit_codebook(rotated, self._mse_bits, max_iter=max_iter)
        self._fitted = True
        return self

    @property
    def is_fitted(self) -> bool:
        """Whether an adaptive codebook has been fitted."""
        return self._fitted

    def encode(self, vectors: np.ndarray) -> CompressedVectors:
        """
        Quantize vectors to compact representation.

        Args:
            vectors: Input vectors, shape (n, dim) or (dim,).

        Returns:
            CompressedVectors with packed indices and metadata.
        """
        single = vectors.ndim == 1
        if single:
            vectors = vectors[np.newaxis, :]

        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Expected dim={self.dim}, got vectors with dim={vectors.shape[1]}"
            )

        vectors = vectors.astype(np.float32)
        n = vectors.shape[0]

        # 1. Normalize and store norms
        normalized, norms = normalize(vectors)

        # 2. Rotate
        rotated = rotate(normalized, self.rotation)

        # 3. Scalar quantize each coordinate
        centroids, boundaries = self.codebook
        indices = quantize_coordinates(rotated, boundaries)

        # 4. QJL for inner-product mode
        qjl_sign_bits = None
        qjl_norms = None
        if self.mode == "inner_product":
            reconstructed_rotated = dequantize_coordinates(indices, centroids)
            residuals = rotated - reconstructed_rotated
            residuals_original = unrotate(residuals, self.rotation)
            qjl_sign_bits, qjl_norms = qjl_encode(residuals_original, self.jl_matrix)

        compressed = CompressedVectors(
            n=n,
            d=self.dim,
            bits=self.bits,
            mode=self.mode,
            rotation_seed=self.seed,
            norms=norms,
            indices=indices,
            qjl_sign_bits=qjl_sign_bits,
            qjl_norms=qjl_norms,
            qjl_seed=self.qjl_seed,
        )

        return compressed

    def decode(self, compressed: CompressedVectors) -> np.ndarray:
        """
        Reconstruct approximate vectors from compressed representation.
        """
        compressed.unpack()

        centroids, _ = self.codebook

        # 1. Dequantize: indices → centroid values
        reconstructed = dequantize_coordinates(compressed.indices, centroids)

        # 2. Unrotate back to original space
        decoded = unrotate(reconstructed, self.rotation)

        # 3. Add QJL residual correction if available
        if (
            compressed.mode == "inner_product"
            and compressed.qjl_sign_bits is not None
        ):
            residual_approx = qjl_decode(
                compressed.qjl_sign_bits,
                compressed.qjl_norms,
                self.jl_matrix,
                self.dim,
            )
            decoded += residual_approx

        # 4. Rescale by original norms
        decoded *= compressed.norms[:, np.newaxis]

        return decoded

    def inner_product(
        self, query: np.ndarray, compressed: CompressedVectors
    ) -> np.ndarray:
        """
        Compute approximate inner products: ⟨query, x_i⟩ for all compressed vectors.
        """
        if query.shape[-1] != self.dim:
            raise ValueError(
                f"Expected dim={self.dim}, got query dim={query.shape[-1]}"
            )

        compressed.unpack()
        centroids, _ = self.codebook

        # MSE component
        reconstructed_rotated = dequantize_coordinates(compressed.indices, centroids)
        decoded_mse = unrotate(reconstructed_rotated, self.rotation)
        decoded_mse *= compressed.norms[:, np.newaxis]

        mse_dots = decoded_mse @ query

        # QJL component (inner_product mode only)
        if (
            compressed.mode == "inner_product"
            and compressed.qjl_sign_bits is not None
        ):
            qjl_dots = qjl_inner_product(
                query,
                compressed.qjl_sign_bits,
                compressed.qjl_norms * compressed.norms,
                self.jl_matrix,
                self.dim,
            )
            return mse_dots + qjl_dots

        return mse_dots

    def cosine_similarity(
        self, query: np.ndarray, compressed: CompressedVectors
    ) -> np.ndarray:
        """
        Compute approximate cosine similarities.
        """
        query = query.astype(np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-10:
            return np.zeros(compressed.n, dtype=np.float32)

        dots = self.inner_product(query / query_norm, compressed)

        norms = np.maximum(compressed.norms, 1e-10)
        return dots / norms

    def mse(self, original: np.ndarray, compressed: CompressedVectors) -> float:
        """
        Compute mean squared error between original and decoded vectors.
        """
        decoded = self.decode(compressed)
        diff = original - decoded
        return float(np.mean(diff ** 2))
