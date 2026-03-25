"""
Beta-optimal scalar quantizer codebooks for TurboQuant.

After random rotation, each coordinate of a unit-norm vector follows:
    f(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2)

This is a symmetric Beta distribution on [-1, 1]. In high dimensions (d ≥ 64),
it concentrates tightly around 0. We precompute optimal Lloyd-Max quantizer
centroids for this distribution at various bit widths.

Paper distortion bounds (MSE per coordinate):
    1-bit: 0.36, 2-bit: 0.117, 3-bit: 0.03, 4-bit: 0.009
"""

import math
from functools import lru_cache

import numpy as np
from scipy import integrate, optimize, special


def beta_pdf(x: float, d: int) -> float:
    """
    PDF of a single coordinate after random rotation of a d-dimensional
    unit vector.

    f(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2)

    Defined on [-1, 1].
    """
    if abs(x) >= 1.0:
        return 0.0
    alpha = (d - 1) / 2.0
    # Use log-gamma for numerical stability
    log_coeff = (
        special.gammaln(d / 2.0)
        - 0.5 * math.log(math.pi)
        - special.gammaln((d - 1) / 2.0)
    )
    log_val = log_coeff + (alpha - 1) * math.log(max(1e-300, 1.0 - x * x))
    return math.exp(log_val)


def _beta_pdf_vec(x: np.ndarray, d: int) -> np.ndarray:
    """Vectorized Beta PDF for integration."""
    alpha = (d - 1) / 2.0
    log_coeff = (
        special.gammaln(d / 2.0)
        - 0.5 * np.log(np.pi)
        - special.gammaln((d - 1) / 2.0)
    )
    safe_val = np.clip(1.0 - x * x, 1e-300, None)
    log_val = log_coeff + (alpha - 1) * np.log(safe_val)
    result = np.exp(log_val)
    result[np.abs(x) >= 1.0] = 0.0
    return result


@lru_cache(maxsize=64)
def compute_codebook(d: int, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute optimal Lloyd-Max codebook for the coordinate Beta distribution.

    Uses iterative Lloyd's algorithm on the continuous PDF.

    Args:
        d: Vector dimension (affects Beta distribution shape).
        bits: Bits per coordinate (1-4).

    Returns:
        (centroids, boundaries) where:
            centroids: shape (2^bits,) — reconstruction values
            boundaries: shape (2^bits - 1,) — decision boundaries between centroids
    """
    n_levels = 2 ** bits

    # For high-d, distribution concentrates around 0. Use tighter init range.
    # Std of the coordinate ≈ 1/√d
    std_approx = 1.0 / math.sqrt(d)
    init_range = min(1.0, 4.0 * std_approx)

    # Initialize centroids uniformly in the effective range
    centroids = np.linspace(-init_range, init_range, n_levels)

    # Lloyd's algorithm: iterate centroid ↔ boundary updates
    for _ in range(200):
        old_centroids = centroids.copy()

        # Boundaries = midpoints between consecutive centroids
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

        # Update centroids: centroid_i = E[x | x in bin_i]
        edges = np.concatenate([[-1.0], boundaries, [1.0]])
        new_centroids = np.zeros(n_levels)

        for i in range(n_levels):
            lo, hi = edges[i], edges[i + 1]
            if hi - lo < 1e-15:
                new_centroids[i] = (lo + hi) / 2.0
                continue

            # Numerator: ∫ x · f(x) dx over [lo, hi]
            num, _ = integrate.quad(
                lambda x: x * beta_pdf(x, d), lo, hi,
                limit=100, epsabs=1e-12,
            )
            # Denominator: ∫ f(x) dx over [lo, hi]
            den, _ = integrate.quad(
                lambda x: beta_pdf(x, d), lo, hi,
                limit=100, epsabs=1e-12,
            )

            if den > 1e-15:
                new_centroids[i] = num / den
            else:
                new_centroids[i] = (lo + hi) / 2.0

        centroids = new_centroids

        # Convergence check
        if np.max(np.abs(centroids - old_centroids)) < 1e-10:
            break

    # Final boundaries
    boundaries = (centroids[:-1] + centroids[1:]) / 2.0

    return centroids.astype(np.float32), boundaries.astype(np.float32)


def quantize_coordinates(
    values: np.ndarray, boundaries: np.ndarray
) -> np.ndarray:
    """
    Quantize coordinate values to codebook indices using searchsorted.

    Vectorized — works on arbitrary shape arrays.

    Args:
        values: Coordinate values to quantize.
        boundaries: Decision boundaries between centroids.

    Returns:
        Integer indices, same shape as values. dtype=uint8 for bits ≤ 8.
    """
    indices = np.searchsorted(boundaries, values.ravel())
    return indices.reshape(values.shape).astype(np.uint8)


def dequantize_coordinates(
    indices: np.ndarray, centroids: np.ndarray
) -> np.ndarray:
    """
    Map codebook indices back to centroid values.

    Args:
        indices: Integer indices from quantize_coordinates.
        centroids: Codebook centroid values.

    Returns:
        Reconstructed coordinate values, same shape as indices.
    """
    return centroids[indices]


@lru_cache(maxsize=16)
def get_codebook(d: int, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Get or compute codebook for given dimension and bit width.

    Cached for repeated use. This is the primary entry point.

    Args:
        d: Vector dimension.
        bits: Bits per coordinate (1-4).

    Returns:
        (centroids, boundaries) tuple.
    """
    if bits < 1 or bits > 8:
        raise ValueError(f"Bits must be 1-8, got {bits}")
    return compute_codebook(d, bits)
