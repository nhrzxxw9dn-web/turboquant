"""
Rotation primitives for TurboQuant.

The key insight from the paper: randomly rotating input vectors induces a
concentrated Beta distribution on each coordinate, enabling optimal scalar
quantization per dimension. The rotation is seeded for reproducibility —
all vectors in a collection share the same rotation matrix.
"""

import numpy as np


def generate_rotation_matrix(d: int, seed: int = 42) -> np.ndarray:
    """
    Generate a d×d orthogonal rotation matrix via QR decomposition.

    The matrix is deterministic for a given (d, seed) pair, so we only
    need to store the seed (4 bytes) instead of the full matrix.

    Args:
        d: Vector dimension.
        seed: Random seed for reproducibility.

    Returns:
        Orthogonal matrix Π ∈ ℝ^(d×d) with Π @ Π.T = I.
    """
    rng = np.random.default_rng(seed)
    # QR decomposition of random Gaussian matrix gives a uniform random
    # orthogonal matrix (Haar measure on O(d)).
    gaussian = rng.standard_normal((d, d)).astype(np.float32)
    q, r = np.linalg.qr(gaussian)
    # Fix the sign ambiguity in QR decomposition for reproducibility:
    # multiply each column of Q by the sign of the diagonal of R.
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    q *= signs[np.newaxis, :]
    return q


def rotate(vectors: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """
    Apply rotation: y = Π · x.

    After rotation, each coordinate of y follows a concentrated Beta
    distribution on [-1, 1] (when x is unit-norm). This is what makes
    scalar quantization near-optimal.

    Args:
        vectors: Input vectors, shape (n, d) or (d,).
        rotation: Orthogonal matrix Π, shape (d, d).

    Returns:
        Rotated vectors, same shape as input.
    """
    if vectors.ndim == 1:
        return vectors @ rotation.T
    return vectors @ rotation.T


def unrotate(vectors: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """
    Reverse rotation: x̃ = Π^T · ỹ.

    Since Π is orthogonal, Π^T = Π^(-1).

    Args:
        vectors: Rotated vectors, shape (n, d) or (d,).
        rotation: Orthogonal matrix Π, shape (d, d).

    Returns:
        Vectors in original space, same shape as input.
    """
    if vectors.ndim == 1:
        return vectors @ rotation
    return vectors @ rotation


def normalize(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    L2-normalize vectors, returning norms separately.

    TurboQuant operates on the unit sphere. We store norms as float32
    scalars and reconstruct: x = norm * x_hat.

    Args:
        vectors: Input vectors, shape (n, d) or (d,).

    Returns:
        (normalized_vectors, norms) — norms shape (n,) or scalar.
    """
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        if norm < 1e-10:
            return vectors, norm
        return vectors / norm, norm
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return vectors / norms, norms.squeeze(axis=1)
