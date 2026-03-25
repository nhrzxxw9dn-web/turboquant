"""
Rotation primitives for TurboQuant.

The key insight from the paper: randomly rotating input vectors induces a
concentrated Beta distribution on each coordinate, enabling optimal scalar
quantization per dimension. The rotation is seeded for reproducibility —
all vectors in a collection share the same rotation matrix.

Two rotation backends:
  1. Hadamard (O(d log d)) — fast, structured, works when d is a power of 2.
     Uses randomized Hadamard: H · D2 · H · D1 · x where D1, D2 are random
     sign-flip diagonals and H is the Walsh-Hadamard transform (applied via
     butterfly operations, never materialized as a dense matrix).
  2. QR (O(d²)) — general-purpose fallback for arbitrary dimensions.
     Pads to next power of 2 internally when beneficial (d > 256).
"""

import math

import numpy as np


# ── Hadamard rotation (O(d log d)) ──────────────────────────────

def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _fwht_inplace(x: np.ndarray) -> None:
    """
    In-place Fast Walsh-Hadamard Transform on last axis.

    O(n · d log d) for shape (n, d). Butterfly factorization —
    never materializes the d×d Hadamard matrix.
    """
    d = x.shape[-1]
    h = 1
    while h < d:
        for i in range(0, d, 2 * h):
            a = x[..., i:i + h].copy()
            b = x[..., i + h:i + 2 * h].copy()
            x[..., i:i + h] = a + b
            x[..., i + h:i + 2 * h] = a - b
        h *= 2
    x /= math.sqrt(d)


def _generate_sign_flips(d: int, seed: int) -> np.ndarray:
    """Generate random ±1 diagonal for randomized Hadamard."""
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=d)


class _HadamardRotation:
    """
    Compact rotation via randomized Walsh-Hadamard transform.

    Two rounds of (sign-flip × Hadamard) approximate a Haar-random
    orthogonal matrix with O(d log d) cost and O(d) storage.
    """
    __slots__ = ('d', 'd1', 'd2')

    def __init__(self, d: int, d1: np.ndarray, d2: np.ndarray):
        self.d = d
        self.d1 = d1
        self.d2 = d2

    def forward(self, vectors: np.ndarray) -> np.ndarray:
        x = vectors.astype(np.float32, copy=True)
        x *= self.d1
        _fwht_inplace(x)
        x *= self.d2
        _fwht_inplace(x)
        return x

    def inverse(self, vectors: np.ndarray) -> np.ndarray:
        x = vectors.astype(np.float32, copy=True)
        _fwht_inplace(x)
        x *= self.d2
        _fwht_inplace(x)
        x *= self.d1
        return x


# ── QR rotation (O(d²)) — fallback ─────────────────────────────

def _generate_qr_rotation(d: int, seed: int) -> np.ndarray:
    """
    Generate a d×d orthogonal rotation matrix via QR decomposition.

    Uniform on O(d) (Haar measure). Deterministic for given (d, seed).
    """
    rng = np.random.default_rng(seed)
    gaussian = rng.standard_normal((d, d)).astype(np.float32)
    q, r = np.linalg.qr(gaussian)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    q *= signs[np.newaxis, :]
    return q


# ── Public API ──────────────────────────────────────────────────

def generate_rotation_matrix(d: int, seed: int = 42):
    """
    Generate a rotation for d-dimensional vectors.

    For power-of-2 dimensions, returns a _HadamardRotation (O(d log d)
    transform, O(d) memory). Otherwise returns a dense d×d orthogonal
    matrix via QR decomposition.

    Args:
        d: Vector dimension.
        seed: Random seed for reproducibility.

    Returns:
        _HadamardRotation or np.ndarray — both work with rotate()/unrotate().
    """
    if _is_power_of_2(d):
        d1 = _generate_sign_flips(d, seed)
        d2 = _generate_sign_flips(d, seed + 1_000_000)
        return _HadamardRotation(d, d1, d2)
    return _generate_qr_rotation(d, seed)


def rotate(vectors: np.ndarray, rotation) -> np.ndarray:
    """
    Apply rotation: y = Π · x.

    After rotation, each coordinate of y follows a concentrated Beta
    distribution on [-1, 1] (when x is unit-norm). This is what makes
    scalar quantization near-optimal.

    Uses Hadamard fast path (O(d log d)) when available, falls back to
    dense matrix multiply (O(d²)).

    Args:
        vectors: Input vectors, shape (n, d) or (d,).
        rotation: _HadamardRotation or orthogonal matrix Π, shape (d, d).

    Returns:
        Rotated vectors, same shape as input.
    """
    if isinstance(rotation, _HadamardRotation):
        return rotation.forward(vectors)
    if vectors.ndim == 1:
        return vectors @ rotation.T
    return vectors @ rotation.T


def unrotate(vectors: np.ndarray, rotation) -> np.ndarray:
    """
    Reverse rotation: x̃ = Π^T · ỹ.

    Since Π is orthogonal, Π^T = Π^(-1).
    """
    if isinstance(rotation, _HadamardRotation):
        return rotation.inverse(vectors)
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
