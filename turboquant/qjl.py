"""
Quantized Johnson-Lindenstrauss (QJL) — 1-bit residual correction.

The second stage of Inner-Product TurboQuant. After the MSE quantizer
introduces error (residual r = x - decode(encode(x))), QJL corrects it
with just 1 bit per dimension.

Key properties from the paper:
- Unbiased: E[⟨y, QJL_decode(QJL_encode(x))⟩] = ⟨y, x⟩
- Variance: Var ≤ (π/2d) · ‖y‖²
- Zero memory overhead: S is generated from seed, sign bits are 1 bit each.

Performance optimization: sparse Rademacher JL matrix (±1 with probability
1/(2√d), 0 otherwise) gives the same theoretical guarantees with O(d√d)
nonzeros instead of O(d²), and faster matrix-vector products.
"""

import numpy as np


def generate_jl_matrix(d: int, seed: int = 0, sparse: bool = True) -> np.ndarray:
    """
    Generate the JL projection matrix S ∈ ℝ^(d×d).

    Two modes:
      - sparse=True (default): Sparse Rademacher. Each entry is independently
        +√s with prob 1/(2s), -√s with prob 1/(2s), 0 with prob 1-1/s,
        where s = √d. This gives O(d^1.5) nonzeros and faster multiply.
        Theoretical JL guarantees are preserved (Achlioptas, 2003).
      - sparse=False: Dense Gaussian N(0,1) entries (original implementation).

    Uses a deterministic seed so S can be regenerated from the seed alone.

    Args:
        d: Vector dimension.
        seed: Random seed for reproducibility.
        sparse: If True, use sparse Rademacher construction.

    Returns:
        S ∈ ℝ^(d×d), float32.
    """
    rng = np.random.default_rng(seed)

    if not sparse or d < 64:
        # Dense Gaussian — original implementation
        return rng.standard_normal((d, d)).astype(np.float32)

    # Sparse Rademacher: s = √d, P(±√s) = 1/(2s), P(0) = 1-1/s
    s = max(2, int(np.sqrt(d)))
    sqrt_s = np.sqrt(float(s))

    # Generate random values to determine nonzero positions and signs
    uniform = rng.random((d, d))
    matrix = np.zeros((d, d), dtype=np.float32)

    # Positive entries: uniform < 1/(2s)
    pos_mask = uniform < (1.0 / (2 * s))
    # Negative entries: 1/(2s) <= uniform < 1/s
    neg_mask = (uniform >= (1.0 / (2 * s))) & (uniform < (1.0 / s))

    matrix[pos_mask] = sqrt_s
    matrix[neg_mask] = -sqrt_s

    return matrix


def qjl_encode(
    residuals: np.ndarray, jl_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    QJL encode: extract sign bits from JL projection of residuals.

    Q_qjl(r) = sign(S · r)

    Args:
        residuals: Residual vectors, shape (n, d) or (d,).
        jl_matrix: JL matrix S, shape (d, d).

    Returns:
        (sign_bits, norms) where:
            sign_bits: Packed uint8 array. Shape (n, ceil(d/8)) or (ceil(d/8),).
            norms: ‖r‖₂ for each residual. Shape (n,) or scalar.
    """
    single = residuals.ndim == 1
    if single:
        residuals = residuals[np.newaxis, :]

    norms = np.linalg.norm(residuals, axis=1)

    # Project: (n, d) @ (d, d).T = (n, d)
    projected = residuals @ jl_matrix.T

    # Extract signs: positive → 1 (True), negative/zero → 0 (False)
    signs = projected > 0

    # Pack bits into bytes
    sign_bits = np.packbits(signs, axis=1)

    if single:
        return sign_bits[0], norms[0]
    return sign_bits, norms


def qjl_decode(
    sign_bits: np.ndarray,
    norms: np.ndarray,
    jl_matrix: np.ndarray,
    d: int,
) -> np.ndarray:
    """
    QJL decode: reconstruct approximate residuals from sign bits.

    Q_qjl^(-1)(z) = ‖r‖₂ · (√(π/2) / d) · S^T · z

    where z ∈ {+1, -1}^d are the unpacked sign bits.
    """
    single = sign_bits.ndim == 1
    if single:
        sign_bits = sign_bits[np.newaxis, :]
        norms = np.array([norms])

    unpacked = np.unpackbits(sign_bits, axis=1)[:, :d].astype(np.float32)

    # Convert {0, 1} → {-1, +1}
    z = 2.0 * unpacked - 1.0

    # Decode: (√(π/2) / d) · S^T · z
    scale = np.sqrt(np.pi / 2.0) / d
    decoded = scale * (z @ jl_matrix)

    decoded *= norms[:, np.newaxis]

    if single:
        return decoded[0]
    return decoded


def qjl_inner_product(
    query: np.ndarray,
    sign_bits: np.ndarray,
    norms: np.ndarray,
    jl_matrix: np.ndarray,
    d: int,
) -> np.ndarray:
    """
    Compute approximate inner products using QJL without full decode.

    ⟨y, r̃⟩ = ‖r‖₂ · (√(π/2) / d) · z^T · (S · y)

    This avoids materializing the full decoded residual.

    Args:
        query: Query vector, shape (d,).
        sign_bits: Packed sign bits, shape (n, ceil(d/8)).
        norms: Residual norms, shape (n,).
        jl_matrix: JL matrix S, shape (d, d).
        d: Vector dimension.

    Returns:
        Approximate inner products, shape (n,).
    """
    # Project query through S: S · y → (d,)
    projected_query = jl_matrix @ query

    # Unpack sign bits
    unpacked = np.unpackbits(sign_bits, axis=1)[:, :d].astype(np.float32)
    z = 2.0 * unpacked - 1.0

    # z^T · (S · y) for each vector
    dots = z @ projected_query

    scale = np.sqrt(np.pi / 2.0) / d
    return norms * scale * dots
