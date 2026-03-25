"""
Compact binary storage for compressed vectors.

At 3 bits/dim, a 768-dim vector compresses to 288 bytes (from 3,072 bytes
at float32). This module handles bit-packing, serialization, and batch ops.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CompressedVectors:
    """
    Container for TurboQuant-compressed vectors.

    Attributes:
        n: Number of vectors.
        d: Original vector dimension.
        bits: Bits per coordinate used for MSE quantization.
        mode: "mse" or "inner_product".
        rotation_seed: Seed for rotation matrix regeneration.
        norms: L2 norms of original vectors, shape (n,), float32.
        indices: Quantized coordinate indices, shape (n, d), uint8.
            Values in [0, 2^bits - 1].
        packed_indices: Bit-packed indices, shape (n, packed_bytes), uint8.
            Used for compact storage. If None, computed on demand from indices.
        qjl_sign_bits: QJL sign bits for inner_product mode, shape (n, ceil(d/8)), uint8.
            None for MSE-only mode.
        qjl_norms: QJL residual norms, shape (n,), float32.
            None for MSE-only mode.
        qjl_seed: Seed for QJL matrix regeneration.
    """
    n: int
    d: int
    bits: int
    mode: str
    rotation_seed: int
    norms: np.ndarray
    indices: np.ndarray
    packed_indices: np.ndarray | None = None
    qjl_sign_bits: np.ndarray | None = None
    qjl_norms: np.ndarray | None = None
    qjl_seed: int = 0

    @property
    def bytes_per_vector(self) -> int:
        """Bytes per vector in packed representation."""
        # MSE indices: d * bits / 8
        mse_bytes = (self.d * self.bits + 7) // 8
        # Norm: 4 bytes (float32)
        norm_bytes = 4
        # QJL: d/8 bytes for sign bits + 4 bytes for residual norm
        qjl_bytes = 0
        if self.mode == "inner_product":
            qjl_bytes = (self.d + 7) // 8 + 4
        return mse_bytes + norm_bytes + qjl_bytes

    @property
    def total_bytes(self) -> int:
        """Total storage for all vectors (excluding metadata)."""
        return self.n * self.bytes_per_vector

    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs float32."""
        original = self.n * self.d * 4
        return original / max(1, self.total_bytes)

    def pack(self) -> None:
        """Pack indices into compact bit representation."""
        if self.packed_indices is not None:
            return
        if self.bits <= 4:
            # For 1-4 bits, pack multiple indices per byte
            self.packed_indices = _pack_indices(self.indices, self.bits)
        else:
            self.packed_indices = self.indices

    def unpack(self) -> None:
        """Unpack compact bit representation to indices."""
        if self.indices is not None and self.indices.shape == (self.n, self.d):
            return
        if self.packed_indices is not None:
            self.indices = _unpack_indices(self.packed_indices, self.d, self.bits)

    def to_bytes(self) -> bytes:
        """
        Serialize to compact binary format.

        Format:
            Header: [magic(4)] [version(1)] [n(4)] [d(4)] [bits(1)] [mode(1)]
                    [rotation_seed(4)] [qjl_seed(4)]
            Body per vector: [norm(4)] [packed_indices(variable)]
                             [qjl_sign_bits(d/8)] [qjl_norm(4)]  (if inner_product)
        """
        self.pack()

        mode_byte = 0 if self.mode == "mse" else 1
        header = struct.pack(
            "<4sBIIBBI I",
            b"TQVQ",  # magic
            1,  # version
            self.n,
            self.d,
            self.bits,
            mode_byte,
            self.rotation_seed,
            self.qjl_seed,
        )

        parts = [header]
        parts.append(self.norms.astype(np.float32).tobytes())
        parts.append(self.packed_indices.tobytes())

        if self.mode == "inner_product" and self.qjl_sign_bits is not None:
            parts.append(self.qjl_sign_bits.tobytes())
            parts.append(self.qjl_norms.astype(np.float32).tobytes())

        return b"".join(parts)

    @classmethod
    def from_bytes(cls, data: bytes) -> CompressedVectors:
        """Deserialize from binary format."""
        offset = 0

        # Parse header
        magic, version, n, d, bits, mode_byte, rot_seed, qjl_seed = struct.unpack_from(
            "<4sBIIBBI I", data, offset
        )
        offset += struct.calcsize("<4sBIIBBI I")

        if magic != b"TQVQ":
            raise ValueError(f"Invalid magic bytes: {magic}")
        if version != 1:
            raise ValueError(f"Unsupported version: {version}")

        mode = "mse" if mode_byte == 0 else "inner_product"

        # Norms
        norms = np.frombuffer(data[offset:offset + n * 4], dtype=np.float32).copy()
        offset += n * 4

        # Packed indices
        packed_bytes_per_vec = (d * bits + 7) // 8
        total_packed = n * packed_bytes_per_vec
        packed_indices = np.frombuffer(
            data[offset:offset + total_packed], dtype=np.uint8
        ).reshape(n, packed_bytes_per_vec).copy()
        offset += total_packed

        # QJL data
        qjl_sign_bits = None
        qjl_norms = None
        if mode == "inner_product":
            sign_bytes_per_vec = (d + 7) // 8
            total_sign = n * sign_bytes_per_vec
            qjl_sign_bits = np.frombuffer(
                data[offset:offset + total_sign], dtype=np.uint8
            ).reshape(n, sign_bytes_per_vec).copy()
            offset += total_sign

            qjl_norms = np.frombuffer(
                data[offset:offset + n * 4], dtype=np.float32
            ).copy()
            offset += n * 4

        indices = _unpack_indices(packed_indices, d, bits)

        return cls(
            n=n, d=d, bits=bits, mode=mode,
            rotation_seed=rot_seed,
            norms=norms,
            indices=indices,
            packed_indices=packed_indices,
            qjl_sign_bits=qjl_sign_bits,
            qjl_norms=qjl_norms,
            qjl_seed=qjl_seed,
        )


def _pack_indices(indices: np.ndarray, bits: int) -> np.ndarray:
    """
    Pack b-bit indices into bytes.

    For 1-4 bits, we pack multiple values per byte.
    For 3 bits: 8 values → 3 bytes (24 bits).
    """
    n, d = indices.shape

    if bits == 8:
        return indices.copy()

    if bits == 4:
        # Two 4-bit values per byte
        packed_d = (d + 1) // 2
        packed = np.zeros((n, packed_d), dtype=np.uint8)
        for i in range(0, d - 1, 2):
            packed[:, i // 2] = (indices[:, i] << 4) | indices[:, i + 1]
        if d % 2:
            packed[:, packed_d - 1] = indices[:, d - 1] << 4
        return packed

    if bits in (1, 2, 3):
        # Generic bit packing
        total_bits = d * bits
        total_bytes = (total_bits + 7) // 8
        packed = np.zeros((n, total_bytes), dtype=np.uint8)

        for vec_idx in range(n):
            bit_pos = 0
            for dim_idx in range(d):
                val = int(indices[vec_idx, dim_idx])
                # Write 'bits' bits starting at bit_pos
                for b in range(bits):
                    if val & (1 << (bits - 1 - b)):
                        byte_idx = bit_pos // 8
                        bit_offset = 7 - (bit_pos % 8)
                        packed[vec_idx, byte_idx] |= 1 << bit_offset
                    bit_pos += 1
        return packed

    raise ValueError(f"Unsupported bits: {bits}")


def _unpack_indices(packed: np.ndarray, d: int, bits: int) -> np.ndarray:
    """Unpack bytes back to b-bit indices."""
    n = packed.shape[0]

    if bits == 8:
        return packed[:, :d].copy()

    if bits == 4:
        indices = np.zeros((n, d), dtype=np.uint8)
        for i in range(0, d - 1, 2):
            indices[:, i] = packed[:, i // 2] >> 4
            indices[:, i + 1] = packed[:, i // 2] & 0x0F
        if d % 2:
            indices[:, d - 1] = packed[:, (d - 1) // 2] >> 4
        return indices

    if bits in (1, 2, 3):
        indices = np.zeros((n, d), dtype=np.uint8)
        mask = (1 << bits) - 1

        for vec_idx in range(n):
            bit_pos = 0
            for dim_idx in range(d):
                val = 0
                for b in range(bits):
                    byte_idx = bit_pos // 8
                    bit_offset = 7 - (bit_pos % 8)
                    if packed[vec_idx, byte_idx] & (1 << bit_offset):
                        val |= 1 << (bits - 1 - b)
                    bit_pos += 1
                indices[vec_idx, dim_idx] = val
        return indices

    raise ValueError(f"Unsupported bits: {bits}")
