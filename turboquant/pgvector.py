"""
pgvector integration helper for TurboQuant.

Provides utilities to compress/decompress vectors stored in PostgreSQL
with the pgvector extension. Supports both:
  1. Transparent compression: store compressed as BYTEA, decompress for HNSW search
  2. Reduced-precision storage: decode to float32 for pgvector, with lower MSE than halfvec

Requires: psycopg[binary]>=3.1, pgvector>=0.3.0
Install with: pip install turboquant[pgvector]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from turboquant.quantizer import TurboQuantizer
from turboquant.storage import CompressedVectors

if TYPE_CHECKING:
    import psycopg


def compress_vectors(
    vectors: np.ndarray,
    dim: int = 768,
    bits: int = 3,
    seed: int = 42,
) -> CompressedVectors:
    """
    Compress a batch of vectors for storage.

    Args:
        vectors: shape (n, dim), float32.
        dim: Vector dimension.
        bits: Bits per coordinate (1-4).
        seed: Rotation seed (must be consistent across encode/decode).

    Returns:
        CompressedVectors ready for serialization.
    """
    tq = TurboQuantizer(dim=dim, bits=bits, seed=seed)
    return tq.encode(vectors)


def decompress_for_pgvector(
    compressed: CompressedVectors,
    seed: int = 42,
) -> np.ndarray:
    """
    Decompress vectors back to float32 for pgvector insertion.

    Use this when you want to store compressed-then-decompressed vectors
    in a pgvector column. The decompressed vectors have lower precision
    than the originals but can still be indexed with HNSW.

    Args:
        compressed: CompressedVectors from compress_vectors.
        seed: Same rotation seed used for compression.

    Returns:
        Approximate float32 vectors, shape (n, dim).
    """
    tq = TurboQuantizer(dim=compressed.d, bits=compressed.bits, seed=seed)
    return tq.decode(compressed)


def compress_pgvector_column(
    conn: psycopg.Connection,
    table: str,
    vector_column: str = "embedding",
    id_column: str = "id",
    compressed_column: str = "embedding_compressed",
    batch_size: int = 500,
    bits: int = 3,
    seed: int = 42,
    dim: int = 768,
) -> dict:
    """
    Compress all vectors in a pgvector table to a BYTEA column.

    Adds a BYTEA column for compressed storage. The original vector column
    is preserved for HNSW search; the compressed column enables 10x
    storage reduction for backup/transfer/caching.

    Requires the compressed column to already exist (ALTER TABLE ADD COLUMN).

    Args:
        conn: psycopg connection.
        table: Table name.
        vector_column: Name of the vector column.
        id_column: Name of the primary key column.
        compressed_column: Name of the BYTEA column for compressed data.
        batch_size: Number of vectors to process at once.
        bits: Compression bits per dimension.
        seed: Rotation seed.
        dim: Vector dimension.

    Returns:
        Stats dict with counts and compression ratio.
    """
    tq = TurboQuantizer(dim=dim, bits=bits, seed=seed)

    total = 0
    total_original_bytes = 0
    total_compressed_bytes = 0

    with conn.cursor() as cur:
        # Count total rows
        cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {vector_column} IS NOT NULL")
        total_rows = cur.fetchone()[0]

        # Process in batches
        offset = 0
        while offset < total_rows:
            cur.execute(
                f"SELECT {id_column}, {vector_column} FROM {table} "
                f"WHERE {vector_column} IS NOT NULL "
                f"ORDER BY {id_column} "
                f"LIMIT %s OFFSET %s",
                (batch_size, offset),
            )
            rows = cur.fetchall()
            if not rows:
                break

            ids = [r[0] for r in rows]
            vectors = np.array([list(r[1]) for r in rows], dtype=np.float32)

            # Compress
            compressed = tq.encode(vectors)
            blob = compressed.to_bytes()

            # For per-row storage, we'd need to split the blob.
            # For now, store each vector individually.
            for i, row_id in enumerate(ids):
                single = tq.encode(vectors[i:i+1])
                single_blob = single.to_bytes()
                cur.execute(
                    f"UPDATE {table} SET {compressed_column} = %s WHERE {id_column} = %s",
                    (single_blob, row_id),
                )
                total_compressed_bytes += len(single_blob)

            total += len(rows)
            total_original_bytes += vectors.nbytes
            offset += batch_size

        conn.commit()

    return {
        "vectors_compressed": total,
        "original_bytes": total_original_bytes,
        "compressed_bytes": total_compressed_bytes,
        "compression_ratio": total_original_bytes / max(1, total_compressed_bytes),
        "bits_per_dim": bits,
    }
