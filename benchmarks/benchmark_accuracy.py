"""
Benchmark TurboQuant accuracy against real embeddings.

Tests:
1. MSE per coordinate at 2/3/4 bits
2. Cosine similarity preservation (original vs decoded)
3. Recall@k: does quantized search return the same top-k as exact search?
4. Serialization roundtrip integrity
"""

import sys
import time

import numpy as np

from turboquant import TurboQuantizer, CompressedVectors


def load_vectors(path: str | None = None) -> np.ndarray:
    """Load test vectors — from file or generate synthetic embeddings."""
    if path:
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            key = list(data.keys())[0]
            return data[key].astype(np.float32)
        return data.astype(np.float32)

    # Generate synthetic embedding-like vectors (unit-ish, 768-dim)
    print("No vector file provided — generating 5,000 synthetic 768-dim vectors")
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((5000, 768)).astype(np.float32)
    # Normalize to unit length (like real embeddings)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= norms
    return vectors


def benchmark_mse(vectors: np.ndarray):
    """Test MSE at different bit widths."""
    print("\n" + "=" * 60)
    print("MSE BENCHMARK")
    print("=" * 60)
    print(f"Vectors: {vectors.shape[0]} × {vectors.shape[1]}")
    print(f"Paper bounds (normalized): 1-bit=0.36, 2-bit=0.117, 3-bit=0.03, 4-bit=0.009")
    print()

    for bits in [2, 3, 4]:
        tq = TurboQuantizer(dim=vectors.shape[1], bits=bits, mode="mse")

        t0 = time.perf_counter()
        compressed = tq.encode(vectors)
        encode_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        decoded = tq.decode(compressed)
        decode_ms = (time.perf_counter() - t0) * 1000

        mse = tq.mse(vectors, compressed)

        # Per-vector cosine similarity
        dots = np.sum(vectors * decoded, axis=1)
        orig_norms = np.linalg.norm(vectors, axis=1)
        dec_norms = np.linalg.norm(decoded, axis=1)
        cosines = dots / (orig_norms * dec_norms + 1e-10)

        print(f"  {bits}-bit:")
        print(f"    MSE/coord:     {mse:.6f}")
        print(f"    Cosine sim:    mean={cosines.mean():.6f}  min={cosines.min():.6f}  p5={np.percentile(cosines, 5):.6f}")
        print(f"    Compression:   {compressed.compression_ratio:.1f}×  ({compressed.bytes_per_vector} bytes/vec vs {vectors.shape[1] * 4} float32)")
        print(f"    Encode:        {encode_ms:.1f}ms  ({encode_ms / vectors.shape[0] * 1000:.1f}µs/vec)")
        print(f"    Decode:        {decode_ms:.1f}ms  ({decode_ms / vectors.shape[0] * 1000:.1f}µs/vec)")
        print()


def benchmark_recall(vectors: np.ndarray, k: int = 10, n_queries: int = 50):
    """Test recall@k: quantized search vs exact search."""
    print("\n" + "=" * 60)
    print(f"RECALL@{k} BENCHMARK")
    print("=" * 60)

    rng = np.random.default_rng(123)
    query_indices = rng.choice(vectors.shape[0], size=n_queries, replace=False)
    queries = vectors[query_indices]

    # Exact top-k via brute force cosine similarity
    norms = np.linalg.norm(vectors, axis=1)
    normalized = vectors / (norms[:, np.newaxis] + 1e-10)

    for bits in [2, 3, 4]:
        tq = TurboQuantizer(dim=vectors.shape[1], bits=bits, mode="mse")
        compressed = tq.encode(vectors)

        recalls = []
        for i, query in enumerate(queries):
            # Exact cosine similarities
            q_norm = query / (np.linalg.norm(query) + 1e-10)
            exact_sims = normalized @ q_norm
            exact_topk = set(np.argsort(exact_sims)[-k:][::-1])

            # Approximate cosine similarities
            approx_sims = tq.cosine_similarity(query, compressed)
            approx_topk = set(np.argsort(approx_sims)[-k:][::-1])

            recall = len(exact_topk & approx_topk) / k
            recalls.append(recall)

        recalls = np.array(recalls)
        print(f"  {bits}-bit: recall@{k} = {recalls.mean():.4f}  (min={recalls.min():.4f}  p5={np.percentile(recalls, 5):.4f})")


def benchmark_serialization(vectors: np.ndarray):
    """Test serialization roundtrip."""
    print("\n" + "=" * 60)
    print("SERIALIZATION BENCHMARK")
    print("=" * 60)

    for bits in [3, 4]:
        for mode in ["mse", "inner_product"]:
            if mode == "inner_product" and bits < 2:
                continue
            tq = TurboQuantizer(dim=vectors.shape[1], bits=bits, mode=mode)
            compressed = tq.encode(vectors)

            t0 = time.perf_counter()
            blob = compressed.to_bytes()
            ser_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            restored = CompressedVectors.from_bytes(blob)
            deser_ms = (time.perf_counter() - t0) * 1000

            # Verify roundtrip
            decoded1 = tq.decode(compressed)
            decoded2 = tq.decode(restored)
            max_diff = np.max(np.abs(decoded1 - decoded2))

            print(f"  {bits}-bit {mode}:")
            print(f"    Blob size:     {len(blob):,} bytes ({len(blob) / vectors.shape[0]:.0f} bytes/vec)")
            print(f"    Serialize:     {ser_ms:.1f}ms")
            print(f"    Deserialize:   {deser_ms:.1f}ms")
            print(f"    Roundtrip err: {max_diff:.2e}")
            print()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    vectors = load_vectors(path)

    benchmark_mse(vectors)
    benchmark_recall(vectors)
    benchmark_serialization(vectors)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
