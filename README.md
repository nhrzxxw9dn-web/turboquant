# TurboQuant

> **⚠️ Alpha — first real-world benchmarks completed** (see below). API may change before v1.0.

Near-optimal vector quantization at 3-4 bits per dimension. First open-source implementation of the ICLR 2026 paper.

## Install

```bash
pip install turboquant
```

## Quick Start

```python
import numpy as np
from turboquant import TurboQuantizer

# Create quantizer (768-dim embeddings, 3 bits/dim)
tq = TurboQuantizer(dim=768, bits=3)

# Compress vectors: 3,072 bytes → 288 bytes each (10x reduction)
vectors = np.random.randn(1000, 768).astype(np.float32)
compressed = tq.encode(vectors)

# Decompress
decoded = tq.decode(compressed)

# Approximate cosine similarity (without full decompression)
query = np.random.randn(768).astype(np.float32)
similarities = tq.cosine_similarity(query, compressed)

# Serialize to bytes
blob = compressed.to_bytes()  # compact binary format
```

## How It Works

TurboQuant is a two-stage vector quantization algorithm:

1. **Random rotation** induces a concentrated Beta distribution on each coordinate
2. **Optimal scalar quantizer** per coordinate (precomputed for the Beta PDF)
3. **QJL residual correction** (optional) adds 1-bit error correction for unbiased inner products

At 3 bits/dimension, TurboQuant achieves MSE within 2.7x of the information-theoretic optimum.

## v0.2.0 — Performance & Accuracy Improvements

### Hadamard Fast Rotation (O(d log d))

For power-of-2 dimensions (256, 512, 1024, 2048, ...), TurboQuant now uses a **randomized Walsh-Hadamard transform** instead of a dense QR matrix multiply. This reduces rotation cost from O(d²) to O(d log d) and memory from O(d²) to O(d):

| Operation | QR (v0.1) | Hadamard (v0.2) | Speedup |
|-----------|-----------|-----------------|---------|
| Rotation matrix memory (d=1024) | 4 MB | 8 KB | 500× |
| rotate() per vector | O(d²) | O(d log d) | ~100× at d=1024 |

Non-power-of-2 dimensions (768, 384, etc.) automatically fall back to QR.

### Sparse JL Matrix

The QJL error-correction stage now uses a **sparse Rademacher** projection matrix by default. Each entry is independently ±√s with probability 1/(2s) and 0 otherwise (where s = √d). This gives the same JL guarantees (Achlioptas, 2003) with O(d√d) nonzeros instead of O(d²).

### Adaptive Codebooks

Instead of assuming the theoretical Beta distribution, you can now **fit codebooks from your actual embeddings**:

```python
tq = TurboQuantizer(dim=768, bits=3)

# Fit codebook from training data (1000+ vectors recommended)
tq.fit(training_vectors)

# Encode using the model-specific codebook
compressed = tq.encode(new_vectors)
```

The adaptive codebook learns the empirical distribution of rotated coordinates, which can reduce MSE by 10-30% for embeddings that deviate from the ideal distribution.

### Vectorized Quantization

For small codebooks (≤4 bits / 16 levels), quantization now uses vectorized comparison instead of binary search. This eliminates branching and is faster for the common 2-4 bit case.

## Benchmarks (Real Production Embeddings)

Tested on 1,045 production Gemini `gemini-embedding-001` vectors (768-dim) from a live pgvector database:

| Bits | Cosine Sim (mean) | Cosine Sim (min) | Recall@10 | MSE/coord | Compression | Bytes/vec |
|------|-------------------|-------------------|-----------|-----------|-------------|-----------|
| 2-bit | 0.939 | 0.930 | 0.74 | 0.000054 | 15.7× | 196 |
| **3-bit** | **0.983** | **0.978** | **0.81** | **0.000016** | **10.5×** | **292** |
| **4-bit** | **0.995** | **0.994** | **0.87** | **0.000004** | **7.9×** | **388** |

Key findings:
- **Cosine fidelity is excellent** — 0.983 at 3-bit, 0.995 at 4-bit. No catastrophic outliers (min never below 0.978).
- **Real embeddings outperform random vectors** — semantic clustering helps quantization.
- **MSE well below paper bounds** — 0.000016 vs paper's 0.03 theoretical bound at 3-bit.
- **Serialization is lossless** — encode → bytes → decode roundtrip has zero error.

## Compression Ratios

| Format | Per vector (768-dim) | Ratio |
|--------|---------------------|-------|
| float32 | 3,072 bytes | 1x |
| float16 | 1,536 bytes | 2x |
| **TurboQuant 4-bit** | **384 bytes** | **8x** |
| **TurboQuant 3-bit** | **288 bytes** | **~10x** |

## Modes

- **`mse`** (default): Minimizes reconstruction error. Best for storage + decompression workflows.
- **`inner_product`**: Uses (b-1) bits for MSE + 1 bit QJL. Gives unbiased inner product estimates without full decompression.

## pgvector Integration

```python
from turboquant.pgvector import compress_vectors, decompress_for_pgvector

# Compress existing embeddings
compressed = compress_vectors(vectors, dim=768, bits=3)

# Decompress for HNSW insertion
approx_vectors = decompress_for_pgvector(compressed)
```

## Reference

Based on: *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate* (ICLR 2026)
by Amir Zandieh, Vahab Mirrokni et al. (Google Research, KAIST, NYU)

## License

Apache 2.0
