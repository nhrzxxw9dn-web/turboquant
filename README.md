# TurboQuant

> **⚠️ Alpha — untested on real workloads.** Unit tests pass but real-world validation against production embeddings is starting now. Expect API changes. Use at your own risk until v0.2.

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
