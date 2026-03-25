"""
TurboQuant — Near-optimal vector quantization at 3-4 bits per dimension.

First open-source implementation of the ICLR 2026 paper:
"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
by Amir Zandieh, Vahab Mirrokni et al. (Google Research, KAIST, NYU)

Usage:
    from turboquant import TurboQuantizer

    tq = TurboQuantizer(dim=768, bits=3)
    compressed = tq.encode(vectors)       # (n, 768) float32 → CompressedVectors
    decoded = tq.decode(compressed)       # CompressedVectors → (n, 768) float32
    sims = tq.cosine_similarity(query, compressed)  # approximate cosine sim
"""

from turboquant.quantizer import TurboQuantizer
from turboquant.storage import CompressedVectors

__version__ = "0.1.0"
__all__ = ["TurboQuantizer", "CompressedVectors"]
