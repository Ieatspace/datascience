"""Letter-level generative handwriting model (conditional VAE).

Phase 1 exports:
- dataset loading from `out/labels.csv` + `out/chars`
- conditional VAE model and training
- inference helpers for generating new glyph pixels from latent samples
"""

from .dataset import LETTERS, LETTER_TO_INDEX, INDEX_TO_LETTER
from .infer import load_model, generate_letter

__all__ = [
    "LETTERS",
    "LETTER_TO_INDEX",
    "INDEX_TO_LETTER",
    "load_model",
    "generate_letter",
]

