from __future__ import annotations

"""Phase 2 model scaffold (line/sentence text-conditioned generator).

Planned direction (not implemented yet):
- Text encoder (char/byte tokenizer + embedding + transformer/GRU)
- Image generator backbone (likely diffusion or autoregressive raster model)
- Optional style encoder for writer-specific conditioning
- Training objective for paired text-line images

This file exists to document the intended interface and provide a stable place to
implement the future architecture without disturbing the Phase 1 letter generator.
"""


class LineGeneratorModel:  # pragma: no cover - scaffold only
    """Placeholder interface for the future line-level handwriting generator."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "Phase 2 line-level model is not implemented yet. Use python_ai.lettergen for Phase 1."
        )

