from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class LineSample:
    text: str
    image_path: Path


def expected_dataset_format() -> str:
    return (
        "Expected Phase 2 dataset format:\n"
        "- metadata CSV/JSONL with columns: split,text,image_path\n"
        "- one line image per sample (single handwritten line)\n"
        "- text must exactly match the image content\n"
        "- images should be cropped to one line with consistent padding"
    )


def load_line_dataset(*args, **kwargs) -> List[LineSample]:
    raise NotImplementedError(
        "Phase 2 line dataset loader is scaffolded only. Collect paired line text+image data first.\n"
        + expected_dataset_format()
    )
