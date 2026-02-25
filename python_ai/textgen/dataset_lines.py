from __future__ import annotations

"""Phase 2 dataset scaffold for line-level text-conditioned handwriting generation.

Expected dataset format (future):

dataset/
  lines/
    labels.csv        # columns: filename,text,split(optional)
    train/
      line_0001.png
      ...
    val/
      line_1001.png
      ...

Alternative flat format is also acceptable:
dataset/lines/labels.csv + dataset/lines/*.png

Prompt sheet collection method (recommended):
1. Create printed prompt sheets with one target text line per row.
2. Handwrite each prompt on the corresponding line.
3. Scan/photo the page.
4. Crop each row into an individual line image.
5. Save the exact typed prompt text in `labels.csv`.

This phase requires paired text+image line examples, not isolated letters.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import csv


@dataclass
class LineExample:
    image_path: Path
    text: str
    split: str = "train"


def load_line_examples(root: Path) -> List[LineExample]:
    """Load line-image/text pairs for future Phase 2 training.

    This loader is intentionally lightweight scaffolding. It validates the dataset
    format so data collection can start before the actual model is implemented.
    """
    root = Path(root)
    labels_csv = root / "labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"Missing {labels_csv}. Create dataset/lines/labels.csv with filename,text[,split]."
        )

    examples: List[LineExample] = []
    with labels_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            filename = row[0].strip()
            text = row[1]
            split = row[2].strip().lower() if len(row) >= 3 and row[2].strip() else "train"
            if not filename or not text:
                continue
            candidates = [
                root / filename,
                root / split / filename,
            ]
            image_path: Optional[Path] = None
            for candidate in candidates:
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path is None:
                # Keep going so users can validate partial datasets early.
                continue
            examples.append(LineExample(image_path=image_path, text=text, split=split))

    if not examples:
        raise ValueError(
            "No valid line examples found. Ensure labels.csv filenames match images under dataset/lines/."
        )
    return examples

