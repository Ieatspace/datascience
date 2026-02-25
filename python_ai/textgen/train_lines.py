from __future__ import annotations

"""Phase 2 training scaffold for sentence/line-level handwriting generation."""

import argparse
from pathlib import Path

from .dataset_lines import load_line_examples


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 2 line-level text-conditioned handwriting training (scaffold)")
    p.add_argument("--dataset-root", type=Path, default=Path("dataset") / "lines")
    p.add_argument("--device", default=None)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    examples = load_line_examples(Path(args.dataset_root))
    raise NotImplementedError(
        "Phase 2 is scaffolded only. Collected {count} line examples.\n"
        "Next steps:\n"
        "1. Implement a text-conditioned line generator in python_ai/textgen/model_lines.py\n"
        "2. Add a tokenizer + paired image/text training loop\n"
        "3. Train on dataset/lines with labels.csv mapping filename,text\n"
        "4. Integrate into generate_handwriting_page.py to replace per-letter assembly".format(
            count=len(examples)
        )
    )


if __name__ == "__main__":
    main()

