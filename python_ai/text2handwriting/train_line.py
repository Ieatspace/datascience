from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 placeholder: full sentence / line handwriting model training"
    )
    parser.add_argument("--dataset", default="(future)")
    parser.parse_args()
    raise NotImplementedError(
        "Phase 2 is intentionally scaffolded only.\n"
        "You need paired data: one text label + one handwritten line image per sample.\n"
        "See README 'Phase 2: Full Sentence Model' for the expected format and collection workflow."
    )


if __name__ == "__main__":
    main()
