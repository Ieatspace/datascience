from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np

from .dataset import LETTERS
from .infer import DEFAULT_WEIGHTS_PATH, generate_glyph


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "out" / "generated"


def _compose_grid(images: List[np.ndarray], cols: int, cell_pad: int = 8) -> np.ndarray:
    if not images:
        raise ValueError("No images to compose")
    h, w = images[0].shape[:2]
    cols = max(1, int(cols))
    rows = int(np.ceil(len(images) / float(cols)))
    grid_h = rows * (h + cell_pad) + cell_pad
    grid_w = cols * (w + cell_pad) + cell_pad
    canvas = np.full((grid_h, grid_w, 3), 248, dtype=np.uint8)

    for idx, rgba in enumerate(images):
        r = idx // cols
        c = idx % cols
        y = cell_pad + r * (h + cell_pad)
        x = cell_pad + c * (w + cell_pad)
        alpha = (rgba[..., 3].astype(np.float32) / 255.0)[..., None]
        ink = np.full((h, w, 3), 24, dtype=np.float32)
        patch = canvas[y : y + h, x : x + w].astype(np.float32)
        canvas[y : y + h, x : x + w] = np.clip(patch * (1.0 - alpha) + ink * alpha, 0, 255).astype(np.uint8)
        cv2.rectangle(canvas, (x - 1, y - 1), (x + w, y + h), (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def _draw_label(canvas: np.ndarray, text: str) -> np.ndarray:
    out = canvas.copy()
    cv2.putText(out, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Sample learned glyphs from the letter model")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH)
    p.add_argument("--letter", default="a", help="'a'..'z' or 'all'")
    p.add_argument("--num", type=int, default=16)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--style-strength", type=float, default=1.0)
    p.add_argument("--output-size", type=int, default=64)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if args.letter.lower() == "all":
        imgs = []
        for i, letter in enumerate(LETTERS):
            glyph = generate_glyph(
                letter=letter,
                seed=int(args.seed) + i,
                style_strength=float(args.style_strength),
                output_size=int(args.output_size),
                weights_path=args.weights,
            )
            imgs.append(glyph.rgba)
        grid = _compose_grid(imgs, cols=max(1, int(args.cols)))
        grid = _draw_label(grid, "letter model samples: all")
        out_path = args.out or (DEFAULT_OUT_DIR / "letter_model_samples_all.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), grid):
            raise SystemExit("Failed to write sample grid: {0}".format(out_path))
        print("[sample] saved {0}".format(out_path))
        return

    letter = args.letter.lower()
    imgs: List[np.ndarray] = []
    for i in range(max(1, int(args.num))):
        glyph = generate_glyph(
            letter=letter,
            seed=int(args.seed) + i,
            style_strength=float(args.style_strength),
            output_size=int(args.output_size),
            weights_path=args.weights,
        )
        imgs.append(glyph.rgba)

    grid = _compose_grid(imgs, cols=max(1, int(args.cols)))
    grid = _draw_label(grid, "letter model samples: {0}".format(letter))
    out_path = args.out or (DEFAULT_OUT_DIR / "letter_model_samples_{0}.png".format(letter))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), grid):
        raise SystemExit("Failed to write sample grid: {0}".format(out_path))
    print("[sample] saved {0}".format(out_path))


if __name__ == "__main__":
    main()
