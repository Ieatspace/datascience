import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

import extract_chars as ec
from train_classifier import IMG_SIZE, SmallCNN


@dataclass
class PredChar:
    line_i: int
    x: int
    y: int
    w: int
    h: int
    pred: str
    conf: float
    topk: list[tuple[str, float]] | None = None


def parse_args():
    parser = argparse.ArgumentParser(description="Predict text from a handwriting page image.")
    parser.add_argument("image", help="Path to input page image (.jpg/.jpeg/.png)")
    parser.add_argument("--model", default="out/char_classifier.pt", help="Path to trained model checkpoint")
    parser.add_argument("--out-dir", default="out/predictions", help="Directory for text/debug outputs")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device",
    )
    parser.add_argument("--strict", action="store_true", help="Disable gentle scan mode")
    parser.add_argument("--dotted", action="store_true", help="Enable dotted-letter extraction mode (i/j pages)")
    parser.add_argument("--soft-dot-merge", action="store_true", help="Enable soft dot+stem merge in line mode")
    parser.add_argument("--no-debug", action="store_true", help="Skip saving debug overlay image")
    parser.add_argument("--no-spaces", action="store_true", help="Disable simple gap-based word spacing heuristic")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON to stdout (suppresses human log lines)",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save a JSON result file alongside the text/debug outputs",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save per-character predictions as CSV",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=1,
        help="Top-K predictions per character (1-5) for JSON/CSV output",
    )
    return parser.parse_args()


def resolve_device(name: str) -> str:
    if name == "cpu":
        return "cpu"
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path: Path, device: str):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train_classifier.py first.")

    ckpt = torch.load(model_path, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = SmallCNN(len(class_to_idx)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, idx_to_class


def predict_char(model, idx_to_class, device: str, char_img: np.ndarray, topk: int = 1):
    arr = char_img.astype(np.float32) / 255.0
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(probs.argmax(dim=1).item())
        conf = float(probs[0, pred_idx].item())
        k = max(1, min(int(topk), probs.shape[1], 5))
        topv, topi = torch.topk(probs, k=k, dim=1)
        topk_pairs = [
            (idx_to_class[int(topi[0, j].item())], float(topv[0, j].item()))
            for j in range(k)
        ]
    return idx_to_class[pred_idx], conf, topk_pairs


def add_spaces(chars: list[PredChar], enabled: bool) -> str:
    if not chars:
        return ""
    if not enabled or len(chars) == 1:
        return "".join(c.pred for c in chars)

    gaps = [chars[i].x - (chars[i - 1].x + chars[i - 1].w) for i in range(1, len(chars))]
    pos_gaps = [g for g in gaps if g > 0]
    if not pos_gaps:
        return "".join(c.pred for c in chars)

    # Conservative heuristic: only insert spaces when a gap is much larger than typical within-word spacing.
    gap_thresh = max(18.0, 2.2 * float(np.median(pos_gaps)))

    parts = [chars[0].pred]
    for i in range(1, len(chars)):
        if gaps[i - 1] > gap_thresh:
            parts.append(" ")
        parts.append(chars[i].pred)
    return "".join(parts)


def cluster_rows_for_dotted(chars: list[PredChar]) -> list[list[PredChar]]:
    if not chars:
        return []
    chars = sorted(chars, key=lambda c: (c.y, c.x))
    rows: list[list[PredChar]] = []
    row_centers: list[float] = []

    for ch in chars:
        cy = ch.y + (ch.h / 2.0)
        assigned = False
        for i, center in enumerate(row_centers):
            tol = max(20.0, 0.7 * np.median([c.h for c in rows[i]]))
            if abs(cy - center) <= tol:
                rows[i].append(ch)
                row_centers[i] = float(np.mean([c.y + (c.h / 2.0) for c in rows[i]]))
                assigned = True
                break
        if not assigned:
            rows.append([ch])
            row_centers.append(cy)

    return [sorted(row, key=lambda c: c.x) for row in rows]


def extract_candidates_normal(thresh: np.ndarray):
    chars: list[PredChar] = []
    line_boxes = ec.find_text_line_boxes(thresh)
    debug_line_boxes = []
    char_crops = []

    for li, (lx, ly, lw, lh) in enumerate(line_boxes):
        lx, ly, lw, lh = map(int, (lx, ly, lw, lh))
        debug_line_boxes.append((lx, ly, lw, lh))

        pad_x = 8
        pad_top = 24 if ec.DOTTED_LETTER_MODE else 8
        pad_bottom = 12 if ec.DOTTED_LETTER_MODE else 8
        x1 = max(0, lx - pad_x)
        y1 = max(0, ly - pad_top)
        x2 = min(thresh.shape[1], lx + lw + pad_x)
        y2 = min(thresh.shape[0], ly + lh + pad_bottom)
        line_bin = thresh[y1:y2, x1:x2]

        char_boxes = ec.extract_char_boxes_from_line(line_bin)
        for cx, cy, cw, ch in char_boxes:
            cx, cy, cw, ch = ec.expand_dot_box_to_stem(line_bin, (cx, cy, cw, ch))
            cx, cy, cw, ch = map(int, (cx, cy, cw, ch))

            cpad = 4
            cx1 = max(0, cx - cpad)
            cy1 = max(0, cy - cpad)
            cx2 = min(line_bin.shape[1], cx + cw + cpad)
            cy2 = min(line_bin.shape[0], cy + ch + cpad)
            char_bin = line_bin[cy1:cy2, cx1:cx2]
            char_img = ec.pad_and_resize(char_bin, out_size=IMG_SIZE, pad=6)

            page_x = x1 + cx1
            page_y = y1 + cy1
            page_w = cx2 - cx1
            page_h = cy2 - cy1

            char_crops.append((li, page_x, page_y, page_w, page_h, char_img))

    return char_crops, debug_line_boxes


def extract_candidates_dotted(thresh: np.ndarray):
    char_crops = []
    boxes = ec.extract_dotted_page_boxes(thresh)
    boxes = [ec.expand_dotted_page_crop(thresh, b) for b in boxes]
    boxes = ec.prune_dotted_boxes(boxes)
    boxes.sort(key=lambda b: (b[1], b[0]))

    for (x, y, w, h) in boxes:
        x, y, w, h = map(int, (x, y, w, h))
        cpad = 6
        x1 = max(0, x - cpad)
        y1 = max(0, y - cpad)
        x2 = min(thresh.shape[1], x + w + cpad)
        y2 = min(thresh.shape[0], y + h + cpad)
        char_bin = thresh[y1:y2, x1:x2]
        char_img = ec.pad_and_resize(char_bin, out_size=IMG_SIZE, pad=6)
        char_crops.append((-1, x1, y1, x2 - x1, y2 - y1, char_img))

    return char_crops, []


def draw_debug(image_bgr: np.ndarray, line_boxes, preds: list[PredChar]):
    out = image_bgr.copy()

    for (x, y, w, h) in line_boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 180, 255), 2)

    for ch in preds:
        cv2.rectangle(out, (ch.x, ch.y), (ch.x + ch.w, ch.y + ch.h), (0, 255, 0), 1)
        label = f"{ch.pred}:{ch.conf:.2f}"
        text_y = max(12, ch.y - 4)
        cv2.putText(
            out,
            label,
            (ch.x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (20, 20, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def save_char_csv(csv_path: Path, preds: list[PredChar]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["line_i,x,y,w,h,pred,conf"]
    for p in preds:
        lines.append(f"{p.line_i},{p.x},{p.y},{p.w},{p.h},{p.pred},{p.conf:.6f}")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_result_payload(
    *,
    image_path: Path,
    model_path: Path,
    device: str,
    text_path: Path,
    debug_path: Path | None,
    json_path: Path | None,
    csv_path: Path | None,
    line_texts: list[str],
    rows: list[list[PredChar]],
    preds: list[PredChar],
):
    avg_conf = (sum(p.conf for p in preds) / len(preds)) if preds else None
    return {
        "imagePath": str(image_path),
        "modelPath": str(model_path),
        "device": device,
        "detectedCharacters": len(preds),
        "averageConfidence": avg_conf,
        "textPath": str(text_path),
        "debugImagePath": str(debug_path) if debug_path else None,
        "jsonPath": str(json_path) if json_path else None,
        "csvPath": str(csv_path) if csv_path else None,
        "text": "\n".join(line_texts),
        "lines": [
            {
                "index": i,
                "text": line_texts[i],
                "chars": [
                    {
                        "x": c.x,
                        "y": c.y,
                        "w": c.w,
                        "h": c.h,
                        "pred": c.pred,
                        "conf": c.conf,
                        **(
                            {
                                "topk": [
                                    {"label": label, "conf": score}
                                    for (label, score) in (c.topk or [])
                                ]
                            }
                            if c.topk
                            else {}
                        ),
                    }
                    for c in row
                ],
            }
            for i, row in enumerate(rows)
        ],
        "extractor": {
            "gentle": bool(ec.GENTLE_SCAN_MODE),
            "dotted": bool(ec.DOTTED_LETTER_MODE),
            "softDotMerge": bool(ec.SOFT_DOT_MERGE),
        },
    }


def main():
    args = parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = resolve_device(args.device)
    topk = max(1, min(int(args.topk), 5))

    # Use extract_chars.py heuristics, with optional overrides for this inference run.
    ec.GENTLE_SCAN_MODE = not args.strict
    if args.dotted:
        ec.DOTTED_LETTER_MODE = True
    if args.soft_dot_merge:
        ec.SOFT_DOT_MERGE = True

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image with OpenCV: {image_path}")

    _, thresh = ec.threshold_image(image_bgr)
    model, idx_to_class = load_model(model_path, device)

    if ec.DOTTED_LETTER_MODE:
        char_crops, line_boxes = extract_candidates_dotted(thresh)
    else:
        char_crops, line_boxes = extract_candidates_normal(thresh)

    preds: list[PredChar] = []
    for li, x, y, w, h, char_img in char_crops:
        pred, conf, topk_pairs = predict_char(model, idx_to_class, device, char_img, topk=topk)
        preds.append(
            PredChar(
                li,
                int(x),
                int(y),
                int(w),
                int(h),
                pred,
                conf,
                topk_pairs if topk > 1 else None,
            )
        )

    if ec.DOTTED_LETTER_MODE:
        rows = cluster_rows_for_dotted(preds)
        line_texts = [add_spaces(row, enabled=(not args.no_spaces)) for row in rows]
    else:
        by_line: dict[int, list[PredChar]] = {}
        for p in preds:
            by_line.setdefault(p.line_i, []).append(p)
        rows = [sorted(by_line[i], key=lambda c: c.x) for i in sorted(by_line)]
        line_texts = [add_spaces(row, enabled=(not args.no_spaces)) for row in rows]

    text_out = "\n".join(line_texts)
    text_path = out_dir / f"{image_path.stem}.txt"
    text_path.write_text(text_out, encoding="utf-8")
    debug_path: Path | None = None
    if not args.no_debug:
        debug_img = draw_debug(image_bgr, line_boxes, preds)
        debug_path = out_dir / f"{image_path.stem}_debug.png"
        cv2.imwrite(str(debug_path), debug_img)

    csv_path: Path | None = None
    if args.save_csv:
        csv_path = out_dir / f"{image_path.stem}_chars.csv"
        save_char_csv(csv_path, preds)

    json_path: Path | None = None
    result_payload = build_result_payload(
        image_path=image_path,
        model_path=model_path,
        device=device,
        text_path=text_path,
        debug_path=debug_path,
        json_path=None,
        csv_path=csv_path,
        line_texts=line_texts,
        rows=rows,
        preds=preds,
    )
    if args.save_json:
        json_path = out_dir / f"{image_path.stem}.json"
        result_payload["jsonPath"] = str(json_path)
        json_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(result_payload))
        return

    print(f"Image: {image_path}")
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(
        f"Extractor modes: gentle={ec.GENTLE_SCAN_MODE} dotted={ec.DOTTED_LETTER_MODE} "
        f"soft_dot_merge={ec.SOFT_DOT_MERGE}"
    )
    print(f"Detected characters: {len(preds)}")
    if preds:
        avg_conf = sum(p.conf for p in preds) / len(preds)
        print(f"Average confidence: {avg_conf:.3f}")
    print(f"Saved text to: {text_path}")
    if csv_path:
        print(f"Saved char CSV to: {csv_path}")
    if json_path:
        print(f"Saved JSON to: {json_path}")
    print("Predicted text:")
    for i, line in enumerate(line_texts):
        print(f"line_{i}: {line}")
    if debug_path:
        print(f"Saved debug overlay to: {debug_path}")


if __name__ == "__main__":
    main()
