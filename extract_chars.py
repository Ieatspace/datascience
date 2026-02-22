import csv
from pathlib import Path

import cv2
import numpy as np

IMAGE_PATH = Path(r"data\page_1_abc.jpg")
OUT_DIR = Path("out/chars")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def threshold_image(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        12,
    )
    return gray, thresh


def find_text_line_boxes(thresh):
    h, w = thresh.shape
    kernel_w = max(25, w // 30)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
    merged = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    min_line_w = max(40, int(0.08 * w))
    min_line_h = max(8, int(0.012 * h))

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < min_line_w or bh < min_line_h:
            continue
        boxes.append((x, y, bw, bh))

    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def extract_char_boxes_from_line(line_bin):
    h, w = line_bin.shape

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(line_bin, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    _ = labels  # keep signature explicit; labels not used downstream

    boxes = []
    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]

        if area < 20:
            continue
        if bh < max(6, int(0.20 * h)):
            continue
        if bw < 3:
            continue
        if bw > 0.50 * w:
            continue

        boxes.append((x, y, bw, bh))

    boxes.sort(key=lambda b: b[0])

    merged = []
    for box in boxes:
        if not merged:
            merged.append(box)
            continue

        x, y, bw, bh = box
        px, py, pw, ph = merged[-1]

        if x <= px + pw + 3:
            nx = min(px, x)
            ny = min(py, y)
            nx2 = max(px + pw, x + bw)
            ny2 = max(py + ph, y + bh)
            merged[-1] = (nx, ny, nx2 - nx, ny2 - ny)
        else:
            merged.append(box)

    return merged


def pad_and_resize(bin_crop, out_size=64, pad=6):
    h, w = bin_crop.shape
    canvas = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.uint8)
    canvas[pad : pad + h, pad : pad + w] = bin_crop

    scale = min((out_size - 10) / canvas.shape[1], (out_size - 10) / canvas.shape[0])
    new_w = max(1, int(canvas.shape[1] * scale))
    new_h = max(1, int(canvas.shape[0] * scale))
    resized = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_AREA)

    out = np.zeros((out_size, out_size), dtype=np.uint8)
    y0 = (out_size - new_h) // 2
    x0 = (out_size - new_w) // 2
    out[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return out


def main():
    image_bgr = cv2.imread(str(IMAGE_PATH))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

    _, thresh = threshold_image(image_bgr)
    line_boxes = find_text_line_boxes(thresh)

    meta_rows = []
    idx = 0

    for li, (lx, ly, lw, lh) in enumerate(line_boxes):
        pad = 8
        x1 = max(0, lx - pad)
        y1 = max(0, ly - pad)
        x2 = min(thresh.shape[1], lx + lw + pad)
        y2 = min(thresh.shape[0], ly + lh + pad)
        line_bin = thresh[y1:y2, x1:x2]

        char_boxes = extract_char_boxes_from_line(line_bin)

        for cx, cy, cw, ch in char_boxes:
            cpad = 4
            cx1 = max(0, cx - cpad)
            cy1 = max(0, cy - cpad)
            cx2 = min(line_bin.shape[1], cx + cw + cpad)
            cy2 = min(line_bin.shape[0], cy + ch + cpad)
            char_bin = line_bin[cy1:cy2, cx1:cx2]

            char_img = pad_and_resize(char_bin, out_size=64, pad=6)

            fname = f"char_{idx:06d}.png"
            cv2.imwrite(str(OUT_DIR / fname), char_img)

            meta_rows.append([fname, li, lx, ly, lw, lh, cx, cy, cw, ch])
            idx += 1

    meta_path = OUT_DIR.parent / "chars_meta.csv"
    with open(meta_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "line_i", "line_x", "line_y", "line_w", "line_h", "cx", "cy", "cw", "ch"])
        writer.writerows(meta_rows)

    print(f"Saved {idx} character crops to {OUT_DIR}")
    print(f"Saved metadata to {meta_path}")
    print("Next: run label_chars.py to label these images.")


if __name__ == "__main__":
    main()
