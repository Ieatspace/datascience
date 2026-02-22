import csv
from pathlib import Path

import cv2
import numpy as np

IMAGE_PATH = Path(r"data/page_3_ghi.jpg")
OUT_DIR = Path("out/chars")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SOFT_DOT_MERGE = False
DOTTED_LETTER_MODE = False
AGGRESSIVE_DOTTED_CROP_MODE = False


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


def make_char_filename(idx: int) -> str:
    page_tag = IMAGE_PATH.stem.replace(" ", "_")
    return f"{page_tag}_char_{idx:06d}.png"


def find_text_line_boxes(thresh):
    h, w = thresh.shape
    if DOTTED_LETTER_MODE:
        ink = (thresh > 0).astype(np.uint8)
        row_counts = ink.sum(axis=1).astype(np.float32)
        row_counts = np.convolve(row_counts, np.ones(5, dtype=np.float32) / 5.0, mode="same")
        active = row_counts > max(25, int(0.008 * w))

        # Only bridge tiny gaps; line crop padding will handle the i/j dot above the stem.
        max_gap = max(4, int(0.002 * h))
        i = 0
        while i < h:
            if active[i]:
                i += 1
                continue
            j = i
            while j < h and not active[j]:
                j += 1
            if i > 0 and j < h and (j - i) <= max_gap and active[i - 1] and active[j]:
                active[i:j] = True
            i = j

        boxes = []
        min_band_h = max(10, int(0.01 * h))
        y = 0
        while y < h:
            if not active[y]:
                y += 1
                continue
            y1 = y
            while y < h and active[y]:
                y += 1
            y2 = y
            if (y2 - y1) < min_band_h:
                continue

            band = ink[y1:y2, :]
            col_counts = band.sum(axis=0)
            active_cols = np.where(col_counts > 1)[0]
            if active_cols.size == 0:
                continue

            x1 = int(active_cols[0])
            x2 = int(active_cols[-1]) + 1
            boxes.append((x1, y1, x2 - x1, y2 - y1))

        boxes.sort(key=lambda b: (b[1], b[0]))
        return boxes

    kernel_w = max(25, w // 30)
    kernel_h = 7 if DOTTED_LETTER_MODE else 3
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    merged = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    if DOTTED_LETTER_MODE:
        # Grow line regions upward/downward a bit so i/j dots stay attached to the same line.
        line_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
        merged = cv2.dilate(merged, line_bridge, iterations=1)

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

    if DOTTED_LETTER_MODE:
        # Avoid aggressive opening (it can erase thin stems). Use a tiny cleanup only.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(line_bin, cv2.MORPH_OPEN, kernel, iterations=1)
        stem_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        cleaned = cv2.dilate(cleaned, stem_kernel, iterations=1)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(line_bin, cv2.MORPH_OPEN, kernel, iterations=1)

    if SOFT_DOT_MERGE:
        # Light vertical close helps reconnect dotted letters (i/j) without heavy over-merge.
        dot_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, dot_kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    _ = labels  # keep signature explicit; labels not used downstream

    boxes = []
    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]

        min_area = 6 if SOFT_DOT_MERGE else 20
        min_h = max(2, int(0.05 * h)) if SOFT_DOT_MERGE else max(6, int(0.20 * h))
        min_w = 2 if SOFT_DOT_MERGE else 3

        if area < min_area:
            continue
        if bh < min_h:
            continue
        if bw < min_w:
            continue
        if bw > 0.50 * w:
            continue

        boxes.append((x, y, bw, bh))

    if DOTTED_LETTER_MODE:
        split_boxes = []
        for x, y, bw, bh in boxes:
            tall_thresh = max(120, int(0.28 * h))
            if bh < tall_thresh:
                split_boxes.append((x, y, bw, bh))
                continue

            comp = cleaned[y : y + bh, x : x + bw]
            row_counts = comp.sum(axis=1).astype(np.float32) / 255.0
            row_counts = np.convolve(row_counts, np.ones(3, dtype=np.float32) / 3.0, mode="same")
            active_rows = row_counts > max(2, int(0.08 * bw))

            # Bridge tiny gaps inside a single character shape, but preserve row-to-row gaps.
            ry = 0
            while ry < bh:
                if active_rows[ry]:
                    ry += 1
                    continue
                rz = ry
                while rz < bh and not active_rows[rz]:
                    rz += 1
                if ry > 0 and rz < bh and (rz - ry) <= 2 and active_rows[ry - 1] and active_rows[rz]:
                    active_rows[ry:rz] = True
                ry = rz

            segments = []
            ry = 0
            while ry < bh:
                if not active_rows[ry]:
                    ry += 1
                    continue
                sy1 = ry
                while ry < bh and active_rows[ry]:
                    ry += 1
                sy2 = ry
                if (sy2 - sy1) < 8:
                    continue

                seg = comp[sy1:sy2, :]
                cols = np.where(seg.sum(axis=0) > 0)[0]
                if cols.size == 0:
                    continue
                sx1 = int(cols[0])
                sx2 = int(cols[-1]) + 1
                segments.append((x + sx1, y + sy1, sx2 - sx1, sy2 - sy1))

            # Keep split result only if it truly broke a tall column into multiple pieces.
            if len(segments) >= 2:
                split_boxes.extend(segments)
            else:
                split_boxes.append((x, y, bw, bh))

        boxes = split_boxes

    boxes.sort(key=lambda b: b[0])

    def should_merge(prev, cur):
        px, py, pw, ph = prev
        x, y, bw, bh = cur

        horizontal_gap = x - (px + pw)
        x_overlap = max(0, min(px + pw, x + bw) - max(px, x))
        overlap_ratio = x_overlap / max(1, min(pw, bw))
        vertical_gap = max(0, max(py, y) - min(py + ph, y + bh))
        y_overlap = max(0, min(py + ph, y + bh) - max(py, y))
        y_overlap_ratio = y_overlap / max(1, min(ph, bh))

        # Broken-stroke merge, but avoid aggressive chaining in dotted-letter mode.
        if (
            not DOTTED_LETTER_MODE
            and horizontal_gap <= 3
            and (vertical_gap <= max(8, int(0.08 * h)) or y_overlap_ratio >= 0.25)
        ):
            return True

        if not SOFT_DOT_MERGE:
            return False

        # Dot-over-stem merge: high x overlap and a small vertical separation.
        small_piece = min(ph, bh) <= max(8, int(0.35 * h))
        close_vertically = vertical_gap <= max(16, int(0.35 * h))
        aligned_x = overlap_ratio >= 0.4
        not_too_far = horizontal_gap <= max(2, int(0.03 * w))
        return small_piece and close_vertically and aligned_x and not_too_far

    merged = []
    for box in boxes:
        if not merged:
            merged.append(box)
            continue

        x, y, bw, bh = box
        px, py, pw, ph = merged[-1]

        if should_merge(merged[-1], box):
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


def expand_dot_box_to_stem(line_bin, box):
    if not DOTTED_LETTER_MODE:
        return box

    x, y, w, h = [int(v) for v in box]
    line_h, line_w = line_bin.shape

    # Heuristic: dot-like components are short and small.
    if h > max(28, int(0.12 * line_h)):
        return box
    if (w * h) > max(900, int(0.02 * line_h * line_w)):
        return box

    search_x_pad = max(10, 2 * w)
    search_y_down = max(60, int(0.35 * line_h))

    sx1 = max(0, x - search_x_pad)
    sx2 = min(line_w, x + w + search_x_pad)
    sy1 = y
    sy2 = min(line_h, y + h + search_y_down)
    roi = line_bin[sy1:sy2, sx1:sx2]

    ys, xs = np.where(roi > 0)
    if xs.size == 0:
        return box

    # Only consider ink below the current dot to avoid grabbing neighboring dots.
    below_mask = ys >= h
    if not np.any(below_mask):
        return box

    ys = ys[below_mask]
    xs = xs[below_mask]

    # Favor stem-like ink roughly under the dot's x-center.
    dot_center_x = x + w / 2.0
    abs_xs = sx1 + xs
    x_dist = np.abs(abs_xs - dot_center_x)
    keep = x_dist <= max(12, 2.5 * w)
    if not np.any(keep):
        return box

    ys = ys[keep]
    xs = xs[keep]
    if xs.size == 0:
        return box

    nx1 = min(x, sx1 + int(xs.min()))
    nx2 = max(x + w, sx1 + int(xs.max()) + 1)
    ny1 = y
    ny2 = max(y + h, sy1 + int(ys.max()) + 1)
    return (nx1, ny1, nx2 - nx1, ny2 - ny1)


def extract_dotted_page_boxes(thresh):
    """Page-level fallback for repeated dotted-letter practice pages (i/j)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    comps = []
    for i in range(1, num_labels):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if area < 6 or w < 2 or h < 2:
            continue
        comps.append((x, y, w, h, area))

    dots = []
    stems = []
    for c in comps:
        x, y, w, h, area = c
        if h <= 24 and area <= 150:
            dots.append(c)
        else:
            stems.append(c)

    used_stems = set()
    merged = []
    for dot in sorted(dots, key=lambda t: (t[1], t[0])):
        dx, dy, dw, dh, _ = dot
        dot_center_x = dx + dw / 2.0
        best_idx = None
        best_score = None

        for si, stem in enumerate(stems):
            if si in used_stems:
                continue
            sx, sy, sw, sh, _ = stem
            if sy <= dy:
                continue

            gap = sy - (dy + dh)
            if gap < 0 or gap > (220 if AGGRESSIVE_DOTTED_CROP_MODE else 140):
                continue

            x_overlap = max(0, min(dx + dw, sx + sw) - max(dx, sx))
            min_overlap = 0 if AGGRESSIVE_DOTTED_CROP_MODE else max(1, int(0.25 * min(dw, sw)))
            if x_overlap < min_overlap:
                continue

            center_dist = abs((sx + sw / 2.0) - dot_center_x)
            max_center_dist = max(28, 3 * max(dw, sw)) if AGGRESSIVE_DOTTED_CROP_MODE else max(18, 2 * max(dw, sw))
            if center_dist > max_center_dist:
                continue

            # Favor a close stem directly underneath the dot.
            score = (gap * 3) + center_dist
            if best_score is None or score < best_score:
                best_score = score
                best_idx = si

        if best_idx is None:
            continue

        used_stems.add(best_idx)
        sx, sy, sw, sh, _ = stems[best_idx]
        x1 = min(dx, sx)
        y1 = min(dy, sy)
        x2 = max(dx + dw, sx + sw)
        y2 = max(dy + dh, sy + sh)
        merged.append((x1, y1, x2 - x1, y2 - y1))

    # Keep unmatched stems too (some i stems may already include the dot or dot may be faint).
    for si, stem in enumerate(stems):
        if si in used_stems:
            continue
        x, y, w, h, _ = stem
        merged.append((x, y, w, h))

    # Deduplicate near-identical boxes.
    merged.sort(key=lambda b: (b[1], b[0]))
    deduped = []
    for box in merged:
        x, y, w, h = box
        matched = False
        for j, (px, py, pw, ph) in enumerate(deduped):
            x_overlap = max(0, min(x + w, px + pw) - max(x, px))
            y_overlap = max(0, min(y + h, py + ph) - max(y, py))
            inter = x_overlap * y_overlap
            union = (w * h) + (pw * ph) - inter
            iou = inter / max(1, union)
            if iou > 0.75:
                deduped[j] = (
                    min(x, px),
                    min(y, py),
                    max(x + w, px + pw) - min(x, px),
                    max(y + h, py + ph) - min(y, py),
                )
                matched = True
                break
        if not matched:
            deduped.append(box)

    return deduped


def expand_dotted_page_crop(thresh, box):
    """Temporary aggressive crop expansion for i/j pages to include the stem under the dot."""
    if not (DOTTED_LETTER_MODE and AGGRESSIVE_DOTTED_CROP_MODE):
        return tuple(int(v) for v in box)

    x, y, w, h = [int(v) for v in box]
    H, W = thresh.shape

    # Only aggressively expand likely dot/small fragments.
    if h > 64 and w > 28:
        return (x, y, w, h)

    x_center = x + w / 2.0
    sx1 = max(0, x - max(18, 4 * w))
    sx2 = min(W, x + w + max(18, 4 * w))
    sy1 = y
    sy2 = min(H, y + h + max(240, 12 * h))
    roi = thresh[sy1:sy2, sx1:sx2]

    # Prefer a stem-shaped connected component below the current box.
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats((roi > 0).astype(np.uint8), connectivity=8)
    best = None
    best_score = None
    for i in range(1, num_labels):
        rx, ry, rw, rh, area = [int(v) for v in stats[i]]
        if area < 6:
            continue
        if ry + rh <= h:
            continue
        if ry <= h and (ry + rh) <= (h + 4):
            continue

        comp_center_x = sx1 + rx + (rw / 2.0)
        center_dist = abs(comp_center_x - x_center)
        if center_dist > max(22, 5 * w):
            continue

        gap = max(0, ry - h)
        if gap > max(220, 10 * h):
            continue

        score = (gap * 3) + center_dist - (0.8 * rh)
        if best_score is None or score < best_score:
            best_score = score
            best = (rx, ry, rw, rh)

    if best is not None:
        rx, ry, rw, rh = best
        nx1 = min(x, sx1 + rx)
        nx2 = max(x + w, sx1 + rx + rw)
        ny1 = y
        ny2 = max(y + h, sy1 + ry + rh)
    else:
        ys, xs = np.where(roi > 0)
        if xs.size == 0:
            return (x, y, w, h)

        below = ys >= h
        if not np.any(below):
            return (x, y, w, h)
        ys = ys[below]
        xs = xs[below]

        abs_x = sx1 + xs
        near_center = np.abs(abs_x - x_center) <= max(18, 4 * w)
        if not np.any(near_center):
            return (x, y, w, h)

        ys = ys[near_center]
        xs = xs[near_center]
        abs_x = sx1 + xs

        nx1 = min(x, int(abs_x.min()))
        nx2 = max(x + w, int(abs_x.max()) + 1)
        ny1 = y
        ny2 = max(y + h, sy1 + int(ys.max()) + 1)

    # Small padding so the crop includes full strokes.
    pad_x = 4
    pad_y = 4
    nx1 = max(0, nx1 - pad_x)
    ny1 = max(0, ny1 - pad_y)
    nx2 = min(W, nx2 + pad_x)
    ny2 = min(H, ny2 + pad_y)
    return (nx1, ny1, nx2 - nx1, ny2 - ny1)


def prune_dotted_boxes(boxes):
    boxes = [tuple(int(v) for v in b) for b in boxes]
    keep = [True] * len(boxes)
    for i, (x, y, w, h) in enumerate(boxes):
        if not keep[i]:
            continue
        area = w * h
        if not (h <= 42 and area <= 1800):
            continue

        for j, (px, py, pw, ph) in enumerate(boxes):
            if i == j or not keep[j]:
                continue
            if (pw * ph) <= area:
                continue

            x_overlap = max(0, min(x + w, px + pw) - max(x, px))
            y_overlap = max(0, min(y + h, py + ph) - max(y, py))
            inter = x_overlap * y_overlap
            containment = inter / max(1, area)
            if containment >= 0.7:
                keep[i] = False
                break

    return [b for b, k in zip(boxes, keep) if k]


def main():
    image_bgr = cv2.imread(str(IMAGE_PATH))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

    _, thresh = threshold_image(image_bgr)

    if DOTTED_LETTER_MODE:
        char_boxes = extract_dotted_page_boxes(thresh)
        char_boxes = [expand_dotted_page_crop(thresh, box) for box in char_boxes]
        char_boxes = prune_dotted_boxes(char_boxes)
        char_boxes.sort(key=lambda b: (b[1], b[0]))
        meta_rows = []
        idx = 0

        for cx, cy, cw, ch in char_boxes:
            cpad = 6
            cx1 = max(0, cx - cpad)
            cy1 = max(0, cy - cpad)
            cx2 = min(thresh.shape[1], cx + cw + cpad)
            cy2 = min(thresh.shape[0], cy + ch + cpad)
            char_bin = thresh[cy1:cy2, cx1:cx2]

            char_img = pad_and_resize(char_bin, out_size=64, pad=6)
            fname = make_char_filename(idx)
            cv2.imwrite(str(OUT_DIR / fname), char_img)

            meta_rows.append([fname, -1, 0, 0, thresh.shape[1], thresh.shape[0], cx, cy, cw, ch])
            idx += 1

        meta_path = OUT_DIR.parent / "chars_meta.csv"
        with open(meta_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "line_i", "line_x", "line_y", "line_w", "line_h", "cx", "cy", "cw", "ch"])
            writer.writerows(meta_rows)

        print(f"Saved {idx} character crops to {OUT_DIR}")
        print(f"Saved metadata to {meta_path}")
        print("Dotted-letter mode: used page-level dot+stem pairing.")
        print("Next: run label_chars.py to label these images.")
        return

    line_boxes = find_text_line_boxes(thresh)

    meta_rows = []
    idx = 0

    for li, (lx, ly, lw, lh) in enumerate(line_boxes):
        pad_x = 8
        pad_top = 24 if DOTTED_LETTER_MODE else 8
        pad_bottom = 12 if DOTTED_LETTER_MODE else 8
        x1 = max(0, lx - pad_x)
        y1 = max(0, ly - pad_top)
        x2 = min(thresh.shape[1], lx + lw + pad_x)
        y2 = min(thresh.shape[0], ly + lh + pad_bottom)
        line_bin = thresh[y1:y2, x1:x2]

        char_boxes = extract_char_boxes_from_line(line_bin)

        for cx, cy, cw, ch in char_boxes:
            cx, cy, cw, ch = expand_dot_box_to_stem(line_bin, (cx, cy, cw, ch))
            cpad = 4
            cx1 = max(0, cx - cpad)
            cy1 = max(0, cy - cpad)
            cx2 = min(line_bin.shape[1], cx + cw + cpad)
            cy2 = min(line_bin.shape[0], cy + ch + cpad)
            char_bin = line_bin[cy1:cy2, cx1:cx2]

            char_img = pad_and_resize(char_bin, out_size=64, pad=6)

            fname = make_char_filename(idx)
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
