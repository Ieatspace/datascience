from __future__ import annotations

import argparse
import csv
import hashlib
import json
import secrets
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
except Exception:
    torch = None

ROOT = Path(__file__).resolve().parent
LABELS_CSV = ROOT / "out" / "labels.csv"
CHARS_DIR = ROOT / "out" / "chars"
MODEL_PATH = ROOT / "out" / "char_classifier.pt"
LETTER_MODEL_PATH = ROOT / "out" / "letter_gen.pt"

STYLES = {
    "pencil": dict(color=(108, 112, 122), opacity=0.78, rot=3.6, jx=1.5, jy=1.6, sj=0.08, gap=0.02, dil=0, blur=0.35, join=1, grain=True, bleed=0.0),
    "ink": dict(color=(28, 31, 36), opacity=0.95, rot=2.2, jx=1.1, jy=1.1, sj=0.05, gap=0.00, dil=0, blur=0.00, join=1, grain=False, bleed=0.0),
    "marker": dict(color=(38, 45, 56), opacity=0.92, rot=2.8, jx=1.3, jy=1.2, sj=0.06, gap=0.03, dil=1, blur=0.65, join=2, grain=False, bleed=0.15),
}

_CACHE_KEY = None
_CACHE_POOLS = None
_LETTER_INFER_MOD = None
_LETTER_INFER_IMPORT_FAILED = False


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def hash_seed(text, style, width, height, line_spacing):
    raw = json.dumps(
        {"text": text, "style": style, "width": width, "height": height, "lineSpacing": round(float(line_spacing), 4)},
        sort_keys=True,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % 2_147_483_647


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--style", choices=sorted(STYLES), default="ink")
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--line-spacing", type=float, required=True, dest="line_spacing")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--json", action="store_true")
    p.add_argument("--debug-json", type=Path, default=None)
    p.add_argument("--use-classifier", action="store_true")
    p.add_argument("--use-letter-model", action="store_true")
    p.add_argument("--letter-model-weights", type=Path, default=LETTER_MODEL_PATH)
    p.add_argument("--letter-style-strength", type=float, default=1.0)
    p.add_argument("--baseline-jitter", type=float, default=1.0)
    p.add_argument("--word-slant", type=float, default=1.0)
    p.add_argument("--letter-rot-jitter", type=float, default=1.0)
    p.add_argument("--ink-variation", type=float, default=0.12)
    p.add_argument("--letter-model-min-samples", type=int, default=4)
    return p.parse_args()


def get_letter_infer_module():
    global _LETTER_INFER_MOD, _LETTER_INFER_IMPORT_FAILED
    if _LETTER_INFER_MOD is not None:
        return _LETTER_INFER_MOD
    if _LETTER_INFER_IMPORT_FAILED:
        return None
    try:
        from python_ai import infer as letter_infer

        _LETTER_INFER_MOD = letter_infer
        return _LETTER_INFER_MOD
    except Exception:
        _LETTER_INFER_IMPORT_FAILED = True
        return None


def mask_bbox(mask):
    ys, xs = np.where(mask > 20)
    if xs.size == 0:
        h, w = mask.shape[:2]
        return 0, 0, w, h
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def pair_kerning_adjust(prev_ch, ch, font_px):
    if not prev_ch or not ch:
        return 0.0
    pair = f"{prev_ch}{ch}"
    tight = {"ri", "rn", "rm", "rv", "rw", "ty", "yo", "wa", "wo", "ov", "ve", "ll", "tt", "ff"}
    loose = {"mm", "mw", "wm", "ww", "nn", "mn"}
    if pair in tight:
        return -0.06 * float(font_px)
    if pair in loose:
        return 0.03 * float(font_px)
    if prev_ch in "ilt" and ch in ".,;:'`\"":
        return -0.04 * float(font_px)
    return 0.0


def maybe_apply_ink_variation(mask, np_rng, amount):
    amt = clamp(float(amount), 0.0, 1.0)
    if amt <= 0.0:
        return mask
    out = mask.copy()
    if np_rng.random() < (0.22 + 0.38 * amt):
        if np_rng.random() < 0.5:
            out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        else:
            out = cv2.erode(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    if np_rng.random() < (0.20 + 0.25 * amt):
        sigma = 0.15 + 0.70 * amt
        out = cv2.GaussianBlur(out, (3, 3), sigma)
    return out


def load_samples(use_classifier=True):
    global _CACHE_KEY, _CACHE_POOLS
    if not LABELS_CSV.exists() or not CHARS_DIR.exists():
        raise FileNotFoundError("Missing out/labels.csv or out/chars")
    key = (LABELS_CSV.stat().st_mtime_ns, len(list(CHARS_DIR.glob("*.png"))), MODEL_PATH.exists(), MODEL_PATH.stat().st_mtime_ns if MODEL_PATH.exists() else 0, bool(use_classifier))
    if _CACHE_KEY == key and _CACHE_POOLS is not None:
        return _CACHE_POOLS

    avail = {p.name for p in CHARS_DIR.glob("*.png")}
    latest = {}
    with open(LABELS_CSV, "r", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) < 2:
                continue
            fn, lab = row[0].strip(), row[1].strip().lower()
            if fn not in avail or len(lab) != 1 or not ("a" <= lab <= "z"):
                continue
            latest[fn] = lab
    items = []
    for fn, lab in latest.items():
        img = cv2.imread(str(CHARS_DIR / fn), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.shape != (64, 64):
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        m = (img > 15).astype(np.uint8) * 255
        ys, xs = np.where(m > 0)
        if xs.size == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        crop = m[y1:y2, x1:x2]
        bh, bw = crop.shape
        area = int(np.count_nonzero(crop))
        if bh <= 0 or bw <= 0 or area < 6:
            continue
        occ = area / max(1, bw * bh)
        touch = int(x1 == 0) + int(y1 == 0) + int(x2 >= 64) + int(y2 >= 64)
        cys, cxs = np.where(crop > 0)
        base_ratio = (float(np.percentile(cys, 90)) if cys.size else bh - 1) / max(1.0, bh - 1)
        items.append(
            dict(
                filename=fn, label=lab, img=img, crop=crop, bw=bw, bh=bh, area=area, occ=occ, touch=touch,
                aspect=bw / max(1.0, bh), base_ratio=base_ratio, adv_ratio=(bw + max(1.0, 0.12 * bh)) / max(1.0, bh)
            )
        )
    if not items:
        raise ValueError("No lowercase labeled glyphs found")

    # optional classifier confidence
    conf_map = {}
    pred_map = {}
    if use_classifier and torch is not None and MODEL_PATH.exists():
        try:
            from train_classifier import SmallCNN

            ckpt = torch.load(MODEL_PATH, map_location="cpu")
            c2i = ckpt.get("class_to_idx", {})
            if isinstance(c2i, dict):
                i2c = {int(v): str(k) for k, v in c2i.items() if isinstance(v, int)}
                model = SmallCNN(len(c2i))
                model.load_state_dict(ckpt["state_dict"])
                model.eval()
                arr = np.stack([it["img"].astype(np.float32) / 255.0 for it in items], axis=0)
                bs = 256
                for s in range(0, len(items), bs):
                    e = min(len(items), s + bs)
                    x = torch.from_numpy(arr[s:e]).unsqueeze(1)
                    with torch.no_grad():
                        probs = torch.softmax(model(x), dim=1).numpy()
                    for i, it in enumerate(items[s:e], start=s):
                        rowp = probs[i - s]
                        idx = c2i.get(it["label"])
                        conf_map[it["filename"]] = float(rowp[idx]) if idx is not None and idx < probs.shape[1] else 0.5
                        top_idx = int(np.argmax(rowp))
                        pred_map[it["filename"]] = (i2c.get(top_idx), float(rowp[top_idx]))
        except Exception:
            conf_map = {}
            pred_map = {}

    med_aspect = {}
    for ch in "abcdefghijklmnopqrstuvwxyz":
        vals = [it["aspect"] for it in items if it["label"] == ch]
        if vals:
            med_aspect[ch] = float(np.median(vals))
    for it in items:
        conf = conf_map.get(it["filename"], 0.70)
        pred_lab, pred_conf = pred_map.get(it["filename"], (None, None))
        occ_score = 0.35 + 0.65 * (1.0 - clamp(abs(it["occ"] - 0.42) / 0.42, 0, 1))
        ar_ref = med_aspect.get(it["label"], it["aspect"])
        ar_score = 0.25 + 0.75 * (1.0 - clamp(abs(it["aspect"] - ar_ref) / max(0.2, ar_ref), 0, 1))
        mismatch_pen = 0.0
        if pred_lab is not None and pred_lab != it["label"]:
            mismatch_pen = 0.30 + 0.55 * clamp(pred_conf if pred_conf is not None else 0.0, 0.0, 1.0)
        clip_pen = 0.08 * it["touch"]
        w = (
            (0.47 * clamp(conf, 0.05, 1.0))
            + (0.28 * occ_score)
            + (0.25 * ar_score)
            - clip_pen
            - mismatch_pen
        )
        it["weight"] = max(0.05, float(w))
        it["conf"] = conf_map.get(it["filename"])
        it["pred_label"] = pred_lab
        it["pred_conf"] = pred_conf

    pools = {}
    for ch in "abcdefghijklmnopqrstuvwxyz":
        group = [it for it in items if it["label"] == ch and it["occ"] > 0.02]
        if not group:
            continue
        # Aggressively filter likely mislabeled crops if the classifier strongly disagrees.
        if use_classifier:
            filtered = [
                it
                for it in group
                if not (
                    it.get("pred_label") not in (None, it["label"])
                    and (it.get("pred_conf") or 0.0) >= 0.62
                )
            ]
            if len(filtered) >= max(8, int(0.6 * len(group))):
                group = filtered
            conf_vals = [float(it["conf"]) for it in group if it.get("conf") is not None]
            if len(conf_vals) >= 10:
                conf_floor = max(0.34, float(np.percentile(conf_vals, 18)))
                filtered = [it for it in group if (it.get("conf") is None) or float(it["conf"]) >= conf_floor]
            if len(filtered) >= max(8, int(0.65 * len(group))):
                group = filtered
        # Prototype similarity penalty to reject outlier shapes within a letter pool.
        if len(group) >= 6:
            vecs = []
            for it in group:
                proto_im = cv2.resize(it["crop"], (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                v = proto_im.reshape(-1)
                n = float(np.linalg.norm(v))
                vecs.append(v / n if n > 1e-8 else v)
            M = np.stack(vecs, axis=0)
            mean_v = M.mean(axis=0)
            mean_n = float(np.linalg.norm(mean_v))
            if mean_n > 1e-8:
                mean_v = mean_v / mean_n
                sims = np.clip(M @ mean_v, 0.0, 1.0)
                sim_floor = float(np.percentile(sims, 18))
                new_group = []
                for it, sim in zip(group, sims.tolist()):
                    it["proto_sim"] = float(sim)
                    it["weight"] = max(0.03, float(it["weight"]) * (0.80 + 0.55 * float(sim)))
                    if sim >= max(0.32, sim_floor - 0.02):
                        new_group.append(it)
                if len(new_group) >= max(8, int(0.7 * len(group))):
                    group = new_group
        # Keep a reasonably clean subset so generation favors readable samples.
        group.sort(key=lambda d: d["weight"], reverse=True)
        keep = max(8, int(round(len(group) * 0.75)))
        if len(group) > keep:
            group = group[:keep]
        weights = np.array([d["weight"] for d in group], dtype=np.float32)
        weights = np.power(np.clip(weights, 1e-5, None), 1.35)
        weights = weights / max(1e-8, float(weights.sum()))
        pools[ch] = dict(
            items=group,
            weights=weights,
            median_adv=float(np.median([d["adv_ratio"] for d in group])),
            median_aspect=float(np.median([d["aspect"] for d in group])),
            median_bh=float(np.median([d["bh"] for d in group])),
        )
    _CACHE_KEY, _CACHE_POOLS = key, pools
    return pools


def sample_pool_item(pool, np_rng, recent_filenames=None):
    items = pool["items"]
    weights = pool["weights"]
    if len(items) == 1:
        return items[0]
    if recent_filenames:
        mask = np.array([it["filename"] not in recent_filenames for it in items], dtype=bool)
        if int(mask.sum()) >= max(1, int(0.35 * len(items))):
            idxs = np.flatnonzero(mask)
            subw = weights[idxs].astype(np.float64)
            subw = subw / max(1e-12, float(subw.sum()))
            return items[int(idxs[int(np_rng.choice(len(idxs), p=subw))])]
    idx = int(np_rng.choice(len(items), p=weights))
    return items[idx]


def width_est(ch, font_px, pools):
    if ch == " ":
        return font_px * 0.42
    if ch in pools:
        return max(font_px * 0.18, pools[ch]["median_adv"] * font_px * 1.02)
    if ch in "ilI'`.,;:!|":
        return font_px * 0.22
    if ch in "mwMW@#%&":
        return font_px * 0.68
    if ch.isupper():
        return font_px * 0.52
    if ch.isdigit():
        return font_px * 0.46
    return font_px * 0.50


def measure(s, font_px, pools):
    return sum(width_est(c, font_px, pools) for c in s)


def wrap_text(text, max_w, font_px, max_rows, pools):
    rows = []
    for para in text.replace("\r\n", "\n").split("\n"):
        if not para.strip():
            chunked = [""]
        else:
            words, chunked, cur = para.strip().split(), [], ""
            for w in words:
                t = f"{cur} {w}" if cur else w
                if measure(t, font_px, pools) <= max_w:
                    cur = t
                    continue
                if cur:
                    chunked.append(cur)
                    cur = ""
                if measure(w, font_px, pools) <= max_w:
                    cur = w
                    continue
                part = ""
                for ch in w:
                    cand = part + ch
                    if part and measure(cand, font_px, pools) > max_w:
                        chunked.append(part)
                        part = ch
                    else:
                        part = cand
                cur = part
            if cur:
                chunked.append(cur)
        for r in chunked:
            if len(rows) >= max_rows:
                return rows, True
            rows.append(r)
    return rows, False


def bg_page(width, height, baseline_step, top_pad, np_rng):
    c = np.full((height, width, 3), (248, 246, 240), np.uint8)
    n = np_rng.normal(0, 2.0, size=(height, width, 1)).astype(np.float32)
    c = np.clip(c.astype(np.float32) + n, 0, 255).astype(np.uint8)
    y = top_pad + int(0.62 * baseline_step)
    while y < height - 24:
        cv2.line(c, (28, int(y)), (width - 28, int(y)), (228, 220, 204), 1, cv2.LINE_AA)
        y += baseline_step
    cv2.line(c, (56, 20), (56, height - 20), (210, 170, 170), 1, cv2.LINE_AA)
    return c


def style_mask(mask, style, np_rng):
    s = STYLES[style]
    out = mask.copy()
    if s["dil"] > 0:
        out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=int(s["dil"]))
    if s["blur"] > 0:
        out = cv2.GaussianBlur(out, (3 if s["blur"] <= 0.6 else 5, 3 if s["blur"] <= 0.6 else 5), s["blur"])
    if s["grain"]:
        out = np.clip(out.astype(np.float32) * np_rng.uniform(0.82, 1.02, size=out.shape), 0, 255).astype(np.uint8)
    if s["bleed"] > 0:
        out = cv2.addWeighted(out, 1.0, cv2.GaussianBlur(out, (5, 5), 1.0), float(s["bleed"]), 0)
    return out


def composite(canvas, mask, x, y, style, opacity_scale=1.0):
    h, w = mask.shape
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
    if x1 >= x2 or y1 >= y2:
        return
    mx1, my1 = x1 - x, y1 - y
    sub = (mask[my1:my1 + (y2 - y1), mx1:mx1 + (x2 - x1)].astype(np.float32) / 255.0)[..., None]
    a = clamp(STYLES[style]["opacity"] * opacity_scale, 0, 1)
    alpha = sub * a
    if not np.any(alpha > 0):
        return
    col = np.array(STYLES[style]["color"], dtype=np.float32).reshape(1, 1, 3)
    patch = canvas[y1:y2, x1:x2].astype(np.float32)
    canvas[y1:y2, x1:x2] = np.clip(patch * (1 - alpha) + col * alpha, 0, 255).astype(np.uint8)


def tf_mask(mask, angle, sx, sy):
    h, w = mask.shape
    sw, sh = max(1, int(round(w * sx))), max(1, int(round(h * sy)))
    m = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_CUBIC)
    ctr = (sw / 2.0, sh / 2.0)
    M = cv2.getRotationMatrix2D(ctr, angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    bw, bh = int(sh * sin + sw * cos) + 2, int(sh * cos + sw * sin) + 2
    M[0, 2] += bw / 2.0 - ctr[0]
    M[1, 2] += bh / 2.0 - ctr[1]
    return cv2.warpAffine(m, M, (bw, bh), flags=cv2.INTER_CUBIC, borderValue=0)


def anchors(mask, x0, y0):
    ys, xs = np.where(mask > 20)
    if xs.size == 0:
        return None, None
    mnx, mxx = int(xs.min()), int(xs.max())
    span = max(1, mxx - mnx)
    base = float(np.percentile(ys, 90))
    band = np.abs(ys - base) <= max(2.0, 0.07 * mask.shape[0])
    if not np.any(band):
        band = np.ones_like(xs, dtype=bool)
    bx, by = xs[band], ys[band]
    left = bx <= (mnx + 0.38 * span)
    right = bx >= (mxx - 0.38 * span)
    la = ra = None
    if np.any(left):
        i = int(np.argmin(bx[left]))
        la = (x0 + float(bx[left][i]), y0 + float(by[left][i]))
    if np.any(right):
        i = int(np.argmax(bx[right]))
        ra = (x0 + float(bx[right][i]), y0 + float(by[right][i]))
    return la, ra


def draw_join(canvas, a, b, style, np_rng):
    x1, y1 = a
    x2, y2 = b
    g = x2 - x1
    if g <= 0 or g > 28 or abs(y2 - y1) > 11:
        return False
    mid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0 + float(np_rng.uniform(-1.5, 1.5)))
    pts = np.array([[x1, y1], [mid[0], mid[1]], [x2, y2]], np.float32)
    curve = []
    for t in np.linspace(0, 1, 8):
        p = ((1 - t) ** 2) * pts[0] + 2 * (1 - t) * t * pts[1] + (t**2) * pts[2]
        curve.append([int(round(p[0])), int(round(p[1]))])
    jm = np.zeros(canvas.shape[:2], np.uint8)
    cv2.polylines(jm, [np.array(curve, np.int32)], False, 255, int(STYLES[style]["join"]), cv2.LINE_AA)
    jm = style_mask(jm, style, np_rng)
    composite(canvas, jm, 0, 0, style, 0.55 if style != "marker" else 0.48)
    return True


def fallback_patch(ch, font_px, style, np_rng):
    if ch.isupper():
        scale_factor = 0.78
    elif ch.isdigit():
        scale_factor = 0.72
    elif ch in ".,;:!?":
        scale_factor = 0.50
    elif ch in "'`\"":
        scale_factor = 0.44
    elif ch in "()[]{}":
        scale_factor = 0.62
    else:
        scale_factor = 0.66
    sc = max(0.22, (font_px / 26.0) * scale_factor)
    font = cv2.FONT_HERSHEY_SIMPLEX if (ch.isdigit() or ch in ".,;:!?") else cv2.FONT_HERSHEY_DUPLEX
    th = 1 if style != "marker" else (2 if (ch.isupper() or ch.isdigit()) else 1)
    (tw, thh), base = cv2.getTextSize(ch, font, sc, th)
    pad = 6
    p = np.zeros((max(1, thh + base + pad * 2), max(1, tw + pad * 2)), np.uint8)
    cv2.putText(p, ch, (pad, pad + thh), font, sc, 255, th, cv2.LINE_AA)
    p = tf_mask(p, float(np_rng.uniform(-STYLES[style]["rot"], STYLES[style]["rot"])), 1.0, 1.0)
    p = style_mask(p, style, np_rng)
    if ch in "'`\"":
        base_ratio = 0.26
    elif ch in ".,;:":
        base_ratio = 0.94
    elif ch in "!?)":
        base_ratio = 0.82
    else:
        base_ratio = 0.76
    adv_ratio = 0.80 if (ch.isupper() or ch.isdigit()) else 0.66
    if ch in ".,;:'`\"":
        adv_ratio = 0.32
    return p, float(base_ratio), float(adv_ratio)


def render(
    text,
    style,
    width,
    height,
    line_spacing,
    seed,
    use_classifier,
    use_letter_model=False,
    letter_model_weights=None,
    letter_style_strength=1.0,
    baseline_jitter=1.0,
    word_slant=1.0,
    letter_rot_jitter=1.0,
    ink_variation=0.12,
    letter_model_min_samples=4,
):
    import random

    warnings = []
    warn_once = set()

    try:
        pools = load_samples(use_classifier=use_classifier)
    except Exception as exc:
        pools = {}
        warnings.append(f"Crop glyph pools unavailable: {exc}")

    letter_infer = None
    letter_runtime = None
    letter_model_enabled = bool(use_letter_model)
    if letter_model_enabled:
        letter_infer = get_letter_infer_module()
        weights_path = Path(letter_model_weights or LETTER_MODEL_PATH)
        if letter_infer is None:
            warnings.append("Letter model module unavailable; falling back to crop sampler")
            letter_model_enabled = False
        elif not weights_path.exists():
            warnings.append(f"Letter model weights missing at {weights_path}; falling back to crop sampler")
            letter_model_enabled = False
        else:
            try:
                letter_runtime = letter_infer.load_generator(weights_path=weights_path)
            except Exception as exc:
                warnings.append(f"Letter model load failed ({exc}); falling back to crop sampler")
                letter_model_enabled = False

    py_rng = random.Random(int(seed))
    np_rng = np.random.default_rng(int(seed))
    pad_x = max(36, round(width * 0.05))
    pad_t = max(28, round(height * 0.045))
    pad_b = max(24, round(height * 0.04))
    font_px = float(max(17, min(38, round(width * 0.028))))
    step = max(font_px * 1.1, font_px * float(line_spacing))
    max_rows = max(1, int((height - pad_t - pad_b - (font_px * 0.4)) // step))
    inner_w = max(60.0, width - (pad_x * 2) - 8.0)
    rows, trunc = wrap_text(text, inner_w, font_px, max_rows, pools)
    if trunc and rows:
        last = rows[-1].rstrip()
        rows[-1] = ((last[:-1] if len(last) > 1 else last) + "...") if last else "..."

    canvas = bg_page(width, height, step, pad_t, np_rng)
    stats = dict(
        rows=len(rows),
        handChars=0,
        modelChars=0,
        cropChars=0,
        fallbackChars=0,
        joins=0,
        letterModelFailures=0,
    )
    meta_rows = []
    missing = "".join([ch for ch in "abcdefghijklmnopqrstuvwxyz" if ch not in pools])
    s = STYLES[style]
    if pools:
        global_median_bh = float(np.median([p["median_bh"] for p in pools.values()]))
    elif letter_runtime is not None:
        global_median_bh = float(max(8.0, letter_runtime.global_median_bh))
    else:
        global_median_bh = 46.0
    recent_by_char = {}

    for ri, row in enumerate(rows):
        baseline = pad_t + font_px + (ri * step) + float(py_rng.uniform(-0.4, 0.8))
        x = float(pad_x + py_rng.uniform(-2.2, 2.2))
        prev = None
        prev_text_char = None
        row_chars = []
        in_word = False
        word_slant_deg = 0.0
        baseline_walk = 0.0
        rot_bias = 0.0

        for ci, ch in enumerate(row):
            if ch == " ":
                x += width_est(ch, font_px, pools) + float(py_rng.uniform(0.05, 0.65))
                prev = None
                prev_text_char = None
                in_word = False
                continue

            if not in_word:
                in_word = True
                word_slant_deg = float(py_rng.uniform(-s["rot"] * 0.35, s["rot"] * 0.35)) * clamp(word_slant, 0.0, 3.0)
                baseline_walk = float(py_rng.uniform(-0.25, 0.25))
                rot_bias = float(py_rng.uniform(-0.20, 0.20))
            else:
                baseline_walk = (baseline_walk * 0.72) + float(py_rng.uniform(-0.55, 0.55))

            x += pair_kerning_adjust(prev_text_char, ch, font_px)
            baseline_corr = baseline_walk * clamp(baseline_jitter, 0.0, 3.0)
            placed = False

            if letter_model_enabled and ("a" <= ch <= "z") and letter_runtime is not None and letter_infer is not None:
                try:
                    glyph_seed = int(seed) + (ri * 10_003) + (ci * 137) + (ord(ch) * 7)
                    gen = letter_infer.generate_glyph(
                        letter=ch,
                        seed=glyph_seed,
                        style_strength=float(letter_style_strength),
                        output_size=letter_runtime.image_size,
                        runtime=letter_runtime,
                        min_class_samples=int(letter_model_min_samples),
                    )
                    raw_patch = gen.alpha
                    x1g, y1g, x2g, y2g = mask_bbox(raw_patch)
                    bh = max(1, int(y2g - y1g))
                    letter_median_bh = letter_runtime.median_bh(ch)
                    if letter_median_bh is None and ch in pools:
                        letter_median_bh = float(pools[ch]["median_bh"])
                    if letter_median_bh is None:
                        letter_median_bh = float(global_median_bh)

                    target_bh_px = max(8.0, (font_px * 1.02) * (float(letter_median_bh) / max(1.0, global_median_bh)))
                    sy = target_bh_px * float(py_rng.uniform(0.94, 1.08)) / max(1.0, float(bh))
                    sx = sy * float(py_rng.uniform(1.0 - s["sj"], 1.0 + s["sj"]))
                    ang = word_slant_deg + (float(py_rng.uniform(-s["rot"] * 0.55, s["rot"] * 0.55)) * clamp(letter_rot_jitter, 0.0, 3.0)) + rot_bias
                    patch = tf_mask(raw_patch, ang, sx, sy)
                    patch = maybe_apply_ink_variation(patch, np_rng, ink_variation)
                    patch = style_mask(patch, style, np_rng)

                    base_off = float(gen.base_ratio * max(1, patch.shape[0] - 1))
                    jx_lo = -0.35 * s["jx"]
                    px = int(round(x + py_rng.uniform(jx_lo, s["jx"])))
                    py = int(round(baseline - base_off + baseline_corr + py_rng.uniform(-s["jy"], s["jy"])))
                    composite(canvas, patch, px, py, style)

                    la, ra = anchors(patch, px, py)
                    if prev and prev["isHand"] and prev["ra"] and la:
                        gap = la[0] - prev["ra"][0]
                        if 0 < gap <= max(10.0, font_px * 0.23) and abs(la[1] - prev["ra"][1]) <= max(7.0, font_px * 0.13):
                            if draw_join(canvas, prev["ra"], la, style, np_rng):
                                stats["joins"] += 1

                    ys, xs = np.where(patch > 20)
                    if xs.size:
                        x1, x2 = int(xs.min()), int(xs.max()) + 1
                        y1, y2 = int(ys.min()), int(ys.max()) + 1
                    else:
                        x1, y1, x2, y2 = 0, 0, patch.shape[1], patch.shape[0]
                    row_chars.append(
                        dict(
                            char=ch,
                            x=px + x1,
                            y=py + y1,
                            w=max(1, x2 - x1),
                            h=max(1, y2 - y1),
                            isHand=True,
                            source="model",
                        )
                    )
                    prev = dict(isHand=True, ra=ra)
                    visible_w = max(1.0, float(x2 - x1))
                    if ch in "iljtfr":
                        gap_px = font_px * 0.08
                    elif ch in "mw":
                        gap_px = font_px * 0.12
                    else:
                        gap_px = font_px * 0.10
                    adv_ref = letter_runtime.median_adv(ch)
                    adv_base = float(adv_ref) if adv_ref is not None else float(gen.adv_ratio)
                    adv_est = max(font_px * 0.18, adv_base * (font_px * float(py_rng.uniform(0.98, 1.05))))
                    x = max(x + adv_est + float(py_rng.uniform(-0.12, 0.32)), float(px + x2) + gap_px)
                    stats["handChars"] += 1
                    stats["modelChars"] += 1
                    placed = True
                except Exception as exc:
                    stats["letterModelFailures"] += 1
                    k = f"{ch}:{str(exc)}"
                    if k not in warn_once:
                        warn_once.add(k)
                        warnings.append(f"Letter model fallback for '{ch}': {exc}")

            if not placed and ("a" <= ch <= "z") and (ch in pools):
                pool = pools[ch]
                rec = recent_by_char.get(ch)
                it = sample_pool_item(pool, np_rng, rec)
                # Normalize to a global lowercase reference to avoid over-scaling short extracted crops (w/x/z/m).
                target_bh_px = max(8.0, (font_px * 1.02) * (pool["median_bh"] / max(1.0, global_median_bh)))
                sy = target_bh_px * float(py_rng.uniform(0.94, 1.08)) / max(1.0, float(it["bh"]))
                sx = sy * float(py_rng.uniform(1.0 - s["sj"], 1.0 + s["sj"]))
                ang = word_slant_deg + (float(py_rng.uniform(-s["rot"], s["rot"])) * clamp(letter_rot_jitter, 0.0, 3.0)) + rot_bias
                patch = tf_mask(it["crop"], ang, sx, sy)
                patch = maybe_apply_ink_variation(patch, np_rng, ink_variation)
                patch = style_mask(patch, style, np_rng)
                base_off = float(it["base_ratio"] * max(1, patch.shape[0] - 1))
                jx_lo = -0.35 * s["jx"]
                px = int(round(x + py_rng.uniform(jx_lo, s["jx"])))
                py = int(round(baseline - base_off + baseline_corr + py_rng.uniform(-s["jy"], s["jy"])))
                composite(canvas, patch, px, py, style)
                la, ra = anchors(patch, px, py)
                if prev and prev["isHand"] and prev["ra"] and la:
                    gap = la[0] - prev["ra"][0]
                    if 0 < gap <= max(10.0, font_px * 0.23) and abs(la[1] - prev["ra"][1]) <= max(7.0, font_px * 0.13):
                        if draw_join(canvas, prev["ra"], la, style, np_rng):
                            stats["joins"] += 1
                ys, xs = np.where(patch > 20)
                if xs.size:
                    x1, x2 = int(xs.min()), int(xs.max()) + 1
                    y1, y2 = int(ys.min()), int(ys.max()) + 1
                else:
                    x1, y1, x2, y2 = 0, 0, patch.shape[1], patch.shape[0]
                row_chars.append(
                    dict(char=ch, x=px + x1, y=py + y1, w=max(1, x2 - x1), h=max(1, y2 - y1), isHand=True, source="crop")
                )
                prev = dict(isHand=True, ra=ra)
                recent_by_char[ch] = [it["filename"], *(rec or [])][:3]
                visible_w = max(1.0, float(x2 - x1))
                if ch in "iljtfr":
                    gap_px = font_px * 0.08
                elif ch in "mw":
                    gap_px = font_px * 0.12
                else:
                    gap_px = font_px * 0.10
                adv_pool = max(font_px * 0.18, pool["median_adv"] * (font_px * float(py_rng.uniform(0.98, 1.04))))
                x = max(x + adv_pool + float(py_rng.uniform(-0.15, 0.35)), float(px + x2) + gap_px)
                stats["handChars"] += 1
                stats["cropChars"] += 1
                placed = True

            if not placed:
                patch, fb_base_ratio, fb_adv_ratio = fallback_patch(ch, font_px, style, np_rng)
                patch = maybe_apply_ink_variation(patch, np_rng, ink_variation * 0.8)
                px = int(round(x + py_rng.uniform(-0.25, 0.55)))
                fb_base_off = float(fb_base_ratio * max(1, patch.shape[0] - 1))
                py = int(round(baseline - fb_base_off + baseline_corr + py_rng.uniform(-0.5, 0.6)))
                fb_opacity = 0.78 if style == "marker" else (0.84 if style == "ink" else 0.80)
                composite(canvas, patch, px, py, style, fb_opacity)
                ys, xs = np.where(patch > 20)
                if xs.size:
                    x1, x2 = int(xs.min()), int(xs.max()) + 1
                    y1, y2 = int(ys.min()), int(ys.max()) + 1
                else:
                    x1, y1, x2, y2 = 0, 0, patch.shape[1], patch.shape[0]
                row_chars.append(
                    dict(char=ch, x=px + x1, y=py + y1, w=max(1, x2 - x1), h=max(1, y2 - y1), isHand=False, source="fallback")
                )
                prev = None
                visible_w = max(1.0, float(x2 - x1))
                adv_est = max(width_est(ch, font_px, pools) * 0.78, visible_w + (font_px * fb_adv_ratio))
                x = max(x + adv_est + float(py_rng.uniform(0.0, 0.35)), float(px + x2) + font_px * 0.05)
                stats["fallbackChars"] += 1

            prev_text_char = ch
            if x > width - pad_x:
                break

        meta_rows.append(dict(index=ri, text=row, chars=row_chars))

    # light vertical vignette
    v = np.linspace(0.98, 1.0, height, dtype=np.float32).reshape(height, 1, 1)
    canvas = np.clip(canvas.astype(np.float32) * v, 0, 255).astype(np.uint8)

    if missing and pools:
        warnings.append(f"Missing glyph pools for: {missing}")
    if letter_model_enabled and letter_runtime is not None:
        model_missing = "".join(
            [ch for ch in "abcdefghijklmnopqrstuvwxyz" if letter_runtime.count_for(ch) < max(1, int(letter_model_min_samples))]
        )
        if model_missing:
            warnings.append(f"Letter model has low/insufficient samples for: {model_missing}")

    deduped_warnings = []
    seen_warn = set()
    for w in warnings:
        if w not in seen_warn:
            seen_warn.add(w)
            deduped_warnings.append(w)

    return canvas, dict(seed=int(seed), rows=meta_rows, truncated=bool(trunc), stats=stats, warnings=deduped_warnings)


def main():
    a = parse_args()
    if not (320 <= a.width <= 4096 and 240 <= a.height <= 4096):
        raise SystemExit("Invalid width/height")
    if len(a.text) > 2000:
        raise SystemExit("Text too long")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    seed = int(a.seed) if a.seed is not None else secrets.randbelow(2_147_483_647)
    auto_use_letter_model = bool(a.use_letter_model or (a.letter_model_weights and Path(a.letter_model_weights).exists()))
    try:
        img, meta = render(
            a.text,
            a.style,
            int(a.width),
            int(a.height),
            float(a.line_spacing),
            seed,
            bool(a.use_classifier),
            use_letter_model=auto_use_letter_model,
            letter_model_weights=a.letter_model_weights,
            letter_style_strength=float(a.letter_style_strength),
            baseline_jitter=float(a.baseline_jitter),
            word_slant=float(a.word_slant),
            letter_rot_jitter=float(a.letter_rot_jitter),
            ink_variation=float(a.ink_variation),
            letter_model_min_samples=int(a.letter_model_min_samples),
        )
    except Exception as e:
        if a.json:
            print(json.dumps({"ok": False, "error": str(e)}))
        raise
    if not cv2.imwrite(str(a.out), img):
        raise SystemExit(f"Failed to write {a.out}")
    payload = {
        "ok": True,
        "outPath": str(a.out),
        "seed": seed,
        "truncated": meta["truncated"],
        "warnings": meta["warnings"],
        "stats": meta["stats"],
        "rows": [{"index": r["index"], "text": r["text"]} for r in meta["rows"]],
    }
    if a.debug_json:
        a.debug_json.parent.mkdir(parents=True, exist_ok=True)
        a.debug_json.write_text(json.dumps({**payload, "meta": meta}, indent=2), encoding="utf-8")
    if a.json:
        print(json.dumps(payload))
        return
    print(f"Saved handwriting image to {a.out}")
    print(f"Seed: {seed}")
    print(f"Rows: {meta['stats']['rows']} | Hand chars: {meta['stats']['handChars']} | Fallback: {meta['stats']['fallbackChars']} | Joins: {meta['stats']['joins']}")
    if meta["truncated"]:
        print("Note: text was truncated")
    for w in meta["warnings"]:
        print(f"Warning: {w}")


if __name__ == "__main__":
    main()
