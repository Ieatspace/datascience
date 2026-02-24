from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - training scripts report a friendlier message
    torch = None

    class Dataset:  # type: ignore[override]
        pass


LETTERS = "abcdefghijklmnopqrstuvwxyz"
LETTER_TO_INDEX = {ch: idx for idx, ch in enumerate(LETTERS)}
INDEX_TO_LETTER = {idx: ch for ch, idx in LETTER_TO_INDEX.items()}


@dataclass
class GlyphRecord:
    letter: str
    label_idx: int
    source_path: Path
    crop_mask: np.ndarray  # uint8, foreground mask (0-255)
    image: np.ndarray  # float32 [H, W] in [0, 1]
    bbox_hw: Tuple[int, int]
    base_ratio: float
    adv_ratio: float


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for python_ai.dataset. Install torch before training or sampling."
        )


def _load_labels(labels_csv: Path) -> Dict[str, str]:
    if not labels_csv.exists():
        raise FileNotFoundError("Missing labels csv: {0}".format(labels_csv))

    mapping: Dict[str, str] = {}
    with labels_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            filename = row[0].strip()
            label = row[1].strip().lower()
            if len(label) != 1 or label not in LETTER_TO_INDEX:
                continue
            mapping[filename] = label
    return mapping


def _crop_foreground(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return mask[y1:y2, x1:x2]


def _center_to_canvas(crop_mask: np.ndarray, image_size: int) -> np.ndarray:
    h, w = crop_mask.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Invalid crop shape")

    pad_ratio = 0.16
    target = max(8, int(round(image_size * (1.0 - (pad_ratio * 2.0)))))
    scale = min(float(target) / float(max(1, h)), float(target) / float(max(1, w)))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(crop_mask, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((image_size, image_size), dtype=np.uint8)
    off_x = (image_size - new_w) // 2
    off_y = (image_size - new_h) // 2
    canvas[off_y : off_y + new_h, off_x : off_x + new_w] = resized
    return canvas.astype(np.float32) / 255.0


def _estimate_metrics(mask_u8: np.ndarray) -> Tuple[Tuple[int, int], float, float]:
    ys, xs = np.where(mask_u8 > 0)
    if xs.size == 0:
        h, w = mask_u8.shape[:2]
        return (h, w), 0.75, 0.66

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    crop = mask_u8[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]

    cys, cxs = np.where(crop > 0)
    if cys.size == 0:
        base_ratio = 0.75
    else:
        denom = max(1.0, float(ch - 1))
        base_ratio = float(np.percentile(cys, 90)) / denom

    adv_ratio = (float(cw) + max(1.0, 0.12 * float(ch))) / max(1.0, float(ch))
    return (ch, cw), float(base_ratio), float(adv_ratio)


def load_letter_records(
    labels_csv: Path,
    chars_dir: Path,
    image_size: int = 64,
    max_samples: Optional[int] = None,
    min_foreground_pixels: int = 6,
) -> List[GlyphRecord]:
    labels = _load_labels(labels_csv)
    if not chars_dir.exists():
        raise FileNotFoundError("Missing chars dir: {0}".format(chars_dir))

    filenames = sorted(labels.keys())
    if max_samples is not None:
        filenames = filenames[: max(0, int(max_samples))]

    records: List[GlyphRecord] = []
    for filename in filenames:
        letter = labels[filename]
        path = chars_dir / filename
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        mask = (img > 15).astype(np.uint8) * 255
        crop = _crop_foreground(mask)
        if crop is None:
            continue
        if int(np.count_nonzero(crop)) < int(min_foreground_pixels):
            continue

        centered = _center_to_canvas(crop, image_size=image_size)
        centered_u8 = (centered * 255.0).astype(np.uint8)
        bbox_hw, base_ratio, adv_ratio = _estimate_metrics(centered_u8)

        records.append(
            GlyphRecord(
                letter=letter,
                label_idx=LETTER_TO_INDEX[letter],
                source_path=path,
                crop_mask=crop.astype(np.uint8),
                image=centered.astype(np.float32),
                bbox_hw=bbox_hw,
                base_ratio=base_ratio,
                adv_ratio=adv_ratio,
            )
        )

    if not records:
        raise ValueError("No usable lowercase letter samples found in dataset")
    return records


def count_by_letter(records: Sequence[GlyphRecord]) -> Dict[str, int]:
    counts = {ch: 0 for ch in LETTERS}
    for rec in records:
        counts[rec.letter] = counts.get(rec.letter, 0) + 1
    return counts


def stratified_split(
    records: Sequence[GlyphRecord],
    val_fraction: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("val_fraction must be in [0, 1)")

    rng = random.Random(int(seed))
    by_label: Dict[int, List[int]] = {}
    for idx, rec in enumerate(records):
        by_label.setdefault(int(rec.label_idx), []).append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []
    for _, idxs in sorted(by_label.items()):
        idxs = list(idxs)
        rng.shuffle(idxs)
        if len(idxs) <= 1 or val_fraction <= 0.0:
            train_indices.extend(idxs)
            continue
        n_val = int(round(len(idxs) * val_fraction))
        if n_val <= 0 and len(idxs) >= 5:
            n_val = 1
        if n_val >= len(idxs):
            n_val = len(idxs) - 1
        val_indices.extend(idxs[:n_val])
        train_indices.extend(idxs[n_val:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def _augment_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
    h, w = out.shape[:2]

    if float(rng.random()) < 0.85:
        angle = float(rng.uniform(-7.0, 7.0))
        tx = float(rng.uniform(-2.0, 2.0))
        ty = float(rng.uniform(-2.0, 2.0))
        scale = float(rng.uniform(0.94, 1.06))
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)

    if float(rng.random()) < 0.45:
        k = 3
        sigma = float(rng.uniform(0.15, 0.8))
        out = cv2.GaussianBlur(out, (k, k), sigmaX=sigma)

    if float(rng.random()) < 0.35:
        if float(rng.random()) < 0.5:
            out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        else:
            out = cv2.erode(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    if float(rng.random()) < 0.30:
        noise = rng.normal(0.0, 6.0, size=out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return np.clip(out.astype(np.float32) / 255.0, 0.0, 1.0)


class GlyphDataset(Dataset):
    def __init__(
        self,
        records: Sequence[GlyphRecord],
        indices: Optional[Sequence[int]] = None,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        _require_torch()
        self.records = list(records)
        self.indices = list(indices) if indices is not None else list(range(len(self.records)))
        self.augment = bool(augment)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item_index: int):  # type: ignore[override]
        rec = self.records[self.indices[item_index]]
        image = rec.image
        if self.augment:
            rng = np.random.default_rng(self.seed + (item_index * 9973))
            image = _augment_mask(image, rng)

        tensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.tensor(int(rec.label_idx), dtype=torch.long)
        return {
            "image": tensor,
            "label": label,
            "letter": rec.letter,
            "path": str(rec.source_path),
        }


def build_train_val_datasets(
    labels_csv: Path,
    chars_dir: Path,
    image_size: int = 64,
    val_fraction: float = 0.1,
    seed: int = 0,
    max_samples: Optional[int] = None,
    augment_train: bool = True,
) -> Tuple[GlyphDataset, GlyphDataset, List[GlyphRecord], Dict[str, int]]:
    records = load_letter_records(
        labels_csv=labels_csv,
        chars_dir=chars_dir,
        image_size=image_size,
        max_samples=max_samples,
    )
    train_indices, val_indices = stratified_split(records, val_fraction=val_fraction, seed=seed)
    train_ds = GlyphDataset(records, train_indices, augment=augment_train, seed=seed)
    val_ds = GlyphDataset(records, val_indices, augment=False, seed=seed + 1)
    return train_ds, val_ds, records, count_by_letter(records)
