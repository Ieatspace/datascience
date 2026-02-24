from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from .dataset import INDEX_TO_LETTER, LETTER_TO_INDEX
from .model import CVAEConfig, ConditionalVAE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "out" / "letter_gen.pt"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "out" / "letter_gen.json"


@dataclass
class GeneratedGlyph:
    letter: str
    rgba: np.ndarray  # H x W x 4, uint8
    alpha: np.ndarray  # H x W, uint8
    bbox: Tuple[int, int, int, int]
    base_ratio: float
    adv_ratio: float
    source: str = "letter_model"


@dataclass
class LetterGeneratorRuntime:
    model: ConditionalVAE
    device: str
    weights_path: Path
    image_size: int
    latent_dim: int
    class_counts: Dict[str, int]
    median_bh_by_letter: Dict[str, float]
    median_adv_by_letter: Dict[str, float]
    global_median_bh: float
    checkpoint_meta: Dict[str, object]

    def count_for(self, letter: str) -> int:
        return int(self.class_counts.get(letter, 0))

    def median_bh(self, letter: str) -> Optional[float]:
        value = self.median_bh_by_letter.get(letter)
        return float(value) if value is not None else None

    def median_adv(self, letter: str) -> Optional[float]:
        value = self.median_adv_by_letter.get(letter)
        return float(value) if value is not None else None


_RUNTIME_CACHE: Dict[Tuple[str, str], LetterGeneratorRuntime] = {}


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for the learned letter generator. Install torch or disable --use-letter-model."
        )


def _resolve_device(device: Optional[str]) -> str:
    if device:
        return device
    _require_torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_checkpoint(weights_path: Path) -> Dict[str, object]:
    _require_torch()
    if not weights_path.exists():
        raise FileNotFoundError("Letter generator weights not found: {0}".format(weights_path))
    ckpt = torch.load(str(weights_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("Invalid checkpoint format in {0}".format(weights_path))
    return ckpt


def _get_model_config(ckpt: Dict[str, object]) -> CVAEConfig:
    raw = ckpt.get("model_config")
    if isinstance(raw, dict):
        return CVAEConfig(
            image_size=int(raw.get("image_size", 64)),
            num_classes=int(raw.get("num_classes", 26)),
            latent_dim=int(raw.get("latent_dim", 32)),
            label_embed_dim=int(raw.get("label_embed_dim", 16)),
            base_channels=int(raw.get("base_channels", 32)),
        )
    return CVAEConfig()


def load_generator(
    weights_path: Optional[Path] = None,
    device: Optional[str] = None,
) -> LetterGeneratorRuntime:
    _require_torch()
    resolved_path = Path(weights_path or DEFAULT_WEIGHTS_PATH).resolve()
    resolved_device = _resolve_device(device)
    key = (str(resolved_path), resolved_device)
    cached = _RUNTIME_CACHE.get(key)
    if cached is not None:
        return cached

    ckpt = _load_checkpoint(resolved_path)
    model_cfg = _get_model_config(ckpt)
    model = ConditionalVAE(model_cfg)
    state_dict = ckpt.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint is missing state_dict")
    model.load_state_dict(state_dict)
    model.eval()
    model.to(resolved_device)

    dataset_meta = ckpt.get("dataset", {}) if isinstance(ckpt.get("dataset"), dict) else {}
    class_counts_raw = dataset_meta.get("class_counts", {})
    class_counts: Dict[str, int] = {}
    if isinstance(class_counts_raw, dict):
        for key_letter, value in class_counts_raw.items():
            if isinstance(key_letter, str) and len(key_letter) == 1:
                try:
                    class_counts[key_letter] = int(value)
                except Exception:
                    continue

    median_bh_raw = dataset_meta.get("median_bh_by_letter", {})
    median_bh_by_letter: Dict[str, float] = {}
    if isinstance(median_bh_raw, dict):
        for key_letter, value in median_bh_raw.items():
            if isinstance(key_letter, str) and len(key_letter) == 1:
                try:
                    median_bh_by_letter[key_letter] = float(value)
                except Exception:
                    continue

    median_adv_raw = dataset_meta.get("median_adv_by_letter", {})
    median_adv_by_letter: Dict[str, float] = {}
    if isinstance(median_adv_raw, dict):
        for key_letter, value in median_adv_raw.items():
            if isinstance(key_letter, str) and len(key_letter) == 1:
                try:
                    median_adv_by_letter[key_letter] = float(value)
                except Exception:
                    continue

    global_median_bh = float(dataset_meta.get("global_median_bh", 32.0)) if isinstance(dataset_meta, dict) else 32.0
    runtime = LetterGeneratorRuntime(
        model=model,
        device=resolved_device,
        weights_path=resolved_path,
        image_size=int(model_cfg.image_size),
        latent_dim=int(model_cfg.latent_dim),
        class_counts=class_counts,
        median_bh_by_letter=median_bh_by_letter,
        median_adv_by_letter=median_adv_by_letter,
        global_median_bh=float(global_median_bh),
        checkpoint_meta=ckpt,
    )
    _RUNTIME_CACHE[key] = runtime
    return runtime


def _decode_with_seed(runtime: LetterGeneratorRuntime, label_idx: int, seed: int, style_strength: float) -> np.ndarray:
    _require_torch()
    rng = np.random.default_rng(int(seed))
    z_np = rng.normal(0.0, max(0.05, float(style_strength)), size=(1, runtime.latent_dim)).astype(np.float32)
    labels = torch.tensor([int(label_idx)], dtype=torch.long, device=runtime.device)
    z = torch.from_numpy(z_np).to(runtime.device)
    with torch.no_grad():
        out = runtime.model.sample(labels=labels, z=z)
    arr = out.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return np.clip(arr, 0.0, 1.0)


def _alpha_metrics(alpha_u8: np.ndarray) -> Tuple[Tuple[int, int, int, int], float, float]:
    ys, xs = np.where(alpha_u8 > 10)
    if xs.size == 0:
        h, w = alpha_u8.shape[:2]
        return (0, 0, w, h), 0.75, 0.66
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    crop = alpha_u8[y1:y2, x1:x2]
    cys, _ = np.where(crop > 10)
    ch = max(1, crop.shape[0])
    cw = max(1, crop.shape[1])
    base_ratio = (float(np.percentile(cys, 90)) if cys.size else (ch - 1)) / max(1.0, float(ch - 1))
    adv_ratio = (float(cw) + max(1.0, 0.12 * float(ch))) / max(1.0, float(ch))
    return (x1, y1, x2, y2), float(base_ratio), float(adv_ratio)


def _postprocess_alpha(mask: np.ndarray, style_strength: float) -> np.ndarray:
    img_u8 = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
    blur_sigma = 0.45 + (0.08 * max(0.0, float(style_strength) - 1.0))
    img_blur = cv2.GaussianBlur(img_u8, (3, 3), blur_sigma)
    _, alpha = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if int(np.count_nonzero(alpha)) < 10:
        _, alpha = cv2.threshold(img_u8, 40, 255, cv2.THRESH_BINARY)

    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    alpha = cv2.medianBlur(alpha, 3)
    return alpha


def _render_rgba_from_alpha(alpha_u8: np.ndarray) -> np.ndarray:
    h, w = alpha_u8.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = 0
    rgba[..., 1] = 0
    rgba[..., 2] = 0
    rgba[..., 3] = alpha_u8
    return rgba


def _fit_to_output(alpha_u8: np.ndarray, output_size: Optional[int]) -> np.ndarray:
    if output_size is None or int(output_size) <= 0 or alpha_u8.shape[0] == int(output_size):
        return alpha_u8

    out_size = int(output_size)
    ys, xs = np.where(alpha_u8 > 10)
    if xs.size == 0:
        return np.zeros((out_size, out_size), dtype=np.uint8)

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    crop = alpha_u8[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    max_box = max(8, int(round(out_size * 0.78)))
    scale = min(float(max_box) / float(max(1, h)), float(max_box) / float(max(1, w)))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_size, out_size), dtype=np.uint8)
    ox = (out_size - nw) // 2
    oy = (out_size - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = resized
    return canvas


def generate_glyph(
    letter: str,
    seed: int,
    style_strength: float = 1.0,
    output_size: Optional[int] = None,
    weights_path: Optional[Path] = None,
    device: Optional[str] = None,
    runtime: Optional[LetterGeneratorRuntime] = None,
    min_class_samples: int = 4,
) -> GeneratedGlyph:
    if not isinstance(letter, str) or len(letter) != 1:
        raise ValueError("letter must be a single character")
    letter = letter.lower()
    if letter not in LETTER_TO_INDEX:
        raise ValueError("Unsupported letter for generator: {0}".format(letter))

    rt = runtime or load_generator(weights_path=weights_path, device=device)
    if rt.count_for(letter) < int(min_class_samples):
        raise ValueError(
            "Too few samples for letter '{0}' in trained model (have {1}, need {2})".format(
                letter, rt.count_for(letter), int(min_class_samples)
            )
        )

    decoded = _decode_with_seed(rt, LETTER_TO_INDEX[letter], int(seed), float(style_strength))
    alpha = _postprocess_alpha(decoded, float(style_strength))
    alpha = _fit_to_output(alpha, output_size)
    bbox, base_ratio, adv_ratio = _alpha_metrics(alpha)
    rgba = _render_rgba_from_alpha(alpha)
    return GeneratedGlyph(
        letter=letter,
        rgba=rgba,
        alpha=alpha,
        bbox=bbox,
        base_ratio=base_ratio,
        adv_ratio=adv_ratio,
    )


def clear_model_cache() -> None:
    _RUNTIME_CACHE.clear()
