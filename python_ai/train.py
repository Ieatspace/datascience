from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover
    torch = None
    DataLoader = None  # type: ignore[assignment]

from .dataset import (
    LETTERS,
    build_train_val_datasets,
)
from .model import CVAEConfig, ConditionalVAE, vae_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS_CSV = PROJECT_ROOT / "out" / "labels.csv"
DEFAULT_CHARS_DIR = PROJECT_ROOT / "out" / "chars"
DEFAULT_WEIGHTS = PROJECT_ROOT / "out" / "letter_gen.pt"
DEFAULT_CONFIG_JSON = PROJECT_ROOT / "out" / "letter_gen.json"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "out" / "checkpoints"


def _require_torch() -> None:
    if torch is None or DataLoader is None:
        raise RuntimeError("PyTorch is required for python_ai.train")


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def choose_device(device: Optional[str]) -> str:
    _require_torch()
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    _require_torch()
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=max(0, int(num_workers)),
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(model, loader, optimizer, device: str, beta: float, train: bool):
    _require_torch()
    mode = "train" if train else "eval"
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kld = 0.0
    total_items = 0

    for batch in loader:
        x = batch["image"].to(device=device, dtype=torch.float32)
        y = batch["label"].to(device=device, dtype=torch.long)
        bs = int(x.shape[0])

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            recon, mu, logvar = model(x, y)
            losses = vae_loss(recon, x, mu, logvar, beta=beta)
            loss = losses["loss"]
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()

        total_items += bs
        total_loss += float(losses["loss"].detach().cpu()) * bs
        total_recon += float(losses["recon"].detach().cpu()) * bs
        total_kld += float(losses["kld"].detach().cpu()) * bs

    denom = max(1, total_items)
    return {
        "mode": mode,
        "loss": total_loss / denom,
        "recon": total_recon / denom,
        "kld": total_kld / denom,
        "items": total_items,
    }


def _dataset_stats(records) -> Dict[str, object]:
    class_counts = {ch: 0 for ch in LETTERS}
    bh_by_letter = {ch: [] for ch in LETTERS}
    adv_by_letter = {ch: [] for ch in LETTERS}
    for rec in records:
        class_counts[rec.letter] = int(class_counts.get(rec.letter, 0)) + 1
        bh_by_letter[rec.letter].append(int(rec.bbox_hw[0]))
        adv_by_letter[rec.letter].append(float(rec.adv_ratio))

    median_bh_by_letter = {}
    median_adv_by_letter = {}
    all_bh: List[float] = []
    for ch in LETTERS:
        if bh_by_letter[ch]:
            median_bh_by_letter[ch] = float(np.median(np.asarray(bh_by_letter[ch], dtype=np.float32)))
            all_bh.extend(bh_by_letter[ch])
        if adv_by_letter[ch]:
            median_adv_by_letter[ch] = float(np.median(np.asarray(adv_by_letter[ch], dtype=np.float32)))

    return {
        "class_counts": class_counts,
        "median_bh_by_letter": median_bh_by_letter,
        "median_adv_by_letter": median_adv_by_letter,
        "global_median_bh": float(np.median(np.asarray(all_bh, dtype=np.float32))) if all_bh else 32.0,
    }


def save_checkpoint(
    path: Path,
    model,
    config: CVAEConfig,
    epoch: int,
    best_val_loss: float,
    dataset_meta: Dict[str, object],
    train_args: Dict[str, object],
    history: List[Dict[str, object]],
) -> None:
    _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "state_dict": model.state_dict(),
        "model_config": config.to_dict(),
        "dataset": dataset_meta,
        "train_args": train_args,
        "history": history[-50:],
        "saved_at": time.time(),
    }
    torch.save(payload, str(path))


def train_letter_model(args: argparse.Namespace) -> Dict[str, object]:
    _require_torch()
    set_seed(int(args.seed), deterministic=not bool(args.non_deterministic))
    device = choose_device(args.device)

    train_ds, val_ds, records, class_counts = build_train_val_datasets(
        labels_csv=Path(args.labels_csv),
        chars_dir=Path(args.chars_dir),
        image_size=int(args.image_size),
        val_fraction=float(args.val_split),
        seed=int(args.seed),
        max_samples=int(args.max_samples) if args.max_samples is not None else None,
        augment_train=not bool(args.no_augment),
    )
    dataset_meta = _dataset_stats(records)

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_cfg = CVAEConfig(
        image_size=int(args.image_size),
        num_classes=26,
        latent_dim=int(args.latent_dim),
        label_embed_dim=int(args.label_embed_dim),
        base_channels=int(args.base_channels),
    )
    model = ConditionalVAE(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    history: List[Dict[str, object]] = []
    best_val = float("inf")
    best_epoch = 0
    start = time.time()

    train_args_meta = {
        "labels_csv": str(Path(args.labels_csv)),
        "chars_dir": str(Path(args.chars_dir)),
        "image_size": int(args.image_size),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "val_split": float(args.val_split),
        "seed": int(args.seed),
        "latent_dim": int(args.latent_dim),
        "label_embed_dim": int(args.label_embed_dim),
        "base_channels": int(args.base_channels),
        "beta": float(args.beta),
        "device": device,
        "max_samples": int(args.max_samples) if args.max_samples is not None else None,
    }

    print("[train] device={0} train={1} val={2}".format(device, len(train_ds), len(val_ds)))
    print("[train] class counts: {0}".format({k: v for k, v in class_counts.items() if v > 0}))

    for epoch in range(1, int(args.epochs) + 1):
        epoch_start = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, device, beta=float(args.beta), train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, beta=float(args.beta), train=False)
        elapsed = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "seconds": round(elapsed, 3),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        print(
            "[epoch {0:03d}] {1:.1f}s train(loss={2:.4f}, recon={3:.4f}, kld={4:.4f}) "
            "val(loss={5:.4f}, recon={6:.4f}, kld={7:.4f})".format(
                epoch,
                elapsed,
                train_metrics["loss"],
                train_metrics["recon"],
                train_metrics["kld"],
                val_metrics["loss"],
                val_metrics["recon"],
                val_metrics["kld"],
            )
        )

        ckpt_dir = Path(args.checkpoint_dir)
        if int(args.checkpoint_every) > 0 and (epoch % int(args.checkpoint_every) == 0):
            ckpt_path = ckpt_dir / "letter_gen_epoch{0:03d}.pt".format(epoch)
            save_checkpoint(ckpt_path, model, model_cfg, epoch, best_val, dataset_meta, train_args_meta, history)

        if float(val_metrics["loss"]) <= best_val:
            best_val = float(val_metrics["loss"])
            best_epoch = epoch
            save_checkpoint(
                Path(args.out_weights),
                model,
                model_cfg,
                epoch,
                best_val,
                dataset_meta,
                train_args_meta,
                history,
            )

    total_seconds = time.time() - start
    config_payload = {
        "version": 1,
        "artifact": {"weights": str(Path(args.out_weights)), "config": str(Path(args.out_config))},
        "model_config": model_cfg.to_dict(),
        "train": {**train_args_meta, "best_epoch": best_epoch, "best_val_loss": best_val},
        "dataset": dataset_meta,
        "history_tail": history[-20:],
        "completed_seconds": round(total_seconds, 3),
    }
    out_config = Path(args.out_config)
    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_config.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    print(
        "[done] best_epoch={0} best_val_loss={1:.4f} weights={2}".format(
            best_epoch, best_val, Path(args.out_weights)
        )
    )
    return config_payload


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a conditional VAE letter generator from out/chars + out/labels.csv")
    p.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS_CSV)
    p.add_argument("--chars-dir", type=Path, default=DEFAULT_CHARS_DIR)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=0.15, help="KL weight")
    p.add_argument("--val-split", type=float, default=0.12)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--label-embed-dim", type=int, default=16)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--checkpoint-every", type=int, default=1)
    p.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--out-weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--out-config", type=Path, default=DEFAULT_CONFIG_JSON)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--non-deterministic", action="store_true")
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if int(args.image_size) < 32 or int(args.image_size) % 16 != 0:
        raise SystemExit("--image-size must be >= 32 and divisible by 16")
    train_letter_model(args)


if __name__ == "__main__":
    main()
