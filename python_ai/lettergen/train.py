from __future__ import annotations

import argparse
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
    make_weighted_sampler,
)
from .model import CVAEConfig, ConditionalVAE, vae_loss


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS_CSV = PROJECT_ROOT / "out" / "labels.csv"
DEFAULT_CHARS_DIR = PROJECT_ROOT / "out" / "chars"
DEFAULT_WEIGHTS = PROJECT_ROOT / "out" / "letter_gen.pt"
DEFAULT_CONFIG_JSON = PROJECT_ROOT / "out" / "letter_gen_config.json"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "out" / "checkpoints"


def make_logger(log_file: Optional[Path]) -> Tuple[Callable[[str], None], Optional[object]]:
    handle = None
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a", encoding="utf-8")

    def _log(message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        if handle is not None:
            handle.write(line + "\n")
            handle.flush()

    return _log, handle


def format_epoch_checkpoint_path(checkpoint_dir: Path, epoch: int) -> Path:
    return Path(checkpoint_dir) / f"epoch_{int(epoch):04d}.pt"


def _extract_epoch_from_checkpoint_name(path: Path) -> int:
    m = re.search(r"epoch_(\d{4,})", path.name)
    if not m:
        return -1
    try:
        return int(m.group(1))
    except Exception:
        return -1


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    cands = list(checkpoint_dir.glob("epoch_*.pt"))
    if not cands:
        return None
    cands.sort(key=lambda p: (_extract_epoch_from_checkpoint_name(p), p.stat().st_mtime_ns))
    return cands[-1]


def _require_torch() -> None:
    if torch is None or DataLoader is None:
        raise RuntimeError("PyTorch is required for python_ai.lettergen.train")


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


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, sampler=None):
    _require_torch()
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle) if sampler is None else False,
        sampler=sampler,
        num_workers=max(0, int(num_workers)),
        pin_memory=torch.cuda.is_available(),
    )


def beta_for_epoch(epoch: int, total_epochs: int, target_beta: float, warmup_epochs: int) -> float:
    target = float(target_beta)
    if target <= 0:
        return 0.0
    warmup = int(max(0, warmup_epochs))
    if warmup <= 0:
        return target
    progress = min(1.0, max(0.0, float(epoch) / float(warmup)))
    # Smooth-ish ramp to reduce posterior collapse risk early in training.
    return target * (progress * progress)


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
    optimizer,
    config: CVAEConfig,
    epoch: int,
    best_val_loss: float,
    best_epoch: int,
    dataset_meta: Dict[str, object],
    train_args: Dict[str, object],
    history: List[Dict[str, object]],
    extra_state: Optional[Dict[str, object]] = None,
) -> None:
    _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 2,
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "model_config": config.to_dict(),
        "dataset": dataset_meta,
        "train_args": train_args,
        "history": history[-50:],
        "saved_at": time.time(),
    }
    if extra_state:
        payload["extra_state"] = dict(extra_state)
    torch.save(payload, str(path))


def load_resume_checkpoint(
    resume: str,
    checkpoint_dir: Path,
    log: Callable[[str], None],
) -> Optional[Tuple[Path, Dict[str, object]]]:
    resume = (resume or "none").strip()
    if resume.lower() in {"none", "off", "false", "0"}:
        return None
    if resume.lower() == "auto":
        ckpt_path = find_latest_checkpoint(Path(checkpoint_dir))
        if ckpt_path is None:
            log(f"[resume] No checkpoint found in {Path(checkpoint_dir)}; starting fresh")
            return None
    else:
        ckpt_path = Path(resume)
        if not ckpt_path.is_absolute():
            ckpt_path = PROJECT_ROOT / ckpt_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

    _require_torch()
    payload = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint payload in {ckpt_path}")
    log(f"[resume] Loaded checkpoint {ckpt_path}")
    return ckpt_path, payload


def train_letter_model(args: argparse.Namespace) -> Dict[str, object]:
    _require_torch()
    log, log_handle = make_logger(Path(args.log_file) if args.log_file else None)
    try:
        set_seed(int(args.seed), deterministic=not bool(args.non_deterministic))
        device = choose_device(args.device)

        resume_loaded = load_resume_checkpoint(str(args.resume), Path(args.checkpoint_dir), log)
        resume_payload: Optional[Dict[str, object]] = resume_loaded[1] if resume_loaded else None
        resume_path = resume_loaded[0] if resume_loaded else None

        resume_model_cfg = None
        if resume_payload and isinstance(resume_payload.get("model_config"), dict):
            raw_cfg = resume_payload["model_config"]
            resume_model_cfg = CVAEConfig(
                image_size=int(raw_cfg.get("image_size", args.image_size)),
                num_classes=int(raw_cfg.get("num_classes", 26)),
                latent_dim=int(raw_cfg.get("latent_dim", args.latent_dim)),
                label_embed_dim=int(raw_cfg.get("label_embed_dim", args.label_embed_dim)),
                base_channels=int(raw_cfg.get("base_channels", args.base_channels)),
            )
            # Keep architecture and image size consistent when resuming.
            args.image_size = int(resume_model_cfg.image_size)
            args.latent_dim = int(resume_model_cfg.latent_dim)
            args.label_embed_dim = int(resume_model_cfg.label_embed_dim)
            args.base_channels = int(resume_model_cfg.base_channels)

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
        train_sampler = make_weighted_sampler(train_ds, records) if bool(args.weighted_sampler) else None
        train_loader = make_loader(
            train_ds,
            args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            sampler=train_sampler,
        )
        val_loader = make_loader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

        model_cfg = resume_model_cfg or CVAEConfig(
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
        start_epoch = 1

        if resume_payload:
            state_dict = resume_payload.get("state_dict")
            if isinstance(state_dict, dict):
                model.load_state_dict(state_dict)
            opt_state = resume_payload.get("optimizer_state")
            if isinstance(opt_state, dict):
                try:
                    optimizer.load_state_dict(opt_state)
                except Exception as exc:
                    log(f"[resume] Warning: optimizer state load failed ({exc}); continuing with fresh optimizer")
            for pg in optimizer.param_groups:
                pg["lr"] = float(args.lr)
            try:
                history = list(resume_payload.get("history", []))  # type: ignore[arg-type]
            except Exception:
                history = []
            best_val = float(resume_payload.get("best_val_loss", best_val))
            best_epoch = int(resume_payload.get("best_epoch", 0))
            start_epoch = int(resume_payload.get("epoch", 0)) + 1

        start_time = time.time()

        train_args_meta = {
            "labels_csv": str(Path(args.labels_csv)),
            "chars_dir": str(Path(args.chars_dir)),
            "image_size": int(args.image_size),
            "epochs": int(args.epochs),
            "infinite": bool(args.infinite),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "val_split": float(args.val_split),
            "seed": int(args.seed),
            "latent_dim": int(args.latent_dim),
            "label_embed_dim": int(args.label_embed_dim),
            "base_channels": int(args.base_channels),
            "beta": float(args.beta),
            "beta_warmup_epochs": int(args.beta_warmup_epochs),
            "device": device,
            "resume": str(args.resume),
            "resume_path": str(resume_path) if resume_path is not None else None,
            "max_samples": int(args.max_samples) if args.max_samples is not None else None,
            "weighted_sampler": bool(args.weighted_sampler),
            "save_every": int(args.save_every),
        }

        total_samples = int(sum(class_counts.values()))
        min_per_letter = min([v for v in class_counts.values() if v > 0] or [0])
        weak = {k: v for k, v in class_counts.items() if 0 < v < int(args.letter_min_warning)}

        log("[startup] Conditional VAE letter generator training")
        log(
            "[startup] device={device} resume={resume} start_epoch={start_epoch} "
            "epochs={epochs} infinite={infinite}".format(
                device=device,
                resume=("none" if resume_path is None else str(resume_path)),
                start_epoch=start_epoch,
                epochs=int(args.epochs),
                infinite=bool(args.infinite),
            )
        )
        log(
            "[startup] dataset train={train_n} val={val_n} total={total} image_size={img} "
            "class_count={cls}".format(
                train_n=len(train_ds),
                val_n=len(val_ds),
                total=total_samples,
                img=int(args.image_size),
                cls=len([v for v in class_counts.values() if v > 0]),
            )
        )
        log(f"[startup] per-letter counts: { {k: v for k, v in class_counts.items() if v > 0} }")
        log(
            "[startup] latent_dim={latent} base_channels={base} lr={lr:g} beta_target={beta:g} "
            "beta_warmup_epochs={warmup} weighted_sampler={ws}".format(
                latent=int(args.latent_dim),
                base=int(args.base_channels),
                lr=float(args.lr),
                beta=float(args.beta),
                warmup=int(args.beta_warmup_epochs),
                ws=bool(args.weighted_sampler),
            )
        )
        log(f"[startup] log_file={args.log_file if args.log_file else 'disabled'} save_every={int(args.save_every)} checkpoint_dir={Path(args.checkpoint_dir)}")

        if total_samples < 26 * 200:
            log("[warn] Dataset is small for a generative model. Aim for 200+ samples per letter (~5200+ total).")
        if min_per_letter < int(args.letter_min_warning):
            log(f"[warn] Some letters have low sample counts; inference may fall back for those letters: {weak}")

        completed_epochs = max(0, start_epoch - 1)
        epoch = start_epoch
        interrupted = False
        try:
            while True:
                if (not bool(args.infinite)) and (epoch > int(args.epochs)):
                    break

                epoch_start = time.time()
                beta_epoch = beta_for_epoch(
                    epoch=epoch,
                    total_epochs=max(int(args.epochs), epoch),
                    target_beta=float(args.beta),
                    warmup_epochs=int(args.beta_warmup_epochs),
                )
                train_metrics = run_epoch(model, train_loader, optimizer, device, beta=beta_epoch, train=True)
                val_metrics = run_epoch(model, val_loader, optimizer, device, beta=beta_epoch, train=False)
                elapsed = time.time() - epoch_start

                row = {
                    "epoch": epoch,
                    "seconds": round(elapsed, 3),
                    "beta": float(beta_epoch),
                    "train": train_metrics,
                    "val": val_metrics,
                }
                history.append(row)

                log(
                    "[epoch {0:04d}] beta={1:.4f} {2:.1f}s "
                    "train(loss={3:.4f}, recon={4:.4f}, kld={5:.4f}) "
                    "val(loss={6:.4f}, recon={7:.4f}, kld={8:.4f})".format(
                        epoch,
                        beta_epoch,
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
                if int(args.save_every) > 0 and (epoch % int(args.save_every) == 0):
                    ckpt_path = format_epoch_checkpoint_path(ckpt_dir, epoch)
                    save_checkpoint(
                        ckpt_path,
                        model,
                        optimizer,
                        model_cfg,
                        epoch,
                        best_val,
                        best_epoch,
                        dataset_meta,
                        train_args_meta,
                        history,
                        extra_state={"last_beta": float(beta_epoch)},
                    )

                if float(val_metrics["loss"]) <= best_val:
                    best_val = float(val_metrics["loss"])
                    best_epoch = epoch
                    for target_path in {Path(args.out_weights), DEFAULT_WEIGHTS}:
                        save_checkpoint(
                            Path(target_path),
                            model,
                            optimizer,
                            model_cfg,
                            epoch,
                            best_val,
                            best_epoch,
                            dataset_meta,
                            train_args_meta,
                            history,
                            extra_state={"last_beta": float(beta_epoch)},
                        )
                    log(f"[best] epoch={epoch} val_loss={best_val:.4f} -> saved {DEFAULT_WEIGHTS}")

                completed_epochs = epoch
                epoch += 1

        except KeyboardInterrupt:
            interrupted = True
            interrupt_path = Path(args.checkpoint_dir) / f"epoch_{max(0, completed_epochs):04d}_interrupt.pt"
            save_checkpoint(
                interrupt_path,
                model,
                optimizer,
                model_cfg,
                completed_epochs,
                best_val,
                best_epoch,
                dataset_meta,
                train_args_meta,
                history,
                extra_state={"interrupted": True},
            )
            log(f"[interrupt] KeyboardInterrupt received. Saved checkpoint to {interrupt_path}")

        total_seconds = time.time() - start_time
        config_payload = {
            "version": 1,
            "artifact": {"weights": str(Path(args.out_weights)), "config": str(Path(args.out_config))},
            "model_config": model_cfg.to_dict(),
            "train": {
                **train_args_meta,
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "completed_epochs": completed_epochs,
                "interrupted": interrupted,
            },
            "dataset": dataset_meta,
            "history_tail": history[-20:],
            "completed_seconds": round(total_seconds, 3),
        }
        out_config = Path(args.out_config)
        out_config.parent.mkdir(parents=True, exist_ok=True)
        out_config.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
        if out_config.resolve() != DEFAULT_CONFIG_JSON.resolve():
            DEFAULT_CONFIG_JSON.parent.mkdir(parents=True, exist_ok=True)
            DEFAULT_CONFIG_JSON.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

        if interrupted:
            log("[done] Interrupted cleanly. Training state saved. Best weights remain at out/letter_gen.pt")
        else:
            log(
                "[done] best_epoch={0} best_val_loss={1:.4f} weights={2}".format(
                    best_epoch, best_val, Path(args.out_weights)
                )
            )
        return config_payload
    finally:
        if log_handle is not None:
            try:
                log_handle.close()
            except Exception:
                pass


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a conditional VAE letter generator from out/chars + out/labels.csv")
    p.add_argument(
        "--mode",
        choices=["twenty", "infinite"],
        default=None,
        help="Convenience mode: 'twenty' runs 20 epochs, 'infinite' trains until manually stopped",
    )
    p.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS_CSV)
    p.add_argument("--chars-dir", type=Path, default=DEFAULT_CHARS_DIR)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", "--batch_size", type=int, default=64, dest="batch_size")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=0.15, help="KL weight")
    p.add_argument("--beta-warmup-epochs", type=int, default=6, help="Ramp KL beta from 0 to target over N epochs")
    p.add_argument("--val-split", type=float, default=0.12)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--latent-dim", "--latent_dim", type=int, default=32, dest="latent_dim")
    p.add_argument("--label-embed-dim", "--label_embed_dim", type=int, default=16, dest="label_embed_dim")
    p.add_argument("--base-channels", "--base_channels", type=int, default=32, dest="base_channels")
    p.add_argument("--save-every", "--checkpoint-every", type=int, default=1, dest="save_every")
    p.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--out-weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--out-config", type=Path, default=DEFAULT_CONFIG_JSON)
    p.add_argument("--resume", type=str, default="none", help="Checkpoint path, or 'auto' to resume from latest in out/checkpoints")
    p.add_argument("--infinite", action="store_true", help="Train until interrupted (ignores --epochs limit)")
    p.add_argument("--log-file", type=Path, default=None, help="Append timestamped logs to a file")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--weighted-sampler", action="store_true", help="Use optional class-balanced sampler for training")
    p.add_argument("--letter-min-warning", type=int, default=12, help="Warn if any letter has fewer than this many samples")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--non-deterministic", action="store_true")
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.mode == "twenty":
        args.epochs = 20
        args.infinite = False
    elif args.mode == "infinite":
        args.infinite = True
    if int(args.image_size) < 32 or int(args.image_size) % 16 != 0:
        raise SystemExit("--image-size must be >= 32 and divisible by 16")
    train_letter_model(args)


if __name__ == "__main__":
    main()
