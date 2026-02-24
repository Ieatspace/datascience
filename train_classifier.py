import argparse
import csv
import importlib.util
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

# If launched with the wrong interpreter (common in VS Code/PowerShell), re-run
# with the project's local venv so imports like `torch` resolve consistently.
if importlib.util.find_spec("torch") is None and os.environ.get("TC_REEXEC") != "1":
    venv_python = SCRIPT_DIR / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        env = os.environ.copy()
        env["TC_REEXEC"] = "1"
        os.execve(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]], env)

import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

LABELS_CSV = Path("out/labels.csv")
CHARS_DIR = Path("out/chars")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 64
DEFAULT_BATCH = 64
DEFAULT_EPOCHS = 16
DEFAULT_LR = 1e-3


def parse_args():
    p = argparse.ArgumentParser(description="Train a lowercase handwriting character classifier.")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--val-split", type=float, default=0.15, dest="val_split")
    p.add_argument("--no-augment", action="store_true")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def augment_array(arr: np.ndarray) -> np.ndarray:
    # arr is float32 [0,1]
    img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L")

    if random.random() < 0.9:
        ang = random.uniform(-8.0, 8.0)
        img = img.rotate(ang, resample=Image.BILINEAR, fillcolor=0)

    if random.random() < 0.85:
        dx = random.randint(-4, 4)
        dy = random.randint(-4, 4)
        img = img.transform(
            (IMG_SIZE, IMG_SIZE),
            Image.AFFINE,
            (1, 0, dx, 0, 1, dy),
            resample=Image.BILINEAR,
            fillcolor=0,
        )

    if random.random() < 0.3:
        img = img.filter(ImageFilter.MaxFilter(3))
    elif random.random() < 0.2:
        img = img.filter(ImageFilter.MinFilter(3))

    arr = np.array(img, dtype=np.float32) / 255.0
    if random.random() < 0.4:
        arr = np.clip(arr * random.uniform(0.9, 1.1), 0.0, 1.0)
    if random.random() < 0.45:
        sigma = random.uniform(0.0, 0.03)
        if sigma > 0:
            arr = np.clip(arr + np.random.normal(0.0, sigma, size=arr.shape).astype(np.float32), 0.0, 1.0)
    return arr


class CharDataset(Dataset):
    def __init__(self, items, class_to_idx, augment=False):
        self.items = items
        self.class_to_idx = class_to_idx
        self.augment = bool(augment)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fname, label = self.items[i]
        img_path = CHARS_DIR / fname
        with Image.open(img_path) as img:
            img = img.convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0

        if arr.shape != (IMG_SIZE, IMG_SIZE):
            raise ValueError(f"Expected {IMG_SIZE}x{IMG_SIZE} image, got {arr.shape} for {img_path}")

        if self.augment:
            arr = augment_array(arr)

        x = torch.from_numpy(arr).unsqueeze(0)
        y = self.class_to_idx[label]
        return x, y


class SmallCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_labels():
    if not LABELS_CSV.exists():
        raise FileNotFoundError("out/labels.csv not found. Run label_chars.py first.")

    available_files = {p.name for p in CHARS_DIR.glob("*.png")} if CHARS_DIR.exists() else set()
    latest_labels = {}
    missing_count = 0
    with open(LABELS_CSV, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                fname, label = row[0], row[1]
                if available_files and fname not in available_files:
                    missing_count += 1
                    continue
                latest_labels[fname] = label

    items = list(latest_labels.items())

    if not items:
        raise ValueError("out/labels.csv is empty. Label some characters first.")

    if missing_count:
        print(f"Warning: skipped {missing_count} stale labels for missing image files.")

    return items


def preflight_checks():
    if not CHARS_DIR.exists():
        print("Missing character crops: out/chars")
        print("Run this first:")
        print(f"  {Path(sys.executable).name} extract_chars.py")
        return False

    char_files = list(CHARS_DIR.glob("*.png"))
    if not char_files:
        print("No character crops found in out/chars.")
        print("Run this first:")
        print(f"  {Path(sys.executable).name} extract_chars.py")
        return False

    if not LABELS_CSV.exists():
        print("Missing labels file: out/labels.csv")
        print("Training needs labeled characters before it can start.")
        print("Run this next:")
        print(f"  {Path(sys.executable).name} label_chars.py")
        print(f"Found {len(char_files)} unlabeled crops in {CHARS_DIR}")
        return False

    return True


def evaluate(model, val_dl):
    model.eval()
    correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            for yt, yp in zip(y.tolist(), preds.tolist()):
                per_class_total[int(yt)] += 1
                if int(yt) == int(yp):
                    per_class_correct[int(yt)] += 1
    return correct / max(1, total), per_class_correct, per_class_total


def stratified_split(items, val_split=0.15):
    buckets = defaultdict(list)
    for fname, label in items:
        buckets[label].append((fname, label))
    train_items = []
    val_items = []
    for label, bucket in buckets.items():
        random.shuffle(bucket)
        if len(bucket) <= 1:
            train_items.extend(bucket)
            continue
        n_val = int(round(len(bucket) * float(val_split)))
        n_val = max(1, min(n_val, len(bucket) - 1))
        val_items.extend(bucket[:n_val])
        train_items.extend(bucket[n_val:])
    random.shuffle(train_items)
    random.shuffle(val_items)
    return train_items, val_items


def main():
    args = parse_args()
    set_seed(int(args.seed))
    if not preflight_checks():
        return

    items = load_labels()
    labels = sorted({lab for _, lab in items})
    class_to_idx = {c: i for i, c in enumerate(labels)}
    n_classes = len(labels)

    print(f"Samples: {len(items)} | Classes: {n_classes} | Device: {DEVICE}")
    if len(items) < 500:
        print("Warning: you should label more (aim 1000+).")
    if n_classes < 2:
        raise ValueError("Need at least 2 classes to train a classifier.")

    train_items, val_items = stratified_split(items, val_split=float(args.val_split))
    if not train_items or not val_items:
        split = int(0.85 * len(items))
        split = max(1, min(split, len(items) - 1))
        train_items = items[:split]
        val_items = items[split:]

    train_ds = CharDataset(train_items, class_to_idx, augment=not args.no_augment)
    val_ds = CharDataset(val_items, class_to_idx)

    train_dl = DataLoader(train_ds, batch_size=int(args.batch), shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=int(args.batch), shuffle=False)

    model = SmallCNN(n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    loss_fn = nn.CrossEntropyLoss()
    best_val = -1.0
    best_state = None

    print(
        f"Train: {len(train_items)} | Val: {len(val_items)} | "
        f"Epochs: {int(args.epochs)} | Batch: {int(args.batch)} | LR: {float(args.lr):g} | "
        f"Augment: {not args.no_augment}"
    )

    for ep in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)
        val_acc, per_class_correct, per_class_total = evaluate(model, val_dl)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        weak = []
        for lab, idx in class_to_idx.items():
            tot = per_class_total.get(int(idx), 0)
            if tot <= 0:
                continue
            acc = per_class_correct.get(int(idx), 0) / tot
            weak.append((acc, lab, tot))
        weak.sort(key=lambda t: (t[0], t[1]))
        weak_str = ", ".join([f"{lab}:{acc:.2f}({tot})" for acc, lab, tot in weak[:4]])

        print(
            f"Epoch {ep}/{int(args.epochs)} | loss {train_loss:.4f} | "
            f"train_acc {train_acc:.3f} | val_acc {val_acc:.3f} | weak [{weak_str}]"
        )

    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
            "meta": {
                "best_val_acc": float(best_val),
                "epochs": int(args.epochs),
                "batch": int(args.batch),
                "lr": float(args.lr),
                "seed": int(args.seed),
                "augment": bool(not args.no_augment),
            },
        },
        out_dir / "char_classifier.pt",
    )
    print(f"Saved best model to out/char_classifier.pt (best_val_acc={best_val:.3f})")


if __name__ == "__main__":
    main()
