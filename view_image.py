"""
Handwriting page utilities:
1) Optional OCR print (Tesseract)
2) Line detection (OpenCV) + debug visualization

Usage:
  python script.py
  python script.py path/to/image.jpg
  python script.py path/to/image.jpg --ocr
"""

import os
import shutil
import sys
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract


# -----------------------
# Config
# -----------------------
DEFAULT_IMAGE_PATH = r"data\page_1_abc.jpg"
PHYSICAL_WIDTH_INCHES = 8.0


# -----------------------
# Tesseract setup (optional OCR)
# -----------------------
def configure_tesseract_cmd() -> None:
    """Find a usable Tesseract executable and set pytesseract.pytesseract.tesseract_cmd."""
    env_tesseract = os.environ.get("TESSERACT_CMD")
    if env_tesseract and Path(env_tesseract).exists():
        pytesseract.pytesseract.tesseract_cmd = env_tesseract
        return

    tesseract_exe = shutil.which("tesseract")
    if tesseract_exe:
        pytesseract.pytesseract.tesseract_cmd = tesseract_exe
        return

    common_paths = [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe",
        Path.home() / "scoop" / "apps" / "tesseract" / "current" / "tesseract.exe",
        Path(r"C:\tools\Tesseract-OCR\tesseract.exe"),
    ]
    for exe_path in common_paths:
        if exe_path.exists():
            pytesseract.pytesseract.tesseract_cmd = str(exe_path)
            return

    raise FileNotFoundError(
        "Tesseract executable not found.\n"
        "Install Tesseract OCR (Windows) and/or add it to PATH.\n"
        "Typical install path: C:\\Program Files\\Tesseract-OCR\\tesseract.exe\n"
        "You can also set the environment variable TESSERACT_CMD to the full path."
    )


def run_ocr(image_path: Path) -> str:
    configure_tesseract_cmd()
    with Image.open(image_path) as img:
        return pytesseract.image_to_string(img)


# -----------------------
# Utilities
# -----------------------
def compute_horizontal_pixel_density(image_path: Path, physical_width_inches: float) -> float:
    if physical_width_inches <= 0:
        raise ValueError("physical_width_inches must be positive")
    with Image.open(image_path) as img:
        return img.width / physical_width_inches


# -----------------------
# Line detection
# -----------------------
def threshold_image(image_bgr):
    """Convert to grayscale + adaptive threshold (ink = white)."""
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
    """Find line-ish blobs by merging ink horizontally with morphology."""
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
    return boxes, merged


def draw_boxes(image_bgr, boxes):
    out = image_bgr.copy()
    for x, y, bw, bh in boxes:
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
    return out


def display_results(thresh, merged, boxed_bgr):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(thresh, cmap="gray")
    axes[0].set_title("Thresholded Image")
    axes[0].axis("off")

    axes[1].imshow(merged, cmap="gray")
    axes[1].set_title("Merged Text Regions")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(boxed_bgr, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Detected Text Lines")
    axes[2].axis("off")

    fig.tight_layout()
    plt.show()


def save_line_crops(image_bgr, boxes, out_dir="out/lines", pad=10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h, w = image_bgr.shape[:2]
    rows = []

    for i, (x, y, bw, bh) in enumerate(boxes):
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)

        crop = image_bgr[y1:y2, x1:x2]
        fname = f"line_{i:04d}.png"
        cv2.imwrite(str(out_dir / fname), crop)

        rows.append([fname, x1, y1, x2 - x1, y2 - y1])

    # save metadata
    with open(out_dir / "lines.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "x", "y", "w", "h"])
        writer.writerows(rows)

    print(f"Saved {len(boxes)} line crops to: {out_dir}")


# -----------------------
# Main
# -----------------------
def parse_args(argv):
    """
    Accept:
      script.py
      script.py path/to/img
      script.py path/to/img --ocr
    """
    image_path = Path(DEFAULT_IMAGE_PATH)
    do_ocr = False

    for arg in argv[1:]:
        if arg.lower() == "--ocr":
            do_ocr = True
        else:
            image_path = Path(arg)

    return image_path, do_ocr

def main():
    image_path, do_ocr = parse_args(sys.argv)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}\n"
            "Run like:\n"
            "  python script.py\n"
            "  python script.py path/to/image.jpg\n"
            "  python script.py path/to/image.jpg --ocr"
        )

    if do_ocr:
        print("=== OCR OUTPUT ===")
        print(run_ocr(image_path))

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image with OpenCV: {image_path}")

    _, thresh = threshold_image(image_bgr)
    ppi_h = compute_horizontal_pixel_density(image_path, PHYSICAL_WIDTH_INCHES)

    boxes, merged = find_text_line_boxes(thresh)
    boxed = draw_boxes(image_bgr, boxes)
    save_line_crops(image_bgr, boxes, out_dir="out/lines", pad=12)
    print(f"Horizontal pixel density: {ppi_h:.2f} PPI")
    print(f"Detected text lines: {len(boxes)}")

    display_results(thresh, merged, boxed)


if __name__ == "__main__":
    main()

