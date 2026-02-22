import csv
from pathlib import Path

import cv2

CHARS_DIR = Path("out/chars")
LABELS_CSV = Path("out/labels.csv")


def load_done_labels():
    done = {}
    if LABELS_CSV.exists():
        with open(LABELS_CSV, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    done[row[0]] = row[1]
    return done


def append_label(filename, label):
    is_new = not LABELS_CSV.exists()
    with open(LABELS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["filename", "label"])
        writer.writerow([filename, label])


def rewrite_labels(labels_dict):
    with open(LABELS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        for k, v in labels_dict.items():
            writer.writerow([k, v])


def main():
    if not CHARS_DIR.exists():
        raise FileNotFoundError("Run extract_chars.py first (out/chars not found).")

    files = sorted(CHARS_DIR.glob("char_*.png"))
    done = load_done_labels()
    queue = [p for p in files if p.name not in done]

    print(f"Total crops: {len(files)} | Already labeled: {len(done)} | Remaining: {len(queue)}")
    print("Controls: press a key to label | SPACE skip | BACKSPACE undo | ESC quit")

    history = []
    i = 0
    while i < len(queue):
        p = queue[i]
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            i += 1
            continue

        show = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Label char (key)", show)

        key = cv2.waitKey(0)

        if key == 27:
            break

        if key == 32:
            i += 1
            continue

        if key in (8, 127):
            if history:
                fname, prev = history.pop()
                if prev is None:
                    done.pop(fname, None)
                else:
                    done[fname] = prev
                rewrite_labels(done)
                print(f"Undid {fname}")
            continue

        label = None
        try:
            ch = chr(key)
            if ch.isprintable():
                label = ch
        except ValueError:
            label = None

        if label is None:
            print("Unrecognized key. Try again.")
            continue

        prev = done.get(p.name)
        done[p.name] = label
        append_label(p.name, label)
        history.append((p.name, prev))

        if (i + 1) % 50 == 0:
            print(f"Labeled {i + 1}/{len(queue)} (session remaining).")

        i += 1

    cv2.destroyAllWindows()
    print(f"Saved labels to {LABELS_CSV}")


if __name__ == "__main__":
    main()
