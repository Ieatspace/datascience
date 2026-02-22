import csv
from pathlib import Path

import cv2

CHARS_DIR = Path("out/chars")
LABELS_CSV = Path("out/labels.csv")
CHARS_META_CSV = Path("out/chars_meta.csv")


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


def load_latest_batch_filenames():
    if not CHARS_META_CSV.exists():
        return []

    latest = []
    try:
        with open(CHARS_META_CSV, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    latest.append(row[0])
    except OSError:
        return []
    return latest


def choose_scope(files, latest_names):
    if not latest_names:
        return files

    file_map = {p.name: p for p in files}
    latest_files = [file_map[name] for name in latest_names if name in file_map]
    if not latest_files:
        return files

    latest_set = {p.name for p in latest_files}
    older_files = [p for p in files if p.name not in latest_set]
    if not older_files:
        return latest_files

    latest_prefix = latest_files[0].name.split("_char_", 1)[0] if "_char_" in latest_files[0].name else "latest"
    print(
        f"Found latest extraction batch '{latest_prefix}' ({len(latest_files)} crops) "
        f"and {len(older_files)} older crops."
    )
    print("Choose labeling scope:")
    print("  1) New batch only (Recommended)")
    print("  2) Old + new (all visible crops)")

    try:
        choice = input("Enter 1 or 2 [default 1]: ").strip()
    except EOFError:
        choice = ""

    if choice == "2":
        return files
    return latest_files


def load_existing_files():
    all_files = sorted(CHARS_DIR.glob("*.png"))
    page_prefixed = [p for p in all_files if "_char_" in p.name]
    legacy = [p for p in all_files if p.name.startswith("char_")]

    if page_prefixed:
        if legacy:
            print(
                f"Info: using {len(page_prefixed)} page-prefixed crops and ignoring "
                f"{len(legacy)} legacy char_*.png files."
            )
        latest_names = load_latest_batch_filenames()
        return choose_scope(page_prefixed, latest_names)

    return all_files


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

    all_existing_files = sorted(CHARS_DIR.glob("*.png"))
    files = load_existing_files()
    done = load_done_labels()
    all_file_names = {p.name for p in all_existing_files}
    stale_labels = [name for name in done if name not in all_file_names]
    if stale_labels:
        print(f"Warning: {len(stale_labels)} labels point to missing files (old extraction).")
        print("They will be ignored while labeling this batch.")
        done = {k: v for k, v in done.items() if k in all_file_names}
        rewrite_labels(done)

    selected_names = {p.name for p in files}
    queue = [p for p in files if p.name not in done]
    done_in_scope = sum(1 for name in selected_names if name in done)

    print(f"Total crops: {len(files)} | Already labeled: {done_in_scope} | Remaining: {len(queue)}")
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
