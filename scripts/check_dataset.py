"""Check image/label pairing, split isolation, and normalized YOLO labels."""

from __future__ import annotations

import argparse
from pathlib import Path

IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npy"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an IDAM-RAW dataset layout.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-train", type=int)
    parser.add_argument("--expected-val", type=int)
    return parser.parse_args()


def collect(folder: Path, suffixes: set[str]) -> dict[str, Path]:
    if not folder.is_dir():
        return {}
    return {
        path.stem: path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    }


def validate_label(path: Path) -> list[str]:
    errors: list[str] = []
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            errors.append(f"{path}:{number}: expected 5 fields, found {len(fields)}")
            continue
        try:
            class_id = float(fields[0])
            coords = [float(value) for value in fields[1:]]
        except ValueError:
            errors.append(f"{path}:{number}: contains a non-numeric field")
            continue
        if class_id < 0 or not class_id.is_integer():
            errors.append(f"{path}:{number}: class ID must be a non-negative integer")
        if any(value < 0.0 or value > 1.0 for value in coords):
            errors.append(f"{path}:{number}: normalized coordinates must lie in [0, 1]")
        if coords[2] <= 0.0 or coords[3] <= 0.0:
            errors.append(f"{path}:{number}: width and height must be positive")
    return errors


def check_split(root: Path, split: str, expected: int | None) -> tuple[set[str], list[str]]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    images = collect(image_dir, IMAGE_SUFFIXES)
    labels = collect(label_dir, {".txt"})
    errors: list[str] = []

    if not image_dir.is_dir():
        errors.append(f"missing directory: {image_dir}")
    if not label_dir.is_dir():
        errors.append(f"missing directory: {label_dir}")

    only_images = sorted(images.keys() - labels.keys())
    only_labels = sorted(labels.keys() - images.keys())
    if only_images:
        errors.append(f"{split}: {len(only_images)} images have no label; examples: {only_images[:10]}")
    if only_labels:
        errors.append(f"{split}: {len(only_labels)} labels have no image; examples: {only_labels[:10]}")
    if expected is not None and len(images) != expected:
        errors.append(f"{split}: expected {expected} images, found {len(images)}")
    if expected is not None and len(labels) != expected:
        errors.append(f"{split}: expected {expected} labels, found {len(labels)}")

    for path in labels.values():
        errors.extend(validate_label(path))

    print(f"{split}: {len(images)} images, {len(labels)} labels, {len(images.keys() & labels.keys())} paired")
    return set(images), errors


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    train_stems, train_errors = check_split(root, "train", args.expected_train)
    val_stems, val_errors = check_split(root, "val", args.expected_val)
    errors = train_errors + val_errors

    overlap = sorted(train_stems & val_stems)
    if overlap:
        errors.append(f"train/val filename overlap: {len(overlap)} stems; examples: {overlap[:10]}")

    if errors:
        print("\nDataset check FAILED:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print("\nDataset check PASSED.")
    print("Filename checks cannot prove physical-scene isolation, RAW decoding, or privacy compliance.")


if __name__ == "__main__":
    main()
