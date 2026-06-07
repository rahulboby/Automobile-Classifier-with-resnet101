import argparse
import random
import shutil
from pathlib import Path

from PIL import Image

import utils.configs as configs


def is_valid_image(filepath):
    try:
        with Image.open(filepath) as image:
            image.verify()
        return True
    except Exception:
        return False


def get_class_names(data_dir=configs.DATA_DIR):
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data folder not found: {data_path}")

    class_names = sorted(folder.name for folder in data_path.iterdir() if folder.is_dir())

    if not class_names:
        raise ValueError(f"No class folders found in {data_path}")

    return class_names


def split_dataset(
    data_dir=configs.DATA_DIR,
    split_dir=configs.SPLIT_DIR,
    train_split=configs.TRAIN_SPLIT,
    val_split=configs.VAL_SPLIT,
    test_split=configs.TEST_SPLIT,
    seed=42,
    overwrite=False,
):
    if round(train_split + val_split + test_split, 8) != 1:
        raise ValueError("train_split + val_split + test_split must equal 1.0")

    source_path = Path(data_dir)
    split_path = Path(split_dir)
    class_names = get_class_names(source_path)

    if split_path.exists() and not overwrite:
        print(f"Using existing split folder: {split_path}")
        print("Pass overwrite=True to rebuild the split.")
        return summarize_split(split_path)

    if split_path.exists():
        shutil.rmtree(split_path)

    rng = random.Random(seed)
    summary = {}

    for class_name in class_names:
        class_path = source_path / class_name
        valid_files = []

        for file in class_path.glob("*"):
            if is_valid_image(file):
                valid_files.append(file)
            else:
                print(f"Skipping corrupted file: {file}")

        rng.shuffle(valid_files)

        total = len(valid_files)
        train_end = int(total * train_split)
        val_end = train_end + int(total * val_split)

        split_files = {
            "train": valid_files[:train_end],
            "val": valid_files[train_end:val_end],
            "test": valid_files[val_end:],
        }

        summary[class_name] = {}

        for split_name, files in split_files.items():
            target_dir = split_path / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)

            for file in files:
                shutil.copy2(file, target_dir / file.name)

            summary[class_name][split_name] = len(files)

    print(f"Dataset split complete: {split_path}")
    print_split_summary(summary)
    return summary


def summarize_split(split_dir=configs.SPLIT_DIR):
    split_path = Path(split_dir)
    summary = {}

    for split_name in ("train", "val", "test"):
        split_folder = split_path / split_name

        if not split_folder.exists():
            continue

        for class_folder in sorted(folder for folder in split_folder.iterdir() if folder.is_dir()):
            summary.setdefault(class_folder.name, {})
            summary[class_folder.name][split_name] = len(list(class_folder.glob("*")))

    if summary:
        print_split_summary(summary)

    return summary


def print_split_summary(summary):
    if not summary:
        print("No split files found.")
        return

    print("Split summary:")
    for class_name, counts in summary.items():
        train_count = counts.get("train", 0)
        val_count = counts.get("val", 0)
        test_count = counts.get("test", 0)
        total = train_count + val_count + test_count
        print(
            f"{class_name}: train={train_count}, val={val_count}, "
            f"test={test_count}, total={total}"
        )


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test folders from raw image data.")
    parser.add_argument("--data-dir", default=configs.DATA_DIR)
    parser.add_argument("--split-dir", default=configs.SPLIT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    split_dataset(
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
