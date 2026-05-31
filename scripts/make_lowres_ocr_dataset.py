import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
    "B", "C", "D", "E", "F", "G", "H", "K", "L", "M",
    "N", "P", "S", "T", "U", "V", "X", "Y", "Z", "0",
]


def stable_unit(name):
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 0xFFFFFFFF


def degrade_lowres(image, name):
    height, width = image.shape[:2]
    u = stable_unit(name)
    scale = 0.28 + 0.32 * u
    small_w = max(8, int(width * scale))
    small_h = max(8, int(height * scale))

    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)

    blur_kernel = 3 if stable_unit(name + ":blur") > 0.45 else 1
    if blur_kernel > 1:
        restored = cv2.GaussianBlur(restored, (blur_kernel, blur_kernel), 0)

    noise_sigma = 3.0 + 5.0 * stable_unit(name + ":noise")
    noise = np.random.default_rng(int(stable_unit(name + ":seed") * 2**32)).normal(
        0, noise_sigma, restored.shape
    )
    restored = np.clip(restored.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return restored


def extract_if_needed(zip_path, raw_dir):
    marker = raw_dir / "OCR" / "images" / "train"
    if marker.exists():
        return
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(raw_dir)


def copy_labels(raw_root, out_root):
    labels_src = raw_root / "labels"
    labels_dst = out_root / "labels"
    if labels_dst.exists():
        shutil.rmtree(labels_dst)
    shutil.copytree(labels_src, labels_dst)


def write_yaml(out_root):
    names = ", ".join(f"'{name}'" for name in CLASS_NAMES)
    yaml_text = (
        f"path: {out_root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: [{names}]\n"
    )
    (out_root / "Letter_detect_lowres.yaml").write_text(yaml_text)


def write_mixed_yaml(out_root):
    names = ", ".join(f"'{name}'" for name in CLASS_NAMES)
    yaml_text = (
        f"path: {out_root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: [{names}]\n"
    )
    (out_root / "Letter_detect_mixed.yaml").write_text(yaml_text)


def build_mixed(raw_root, lowres_root, mixed_root):
    if mixed_root.exists():
        shutil.rmtree(mixed_root)
    for split in ("train", "val"):
        for subset, source_root in (("raw", raw_root), ("lowres", lowres_root)):
            image_src = source_root / "images" / split
            label_src = source_root / "labels" / split
            image_dst = mixed_root / "images" / split
            label_dst = mixed_root / "labels" / split
            image_dst.mkdir(parents=True, exist_ok=True)
            label_dst.mkdir(parents=True, exist_ok=True)
            for image_path in sorted(image_src.iterdir()):
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                name = f"{subset}_{image_path.name}"
                shutil.copy2(image_path, image_dst / name)
                shutil.copy2(label_src / f"{image_path.stem}.txt", label_dst / f"{Path(name).stem}.txt")
    write_mixed_yaml(mixed_root)


def build_lowres(raw_root, out_root):
    copy_labels(raw_root, out_root)
    for split in ("train", "val"):
        image_src = raw_root / "images" / split
        image_dst = out_root / "images" / split
        if image_dst.exists():
            shutil.rmtree(image_dst)
        image_dst.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(image_src.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Failed to read {image_path}")
            lowres = degrade_lowres(image, f"{split}/{image_path.name}")
            cv2.imwrite(str(image_dst / image_path.name), lowres)
    write_yaml(out_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default="OCR.zip", help="path to downloaded OCR.zip")
    parser.add_argument("--raw-dir", default="datasets/ocr_raw", help="raw extraction directory")
    parser.add_argument("--out-dir", default="datasets/ocr_lowres", help="low-res output directory")
    parser.add_argument("--mixed-dir", default="datasets/ocr_mixed", help="raw + low-res output directory")
    parser.add_argument("--skip-mixed", action="store_true", help="do not create the mixed dataset")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    raw_dir = Path(args.raw_dir)
    out_root = Path(args.out_dir)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    extract_if_needed(zip_path, raw_dir)
    raw_root = raw_dir / "OCR"
    lowres_root = out_root
    build_lowres(raw_root, lowres_root)
    if not args.skip_mixed:
        build_mixed(raw_root, lowres_root, Path(args.mixed_dir))


if __name__ == "__main__":
    main()
