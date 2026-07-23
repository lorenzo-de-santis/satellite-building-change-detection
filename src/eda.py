from __future__ import annotations

from pathlib import Path
from typing import Dict
import json

from PIL import Image
import numpy as np

from .preprocessing import IMAGE_EXTENSIONS


def summarize_split(root: str | Path, split: str, patch_size: int = 256) -> Dict:
    root = Path(root)
    a_dir = root / split / "A"
    b_dir = root / split / "B"
    mask_dir = root / split / "label"
    files = sorted(p for p in a_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

    total_patches = 0
    total_pixels = 0
    changed_pixels = 0
    sizes = {}

    for a_path in files:
        b_path = b_dir / a_path.name
        mask_path = mask_dir / a_path.name
        with Image.open(a_path) as a_img, Image.open(b_path) as b_img, Image.open(mask_path) as mask_img:
            if not (a_img.size == b_img.size == mask_img.size):
                raise ValueError(f"Unaligned sizes for {a_path.name}")
            width, height = a_img.size
            sizes[f"{width}x{height}"] = sizes.get(f"{width}x{height}", 0) + 1
            total_patches += (width // patch_size) * (height // patch_size)

            mask = np.asarray(mask_img.convert("L"), dtype=np.uint8)
            total_pixels += width * height
            changed_pixels += int((mask > 127).sum())

    return {
        "split": split,
        "images": len(files),
        "patches": total_patches,
        "patch_size": patch_size,
        "sizes": sizes,
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "change_ratio": changed_pixels / max(total_pixels, 1),
    }


def summarize_dataset(root: str | Path, patch_size: int = 256) -> Dict:
    return {
        split: summarize_split(root, split, patch_size=patch_size)
        for split in ("train", "val", "test")
    }


def save_eda_report(root: str | Path, out_json: str | Path, patch_size: int = 256) -> Dict:
    report = summarize_dataset(root, patch_size=patch_size)
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(report, f, indent=2)
    return report
