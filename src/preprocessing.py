from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return torch.from_numpy(arr.transpose(2, 0, 1))


def normalize_rgb(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    mean = mean.to(dtype=x.dtype).view(3, 1, 1)
    std = std.to(dtype=x.dtype).view(3, 1, 1)
    return (x - mean) / std.clamp_min(1e-7)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class JointTransform:
    def __init__(
        self,
        size: int = 256,
        train: bool = True,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
    ):
        self.size = size
        self.train = train
        self.mean = mean
        self.std = std

    def __call__(self, a: Image.Image, b: Image.Image, mask: Image.Image):
        a = a.resize((self.size, self.size), Image.LANCZOS)
        b = b.resize((self.size, self.size), Image.LANCZOS)
        mask = mask.resize((self.size, self.size), Image.NEAREST)

        if self.train:
            if random.random() < 0.5:
                a = a.transpose(Image.FLIP_LEFT_RIGHT)
                b = b.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                a = a.transpose(Image.FLIP_TOP_BOTTOM)
                b = b.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

            angle = random.uniform(-15, 15)
            a = a.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
            b = b.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
            mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)

        a_t = pil_to_tensor(a.convert("RGB"))
        b_t = pil_to_tensor(b.convert("RGB"))
        mask_t = (pil_to_tensor(mask.convert("L")) > 0.5).float()

        if self.mean is not None and self.std is not None:
            a_t = normalize_rgb(a_t, self.mean, self.std)
            b_t = normalize_rgb(b_t, self.mean, self.std)

        return a_t, b_t, mask_t


class LEVIRPatchDataset(Dataset):
    """LEVIR-CD dataset split into aligned 256x256 A/B/mask patches."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        patch_size: int = 256,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        augment: Optional[bool] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.patch_size = patch_size
        base = self.root / split
        self.a_dir = base / "A"
        self.b_dir = base / "B"
        self.mask_dir = base / "label"

        for directory in (self.a_dir, self.b_dir, self.mask_dir):
            if not directory.exists():
                raise FileNotFoundError(f"Missing directory: {directory}")

        self.files = sorted(p for p in self.a_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.files:
            raise FileNotFoundError(f"No images found in {self.a_dir}")

        self.samples: List[Tuple[Path, int, int]] = []
        for a_path in self.files:
            b_path = self.b_dir / a_path.name
            mask_path = self.mask_dir / a_path.name
            if not b_path.exists() or not mask_path.exists():
                raise FileNotFoundError(f"Missing B image or mask for {a_path.name}")

            with Image.open(a_path) as a_img, Image.open(b_path) as b_img, Image.open(mask_path) as mask_img:
                if not (a_img.size == b_img.size == mask_img.size):
                    raise ValueError(f"Unaligned sizes for {a_path.name}: {a_img.size}, {b_img.size}, {mask_img.size}")
                width, height = a_img.size

            if width % patch_size != 0 or height % patch_size != 0:
                raise ValueError(f"{a_path.name} size {width}x{height} is not divisible by patch_size={patch_size}")

            for top in range(0, height, patch_size):
                for left in range(0, width, patch_size):
                    self.samples.append((a_path, top, left))

        if augment is None:
            augment = split == "train"
        self.transform = JointTransform(size=patch_size, train=augment, mean=mean, std=std)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        a_path, top, left = self.samples[idx]
        box = (left, top, left + self.patch_size, top + self.patch_size)
        b_path = self.b_dir / a_path.name
        mask_path = self.mask_dir / a_path.name
        with Image.open(a_path) as a_img, Image.open(b_path) as b_img, Image.open(mask_path) as mask_img:
            a_crop = a_img.crop(box).copy()
            b_crop = b_img.crop(box).copy()
            mask_crop = mask_img.crop(box).copy()
        a, b, mask = self.transform(
            a_crop,
            b_crop,
            mask_crop,
        )
        return {
            "name": f"{a_path.stem}_y{top}_x{left}",
            "a": a,
            "b": b,
            "mask": mask,
        }


def compute_mean_std(
    dataset: Dataset,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sum_sq = torch.zeros(3, dtype=torch.float64)
    num_pixels = 0

    for batch in loader:
        imgs = torch.cat([batch["a"], batch["b"]], dim=0).to(torch.float64)
        bsz, _, height, width = imgs.shape
        num_pixels += bsz * height * width
        channel_sum += imgs.sum(dim=[0, 2, 3])
        channel_sum_sq += (imgs ** 2).sum(dim=[0, 2, 3])

    mean = channel_sum / num_pixels
    variance = (channel_sum_sq / num_pixels - mean ** 2).clamp_min(0.0)
    std = torch.sqrt(variance)
    return mean.float(), std.float()


def compute_mean_std_from_images(root: str | Path, split: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute RGB mean/std from full A/B images.

    LEVIR-CD patches are a non-overlapping tiling of the full images, so this is
    equivalent to computing stats over all extracted patches and much faster.
    """
    root = Path(root)
    a_dir = root / split / "A"
    b_dir = root / split / "B"
    files = sorted(p for p in a_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)
    num_pixels = 0

    for a_path in files:
        for path in (a_path, b_dir / a_path.name):
            with Image.open(path) as img:
                arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
            channel_sum += arr.sum(axis=(0, 1), dtype=np.float64)
            channel_sum_sq += (arr ** 2).sum(axis=(0, 1), dtype=np.float64)
            num_pixels += arr.shape[0] * arr.shape[1]

    mean = channel_sum / num_pixels
    std = np.sqrt(channel_sum_sq / num_pixels - mean ** 2)
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


def estimate_pos_weight(dataset: Dataset, max_samples: Optional[int] = None, seed: int = 42) -> float:
    indices = list(range(len(dataset)))
    if max_samples is not None and max_samples > 0 and max_samples < len(indices):
        indices = random.Random(seed).sample(indices, max_samples)

    positive = 0.0
    total = 0.0
    for idx in indices:
        mask = dataset[idx]["mask"]
        positive += mask.sum().item()
        total += mask.numel()
    negative = total - positive
    return negative / max(positive, 1.0)


def build_datasets(root: str | Path, patch_size: int, mean: torch.Tensor, std: torch.Tensor):
    return {
        "train": LEVIRPatchDataset(root, "train", patch_size=patch_size, mean=mean, std=std, augment=True),
        "val": LEVIRPatchDataset(root, "val", patch_size=patch_size, mean=mean, std=std, augment=False),
        "test": LEVIRPatchDataset(root, "test", patch_size=patch_size, mean=mean, std=std, augment=False),
    }


def build_loaders(
    datasets: Dict[str, Dataset],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int = 42,
):
    generator = torch.Generator()
    generator.manual_seed(seed)
    worker_init_fn = seed_worker if num_workers > 0 else None
    return {
        "train": DataLoader(
            datasets["train"], batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
            worker_init_fn=worker_init_fn, generator=generator,
        ),
        "val": DataLoader(
            datasets["val"], batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
        ),
        "test": DataLoader(
            datasets["test"], batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
        ),
    }
