#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.checkpoints import load_model_checkpoint
from src.models import build_model
from src.preprocessing import (
    IMAGE_EXTENSIONS,
    LEVIRPatchDataset,
    normalize_rgb,
    pil_to_tensor,
    seed_worker,
)


MODEL_LABELS = {
    "simple_cnn": "Simple CNN",
    "unet": "U-Net",
    "siamese_temporal_attention_unet": "Siamese Temporal Attention",
}


def detect_data_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()

    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "data" / "LEVIR-CD",
        here.parent / "data",
        here.parent / "LEVIR-CD",
        Path.cwd() / "data" / "LEVIR-CD",
        Path.cwd() / "data",
        Path.cwd() / "LEVIR-CD",
    ]
    for candidate in candidates:
        if (candidate / "train" / "A").exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not detect LEVIR-CD root. Pass --data_root.")


def load_stats_json(path: str | Path) -> tuple[torch.Tensor, torch.Tensor, dict]:
    stats_path = Path(path).expanduser().resolve()
    with stats_path.open() as f:
        stats = json.load(f)
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = torch.tensor(stats["std"], dtype=torch.float32)
    return mean, std, stats


def resolve_project_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parent / path).resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def tensor_to_device(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a = batch["a"].to(device, non_blocking=True)
    b = batch["b"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    return a, b, mask


def init_counts(thresholds: Iterable[float]) -> dict[float, dict[str, float]]:
    return {float(t): {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0} for t in thresholds}


def update_counts_from_probs(
    probs: torch.Tensor,
    targets: torch.Tensor,
    counts_by_threshold: dict[float, dict[str, float]],
) -> None:
    targets_flat = targets.float().reshape(-1)
    probs_flat = probs.reshape(-1)
    for threshold, counts in counts_by_threshold.items():
        preds = (probs_flat >= threshold).float()
        counts["tp"] += (preds * targets_flat).sum().item()
        counts["fp"] += (preds * (1.0 - targets_flat)).sum().item()
        counts["fn"] += ((1.0 - preds) * targets_flat).sum().item()
        counts["tn"] += ((1.0 - preds) * (1.0 - targets_flat)).sum().item()


def metrics_from_counts(counts: dict[str, float]) -> dict[str, float]:
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2.0 * tp / (2.0 * tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    pred_change_ratio = (tp + fp) / (tp + fp + fn + tn + eps)
    true_change_ratio = (tp + fn) / (tp + fp + fn + tn + eps)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dice": dice,
        "accuracy": accuracy,
        "pred_change_ratio": pred_change_ratio,
        "true_change_ratio": true_change_ratio,
    }


def save_json(path: Path, payload: dict | list) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def short_label(label: str) -> str:
    return MODEL_LABELS.get(label, label)


def save_bar(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    out_path: Path,
    color: str = "#2f6f9f",
    value_format: str = "{:.3f}",
) -> None:
    fig = plt.figure(figsize=(7.0, 4.8), dpi=180)
    ax = fig.add_subplot(111)
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=color, edgecolor="#1f2933", linewidth=0.7)
    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    max_value = max(values) if values else 0.0
    offset = max(max_value * 0.015, 1e-6)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + offset,
            value_format.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_line(
    x_values: list[float],
    y_values: list[float],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    color: str = "#2f6f9f",
) -> None:
    fig = plt.figure(figsize=(7.0, 4.8), dpi=180)
    ax = fig.add_subplot(111)
    ax.plot(x_values, y_values, color=color, linewidth=2.0)
    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_hist(
    values: list[float],
    title: str,
    xlabel: str,
    out_path: Path,
    bins: int = 40,
    color: str = "#2f6f9f",
) -> None:
    fig = plt.figure(figsize=(7.0, 4.8), dpi=180)
    ax = fig.add_subplot(111)
    ax.hist(values, bins=bins, color=color, edgecolor="#1f2933", linewidth=0.4)
    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Patch count")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_image(path: Path, image: Image.Image) -> None:
    image.save(path)


def save_binary_mask(mask: np.ndarray, out_path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(out_path)


def save_probability_heatmap(prob: np.ndarray, out_path: Path) -> None:
    plt.imsave(out_path, prob, cmap="inferno", vmin=0.0, vmax=1.0)


def save_overlay(rgb_image: Image.Image, mask: np.ndarray, out_path: Path, color=(255, 40, 40), alpha=0.45) -> None:
    base = np.asarray(rgb_image.convert("RGB"), dtype=np.float32)
    overlay = np.zeros_like(base)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    out = base.copy()
    m = mask.astype(bool)
    out[m] = (1.0 - alpha) * base[m] + alpha * overlay[m]
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB").save(out_path)


def save_error_map(pred: np.ndarray, target: np.ndarray, out_path: Path) -> None:
    pred = pred.astype(bool)
    target = target.astype(bool)
    out = np.zeros((*target.shape, 3), dtype=np.uint8)
    out[pred & target] = (35, 170, 80)       # true positive
    out[pred & ~target] = (220, 50, 47)      # false positive
    out[~pred & target] = (45, 105, 210)     # false negative
    Image.fromarray(out, mode="RGB").save(out_path)


def save_abs_rgb_difference(a_image: Image.Image, b_image: Image.Image, out_path: Path) -> None:
    a = np.asarray(a_image.convert("RGB"), dtype=np.int16)
    b = np.asarray(b_image.convert("RGB"), dtype=np.int16)
    diff = np.abs(b - a).astype(np.uint8)
    Image.fromarray(diff, mode="RGB").save(out_path)


def crop_raw_triplet(dataset: LEVIRPatchDataset, idx: int) -> tuple[Image.Image, Image.Image, Image.Image, dict]:
    a_path, top, left = dataset.samples[idx]
    b_path = dataset.b_dir / a_path.name
    mask_path = dataset.mask_dir / a_path.name
    box = (left, top, left + dataset.patch_size, top + dataset.patch_size)
    with Image.open(a_path) as a_img, Image.open(b_path) as b_img, Image.open(mask_path) as mask_img:
        a_crop = a_img.crop(box).convert("RGB")
        b_crop = b_img.crop(box).convert("RGB")
        mask_crop = mask_img.crop(box).convert("L")
    metadata = {
        "source_image": a_path.name,
        "patch_top": top,
        "patch_left": left,
        "patch_size": dataset.patch_size,
        "sample_name": f"{a_path.stem}_y{top}_x{left}",
    }
    return a_crop, b_crop, mask_crop, metadata


def mask_change_ratio(mask_image: Image.Image) -> float:
    arr = np.asarray(mask_image.convert("L"))
    return float((arr > 127).mean())


def choose_sample_index(
    dataset: LEVIRPatchDataset,
    seed: int,
    sample_index: int | None,
    min_change_ratio: float,
) -> int:
    if sample_index is not None:
        if sample_index < 0 or sample_index >= len(dataset):
            raise IndexError(f"sample_index={sample_index} outside dataset length {len(dataset)}")
        return sample_index

    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    fallback = indices[0]
    for idx in indices:
        _, _, mask_crop, _ = crop_raw_triplet(dataset, idx)
        ratio = mask_change_ratio(mask_crop)
        if ratio > 0.0:
            fallback = idx
        if ratio >= min_change_ratio:
            return idx
    return fallback


def save_eda_assets(data_root: Path, patch_size: int, out_dir: Path) -> dict:
    ensure_dir(out_dir)
    summary: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        a_dir = data_root / split / "A"
        mask_dir = data_root / split / "label"
        files = sorted(p for p in a_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        patch_ratios: list[float] = []
        changed_pixels = 0
        total_pixels = 0
        patch_count = 0
        for a_path in files:
            mask_path = mask_dir / a_path.name
            with Image.open(mask_path) as mask_img:
                mask = np.asarray(mask_img.convert("L")) > 127
            height, width = mask.shape
            for top in range(0, height, patch_size):
                for left in range(0, width, patch_size):
                    crop = mask[top:top + patch_size, left:left + patch_size]
                    positives = int(crop.sum())
                    total = int(crop.size)
                    changed_pixels += positives
                    total_pixels += total
                    patch_count += 1
                    patch_ratios.append(positives / max(total, 1))
        summary[split] = {
            "images": len(files),
            "patches": patch_count,
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "change_ratio": changed_pixels / max(total_pixels, 1),
            "empty_patch_fraction": float(np.mean(np.asarray(patch_ratios) == 0.0)) if patch_ratios else 0.0,
            "patch_change_ratio_mean": float(np.mean(patch_ratios)) if patch_ratios else 0.0,
            "patch_change_ratio_median": float(np.median(patch_ratios)) if patch_ratios else 0.0,
        }
        save_hist(
            patch_ratios,
            title=f"{split} patch change-ratio distribution",
            xlabel="Changed-pixel ratio per patch",
            out_path=out_dir / f"{split}_patch_change_ratio_histogram.png",
        )

    split_labels = list(summary)
    save_bar(
        split_labels,
        [summary[s]["images"] for s in split_labels],
        "Image pairs by split",
        "Image pairs",
        out_dir / "split_image_pairs.png",
        value_format="{:.0f}",
    )
    save_bar(
        split_labels,
        [summary[s]["patches"] for s in split_labels],
        "Patch count by split",
        "Patches",
        out_dir / "split_patch_counts.png",
        value_format="{:.0f}",
    )
    save_bar(
        split_labels,
        [summary[s]["change_ratio"] for s in split_labels],
        "Changed-pixel ratio by split",
        "Changed-pixel ratio",
        out_dir / "split_change_ratio.png",
        value_format="{:.4f}",
    )
    save_bar(
        split_labels,
        [summary[s]["empty_patch_fraction"] for s in split_labels],
        "Empty-patch fraction by split",
        "Fraction of patches with no changed pixels",
        out_dir / "split_empty_patch_fraction.png",
        value_format="{:.3f}",
    )
    save_json(out_dir / "eda_summary.json", summary)
    return summary


def load_models(checkpoints: dict[str, Path], device: torch.device) -> dict[str, torch.nn.Module]:
    models: dict[str, torch.nn.Module] = {}
    for model_name, checkpoint_path in checkpoints.items():
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint for {model_name} not found: {checkpoint_path}")
        model = build_model(model_name).to(device)
        load_model_checkpoint(
            model,
            checkpoint_path,
            map_location=device,
        )
        model.eval()
        models[model_name] = model
    return models


def save_model_parameter_assets(models: dict[str, torch.nn.Module], out_dir: Path) -> dict[str, int]:
    ensure_dir(out_dir)
    params = {
        model_name: int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        for model_name, model in models.items()
    }
    save_bar(
        [short_label(name) for name in params],
        list(params.values()),
        "Trainable parameters",
        "Parameters",
        out_dir / "model_trainable_parameters.png",
        color="#466d4f",
        value_format="{:.0f}",
    )
    save_json(out_dir / "model_parameters.json", params)
    return params


def read_history(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def save_history_assets(history_paths: dict[str, Path | None], out_dir: Path) -> dict[str, str]:
    ensure_dir(out_dir)
    saved: dict[str, str] = {}
    columns = [
        "train_loss",
        "val_loss",
        "train_f1",
        "val_f1",
        "train_iou",
        "val_iou",
        "train_precision",
        "val_precision",
        "train_recall",
        "val_recall",
    ]
    for model_name, history_path in history_paths.items():
        if history_path is None or not history_path.exists():
            continue
        rows = read_history(history_path)
        if not rows or "epoch" not in rows[0]:
            continue
        epochs = [float(row["epoch"]) for row in rows if isinstance(row.get("epoch"), (int, float))]
        for column in columns:
            values = [row.get(column) for row in rows]
            if not values or any(not isinstance(v, (int, float)) for v in values):
                continue
            file_name = f"{model_name}_{column}.png"
            save_line(
                epochs,
                [float(v) for v in values],
                title=f"{short_label(model_name)} {column.replace('_', ' ')}",
                xlabel="Epoch",
                ylabel=column.replace("_", " "),
                out_path=out_dir / file_name,
            )
            saved[f"{model_name}:{column}"] = str(out_dir / file_name)
    save_json(out_dir / "history_assets.json", saved)
    return saved


@torch.no_grad()
def evaluate_models(
    models: dict[str, torch.nn.Module],
    dataset: LEVIRPatchDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    thresholds: list[float],
    max_eval_patches: int,
    seed: int,
) -> tuple[dict, list[dict]]:
    if max_eval_patches and max_eval_patches > 0 and max_eval_patches < len(dataset):
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(dataset)), max_eval_patches))
        eval_dataset = Subset(dataset, indices)
    else:
        indices = list(range(len(dataset)))
        eval_dataset = dataset

    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )

    metrics_by_model = {}
    threshold_rows: list[dict] = []
    for model_name, model in models.items():
        counts_by_threshold = init_counts(thresholds)
        for batch in loader:
            a, b, mask = tensor_to_device(batch, device)
            probs = torch.sigmoid(model(a, b))
            update_counts_from_probs(probs, mask, counts_by_threshold)

        model_threshold_metrics = {
            str(threshold): metrics_from_counts(counts)
            for threshold, counts in counts_by_threshold.items()
        }
        metrics_by_model[model_name] = {
            "num_eval_patches": len(indices),
            "threshold_metrics": model_threshold_metrics,
        }
        for threshold, values in model_threshold_metrics.items():
            threshold_float = float(threshold)
            threshold_rows.append({"model": model_name, "threshold": threshold_float, **values})
    return metrics_by_model, threshold_rows


def save_metric_assets(threshold_rows: list[dict], threshold: float, out_dir: Path) -> None:
    ensure_dir(out_dir)
    chosen = [row for row in threshold_rows if abs(float(row["threshold"]) - threshold) < 1e-9]
    chosen.sort(key=lambda row: row["model"])
    if not chosen:
        return

    labels = [short_label(row["model"]) for row in chosen]
    for metric in ("f1", "iou", "dice", "precision", "recall", "accuracy"):
        save_bar(
            labels,
            [float(row[metric]) for row in chosen],
            f"Test {metric.upper()} at threshold {threshold:.2f}",
            metric.upper(),
            out_dir / f"test_{metric}_bar.png",
            color="#7c5d3b",
            value_format="{:.4f}",
        )

    gt_change_ratio = float(chosen[0]["true_change_ratio"])
    save_bar(
        ["Ground truth"] + labels,
        [gt_change_ratio] + [float(row["pred_change_ratio"]) for row in chosen],
        f"Predicted changed-pixel ratio at threshold {threshold:.2f}",
        "Changed-pixel ratio",
        out_dir / "predicted_change_ratio.png",
        color="#5f6f52",
        value_format="{:.4f}",
    )

    by_model: dict[str, list[dict]] = {}
    for row in threshold_rows:
        by_model.setdefault(row["model"], []).append(row)
    for model_name, rows in by_model.items():
        rows.sort(key=lambda row: float(row["threshold"]))
        x = [float(row["threshold"]) for row in rows]
        for metric in ("f1", "precision", "recall"):
            save_line(
                x,
                [float(row[metric]) for row in rows],
                title=f"{short_label(model_name)} threshold sensitivity: {metric}",
                xlabel="Threshold",
                ylabel=metric.upper(),
                out_path=out_dir / f"{model_name}_threshold_{metric}.png",
                color="#7c5d3b",
            )


def save_sample_assets(
    models: dict[str, torch.nn.Module],
    dataset: LEVIRPatchDataset,
    idx: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    threshold: float,
    device: torch.device,
    out_dir: Path,
) -> dict:
    ensure_dir(out_dir)
    a_raw, b_raw, mask_raw, metadata = crop_raw_triplet(dataset, idx)
    mask_np = np.asarray(mask_raw, dtype=np.uint8) > 127

    save_image(out_dir / "image_a.png", a_raw)
    save_image(out_dir / "image_b.png", b_raw)
    save_binary_mask(mask_np, out_dir / "ground_truth_mask.png")
    save_overlay(b_raw, mask_np, out_dir / "ground_truth_overlay_on_image_b.png", color=(40, 210, 210), alpha=0.45)
    save_abs_rgb_difference(a_raw, b_raw, out_dir / "rgb_absolute_difference.png")

    a_tensor = normalize_rgb(pil_to_tensor(a_raw), mean, std).unsqueeze(0).to(device)
    b_tensor = normalize_rgb(pil_to_tensor(b_raw), mean, std).unsqueeze(0).to(device)

    sample_rows: list[dict] = []
    with torch.no_grad():
        for model_name, model in models.items():
            prob = torch.sigmoid(model(a_tensor, b_tensor))[0, 0].detach().cpu().numpy()
            pred = prob >= threshold
            save_probability_heatmap(prob, out_dir / f"{model_name}_probability_heatmap.png")
            save_binary_mask(pred, out_dir / f"{model_name}_predicted_mask.png")
            save_overlay(b_raw, pred, out_dir / f"{model_name}_prediction_overlay_on_image_b.png")
            save_error_map(pred, mask_np, out_dir / f"{model_name}_error_map.png")

            counts = init_counts([threshold])[threshold]
            update_counts_from_probs(
                torch.from_numpy(prob).view(1, 1, *prob.shape),
                torch.from_numpy(mask_np.astype(np.float32)).view(1, 1, *mask_np.shape),
                {threshold: counts},
            )
            sample_rows.append({
                "model": model_name,
                "threshold": threshold,
                **metrics_from_counts(counts),
            })

    metadata["dataset_index"] = idx
    metadata["change_ratio"] = float(mask_np.mean())
    metadata["threshold"] = threshold
    save_json(out_dir / "sample_metadata.json", metadata)
    save_csv(out_dir / "sample_metrics.csv", sample_rows)
    return {"metadata": metadata, "sample_metrics": sample_rows}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate atomic LEVIR-CD report assets and per-model prediction masks."
    )
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/prediction_assets")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--stats_json", type=str, default="checkpoints/shared_dataset_stats.json")
    parser.add_argument("--simple_cnn_checkpoint", type=str, required=True)
    parser.add_argument("--unet_checkpoint", type=str, required=True)
    parser.add_argument("--siamese_temporal_attention_checkpoint", type=str, required=True)
    parser.add_argument("--simple_cnn_history", type=str, default=None)
    parser.add_argument("--unet_history", type=str, default=None)
    parser.add_argument("--siamese_temporal_attention_history", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--sample_index", type=int, default=None)
    parser.add_argument("--min_change_ratio", type=float, default=0.002)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--thresholds", type=str, default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--max_eval_patches", type=int, default=0, help="0 means evaluate the full split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_cuda", action="store_true")
    parser.add_argument("--disable_mps", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    data_root = detect_data_root(args.data_root)
    stats_json = resolve_project_path(args.stats_json)
    if stats_json is None or not stats_json.exists():
        raise FileNotFoundError(f"Stats JSON not found: {stats_json}")
    mean, std, stats = load_stats_json(stats_json)

    run_name = args.run_name or time.strftime("prediction_assets_%Y%m%d_%H%M%S")
    output_root = ensure_dir(resolve_project_path(args.output_dir) / run_name)
    eda_dir = ensure_dir(output_root / "eda")
    metrics_dir = ensure_dir(output_root / "metrics")
    history_dir = ensure_dir(output_root / "history")
    sample_dir = ensure_dir(output_root / "sample")

    if torch.cuda.is_available() and not args.disable_cuda:
        device = torch.device("cuda")
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and not args.disable_mps
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    checkpoints = {
        "simple_cnn": resolve_project_path(args.simple_cnn_checkpoint),
        "unet": resolve_project_path(args.unet_checkpoint),
        "siamese_temporal_attention_unet": resolve_project_path(args.siamese_temporal_attention_checkpoint),
    }
    checkpoints = {name: path for name, path in checkpoints.items() if path is not None}

    config = {
        **vars(args),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "device": str(device),
        "stats_json": str(stats_json),
        "stats": stats,
        "checkpoints": {name: str(path) for name, path in checkpoints.items()},
    }
    save_json(output_root / "asset_run_config.json", config)

    print("=" * 80, flush=True)
    print("LEVIR-CD atomic asset generation", flush=True)
    print(f"data_root : {data_root}", flush=True)
    print(f"output    : {output_root}", flush=True)
    print(f"device    : {device}", flush=True)
    print("=" * 80, flush=True)

    print("[1/6] EDA assets", flush=True)
    eda_summary = save_eda_assets(data_root, args.patch_size, eda_dir)

    print("[2/6] Loading models", flush=True)
    models = load_models(checkpoints, device)
    save_model_parameter_assets(models, metrics_dir)

    print("[3/6] History assets", flush=True)
    history_paths = {
        "simple_cnn": resolve_project_path(args.simple_cnn_history),
        "unet": resolve_project_path(args.unet_history),
        "siamese_temporal_attention_unet": resolve_project_path(args.siamese_temporal_attention_history),
    }
    save_history_assets(history_paths, history_dir)

    print("[4/6] Dataset and global inference metrics", flush=True)
    eval_dataset = LEVIRPatchDataset(
        data_root,
        args.split,
        patch_size=args.patch_size,
        mean=mean,
        std=std,
        augment=False,
    )
    thresholds = [float(value.strip()) for value in args.thresholds.split(",") if value.strip()]
    if args.threshold not in thresholds:
        thresholds.append(float(args.threshold))
        thresholds = sorted(set(thresholds))
    metrics_by_model, threshold_rows = evaluate_models(
        models=models,
        dataset=eval_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        thresholds=thresholds,
        max_eval_patches=args.max_eval_patches,
        seed=args.seed,
    )
    save_json(metrics_dir / "computed_metrics.json", metrics_by_model)
    save_csv(metrics_dir / "threshold_metrics.csv", threshold_rows)
    save_metric_assets(threshold_rows, args.threshold, metrics_dir)

    print("[5/6] Random sample prediction assets", flush=True)
    sample_idx = choose_sample_index(
        eval_dataset,
        seed=args.seed,
        sample_index=args.sample_index,
        min_change_ratio=args.min_change_ratio,
    )
    sample_summary = save_sample_assets(
        models=models,
        dataset=eval_dataset,
        idx=sample_idx,
        mean=mean,
        std=std,
        threshold=args.threshold,
        device=device,
        out_dir=sample_dir,
    )

    print("[6/6] Manifest", flush=True)
    png_files = sorted(str(path.relative_to(output_root)) for path in output_root.rglob("*.png"))
    manifest = {
        "output_root": str(output_root),
        "png_count": len(png_files),
        "png_files": png_files,
        "eda_summary": eda_summary,
        "sample": sample_summary,
    }
    save_json(output_root / "asset_manifest.json", manifest)

    print(f"Saved {len(png_files)} PNG files in {output_root}", flush=True)


if __name__ == "__main__":
    main()
