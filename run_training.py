#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.eda import save_eda_report
from src.losses import BCEDiceLoss
from src.models import MODEL_REGISTRY, build_model
from src.preprocessing import build_datasets, build_loaders, compute_mean_std_from_images
from src.training import train_model


DEFAULT_MODELS = ["simple_cnn", "unet", "siamese_temporal_attention_unet"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def detect_data_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "data" / "LEVIR-CD",
        here.parent / "LEVIR-CD",
        Path.cwd() / "data" / "LEVIR-CD",
        Path.cwd() / "LEVIR-CD",
    ]
    for candidate in candidates:
        if (candidate / "train" / "A").exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not detect LEVIR-CD root. Pass --data_root.")


def resolve_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parent / path).resolve()


def load_stats_json(path: Path) -> tuple[torch.Tensor, torch.Tensor, float | None, dict]:
    with path.open() as f:
        stats = json.load(f)
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = torch.tensor(stats["std"], dtype=torch.float32)
    pos_weight = stats.get("pos_weight", stats.get("pos_weight_est"))
    return mean, std, (float(pos_weight) if pos_weight is not None else None), stats


def load_model_configs(path: Path) -> dict:
    with path.open() as f:
        configs = json.load(f)
    missing = sorted(set(DEFAULT_MODELS) - set(configs))
    if missing:
        raise ValueError(f"Missing model configs: {missing}")
    return configs


def write_comparison(rows: list[dict], out_dir: Path) -> None:
    fields = [
        "model",
        "best_epoch",
        "epochs_ran",
        "best_val_f1",
        "test_f1",
        "test_iou",
        "test_dice",
        "test_precision",
        "test_recall",
        "test_loss",
        "test_accuracy",
    ]
    with (out_dir / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    with (out_dir / "comparison.json").open("w") as f:
        json.dump(rows, f, indent=2)

    lines = [
        "# Final LEVIR-CD Comparison",
        "",
        "| Model | Best epoch | Epochs ran | Best val F1 | Test F1 | Test IoU | Test Dice | Precision | Recall | Test loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['best_epoch']} | {row['epochs_ran']} | "
            f"{row['best_val_f1']:.4f} | {row['test_f1']:.4f} | "
            f"{row['test_iou']:.4f} | {row['test_dice']:.4f} | "
            f"{row['test_precision']:.4f} | {row['test_recall']:.4f} | "
            f"{row['test_loss']:.4f} |"
        )
    (out_dir / "RUN_RESULTS.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final LEVIR-CD models with fixed best hyperparameters.")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/final_models.json")
    parser.add_argument("--stats_json", type=str, default="checkpoints/shared_dataset_stats.json")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None, help="Override config batch_size.")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_root = detect_data_root(args.data_root)
    config_path = resolve_path(args.config)
    stats_path = resolve_path(args.stats_json)
    if config_path is None:
        raise FileNotFoundError("Missing config path.")
    model_configs = load_model_configs(config_path)

    run_name = args.run_name or time.strftime("final_run_%Y%m%d_%H%M%S")
    out_dir = (Path(args.output_dir) / run_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    use_amp = device.type == "cuda" and not args.disable_amp

    print("=" * 80, flush=True)
    print("LEVIR-CD FINAL TRAINING", flush=True)
    print(f"data_root: {data_root}", flush=True)
    print(f"out_dir  : {out_dir}", flush=True)
    print(f"device   : {device} | amp={use_amp}", flush=True)
    print(f"models   : {args.models}", flush=True)
    print("=" * 80, flush=True)

    eda_report = save_eda_report(data_root, out_dir / "eda_summary.json", patch_size=args.patch_size)
    computed_pos_weight = (
        (eda_report["train"]["total_pixels"] - eda_report["train"]["changed_pixels"])
        / max(eda_report["train"]["changed_pixels"], 1)
    )

    loaded_stats = None
    if stats_path is not None and stats_path.exists():
        mean, std, base_pos_weight, loaded_stats = load_stats_json(stats_path)
        if base_pos_weight is None:
            base_pos_weight = computed_pos_weight
    else:
        mean, std = compute_mean_std_from_images(data_root, split="train")
        base_pos_weight = computed_pos_weight

    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "base_pos_weight": base_pos_weight,
        "computed_pos_weight": computed_pos_weight,
        "loaded_stats": loaded_stats,
    }
    with (out_dir / "dataset_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)

    run_config = {
        "data_root": str(data_root),
        "models": args.models,
        "device": str(device),
        "amp": use_amp,
        "seed": args.seed,
        "patch_size": args.patch_size,
        "num_workers": args.num_workers,
        "model_configs": {name: model_configs[name] for name in args.models},
        "dataset_stats": stats,
    }
    with (out_dir / "run_config.json").open("w") as f:
        json.dump(run_config, f, indent=2)

    summaries: list[dict] = []
    for model_name in args.models:
        cfg = dict(model_configs[model_name])
        if args.batch_size is not None:
            cfg["batch_size"] = args.batch_size
        if args.epochs is not None:
            cfg["epochs"] = args.epochs

        datasets = build_datasets(data_root, patch_size=args.patch_size, mean=mean, std=std)
        loaders = build_loaders(
            datasets,
            batch_size=int(cfg["batch_size"]),
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            seed=args.seed,
        )

        effective_pos_weight = float(base_pos_weight) * float(cfg["pos_weight_scale"])
        criterion = BCEDiceLoss(pos_weight=effective_pos_weight, bce_weight=float(cfg["bce_weight"])).to(device)
        model = build_model(model_name).to(device)
        model_run_config = {
            **run_config,
            "model": model_name,
            "hyperparameters": cfg,
            "effective_pos_weight": effective_pos_weight,
        }

        summary = train_model(
            model_name=model_name,
            model=model,
            loaders=loaders,
            criterion=criterion,
            device=device,
            out_dir=out_dir / model_name,
            epochs=int(cfg["epochs"]),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
            threshold=float(cfg["threshold"]),
            patience=int(cfg["patience"]),
            min_delta=float(cfg["min_delta"]),
            use_amp=use_amp,
            run_config=model_run_config,
            grad_clip=float(cfg["grad_clip"]) if float(cfg["grad_clip"]) > 0 else None,
            early_stop_metric=str(cfg["early_stop_metric"]),
            early_stop_mode=str(cfg["early_stop_mode"]),
        )
        summaries.append(summary)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_comparison(summaries, out_dir)
    print(f"Saved final run artifacts in: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
