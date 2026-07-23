from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import csv
import json
import time

import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .early_stopping import EarlyStopping
from .metrics import make_counts, metrics_from_counts, update_counts


def move_to_device(batch: Dict, device: torch.device) -> Dict:
    return {
        key: (value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value)
        for key, value in batch.items()
    }


def run_epoch(
    model,
    loader,
    criterion,
    device,
    threshold: float,
    train: bool,
    optimizer=None,
    scaler: GradScaler | None = None,
    grad_clip: float | None = None,
):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_samples = 0
    counts = make_counts()

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            batch = move_to_device(batch, device)
            a, b, mask = batch["a"], batch["b"], batch["mask"]

            with autocast(device_type=device.type, enabled=(scaler is not None and device.type == "cuda")):
                logits = model(a, b)
                loss = criterion(logits, mask)

            if not torch.isfinite(loss):
                phase = "train" if train else "eval"
                raise FloatingPointError(f"Non-finite {phase} loss detected: {loss.item()}")

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if grad_clip is not None and grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip is not None and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            batch_size = a.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            update_counts(logits, mask, counts, threshold=threshold)

    metrics = metrics_from_counts(counts)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics


def save_history(history: List[Dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)
    if history:
        fields = list(history[0].keys())
        with (out_dir / "history.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)


def train_model(
    model_name: str,
    model,
    loaders: Dict,
    criterion,
    device,
    out_dir: Path,
    epochs: int,
    lr: float,
    weight_decay: float,
    threshold: float,
    patience: int,
    min_delta: float,
    use_amp: bool,
    run_config: Dict,
    grad_clip: float | None = None,
    early_stop_metric: str = "val_loss",
    early_stop_mode: str = "min",
):
    out_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda") if use_amp else None
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, mode=early_stop_mode)

    best_f1 = -1.0
    best_epoch = -1
    best_path = out_dir / "best.pt"
    history = []

    print(f"\n=== Training {model_name} ===", flush=True)
    for epoch in range(1, epochs + 1):
        started = time.time()
        train_metrics = run_epoch(
            model, loaders["train"], criterion, device, threshold,
            train=True, optimizer=optimizer, scaler=scaler, grad_clip=grad_clip,
        )
        val_metrics = run_epoch(
            model, loaders["val"], criterion, device, threshold,
            train=False,
        )
        scheduler.step()

        row = {
            "model": model_name,
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.time() - started,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        save_history(history, out_dir)

        is_best = val_metrics["f1"] > best_f1
        if is_best:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            torch.save(
                {
                    "model_name": model_name,
                    "epoch": epoch,
                    "best_f1": best_f1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "run_config": run_config,
                },
                best_path,
            )

        print(
            f"{model_name} | epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | val_iou={val_metrics['iou']:.4f} | "
            f"best_f1={best_f1:.4f}@{best_epoch} | "
            f"early_bad={early_stopping.bad_epochs}/{patience}",
            flush=True,
        )

        monitor_values = {
            "val_loss": val_metrics["loss"],
            "val_f1": val_metrics["f1"],
            "val_iou": val_metrics["iou"],
        }
        if early_stop_metric not in monitor_values:
            raise ValueError(f"Unknown early_stop_metric: {early_stop_metric}")

        monitor_value = monitor_values[early_stop_metric]
        should_stop = early_stopping.step(monitor_value)
        if should_stop:
            print(
                f"Early stopping {model_name}: {early_stop_metric} did not improve by "
                f"{min_delta} for {patience} epochs.",
                flush=True,
            )
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = run_epoch(model, loaders["test"], criterion, device, threshold, train=False)
    with (out_dir / "test_metrics.json").open("w") as f:
        json.dump(test_metrics, f, indent=2)

    summary = {
        "model": model_name,
        "source": "trained",
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "epochs_ran": len(history),
        "checkpoint_best_f1": None,
        "checkpoint_best_threshold": None,
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary
