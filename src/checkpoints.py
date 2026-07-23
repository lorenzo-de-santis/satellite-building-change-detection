from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def extract_model_state(checkpoint: Any) -> tuple[dict, str]:
    if isinstance(checkpoint, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value, key
        if checkpoint and all(hasattr(value, "shape") for value in checkpoint.values()):
            return checkpoint, "raw"
    raise ValueError("Could not find a model state_dict in checkpoint.")


def normalize_state_keys(state_dict: dict) -> dict:
    normalized = {}
    for key, value in state_dict.items():
        new_key = key.removeprefix("module.")
        normalized[new_key] = value
    return normalized


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    checkpoint_path = Path(path).expanduser()
    return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def load_model_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    map_location="cpu",
) -> dict:
    checkpoint = load_checkpoint(path, map_location=map_location)
    state_dict, state_key = extract_model_state(checkpoint)
    state_dict = normalize_state_keys(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match model after key normalization. "
            f"missing={list(missing)[:10]} unexpected={list(unexpected)[:10]}"
        )
    if isinstance(checkpoint, dict):
        checkpoint["_loaded_state_key"] = state_key
    return checkpoint
