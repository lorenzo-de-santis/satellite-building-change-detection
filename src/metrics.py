import torch


def make_counts() -> dict:
    return {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}


@torch.no_grad()
def update_counts(logits, targets, counts: dict, threshold: float = 0.5) -> None:
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits and targets must have the same shape, got "
            f"{tuple(logits.shape)} vs {tuple(targets.shape)}"
        )
    preds = (torch.sigmoid(logits) >= threshold).float().reshape(-1)
    targets = targets.float().reshape(-1)
    counts["tp"] += (preds * targets).sum().item()
    counts["fp"] += (preds * (1.0 - targets)).sum().item()
    counts["fn"] += ((1.0 - preds) * targets).sum().item()
    counts["tn"] += ((1.0 - preds) * (1.0 - targets)).sum().item()


def metrics_from_counts(counts: dict) -> dict:
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2.0 * tp / (2.0 * tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dice": dice,
        "accuracy": accuracy,
    }
