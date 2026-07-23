from typing import Optional

import torch
import torch.nn as nn


def assert_same_shape(logits: torch.Tensor, targets: torch.Tensor) -> None:
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits and targets must have the same shape, got "
            f"{tuple(logits.shape)} vs {tuple(targets.shape)}"
        )


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        assert_same_shape(logits, targets)
        targets = targets.float()
        probs = torch.sigmoid(logits).flatten(1)
        targets = targets.flatten(1)
        intersection = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: Optional[float] = None, bce_weight: float = 0.5):
        super().__init__()
        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            self.register_buffer("pos_weight", torch.tensor([float(pos_weight)]))
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.dice = DiceLoss()
        self.bce_weight = float(bce_weight)

    def forward(self, logits, targets):
        assert_same_shape(logits, targets)
        bce = self.bce(logits, targets.float())
        dice = self.dice(logits, targets)
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice
