from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int = 35
    min_delta: float = 1e-5
    mode: str = "min"

    def __post_init__(self):
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.best = None
        self.bad_epochs = 0

    def improved(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value: float) -> bool:
        if self.improved(value):
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience
