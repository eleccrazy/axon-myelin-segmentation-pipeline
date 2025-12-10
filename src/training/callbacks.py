"""
File: callbacks.py
Description: Reusable training callbacks such as early stopping.

Author: Gizachew Kassa
Date Created: 10/12/2025
"""

from __future__ import annotations


class EarlyStopping:
    """
    Early stopping on a monitored metric.

    This matches the logic used in the original scripts: the best value of the
    validation loss is tracked and training stops after a given number of
    epochs without improvement.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        if mode == "min":
            self.best = float("inf")
        else:
            self.best = -float("inf")

        self.num_bad_epochs = 0
        self.should_stop = False

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "min":
            return value + self.min_delta < self.best
        return value - self.min_delta > self.best

    def step(self, value: float) -> bool:
        """
        Update the early stopping state with a new metric value.

        Returns True if training should stop.
        """
        if self._is_improvement(value):
            self.best = value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.should_stop = True

        return self.should_stop
