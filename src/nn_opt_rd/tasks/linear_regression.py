from __future__ import annotations

import random


class LinearRegressionTask:
    """Synthetic task with known underlying slope."""

    def __init__(self, true_weight: float = 2.5, noise_std: float = 0.1, seed: int = 0) -> None:
        self.true_weight = true_weight
        self.noise_std = noise_std
        self._rng = random.Random(seed)

    def sample_batch(self, batch_size: int) -> tuple[list[float], list[float]]:
        xs: list[float] = []
        ys: list[float] = []
        for _ in range(batch_size):
            x = self._rng.uniform(-1.0, 1.0)
            y = self.true_weight * x + self._rng.gauss(0.0, self.noise_std)
            xs.append(x)
            ys.append(y)
        return xs, ys
