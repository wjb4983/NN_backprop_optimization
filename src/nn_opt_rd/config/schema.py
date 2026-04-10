from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    seed: int = 0
    steps: int = 100
    batch_size: int = 32
    optimizer: str = "adam"
    learning_rate: float = 0.03
    output_dir: str = "runs/default"


@dataclass
class BenchmarkConfig:
    seed: int = 0
    steps: int = 80
    batch_size: int = 32
    optimizers: list[str] = None
    learning_rates: list[float] = None
    output_dir: str = "runs/benchmark"

    def __post_init__(self) -> None:
        if self.optimizers is None:
            self.optimizers = ["sgd", "momentum", "adam"]
        if self.learning_rates is None:
            self.learning_rates = [0.01, 0.03]
