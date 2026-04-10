from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Parameters:
    value: float


class Task(Protocol):
    def sample_batch(self, batch_size: int) -> tuple[list[float], list[float]]:
        ...


class Model(Protocol):
    @property
    def parameters(self) -> Parameters:
        ...

    def predict(self, x: float) -> float:
        ...


class Loss(Protocol):
    def value_and_grad(
        self,
        model: Model,
        xs: list[float],
        ys: list[float],
    ) -> tuple[float, float]:
        ...


class Optimizer(Protocol):
    def step(self, params: Parameters, grad: float) -> None:
        ...
