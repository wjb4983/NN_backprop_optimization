from __future__ import annotations

from nn_opt_rd.interfaces.core import Parameters


class LinearModel:
    """Simple y = w * x model for baseline optimization tests."""

    def __init__(self, init_weight: float = 0.0) -> None:
        self._params = Parameters(value=init_weight)

    @property
    def parameters(self) -> Parameters:
        return self._params

    def predict(self, x: float) -> float:
        return self._params.value * x
