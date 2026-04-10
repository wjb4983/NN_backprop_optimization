from __future__ import annotations

from dataclasses import dataclass

from nn_opt_rd.interfaces.core import Parameters


class AdaptiveScalarOptimizer:
    def __init__(self, lr: float) -> None:
        self.base_lr = lr
        self.lr_multiplier = 1.0
        self.trust = 1.0

    def current_lr(self) -> float:
        return self.base_lr * self.lr_multiplier

    def set_lr_multiplier(self, value: float) -> None:
        self.lr_multiplier = value

    def set_trust(self, value: float) -> None:
        self.trust = value


@dataclass
class SGD(AdaptiveScalarOptimizer):
    lr: float

    def __post_init__(self) -> None:
        super().__init__(self.lr)

    def step(self, params: Parameters, grad: float) -> None:
        params.value -= self.current_lr() * self.trust * grad


@dataclass
class SGDMomentum(AdaptiveScalarOptimizer):
    lr: float
    momentum: float = 0.9
    velocity: float = 0.0

    def __post_init__(self) -> None:
        super().__init__(self.lr)

    def set_momentum(self, value: float) -> None:
        self.momentum = value

    def step(self, params: Parameters, grad: float) -> None:
        self.velocity = self.momentum * self.velocity + (1.0 - self.momentum) * grad
        params.value -= self.current_lr() * self.trust * self.velocity


@dataclass
class Adam(AdaptiveScalarOptimizer):
    lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    m: float = 0.0
    v: float = 0.0
    t: int = 0

    def __post_init__(self) -> None:
        super().__init__(self.lr)

    def set_momentum(self, value: float) -> None:
        self.beta1 = value

    def step(self, params: Parameters, grad: float) -> None:
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad * grad)
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        params.value -= self.current_lr() * self.trust * m_hat / (v_hat**0.5 + self.eps)


@dataclass
class AdamW(Adam):
    weight_decay: float = 1e-2

    def step(self, params: Parameters, grad: float) -> None:
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad * grad)
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        step = self.current_lr() * self.trust
        params.value -= step * (m_hat / (v_hat**0.5 + self.eps) + self.weight_decay * params.value)


def build_optimizer(name: str, lr: float):
    name = name.lower()
    if name == "sgd":
        return SGD(lr=lr)
    if name == "momentum":
        return SGDMomentum(lr=lr)
    if name == "adam":
        return Adam(lr=lr)
    if name == "adamw":
        return AdamW(lr=lr)
    raise ValueError(f"Unknown optimizer: {name}")
