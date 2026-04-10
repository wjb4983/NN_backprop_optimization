from __future__ import annotations

from dataclasses import dataclass

from nn_opt_rd.interfaces.core import Parameters


@dataclass
class SGD:
    lr: float

    def step(self, params: Parameters, grad: float) -> None:
        params.value -= self.lr * grad


@dataclass
class SGDMomentum:
    lr: float
    momentum: float = 0.9
    velocity: float = 0.0

    def step(self, params: Parameters, grad: float) -> None:
        self.velocity = self.momentum * self.velocity + (1.0 - self.momentum) * grad
        params.value -= self.lr * self.velocity


@dataclass
class Adam:
    lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    m: float = 0.0
    v: float = 0.0
    t: int = 0

    def step(self, params: Parameters, grad: float) -> None:
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad * grad)
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        params.value -= self.lr * m_hat / (v_hat**0.5 + self.eps)


def build_optimizer(name: str, lr: float):
    name = name.lower()
    if name == "sgd":
        return SGD(lr=lr)
    if name == "momentum":
        return SGDMomentum(lr=lr)
    if name == "adam":
        return Adam(lr=lr)
    raise ValueError(f"Unknown optimizer: {name}")
