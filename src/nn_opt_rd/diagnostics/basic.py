from __future__ import annotations


def summarize_step(loss: float, grad: float, weight: float) -> dict:
    return {
        "loss": loss,
        "grad_abs": abs(grad),
        "weight": weight,
    }
