from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


@dataclass
class RollingDiagnostics:
    window: int
    losses: deque[float] = field(init=False)
    grad_norms: deque[float] = field(init=False)
    update_norms: deque[float] = field(init=False)
    alignments: deque[float] = field(init=False)
    clip_flags: deque[float] = field(init=False)
    clip_ratios: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        self.losses = deque(maxlen=self.window)
        self.grad_norms = deque(maxlen=self.window)
        self.update_norms = deque(maxlen=self.window)
        self.alignments = deque(maxlen=self.window)
        self.clip_flags = deque(maxlen=self.window)
        self.clip_ratios = deque(maxlen=self.window)

    def update(self, *, loss: float, grad_norm: float, update_norm: float, alignment: float, clipped: bool, clip_ratio: float) -> dict:
        self.losses.append(loss)
        self.grad_norms.append(grad_norm)
        self.update_norms.append(update_norm)
        self.alignments.append(alignment)
        self.clip_flags.append(1.0 if clipped else 0.0)
        self.clip_ratios.append(clip_ratio)
        return self.features()

    def _mean(self, values: deque[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _std(self, values: deque[float], mean: float) -> float:
        if not values:
            return 0.0
        return (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5

    def _loss_slope(self) -> float:
        if len(self.losses) < 2:
            return 0.0
        return (self.losses[-1] - self.losses[0]) / (len(self.losses) - 1)

    def features(self) -> dict:
        mean_loss = self._mean(self.losses)
        std_loss = self._std(self.losses, mean_loss)
        mean_grad = self._mean(self.grad_norms)
        std_grad = self._std(self.grad_norms, mean_grad)
        mean_update = self._mean(self.update_norms)
        return {
            "loss_mean": mean_loss,
            "loss_std": std_loss,
            "loss_cv": _safe_div(std_loss, mean_loss + 1e-12),
            "loss_slope": self._loss_slope(),
            "grad_norm_mean": mean_grad,
            "grad_norm_cv": _safe_div(std_grad, mean_grad + 1e-12),
            "update_norm_mean": mean_update,
            "update_to_grad_ratio": _safe_div(mean_update, mean_grad + 1e-12),
            "update_alignment_mean": self._mean(self.alignments),
            "clip_rate": self._mean(self.clip_flags),
            "clip_ratio_mean": self._mean(self.clip_ratios),
            "window_count": len(self.losses),
        }


def summarize_step(loss: float, grad: float, weight: float, update: float, clipped: bool, clip_ratio: float) -> dict:
    grad_norm = abs(grad)
    update_norm = abs(update)
    alignment = 0.0
    if grad_norm > 0.0 and update_norm > 0.0:
        alignment = max(-1.0, min(1.0, (-grad * update) / (grad_norm * update_norm)))
    return {
        "loss": loss,
        "grad_abs": grad_norm,
        "weight": weight,
        "update_abs": update_norm,
        "update_alignment": alignment,
        "clipped": 1 if clipped else 0,
        "clip_ratio": clip_ratio,
    }
