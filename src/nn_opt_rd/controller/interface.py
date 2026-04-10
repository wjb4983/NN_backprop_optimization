from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from nn_opt_rd.config.schema import ControllerConfig


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass
class ControllerState:
    step: int = 0
    mode: str = "hold"
    lr_multiplier: float = 1.0
    clip_threshold: float = 1.0
    momentum: float = 0.9
    trust: float = 1.0
    last_intervention_step: int = -10_000
    intervention_steps: deque[int] = field(default_factory=deque)
    intervention_count: int = 0
    rollback_reference_loss: float = 0.0
    rollback_guard_step: int = -1
    diagnostics: dict = field(default_factory=dict)


class Controller:
    def __init__(self, config: ControllerConfig) -> None:
        self.config = config

    def bootstrap(self, state: ControllerState) -> None:
        state.clip_threshold = self.config.clip_threshold

    def _limited_by_frequency(self, state: ControllerState) -> bool:
        while state.intervention_steps and state.intervention_steps[0] < state.step - self.config.safety.intervention_window:
            state.intervention_steps.popleft()
        return len(state.intervention_steps) >= self.config.safety.max_interventions_per_window

    def _infer_health(self, features: dict) -> tuple[str, list[str]]:
        hints: list[str] = []
        if features["clip_rate"] > self.config.thresholds.instability_clip_rate:
            hints.append("clip")
        if features["update_alignment_mean"] < 0.4:
            hints.append("trust")
        if features["loss_slope"] >= 0.0:
            hints.append("lr")
        if features["grad_norm_cv"] > self.config.thresholds.instability_grad_cv:
            hints.append("momentum")

        if (
            features["grad_norm_cv"] > self.config.thresholds.instability_grad_cv
            or features["clip_rate"] > self.config.thresholds.instability_clip_rate
        ):
            return "instability", hints
        if abs(features["loss_slope"]) <= self.config.thresholds.plateau_slope_abs:
            return "stall", hints
        if features["loss_cv"] > self.config.thresholds.noisy_loss_cv:
            return "noisy", hints
        if features["loss_slope"] < self.config.thresholds.healthy_loss_slope:
            return "healthy", hints
        return "noisy", hints

    def _apply_action(self, state: ControllerState, health: str, hints: list[str]) -> bool:
        bounds = self.config.bounds
        steps = self.config.adjustments
        did_intervene = False

        if health == "instability":
            state.mode = "backoff"
            state.lr_multiplier = _clamp(state.lr_multiplier * (1.0 - steps.lr_step), bounds.min_lr_multiplier, bounds.max_lr_multiplier)
            state.clip_threshold = _clamp(state.clip_threshold * (1.0 - steps.clip_step), bounds.min_clip_threshold, bounds.max_clip_threshold)
            if self.config.enable_trust_tweaks:
                state.trust = _clamp(state.trust * (1.0 - steps.trust_step), bounds.min_trust, bounds.max_trust)
            if self.config.enable_momentum_tweaks and hasattr(state, "momentum"):
                state.momentum = _clamp(state.momentum - steps.momentum_step, bounds.min_momentum, bounds.max_momentum)
            did_intervene = True
        elif health == "stall":
            state.mode = "probe"
            state.lr_multiplier = _clamp(state.lr_multiplier * (1.0 + steps.lr_step), bounds.min_lr_multiplier, bounds.max_lr_multiplier)
            state.clip_threshold = _clamp(state.clip_threshold * (1.0 + (0.5 * steps.clip_step)), bounds.min_clip_threshold, bounds.max_clip_threshold)
            if self.config.enable_trust_tweaks:
                state.trust = _clamp(state.trust * (1.0 + 0.5 * steps.trust_step), bounds.min_trust, bounds.max_trust)
            did_intervene = True
        else:
            state.mode = "hold"

        if health == "noisy" and "clip" in hints:
            state.clip_threshold = _clamp(state.clip_threshold * (1.0 - 0.5 * steps.clip_step), bounds.min_clip_threshold, bounds.max_clip_threshold)
            did_intervene = True

        if health == "healthy":
            state.mode = "hold"
        return did_intervene

    def maybe_update(self, state: ControllerState, step_metrics: dict, rolling_features: dict) -> dict:
        state.step += 1
        state.diagnostics = rolling_features

        if not self.config.enabled:
            return {"controller_enabled": 0, "controller_mode": "disabled"}
        if rolling_features["window_count"] < 3:
            return {"controller_enabled": 1, "controller_mode": "warmup"}
        if state.step % self.config.cadence != 0:
            return {"controller_enabled": 1, "controller_mode": state.mode}

        if state.step - state.last_intervention_step < self.config.safety.cooldown_steps:
            return {"controller_enabled": 1, "controller_mode": "cooldown"}

        if self._limited_by_frequency(state):
            return {"controller_enabled": 1, "controller_mode": "frequency_limited"}

        health, hints = self._infer_health(rolling_features)
        did_intervene = self._apply_action(state, health, hints)

        if did_intervene:
            state.intervention_steps.append(state.step)
            state.intervention_count += 1
            state.last_intervention_step = state.step
            state.rollback_reference_loss = step_metrics["loss"]
            state.rollback_guard_step = state.step

        if (
            state.rollback_guard_step > 0
            and state.step - state.rollback_guard_step >= self.config.safety.rollback_horizon
            and step_metrics["loss"] > state.rollback_reference_loss * (1.0 + self.config.safety.rollback_loss_tol)
        ):
            bounds = self.config.bounds
            steps = self.config.adjustments
            state.lr_multiplier = _clamp(state.lr_multiplier * (1.0 - steps.lr_step), bounds.min_lr_multiplier, bounds.max_lr_multiplier)
            state.clip_threshold = _clamp(state.clip_threshold * (1.0 - steps.clip_step), bounds.min_clip_threshold, bounds.max_clip_threshold)
            state.mode = "rollback_backoff"
            state.rollback_guard_step = -1

        return {
            "controller_enabled": 1,
            "controller_mode": state.mode,
            "health": health,
            "hints": ",".join(hints),
            "lr_multiplier": state.lr_multiplier,
            "clip_threshold": state.clip_threshold,
            "trust": state.trust,
            "momentum": state.momentum,
            "interventions": state.intervention_count,
        }
