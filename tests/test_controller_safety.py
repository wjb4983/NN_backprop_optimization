from __future__ import annotations

from nn_opt_rd.config.schema import ControllerConfig
from nn_opt_rd.controller.interface import Controller, ControllerState


def test_controller_respects_clamps() -> None:
    cfg = ControllerConfig(
        enabled=True,
        cadence=1,
        clip_threshold=1.0,
    )
    controller = Controller(cfg)
    state = ControllerState()
    controller.bootstrap(state)

    features = {
        "window_count": 12,
        "clip_rate": 0.9,
        "update_alignment_mean": 0.2,
        "loss_slope": 0.001,
        "grad_norm_cv": 0.95,
        "loss_cv": 0.6,
    }
    step = {"loss": 1.0}

    for _ in range(60):
        controller.maybe_update(state, step, features)

    assert cfg.bounds.min_lr_multiplier <= state.lr_multiplier <= cfg.bounds.max_lr_multiplier
    assert cfg.bounds.min_clip_threshold <= state.clip_threshold <= cfg.bounds.max_clip_threshold


def test_controller_enforces_frequency_limit() -> None:
    cfg = ControllerConfig(enabled=True, cadence=1)
    cfg.safety.max_interventions_per_window = 1
    cfg.safety.intervention_window = 100
    cfg.safety.cooldown_steps = 0
    controller = Controller(cfg)
    state = ControllerState()
    controller.bootstrap(state)

    features = {
        "window_count": 12,
        "clip_rate": 0.7,
        "update_alignment_mean": 0.3,
        "loss_slope": 0.001,
        "grad_norm_cv": 0.8,
        "loss_cv": 0.4,
    }

    out1 = controller.maybe_update(state, {"loss": 1.0}, features)
    out2 = controller.maybe_update(state, {"loss": 1.0}, features)
    assert out1["controller_mode"] in {"backoff", "rollback_backoff", "hold"}
    assert out2["controller_mode"] == "frequency_limited"


def test_controller_rolls_back_on_worsening() -> None:
    cfg = ControllerConfig(enabled=True, cadence=1)
    cfg.safety.cooldown_steps = 0
    cfg.safety.rollback_horizon = 2
    cfg.safety.rollback_loss_tol = 0.01
    controller = Controller(cfg)
    state = ControllerState()
    controller.bootstrap(state)

    unstable = {
        "window_count": 12,
        "clip_rate": 0.8,
        "update_alignment_mean": 0.2,
        "loss_slope": 0.001,
        "grad_norm_cv": 0.9,
        "loss_cv": 0.5,
    }
    healthy = {
        "window_count": 12,
        "clip_rate": 0.0,
        "update_alignment_mean": 0.9,
        "loss_slope": -0.01,
        "grad_norm_cv": 0.1,
        "loss_cv": 0.1,
    }

    controller.maybe_update(state, {"loss": 1.0}, unstable)
    controller.maybe_update(state, {"loss": 1.2}, healthy)
    out = controller.maybe_update(state, {"loss": 1.4}, healthy)

    assert out["controller_mode"] in {"rollback_backoff", "hold"}
