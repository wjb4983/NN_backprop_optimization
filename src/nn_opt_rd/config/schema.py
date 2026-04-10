from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ControllerThresholds:
    healthy_loss_slope: float = -1e-4
    noisy_loss_cv: float = 0.30
    plateau_slope_abs: float = 8e-5
    instability_grad_cv: float = 0.70
    instability_clip_rate: float = 0.40


@dataclass
class ControllerBounds:
    min_lr_multiplier: float = 0.30
    max_lr_multiplier: float = 2.00
    min_clip_threshold: float = 0.05
    max_clip_threshold: float = 10.0
    min_momentum: float = 0.70
    max_momentum: float = 0.99
    min_trust: float = 0.30
    max_trust: float = 1.00


@dataclass
class ControllerAdjustments:
    lr_step: float = 0.08
    clip_step: float = 0.10
    momentum_step: float = 0.03
    trust_step: float = 0.08


@dataclass
class ControllerSafety:
    cooldown_steps: int = 6
    max_interventions_per_window: int = 2
    intervention_window: int = 24
    rollback_horizon: int = 6
    rollback_loss_tol: float = 0.05


@dataclass
class ControllerConfig:
    enabled: bool = False
    cadence: int = 4
    rolling_window: int = 12
    enable_momentum_tweaks: bool = False
    enable_trust_tweaks: bool = False
    clip_threshold: float = 1.0
    thresholds: ControllerThresholds = field(default_factory=ControllerThresholds)
    bounds: ControllerBounds = field(default_factory=ControllerBounds)
    adjustments: ControllerAdjustments = field(default_factory=ControllerAdjustments)
    safety: ControllerSafety = field(default_factory=ControllerSafety)


@dataclass
class TrainConfig:
    seed: int = 0
    steps: int = 100
    batch_size: int = 32
    optimizer: str = "adamw"
    learning_rate: float = 0.03
    output_dir: str = "runs/default"
    controller: ControllerConfig = field(default_factory=ControllerConfig)


@dataclass
class BenchmarkConfig:
    seed: int = 0
    steps: int = 80
    batch_size: int = 32
    optimizers: list[str] = None
    learning_rates: list[float] = None
    output_dir: str = "runs/benchmark"
    controller: ControllerConfig = field(default_factory=ControllerConfig)

    def __post_init__(self) -> None:
        if self.optimizers is None:
            self.optimizers = ["sgd", "momentum", "adam", "adamw"]
        if self.learning_rates is None:
            self.learning_rates = [0.01, 0.03]
