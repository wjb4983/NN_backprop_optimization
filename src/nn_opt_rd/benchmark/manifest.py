from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from nn_opt_rd.config.loader import _load_yaml
from nn_opt_rd.config.schema import (
    ControllerAdjustments,
    ControllerBounds,
    ControllerConfig,
    ControllerSafety,
    ControllerThresholds,
)


@dataclass
class RunDefinition:
    name: str
    optimizer: str
    learning_rate: float
    controller_enabled: bool = False
    controller_overrides: dict = field(default_factory=dict)
    seeds: list[int] = field(default_factory=list)
    steps: int | None = None
    batch_size: int | None = None


@dataclass
class StageDefinition:
    name: str
    description: str = ""
    runs: list[RunDefinition] = field(default_factory=list)


@dataclass
class AblationVariant:
    name: str
    group: str
    controller_overrides: dict = field(default_factory=dict)


@dataclass
class ExperimentManifest:
    suite_name: str
    output_dir: str
    default_steps: int = 80
    default_batch_size: int = 32
    default_seeds: list[int] = field(default_factory=lambda: [0])
    stages: list[StageDefinition] = field(default_factory=list)
    ablations: list[AblationVariant] = field(default_factory=list)


def _parse_stage(data: dict) -> StageDefinition:
    runs: list[RunDefinition] = []
    for run in data.get("runs", []):
        runs.append(
            RunDefinition(
                name=run["name"],
                optimizer=run["optimizer"],
                learning_rate=float(run["learning_rate"]),
                controller_enabled=bool(run.get("controller_enabled", False)),
                controller_overrides=dict(run.get("controller_overrides", {})),
                seeds=list(run.get("seeds", [])),
                steps=run.get("steps"),
                batch_size=run.get("batch_size"),
            )
        )
    return StageDefinition(name=data["name"], description=data.get("description", ""), runs=runs)


def _parse_ablations(data: list[dict]) -> list[AblationVariant]:
    variants: list[AblationVariant] = []
    for group in data:
        group_name = group["name"]
        for variant in group.get("variants", []):
            variants.append(
                AblationVariant(
                    name=variant["name"],
                    group=group_name,
                    controller_overrides=dict(variant.get("controller_overrides", {})),
                )
            )
    return variants


def load_experiment_manifest(path: str) -> ExperimentManifest:
    data = _load_yaml(path)
    defaults = data.get("defaults", {})
    stages = [_parse_stage(stage) for stage in data.get("stages", [])]
    manifest = ExperimentManifest(
        suite_name=data.get("suite_name", Path(path).stem),
        output_dir=data.get("output_dir", f"runs/{Path(path).stem}"),
        default_steps=int(defaults.get("steps", 80)),
        default_batch_size=int(defaults.get("batch_size", 32)),
        default_seeds=list(data.get("seeds", [0])),
        stages=stages,
        ablations=_parse_ablations(data.get("ablations", [])),
    )
    return manifest


def apply_controller_overrides(base: ControllerConfig, enabled: bool, overrides: dict) -> ControllerConfig:
    cfg = ControllerConfig(
        enabled=enabled,
        cadence=base.cadence,
        rolling_window=base.rolling_window,
        enable_momentum_tweaks=base.enable_momentum_tweaks,
        enable_trust_tweaks=base.enable_trust_tweaks,
        clip_threshold=base.clip_threshold,
        thresholds=ControllerThresholds(**base.thresholds.__dict__),
        bounds=ControllerBounds(**base.bounds.__dict__),
        adjustments=ControllerAdjustments(**base.adjustments.__dict__),
        safety=ControllerSafety(**base.safety.__dict__),
    )
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        elif key.startswith("thresholds."):
            setattr(cfg.thresholds, key.split(".", 1)[1], value)
        elif key.startswith("bounds."):
            setattr(cfg.bounds, key.split(".", 1)[1], value)
        elif key.startswith("adjustments."):
            setattr(cfg.adjustments, key.split(".", 1)[1], value)
        elif key.startswith("safety."):
            setattr(cfg.safety, key.split(".", 1)[1], value)
    return cfg
