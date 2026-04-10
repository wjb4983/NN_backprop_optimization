# Roadmap

## v1: Controller integration
- Replace `controller.interface.Controller` with a strategy interface supporting:
  - lr schedule control
  - gradient clipping/normalization policy
  - optimizer hyperparameter adaptation
- Add config blocks for controller policy variants.
- Add A/B benchmark mode vs fixed baselines.

## v2: Probe module
- Expand `probe.interface.Probe` into pluggable probes:
  - gradient statistics probe
  - optimizer internal state probe
  - activation/representation probe (for richer models)
- Add per-step artifact persistence and structured probe output schemas.

## v3: Full experimental platform
- Multi-task registry (vision/language/tabular toy tasks).
- Multi-parameter models and optional torch backend.
- Reproducible experiment tracking and report generation.
- Distributed benchmark orchestration.
