# Roadmap

## v1: Controller integration (implemented baseline)
- ✅ Strategy-style controller with:
  - lr multiplier control
  - gradient clipping policy control
  - optional momentum/trust bounded adaptation
- ✅ Config blocks for controller policy variants and safety rails.
- ✅ A/B benchmark mode via controller enabled/disabled config.

## v1.1: Calibration + richer diagnostics
- Add richer synthetic stress tasks for calibrating thresholds.
- Add decision explainability reports from logged health/hints/mode.
- Validate threshold defaults against wider optimizer/lr grids.

## v2: Probe module
- Expand `probe.interface.Probe` into pluggable probes:
  - optimizer internal state probe
  - activation/representation probe (for richer models)
- Add per-step artifact persistence and structured probe output schemas.

## v3: Full experimental platform
- Multi-task registry (vision/language/tabular toy tasks).
- Multi-parameter models and optional torch backend.
- Reproducible experiment tracking and report generation.
- Distributed benchmark orchestration.
