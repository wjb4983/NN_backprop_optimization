# NN Backprop Optimization R&D

A clean baseline repository for optimizer research and development.

## What is included
- Baseline training harness and benchmarking pipeline.
- YAML-based config system with defaults.
- Baseline optimizers (SGD, momentum SGD, Adam, AdamW).
- Optional bounded v1 controller with hard safety rails.
- Rolling diagnostics + health inference hooks.
- Logging and metrics sinks.
- Smoke/unit/integration tests.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]

nn-opt-train --config configs/train_default.yaml
nn-opt-benchmark --config configs/benchmark_default.yaml
pytest -q
```

## Controller v1 (optional)
Controller is fully optional and can be disabled with:

```yaml
controller:
  enabled: false
```

When enabled, controller consumes rolling diagnostics from:
- loss history and trend/volatility
- gradient norm volatility
- update norm magnitude and alignment
- clipping rate/severity

It infers health modes:
- healthy learning
- noisy descent
- stall/plateau
- instability/exploding risk

And chooses bounded actions:
- hold steady
- back off (reduce LR/clip/trust)
- probe (small LR upshift)

Safety constraints are always enforced:
- hard min/max clamps
- cooldown between interventions
- per-window intervention limit
- rollback/backoff if short-horizon loss worsens after intervention

## Heuristic tradeoffs
- We use **cheap scalar rolling statistics** instead of expensive second-order probes (faster iteration, less fidelity).
- Thresholds are **intentionally conservative defaults**; they may under-react on some tasks but reduce overshoot risk.
- Probe mode uses small LR increases to avoid destabilizing the baseline.
- Rollback is loss-based and short-horizon, which is robust/simple but can be noisy on highly stochastic tasks.

See `docs/architecture.md` and `docs/roadmap.md` for details.


## Experiment suites and reporting
Use staged manifests for reproducible multi-seed comparisons:

```bash
nn-opt-experiment --manifest configs/experiments/starter_narrow.yaml --target-loss 0.02
```

Artifacts per run:
- `metrics_step.jsonl`: per-step metrics (also mirrored to `metrics.jsonl` for compatibility)
- `overhead.jsonl`: per-step wall-clock and controller overhead
- `events.jsonl`: intervention events
- `run_meta.json`: seed/config/git-hash metadata

Report outputs (`report/summary.json`, `report/summary.md`) include:
- multi-seed aggregates
- time-to-target (if a target is set)
- AULC-style area under learning curve
- stability/failure rates

Starter manifests:
- `configs/experiments/starter_narrow.yaml`
- `configs/experiments/medium.yaml`
- `configs/experiments/later_scaffold.yaml`
