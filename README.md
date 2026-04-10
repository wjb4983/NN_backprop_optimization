# NN Backprop Optimization R&D

A clean baseline repository for optimizer research and development.

## What is included
- Baseline training harness and benchmarking pipeline.
- YAML-based config system with defaults.
- Baseline optimizers (SGD, momentum SGD, Adam).
- Logging and metrics sinks.
- Diagnostics helpers.
- Placeholder extension interfaces for future controller (v1) and probes (v2).
- Smoke tests and architecture/roadmap docs.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]

nn-opt-train --config configs/train_default.yaml
nn-opt-benchmark --config configs/benchmark_default.yaml
pytest -q
```

## Design goals
- Simple, explicit modules over clever abstractions.
- Fast local iteration.
- Easy evolution path for controller/probe experiments.

See `docs/architecture.md` and `docs/roadmap.md` for details.
