# Architecture

## High-level modules
- `config/`: YAML loaders + typed config dataclasses.
- `interfaces/`: Protocol-style contracts for task/model/loss/optimizer.
- `tasks/` and `models/`: baseline synthetic task and toy model.
- `optimizers/`: baseline optimizers for comparison.
- `harness/`: training loop and loss definitions.
- `logging/`: metric sinks (jsonl + stdout).
- `benchmark/`: batch evaluation across optimizer/lr combinations.
- `controller/`: placeholder hook for v1 adaptive logic.
- `probe/`: placeholder hook for v2 state probes.
- `diagnostics/`: reusable step diagnostics.

## Baseline dataflow
1. CLI loads YAML config.
2. Trainer instantiates task, model, optimizer, hooks.
3. Per step: sample data -> compute loss/grad -> controller hook -> optimizer step -> probe hook -> log metrics.
4. Benchmark repeats trainer runs across a sweep and writes a leaderboard.

## Key tradeoffs
- **Pure-Python scalar baseline** over framework-heavy setup for speed and reviewability.
- **Explicit wiring** in trainer over inversion-of-control to keep extension points clear.
- **Simple jsonl logging** as lowest-friction default while preserving compatibility with future richer sinks.
