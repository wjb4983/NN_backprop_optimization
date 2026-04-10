# Architecture

## High-level modules
- `config/`: YAML loaders + typed config dataclasses.
- `interfaces/`: Protocol-style contracts for task/model/loss/optimizer.
- `tasks/` and `models/`: baseline synthetic task and toy model.
- `optimizers/`: baseline optimizers for comparison (includes AdamW).
- `harness/`: training loop and loss definitions.
- `logging/`: metric sinks (jsonl + stdout).
- `benchmark/`: batch evaluation across optimizer/lr combinations.
- `controller/`: bounded adaptive logic for v1.
- `probe/`: placeholder hook for v2 state probes.
- `diagnostics/`: rolling step diagnostics and feature extraction.

## Baseline dataflow
1. CLI loads YAML config.
2. Trainer instantiates task, model, optimizer, hooks.
3. Per step: sample data -> compute loss/grad -> optional clipping -> optimizer step -> rolling diagnostics -> controller decision -> probe capture -> log metrics.
4. Benchmark repeats trainer runs across a sweep and writes a leaderboard.

## Controller dataflow (v1)
1. `RollingDiagnostics` updates O(1)-memory windows with loss, grad norm, update norm/alignment, clip stats.
2. Controller maps features to health signal:
   - healthy / noisy / stall / instability
   - plus mismatch hints (lr/clip/momentum/trust)
3. Controller chooses mode:
   - hold / probe / backoff
4. Interventions are bounded by:
   - hard clamps, cooldowns, frequency caps, rollback checks.

## Key tradeoffs
- **Pure-Python scalar baseline** over framework-heavy setup for speed and reviewability.
- **Explicit wiring** in trainer over inversion-of-control to keep extension points clear.
- **Simple jsonl logging** as lowest-friction default while preserving compatibility with future richer sinks.
- **Heuristic bounded controller** over learned policy for debuggability and deterministic guardrails.
- **Cadence-based interventions** reduce overhead and policy chatter at the cost of delayed reactions.
