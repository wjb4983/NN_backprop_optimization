from __future__ import annotations

from pathlib import Path

from nn_opt_rd.benchmark.runner import run_benchmark
from nn_opt_rd.config.schema import BenchmarkConfig


def test_smoke_benchmark(tmp_path: Path) -> None:
    cfg = BenchmarkConfig(
        steps=6,
        batch_size=8,
        optimizers=["sgd", "adam"],
        learning_rates=[0.01],
        output_dir=str(tmp_path / "bench"),
    )
    rows = run_benchmark(cfg)
    assert len(rows) == 2
    assert (tmp_path / "bench" / "leaderboard.txt").exists()
