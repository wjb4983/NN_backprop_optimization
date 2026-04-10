from __future__ import annotations

from pathlib import Path

from nn_opt_rd.config.schema import TrainConfig
from nn_opt_rd.harness.trainer import run_training


def test_smoke_training(tmp_path: Path) -> None:
    cfg = TrainConfig(steps=8, batch_size=16, output_dir=str(tmp_path / "train"))
    summary = run_training(cfg)
    assert summary["steps"] == 8
    assert "final_loss" in summary
    assert (tmp_path / "train" / "metrics.jsonl").exists()
