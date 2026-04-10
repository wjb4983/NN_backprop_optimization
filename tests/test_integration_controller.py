from __future__ import annotations

from pathlib import Path

from nn_opt_rd.config.schema import ControllerConfig, TrainConfig
from nn_opt_rd.harness.trainer import run_training


def test_controller_optional_and_outputs_metrics(tmp_path: Path) -> None:
    off_cfg = TrainConfig(
        steps=10,
        batch_size=16,
        optimizer="adamw",
        output_dir=str(tmp_path / "off"),
        controller=ControllerConfig(enabled=False),
    )
    on_cfg = TrainConfig(
        steps=10,
        batch_size=16,
        optimizer="adamw",
        output_dir=str(tmp_path / "on"),
        controller=ControllerConfig(enabled=True, cadence=1),
    )

    off = run_training(off_cfg)
    on = run_training(on_cfg)

    assert off["controller_enabled"] == 0
    assert on["controller_enabled"] == 1
    assert (tmp_path / "on" / "metrics.jsonl").exists()
