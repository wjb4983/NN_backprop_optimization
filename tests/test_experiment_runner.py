from __future__ import annotations

from pathlib import Path

from nn_opt_rd.benchmark.manifest import load_experiment_manifest
from nn_opt_rd.benchmark.runner import run_manifest
from nn_opt_rd.config.schema import ControllerConfig


def test_run_manifest_creates_records_and_logging_artifacts(tmp_path: Path) -> None:
    manifest = load_experiment_manifest("configs/experiments/starter_narrow.yaml")
    manifest.output_dir = str(tmp_path / "exp")
    records = run_manifest(manifest, base_controller=ControllerConfig())

    assert records
    first_run = Path(records[0]["run_dir"])
    assert (first_run / "metrics_step.jsonl").exists()
    assert (first_run / "events.jsonl").exists()
    assert (first_run / "overhead.jsonl").exists()
    assert (first_run / "run_meta.json").exists()
