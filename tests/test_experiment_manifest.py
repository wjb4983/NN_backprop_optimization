from __future__ import annotations

from pathlib import Path

from nn_opt_rd.benchmark.manifest import load_experiment_manifest


def test_load_starter_manifest_has_required_stages_and_baselines() -> None:
    manifest = load_experiment_manifest("configs/experiments/starter_narrow.yaml")
    assert manifest.suite_name == "starter_narrow"
    assert manifest.stages
    run_names = [run.name for stage in manifest.stages for run in stage.runs]
    assert "adamw_baseline" in run_names
    assert "adamw_controller" in run_names


def test_manifest_file_exists_for_medium_and_later() -> None:
    assert Path("configs/experiments/medium.yaml").exists()
    assert Path("configs/experiments/later_scaffold.yaml").exists()
