from __future__ import annotations

from nn_opt_rd.benchmark.reporting import compute_aulc, summarize_runs, time_to_target


def test_time_to_target_and_aulc_basic() -> None:
    rows = [
        {"step": 0, "loss": 1.0, "step_wall_ms": 10.0},
        {"step": 1, "loss": 0.5, "step_wall_ms": 10.0},
        {"step": 2, "loss": 0.2, "step_wall_ms": 10.0},
    ]
    assert time_to_target(rows, 0.5) is not None
    assert compute_aulc(rows) > 0.0


def test_summarize_runs_includes_failure_rate() -> None:
    records = [
        {
            "stage": "s",
            "run": "r",
            "final_loss": 0.2,
            "wall_clock_s": 0.1,
            "failed": 0,
            "run_dir": ".",
        },
        {
            "stage": "s",
            "run": "r",
            "final_loss": 0.3,
            "wall_clock_s": 0.2,
            "failed": 1,
            "run_dir": ".",
        },
    ]
    summary = summarize_runs(records)
    assert summary[0]["failure_rate"] == 0.5
