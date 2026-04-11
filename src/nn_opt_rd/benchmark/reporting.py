from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def compute_aulc(step_rows: list[dict]) -> float:
    points = [r for r in step_rows if "step" in r and "loss" in r]
    if len(points) < 2:
        return 0.0
    points = sorted(points, key=lambda r: r["step"])
    area = 0.0
    for a, b in zip(points[:-1], points[1:]):
        dx = max(1e-12, (b.get("step_wall_ms", 0.0) + a.get("step_wall_ms", 0.0)) / 2000.0)
        area += dx * (a["loss"] + b["loss"]) / 2.0
    return area


def time_to_target(step_rows: list[dict], target_loss: float) -> float | None:
    elapsed = 0.0
    for row in sorted([r for r in step_rows if "step" in r], key=lambda r: r["step"]):
        elapsed += row.get("step_wall_ms", 0.0) / 1000.0
        if row.get("loss", float("inf")) <= target_loss:
            return elapsed
    return None


def summarize_runs(records: list[dict], target_loss: float | None = None) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        grouped[(r["stage"], r["run"])].append(r)

    rows: list[dict] = []
    for (stage, run), vals in grouped.items():
        final_losses = [v["final_loss"] for v in vals]
        wall = [v.get("wall_clock_s", 0.0) for v in vals]
        failures = sum(v.get("failed", 0) for v in vals)

        ttt_values = []
        aulc_values = []
        for v in vals:
            step_rows = _read_jsonl(Path(v["run_dir"]) / "metrics_step.jsonl")
            if target_loss is not None:
                ttt = time_to_target(step_rows, target_loss)
                if ttt is not None:
                    ttt_values.append(ttt)
            aulc_values.append(compute_aulc(step_rows))

        rows.append(
            {
                "stage": stage,
                "run": run,
                "seeds": len(vals),
                "final_loss_mean": sum(final_losses) / len(final_losses),
                "final_loss_min": min(final_losses),
                "final_loss_max": max(final_losses),
                "wall_clock_s_mean": sum(wall) / len(wall),
                "aulc_mean": (sum(aulc_values) / len(aulc_values)) if aulc_values else 0.0,
                "time_to_target_s_mean": (sum(ttt_values) / len(ttt_values)) if ttt_values else None,
                "target_hit_rate": (len(ttt_values) / len(vals)) if target_loss is not None else None,
                "failure_rate": failures / len(vals),
            }
        )
    return sorted(rows, key=lambda r: (r["stage"], r["final_loss_mean"]))


def write_report(records: list[dict], out_dir: str, target_loss: float | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize_runs(records, target_loss=target_loss)
    json_path = out / "summary.json"
    md_path = out / "summary.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Experiment Summary\n\n")
        f.write("| stage | run | seeds | final_loss_mean | wall_clock_s_mean | aulc_mean | time_to_target_s_mean | target_hit_rate | failure_rate |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary:
            f.write(
                f"| {row['stage']} | {row['run']} | {row['seeds']} | {row['final_loss_mean']:.6f} | {row['wall_clock_s_mean']:.4f} | {row['aulc_mean']:.6f} | {row['time_to_target_s_mean']} | {row['target_hit_rate']} | {row['failure_rate']:.3f} |\n"
            )

    return md_path
