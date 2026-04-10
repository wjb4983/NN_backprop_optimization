from __future__ import annotations

from pathlib import Path

from nn_opt_rd.config.schema import BenchmarkConfig, TrainConfig
from nn_opt_rd.harness.trainer import run_training


def run_benchmark(config: BenchmarkConfig) -> list[dict]:
    records: list[dict] = []
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    for optimizer_name in config.optimizers:
        for lr in config.learning_rates:
            run_name = f"{optimizer_name}_lr{lr}"
            run_dir = root / run_name
            train_cfg = TrainConfig(
                seed=config.seed,
                steps=config.steps,
                batch_size=config.batch_size,
                optimizer=optimizer_name,
                learning_rate=lr,
                output_dir=str(run_dir),
                controller=config.controller,
            )
            summary = run_training(train_cfg)
            record = {
                "optimizer": optimizer_name,
                "lr": lr,
                "final_loss": summary["final_loss"],
                "controller_enabled": summary["controller_enabled"],
                "interventions": summary["interventions"],
                "run_dir": str(run_dir),
            }
            records.append(record)

    records = sorted(records, key=lambda x: x["final_loss"])
    leaderboard = root / "leaderboard.txt"
    with leaderboard.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(
                f"{row['optimizer']}\tlr={row['lr']}\tfinal_loss={row['final_loss']:.6f}"
                f"\tcontroller={row['controller_enabled']}\tinterventions={row['interventions']}\n"
            )
    return records
