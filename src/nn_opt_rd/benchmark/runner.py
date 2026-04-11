from __future__ import annotations

import json
from pathlib import Path

from nn_opt_rd.benchmark.manifest import ExperimentManifest, apply_controller_overrides
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
                "variant": f"{optimizer_name}{'+controller' if config.controller.enabled else ''}",
                "lr": lr,
                "final_loss": summary["final_loss"],
                "controller_enabled": summary["controller_enabled"],
                "interventions": summary["interventions"],
                "wall_clock_s": summary["wall_clock_s"],
                "failed": summary["failed"],
                "run_dir": str(run_dir),
            }
            records.append(record)

    records = sorted(records, key=lambda x: x["final_loss"])
    leaderboard = root / "leaderboard.txt"
    with leaderboard.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(
                f"{row['optimizer']}\tlr={row['lr']}\tfinal_loss={row['final_loss']:.6f}"
                f"\tcontroller={row['controller_enabled']}\tinterventions={row['interventions']}"
                f"\twall_clock_s={row['wall_clock_s']:.4f}\n"
            )
    return records


def run_manifest(manifest: ExperimentManifest, base_controller) -> list[dict]:
    root = Path(manifest.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for stage in manifest.stages:
        for run_def in stage.runs:
            seeds = run_def.seeds or manifest.default_seeds
            for seed in seeds:
                run_dir = root / stage.name / run_def.name / f"seed_{seed}"
                controller_cfg = apply_controller_overrides(
                    base=base_controller,
                    enabled=run_def.controller_enabled,
                    overrides=run_def.controller_overrides,
                )
                cfg = TrainConfig(
                    seed=seed,
                    steps=run_def.steps or manifest.default_steps,
                    batch_size=run_def.batch_size or manifest.default_batch_size,
                    optimizer=run_def.optimizer,
                    learning_rate=run_def.learning_rate,
                    output_dir=str(run_dir),
                    controller=controller_cfg,
                )
                summary = run_training(cfg)
                records.append(
                    {
                        "suite_name": manifest.suite_name,
                        "stage": stage.name,
                        "run": run_def.name,
                        "seed": seed,
                        "optimizer": run_def.optimizer,
                        "variant": f"{run_def.optimizer}{'+controller' if run_def.controller_enabled else ''}",
                        "learning_rate": run_def.learning_rate,
                        "controller_enabled": int(run_def.controller_enabled),
                        "final_loss": summary["final_loss"],
                        "interventions": summary["interventions"],
                        "wall_clock_s": summary["wall_clock_s"],
                        "failed": summary["failed"],
                        "run_dir": str(run_dir),
                    }
                )

    records_path = root / "records.json"
    with records_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, sort_keys=True)
    return records
