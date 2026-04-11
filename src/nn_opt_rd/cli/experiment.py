from __future__ import annotations

import argparse
from pathlib import Path

from nn_opt_rd.benchmark.manifest import load_experiment_manifest
from nn_opt_rd.benchmark.reporting import write_report
from nn_opt_rd.benchmark.runner import run_manifest
from nn_opt_rd.config.schema import ControllerConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run staged experiment manifest")
    parser.add_argument("--manifest", required=True, help="Path to experiment manifest YAML")
    parser.add_argument("--target-loss", type=float, default=None, help="Optional target loss for time-to-target")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = load_experiment_manifest(args.manifest)
    records = run_manifest(manifest, base_controller=ControllerConfig())
    report_path = write_report(records, out_dir=str(Path(manifest.output_dir) / "report"), target_loss=args.target_loss)
    print("EXPERIMENT_RECORDS", len(records))
    print("EXPERIMENT_REPORT", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
