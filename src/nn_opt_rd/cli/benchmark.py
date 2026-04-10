from __future__ import annotations

import argparse

from nn_opt_rd.benchmark.runner import run_benchmark
from nn_opt_rd.config.loader import load_benchmark_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline benchmark")
    parser.add_argument("--config", required=True, help="Path to benchmark YAML config")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_benchmark_config(args.config)
    rows = run_benchmark(config)
    print("BENCHMARK_TOP", rows[0] if rows else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
