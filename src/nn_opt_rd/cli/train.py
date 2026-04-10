from __future__ import annotations

import argparse

from nn_opt_rd.config.loader import load_train_config
from nn_opt_rd.harness.trainer import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline training")
    parser.add_argument("--config", required=True, help="Path to train YAML config")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_train_config(args.config)
    summary = run_training(config)
    print("TRAIN_SUMMARY", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
