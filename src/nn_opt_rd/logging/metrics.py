from __future__ import annotations

import json
from pathlib import Path


class MetricLogger:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.output_dir / "metrics.jsonl"

    def log(self, payload: dict) -> None:
        with self.metrics_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
        print(payload)
