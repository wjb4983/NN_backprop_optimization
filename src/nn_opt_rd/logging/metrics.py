from __future__ import annotations

import json
from pathlib import Path


class MetricLogger:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.output_dir / "metrics_step.jsonl"
        self.legacy_metrics_file = self.output_dir / "metrics.jsonl"
        self.events_file = self.output_dir / "events.jsonl"
        self.overhead_file = self.output_dir / "overhead.jsonl"
        self.meta_file = self.output_dir / "run_meta.json"
        for path in (self.metrics_file, self.legacy_metrics_file, self.events_file, self.overhead_file):
            path.touch(exist_ok=True)

    def _append_jsonl(self, path: Path, payload: dict) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")

    def log(self, payload: dict) -> None:
        self.log_step(payload)

    def log_step(self, payload: dict) -> None:
        self._append_jsonl(self.metrics_file, payload)
        self._append_jsonl(self.legacy_metrics_file, payload)

    def log_event(self, payload: dict) -> None:
        self._append_jsonl(self.events_file, payload)

    def log_overhead(self, payload: dict) -> None:
        self._append_jsonl(self.overhead_file, payload)

    def write_metadata(self, payload: dict) -> None:
        with self.meta_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
