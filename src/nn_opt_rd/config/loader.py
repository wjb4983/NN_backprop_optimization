from __future__ import annotations

from pathlib import Path

from nn_opt_rd.config.schema import BenchmarkConfig, TrainConfig


def _fallback_parse_yaml(text: str) -> dict:
    data: dict = {}
    current_list_key: str | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("- ") and current_list_key:
            item = line.strip()[2:].strip()
            try:
                cast_item = float(item) if "." in item else int(item)
            except ValueError:
                cast_item = item
            data[current_list_key].append(cast_item)
            continue

        current_list_key = None
        if ":" not in line:
            raise ValueError(f"Unsupported config line: {raw}")
        key, value = [x.strip() for x in line.split(":", 1)]
        if value == "":
            data[key] = []
            current_list_key = key
            continue
        if value.lower() in {"true", "false"}:
            data[key] = value.lower() == "true"
        else:
            try:
                data[key] = float(value) if "." in value else int(value)
            except ValueError:
                data[key] = value
    return data


def _load_yaml(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        data = _fallback_parse_yaml(text)

    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping")
    return data


def load_train_config(path: str) -> TrainConfig:
    data = _load_yaml(path)
    return TrainConfig(**data)


def load_benchmark_config(path: str) -> BenchmarkConfig:
    data = _load_yaml(path)
    return BenchmarkConfig(**data)
