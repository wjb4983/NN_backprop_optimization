from __future__ import annotations

from pathlib import Path

from nn_opt_rd.config.schema import (
    BenchmarkConfig,
    ControllerAdjustments,
    ControllerBounds,
    ControllerConfig,
    ControllerSafety,
    ControllerThresholds,
    TrainConfig,
)


def _cast_scalar(value: str):
    v = value.strip()
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:
        return float(v) if "." in v else int(v)
    except ValueError:
        return v


def _fallback_parse_yaml(text: str) -> dict:
    lines = text.splitlines()
    root: dict = {}
    stack: list[tuple[int, dict | list]] = [(-1, root)]

    for idx, raw in enumerate(lines):
        line = raw.split("#", 1)[0].rstrip("\n")
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        token = line.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if token.startswith("- "):
            item = _cast_scalar(token[2:])
            if not isinstance(parent, list):
                raise ValueError(f"Invalid list item placement: {raw}")
            parent.append(item)
            continue

        if ":" not in token:
            raise ValueError(f"Unsupported config line: {raw}")
        key, value = [x.strip() for x in token.split(":", 1)]

        if value == "":
            next_non_empty = ""
            for candidate in lines[idx + 1 :]:
                nxt = candidate.split("#", 1)[0].strip()
                if nxt:
                    next_non_empty = nxt
                    break
            container: dict | list = [] if next_non_empty.startswith("- ") else {}
            if not isinstance(parent, dict):
                raise ValueError(f"Mapping key inside non-dict parent: {raw}")
            parent[key] = container
            stack.append((indent, container))
        else:
            if not isinstance(parent, dict):
                raise ValueError(f"Scalar key inside non-dict parent: {raw}")
            parent[key] = _cast_scalar(value)
    return root


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


def _parse_controller_config(data: dict | None) -> ControllerConfig:
    if not data:
        return ControllerConfig()
    thresholds = ControllerThresholds(**data.get("thresholds", {}))
    bounds = ControllerBounds(**data.get("bounds", {}))
    adjustments = ControllerAdjustments(**data.get("adjustments", {}))
    safety = ControllerSafety(**data.get("safety", {}))

    top_level = {
        k: v
        for k, v in data.items()
        if k not in {"thresholds", "bounds", "adjustments", "safety"}
    }
    return ControllerConfig(
        **top_level,
        thresholds=thresholds,
        bounds=bounds,
        adjustments=adjustments,
        safety=safety,
    )


def load_train_config(path: str) -> TrainConfig:
    data = _load_yaml(path)
    controller = _parse_controller_config(data.pop("controller", None))
    return TrainConfig(**data, controller=controller)


def load_benchmark_config(path: str) -> BenchmarkConfig:
    data = _load_yaml(path)
    controller = _parse_controller_config(data.pop("controller", None))
    return BenchmarkConfig(**data, controller=controller)
