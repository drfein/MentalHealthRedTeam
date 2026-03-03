from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_DATA_PATHS_CONFIG = Path("configs/data_paths.yaml")


def load_data_paths(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path is not None else DEFAULT_DATA_PATHS_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def resolve_data_path(registry: dict[str, Any], key: str) -> Path:
    parts = key.split(".")
    node: Any = registry
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"Unknown data path key: {key}")
        node = node[part]
    if not isinstance(node, str):
        raise TypeError(f"Data path key '{key}' must resolve to a string")
    return Path(node)


def resolve_path_spec(spec: str | Path, registry: dict[str, Any]) -> Path:
    if isinstance(spec, Path):
        return spec
    if isinstance(spec, str) and spec.startswith("dp://"):
        return resolve_data_path(registry, spec[len("dp://") :])
    return Path(spec)


__all__ = [
    "DEFAULT_DATA_PATHS_CONFIG",
    "load_data_paths",
    "resolve_data_path",
    "resolve_path_spec",
]
