from __future__ import annotations

# Loads plain YAML config files used by the experiment modules directly.
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "PyYAML is required to load run configs. Install dependencies with "
        "`python3 -m pip install -r requirements.txt`."
    ) from exc

def load_yaml(path: str | Path) -> dict:
    path_obj = Path(path)
    with open(path_obj, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at {path_obj}, got {type(data).__name__}.")
    return data


def load_text(path: str | Path) -> str:
    path_obj = Path(path)
    with open(path_obj, "r", encoding="utf-8") as f:
        return f.read().strip()
