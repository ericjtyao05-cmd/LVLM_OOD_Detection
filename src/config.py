"""Load and lightly validate experiment.yaml."""

from __future__ import annotations

from pathlib import Path


def load_config(path: str | Path) -> dict:
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k in ("model", "detector", "data", "conditions"):
        if k not in cfg:
            raise KeyError(f"experiment config missing top-level key: {k}")
    return cfg


def id_class_names(cfg: dict) -> list[str]:
    """Sorted ID class names — must match build_datasets' sorted() convention."""
    return sorted(c["name"] for c in cfg["data"]["id"]["classes"])
