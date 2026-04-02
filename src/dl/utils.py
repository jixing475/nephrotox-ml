"""Utility functions for I/O and file operations."""

from __future__ import annotations

import json
import pathlib
import platform
import sys
from typing import Any

import pandas as pd

from . import config


def ensure_output_dir(path: pathlib.Path | str):
    """Ensure output directory exists."""
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_output_dir_with_confirm(base_dir: pathlib.Path | str, config_key: str):
    """Ensure output directory exists; files will be overwritten if they exist."""
    base_dir = pathlib.Path(base_dir)
    out_path = base_dir / config_key
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def save_json(data: dict, path: pathlib.Path | str):
    """Save dictionary to JSON file."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def update_json(path: pathlib.Path | str, updates: dict[str, Any]):
    """
    Update a JSON file in-place (shallow merge).

    - If the file does not exist, it will be created.
    - If `updates["warnings"]` is a list, it will be appended to any existing
      `warnings` list rather than overwritten.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, Any] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f) or {}
            except json.JSONDecodeError:
                existing = {}

    merged = dict(existing)
    for k, v in updates.items():
        if k == "warnings" and isinstance(v, list):
            prev = merged.get("warnings")
            if isinstance(prev, list):
                merged["warnings"] = prev + v
            else:
                merged["warnings"] = v
        else:
            merged[k] = v

    save_json(merged, path)


def save_dataframe(df: pd.DataFrame, path: pathlib.Path | str):
    """Save DataFrame to CSV file."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def log_versions(path: pathlib.Path | str):
    """Log Python and library versions."""
    import deepchem
    import torch
    import sklearn
    import pandas
    import numpy

    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "deepchem": deepchem.__version__,
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
    }
    save_json(info, path)

