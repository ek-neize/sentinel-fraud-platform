"""Load project paths and business assumptions."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def _find_root() -> Path:
    """Repo root: SENTINEL_HOME if set, else the source checkout this file lives in
    (editable install), else the current directory (non-editable install, e.g. Docker
    with WORKDIR /app containing config/)."""
    if os.environ.get("SENTINEL_HOME"):
        return Path(os.environ["SENTINEL_HOME"]).resolve()
    here = Path(__file__).resolve().parents[3]
    if (here / "config" / "assumptions.yaml").exists():
        return here
    return Path.cwd()


REPO_ROOT = _find_root()
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
REFERENCE_DIR = DATA_DIR / "reference"
SOP_DIR = REPO_ROOT / "docs" / "sops"


@lru_cache(maxsize=1)
def load_assumptions(path: str | Path | None = None) -> dict[str, Any]:
    """Return the parsed assumptions.yaml (cached)."""
    path = Path(path) if path else CONFIG_DIR / "assumptions.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)
