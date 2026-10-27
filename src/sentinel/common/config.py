"""Load project paths and business assumptions."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
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
