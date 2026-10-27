"""Run a titled cell from a Databricks source notebook in a plain namespace."""

import re
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "databricks"


def load_cell(notebook: str, title: str) -> dict:
    path = NOTEBOOK_DIR / notebook
    cells = re.split(r"^# COMMAND -+$", path.read_text(), flags=re.M)
    cell = next(c for c in cells if f"# DBTITLE 1,{title}" in c)
    ns: dict = {}
    exec(compile(cell, str(path), "exec"), ns)
    return ns
