"""Export Kafka event JSON Schemas to schemas/ (run after editing events.py)."""

from sentinel.common.config import REPO_ROOT
from sentinel.common.events import export_json_schemas

if __name__ == "__main__":
    export_json_schemas(REPO_ROOT / "schemas")
    print(f"Wrote schemas to {REPO_ROOT / 'schemas'}")
