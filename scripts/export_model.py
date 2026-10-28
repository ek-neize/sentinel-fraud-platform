"""Download fraud_model@champion from Unity Catalog for the FastAPI service / Docker.

    python scripts/export_model.py                    # -> models/champion/
    python scripts/export_model.py --alias baseline --out models/baseline

Writes the MLflow model plus sentinel_meta.json (version, operating threshold and budget
tagged by notebook 06). Uses your Databricks CLI profile; no token is stored.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from sentinel.common.config import REPO_ROOT


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--profile", default="sentinel")
    p.add_argument("--model", default="workspace.sentinel.fraud_model")
    p.add_argument("--alias", default="champion")
    p.add_argument("--out", default=str(REPO_ROOT / "models" / "champion"))
    args = p.parse_args()

    os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mv = MlflowClient().get_model_version_by_alias(args.model, args.alias)

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(f"models:/{args.model}@{args.alias}", dst_path=str(out))
    meta = {
        "model": args.model,
        "alias": args.alias,
        "version": mv.version,
        "threshold": float(mv.tags["operating_threshold"])
        if "operating_threshold" in mv.tags
        else float(mv.tags.get("threshold_25_per_day", 0.5)),
        "operating_k": int(mv.tags["operating_k"]) if "operating_k" in mv.tags else None,
        "experiment": mv.tags.get("experiment"),
    }
    (out / "sentinel_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Exported {args.model}@{args.alias} v{mv.version} -> {out}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
