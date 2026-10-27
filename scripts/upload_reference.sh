#!/usr/bin/env bash
# Upload the watchlist to the Databricks landing volume, then run
# notebooks/databricks/02_load_reference to load it into Delta.
#
# Usage: scripts/upload_reference.sh [profile] [catalog] [schema]
set -euo pipefail

PROFILE="${1:-sentinel}"
CATALOG="${2:-workspace}"
SCHEMA="${3:-sentinel}"
SRC="data/synthetic/watchlist.parquet"
DEST="dbfs:/Volumes/${CATALOG}/${SCHEMA}/landing/watchlist.parquet"

if [[ ! -f "$SRC" ]]; then
  echo "Missing $SRC — run: python scripts/build_reference_data.py" >&2
  exit 1
fi

databricks fs cp "$SRC" "$DEST" --overwrite --profile "$PROFILE"
databricks fs ls "dbfs:/Volumes/${CATALOG}/${SCHEMA}/landing" --profile "$PROFILE"
echo "Uploaded. Now run notebook 02_load_reference in Databricks."
