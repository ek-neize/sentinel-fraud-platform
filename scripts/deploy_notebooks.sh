#!/usr/bin/env bash
# Import the Databricks notebooks into your workspace.
#
# Usage: scripts/deploy_notebooks.sh [profile]
# Notebooks land in /Users/<you>/sentinel. Re-running overwrites them.
set -euo pipefail

PROFILE="${1:-sentinel}"
USER_NAME=$(databricks current-user me --profile "$PROFILE" --output json | python3 -c "import sys,json; print(json.load(sys.stdin)['userName'])")
TARGET="/Users/${USER_NAME}/sentinel"

databricks workspace mkdirs "$TARGET" --profile "$PROFILE"
databricks workspace import-dir notebooks/databricks "$TARGET" --overwrite --profile "$PROFILE"
databricks workspace list "$TARGET" --profile "$PROFILE"
echo "Notebooks imported to $TARGET"
