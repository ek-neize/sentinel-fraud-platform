"""Download the dashboard snapshot written by notebook 08 into dashboard/snapshot/.

    python scripts/pull_snapshot.py
    git add dashboard/snapshot && git commit -m "Update dashboard snapshot" && git push

Uses your Databricks CLI profile (default: sentinel). The public Streamlit app reads
these committed files, so it never needs Databricks credentials.
"""

from __future__ import annotations

import argparse
import shutil

from sentinel.common.config import REPO_ROOT

DEST = REPO_ROOT / "dashboard" / "snapshot"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--profile", default="sentinel")
    p.add_argument("--catalog", default="workspace")
    p.add_argument("--schema", default="sentinel")
    args = p.parse_args()

    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient(profile=args.profile)
    folder = f"/Volumes/{args.catalog}/{args.schema}/dashboard_snapshot"
    files = [
        e.name
        for e in w.files.list_directory_contents(folder)
        if e.name and (e.name.endswith(".parquet") or e.name == "meta.json")
    ]
    if "meta.json" not in files:
        raise SystemExit(f"No snapshot in {folder}. Run notebook 08_dashboard_snapshot first.")
    tmp = DEST.with_name("snapshot_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    total = 0
    for name in sorted(files):
        data = w.files.download(f"{folder}/{name}").contents.read()
        (tmp / name).write_bytes(data)
        total += len(data)
    if (DEST / "README.md").exists():
        shutil.copy(DEST / "README.md", tmp / "README.md")
    shutil.rmtree(DEST, ignore_errors=True)
    tmp.rename(DEST)
    print(f"Pulled {len(files)} files ({total / 1024:,.0f} KB) into {DEST}")


if __name__ == "__main__":
    main()
