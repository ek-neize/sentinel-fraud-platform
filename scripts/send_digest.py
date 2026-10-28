"""Send the latest daily alert digest by email (or just show where it is).

Databricks writes digests to the volume /Volumes/<catalog>/<schema>/alerts/. Free Edition
blocks outbound SMTP, so delivery runs here (laptop or a scheduled GitHub Action).

Examples
  python scripts/send_digest.py --from-databricks             # latest digest from the volume
  python scripts/send_digest.py --from-databricks --day 2020-12-14
  python scripts/send_digest.py --file data/alerts/digest_2020-12-14.html
  python scripts/send_digest.py --from-databricks --dry-run   # download + build, don't send

Email settings come from .env (see src/sentinel/alerts/notify.py). Without them the
digest is downloaded and saved, and the script says email is disabled.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from sentinel.alerts.notify import EmailSettings, send_email
from sentinel.common.config import REPO_ROOT

LOCAL_DIR = REPO_ROOT / "data" / "alerts"


def download_from_databricks(profile: str, catalog: str, schema: str, day: str | None) -> Path:
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient(profile=profile)
    folder = f"/Volumes/{catalog}/{schema}/alerts"
    names = sorted(
        e.name
        for e in w.files.list_directory_contents(folder)
        if e.name and re.fullmatch(r"digest_\d{4}-\d{2}-\d{2}\.html", e.name)
    )
    if not names:
        raise SystemExit(f"No digests in {folder}. Run notebook 07_score_and_alert first.")
    name = f"digest_{day}.html" if day else names[-1]
    if name not in names:
        raise SystemExit(f"{name} not found. Available: {', '.join(names[-5:])}")
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    local = LOCAL_DIR / name
    local.write_bytes(w.files.download(f"{folder}/{name}").contents.read())
    return local


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="local digest HTML")
    src.add_argument("--from-databricks", action="store_true")
    p.add_argument("--day", help="YYYY-MM-DD (default: latest)")
    p.add_argument("--profile", default="sentinel")
    p.add_argument("--catalog", default="workspace")
    p.add_argument("--schema", default="sentinel")
    p.add_argument("--dry-run", action="store_true", help="never send")
    args = p.parse_args()

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    path = (
        download_from_databricks(args.profile, args.catalog, args.schema, args.day)
        if args.from_databricks
        else Path(args.file)
    )
    html = path.read_text(encoding="utf-8")
    day = re.search(r"\d{4}-\d{2}-\d{2}", path.name)
    subject = f"Sentinel fraud alerts — {day.group(0) if day else path.stem}"
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()[:4000]
    print(f"Digest: {path}")
    if args.dry_run:
        print("dry run: not sending")
        return 0
    print(send_email(subject, html, text, EmailSettings.from_env()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
