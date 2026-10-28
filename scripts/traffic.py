"""Replay demo transactions against the scoring API (standard library only).

    python scripts/traffic.py --url http://localhost:8000 --file models/demo_traffic.jsonl

--drift-after N: after N seconds, simulate a data shift (amounts x4, night-time hours) so
the Prometheus drift and alert-rate rules fire and show up in Grafana.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request


def post(url: str, body: dict) -> int:
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, TimeoutError):
        return 0


def shifted(rec: dict) -> dict:
    r = dict(rec)
    for k in ("amount", "amt_1h", "amt_24h"):
        if r.get(k) is not None:
            r[k] = r[k] * 4
    r["hour_of_day"] = random.choice([22, 23, 0, 1, 2, 3])
    r.pop("log_amount", None)
    r.pop("is_night", None)
    return r


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--url", default=os.environ.get("API_URL", "http://localhost:8000"))
    p.add_argument("--file", default=os.environ.get("TRAFFIC_FILE", "models/demo_traffic.jsonl"))
    p.add_argument("--rate", type=float, default=float(os.environ.get("RATE", 5)))
    p.add_argument(
        "--drift-after", type=float, default=float(os.environ.get("DRIFT_AFTER_S", 0)) or None
    )
    p.add_argument(
        "--invalid-share", type=float, default=float(os.environ.get("INVALID_SHARE", 0.01))
    )
    p.add_argument("--duration", type=float, default=float(os.environ.get("DURATION_S", 0)))
    args = p.parse_args()
    with open(args.file) as fh:
        records = [json.loads(x) for x in fh if x.strip()]
    print(
        f"{len(records)} records, {args.rate}/s to {args.url}"
        + (f", drift after {args.drift_after:.0f}s" if args.drift_after else ""),
        flush=True,
    )
    start, n, codes = time.time(), 0, {}
    while not args.duration or time.time() - start < args.duration:
        rec = random.choice(records)
        if args.drift_after and time.time() - start > args.drift_after:
            rec = shifted(rec)
        if random.random() < args.invalid_share:
            rec = dict(rec, amount=-1)  # a few malformed requests (422s) for the error panel
        code = post(f"{args.url}/score", rec)
        codes[code] = codes.get(code, 0) + 1
        n += 1
        if n % 500 == 0:
            print(f"{n} sent, status codes {codes}", flush=True)
        time.sleep(max(0.0, 1 / args.rate))
    print(f"done: {n} sent, status codes {codes}")


if __name__ == "__main__":
    main()
