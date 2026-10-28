"""Daily alert digest as self-contained HTML (inline CSS, works in email clients).

The same HTML is written to a file/volume every day and, if enabled, emailed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from html import escape

import pandas as pd

DISCLAIMER = (
    "Risk scores and reasons are model signals, not verified facts. "
    "An investigator reviews the evidence and makes the final decision."
)


def _kpi(label: str, value) -> str:
    return (
        '<td style="padding:10px 14px;border:1px solid #e3e6ea;border-radius:6px;'
        'text-align:center;min-width:110px">'
        f'<div style="font-size:22px;font-weight:700;color:#1b2733">{escape(str(value))}</div>'
        f'<div style="font-size:12px;color:#5b6773">{escape(label)}</div></td>'
    )


def _band(score: float, threshold: float) -> tuple[str, str]:
    """Same bands as the API: high >= max(0.9, threshold) > elevated >= threshold."""
    if score >= max(0.9, threshold):
        return "High", "#b42318"
    if score >= threshold:
        return "Elevated", "#b54708"
    return "Watch", "#475467"


def render_digest(
    day, alerts: pd.DataFrame, summary: dict, model_info: dict, top_n: int = 20
) -> str:
    """alerts: columns transaction_id, event_ts, card_id, merchant_name, category, amount,
    score, status, reason_1..reason_3 (sorted by priority, highest first)."""
    day_s = pd.Timestamp(day).strftime("%Y-%m-%d")
    kpis = "".join(
        _kpi(lbl, summary.get(key, "–"))
        for lbl, key in (
            ("New alerts", "new_alerts"),
            ("Reviewed today", "reviewed"),
            ("Waiting in backlog", "open_end"),
            ("Expired unreviewed", "expired"),
        )
    )
    rows = []
    for rank, (_, a) in enumerate(alerts.head(top_n).iterrows(), start=1):
        band, color = _band(float(a["score"]), float(model_info.get("threshold", 0.5)))
        reasons = "".join(
            f"<li>{escape(str(a[c]))}</li>"
            for c in ("reason_1", "reason_2", "reason_3")
            if c in a and pd.notna(a[c]) and a[c]
        )
        card = str(a.get("card_id", ""))[-4:]
        rows.append(
            "<tr>"
            f'<td style="padding:8px;border-bottom:1px solid #eef0f3;color:#5b6773">{rank}</td>'
            f'<td style="padding:8px;border-bottom:1px solid #eef0f3">'
            f'<span style="color:{color};font-weight:700">{band}</span><br>'
            f'<span style="font-size:12px;color:#5b6773">score {float(a["score"]):.3f}</span></td>'
            f'<td style="padding:8px;border-bottom:1px solid #eef0f3">'
            f'<b>${float(a["amount"]):,.2f}</b><br><span style="font-size:12px;color:#5b6773">'
            f"{escape(str(a.get('category', '')))} · card …{escape(card)}</span></td>"
            f'<td style="padding:8px;border-bottom:1px solid #eef0f3;font-size:13px">'
            f'{escape(str(a.get("merchant_name", "")))}<br><span style="color:#5b6773">'
            f"{pd.Timestamp(a['event_ts']):%Y-%m-%d %H:%M} UTC</span></td>"
            f'<td style="padding:8px;border-bottom:1px solid #eef0f3;font-size:13px">'
            f'<ul style="margin:0;padding-left:16px">{reasons}</ul></td>'
            f'<td style="padding:8px;border-bottom:1px solid #eef0f3;font-size:12px">'
            f'{escape(str(a.get("status", "")))}<br><span style="color:#98a2b3">'
            f"…{escape(str(a['transaction_id'])[-8:])}</span></td>"
            "</tr>"
        )
    table = (
        '<table style="border-collapse:collapse;width:100%;font-size:14px">'
        '<tr style="text-align:left;background:#f6f7f9;color:#344054">'
        '<th style="padding:8px">#</th><th style="padding:8px">Risk</th>'
        '<th style="padding:8px">Amount</th><th style="padding:8px">Merchant / time</th>'
        '<th style="padding:8px">Why it was flagged (model signals)</th>'
        '<th style="padding:8px">Status</th></tr>' + "".join(rows) + "</table>"
        if rows
        else '<p style="color:#5b6773">No alerts today.</p>'
    )
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Sentinel alert digest {day_s}</title></head>
<body style="margin:0;padding:24px;background:#ffffff;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1b2733">
<div style="max-width:980px;margin:0 auto">
<h2 style="margin:0 0 4px">Sentinel · daily alert digest</h2>
<div style="color:#5b6773;margin-bottom:16px">{day_s} · review budget {escape(str(summary.get("capacity", "–")))} alerts/day</div>
<table style="border-collapse:separate;border-spacing:8px 0;margin:0 -8px 16px"><tr>{kpis}</tr></table>
<div style="background:#fffaeb;border:1px solid #fedf89;color:#7a2e0e;padding:10px 12px;border-radius:6px;font-size:13px;margin-bottom:16px">{escape(DISCLAIMER)}</div>
<h3 style="margin:8px 0">Highest-priority alerts</h3>
{table}
<p style="color:#98a2b3;font-size:12px;margin-top:20px">Model {escape(str(model_info.get("name", "")))} v{escape(str(model_info.get("version", "")))} · threshold {float(model_info.get("threshold", 0)):.4f} · generated {generated}. Synthetic data (Sparkov) — demo only.</p>
</div></body></html>"""


def digest_text(day, alerts: pd.DataFrame, summary: dict, top_n: int = 10) -> str:
    """Plain-text alternative for email clients without HTML."""
    lines = [
        f"Sentinel daily alert digest — {pd.Timestamp(day):%Y-%m-%d}",
        f"New alerts: {summary.get('new_alerts')} | reviewed: {summary.get('reviewed')} | "
        f"backlog: {summary.get('open_end')} | expired: {summary.get('expired')}",
        DISCLAIMER,
        "",
    ]
    for rank, (_, a) in enumerate(alerts.head(top_n).iterrows(), start=1):
        reasons = "; ".join(
            str(a[c]) for c in ("reason_1", "reason_2", "reason_3") if c in a and pd.notna(a[c])
        )
        lines.append(
            f"{rank}. ${float(a['amount']):,.2f} score {float(a['score']):.3f} — {reasons}"
        )
    return "\n".join(lines)
