"""Deliver the digest: always as a file, optionally by email (Gmail SMTP).

Email is off unless ALERT_EMAIL_ENABLED=true and the SMTP settings are present, so the
project runs (and CI passes) without any mail account. Secrets come from the
environment / .env only.

    ALERT_EMAIL_ENABLED=true
    SMTP_HOST=smtp.gmail.com        # default
    SMTP_PORT=465                   # default (SSL)
    SMTP_USER=you@gmail.com
    SMTP_PASSWORD=<16-char Gmail app password>
    ALERT_EMAIL_TO=you@gmail.com[,other@example.com]
"""

from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path


def write_digest_file(html: str, out_dir: str | Path, day) -> Path:
    import pandas as pd

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"digest_{pd.Timestamp(day):%Y-%m-%d}.html"
    path.write_text(html, encoding="utf-8")
    return path


@dataclass
class EmailSettings:
    enabled: bool
    host: str
    port: int
    user: str | None
    password: str | None
    to: list[str]

    @classmethod
    def from_env(cls) -> EmailSettings:
        return cls(
            enabled=os.environ.get("ALERT_EMAIL_ENABLED", "false").lower() == "true",
            host=os.environ.get("SMTP_HOST", "smtp.gmail.com"),
            port=int(os.environ.get("SMTP_PORT", "465")),
            user=os.environ.get("SMTP_USER"),
            password=os.environ.get("SMTP_PASSWORD"),
            to=[a.strip() for a in os.environ.get("ALERT_EMAIL_TO", "").split(",") if a.strip()],
        )

    def problems(self) -> list[str]:
        missing = [
            n for n, v in (("SMTP_USER", self.user), ("SMTP_PASSWORD", self.password)) if not v
        ]
        if not self.to:
            missing.append("ALERT_EMAIL_TO")
        return missing


def build_message(subject: str, html: str, text: str, sender: str, to: list[str]) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(to)
    msg.set_content(text)
    msg.add_alternative(html, subtype="html")
    return msg


def send_email(
    subject: str, html: str, text: str, settings: EmailSettings, smtp_factory=smtplib.SMTP_SSL
) -> str:
    """Send if enabled and configured. Returns a short status string (never raises on
    'disabled'; raises on real SMTP errors so a failed send is visible)."""
    if not settings.enabled:
        return "email disabled (set ALERT_EMAIL_ENABLED=true to send)"
    missing = settings.problems()
    if missing:
        return f"email not sent: missing {', '.join(missing)}"
    msg = build_message(subject, html, text, settings.user, settings.to)
    with smtp_factory(settings.host, settings.port, timeout=30) as smtp:
        smtp.login(settings.user, settings.password)
        smtp.send_message(msg)
    return f"email sent to {', '.join(settings.to)}"
