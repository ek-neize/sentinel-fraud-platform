"""Parse the synthetic SOP markdown files into citable sections ("SOP-003 §2")."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sentinel.common.config import SOP_DIR

_FRONT = re.compile(r"^---\n(.*?)\n---\n", re.S)
_SECTION = re.compile(r"^## §(\d+) (.+)$", re.M)


@dataclass(frozen=True)
class Section:
    id: str  # "SOP-003 §2"
    sop_id: str  # "SOP-003"
    sop_title: str
    heading: str
    text: str

    @property
    def label(self) -> str:
        return f"{self.id} {self.sop_title}: {self.heading}"


def _front_matter(raw: str) -> dict[str, str]:
    m = _FRONT.match(raw)
    if not m:
        return {}
    out = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip().strip('"')
    return out


def parse_sop(raw: str) -> list[Section]:
    meta = _front_matter(raw)
    sop_id, title = meta["id"], meta["title"]
    body = _FRONT.sub("", raw, count=1)
    heads = list(_SECTION.finditer(body))
    sections = []
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(body)
        text = " ".join(body[h.end() : end].split())
        sections.append(Section(f"{sop_id} §{h.group(1)}", sop_id, title, h.group(2), text))
    return sections


def load_sections(folder: str | Path = SOP_DIR) -> list[Section]:
    files = sorted(Path(folder).glob("SOP-*.md"))
    if not files:
        raise FileNotFoundError(f"No SOP-*.md files in {folder}")
    return [s for f in files for s in parse_sop(f.read_text(encoding="utf-8"))]
