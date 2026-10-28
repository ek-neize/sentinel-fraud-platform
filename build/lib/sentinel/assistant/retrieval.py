"""BM25 retrieval over SOP sections. Small corpus, so no vector database is needed:
lexical retrieval is transparent, deterministic and easy to evaluate (recall@k)."""

from __future__ import annotations

import math
import re
from collections import Counter

from sentinel.assistant.sops import Section

_STOP = set(
    "a an the and or of to in on at for by with is are be as it this that from was were "
    "not no any all its their there than then when which who what into up out if so do "
    "does must may can only each per one".split()
)
# Domain normalisation: data vocabulary -> procedure vocabulary.
_EXPAND = {
    "_net": "online card-not-present cnp",
    "_pos": "in-person point-of-sale",
    "hot-list": "watchlist hot-list listed",
    "earlier": "burst velocity",
    "average": "out-of-pattern",
    "largest": "out-of-pattern",
    "population": "context-only",
    "state": "context-only",
    "km": "distance location",
}


def tokenize(text: str) -> list[str]:
    t = text.lower()
    for k, v in _EXPAND.items():
        if k in t:
            t += " " + v
    words = []
    for w in re.findall(r"[a-z][a-z\-]+", t):
        words.append(w)
        if "-" in w:  # "night-time" also matches "night"
            words += w.split("-")
    return [w for w in words if w not in _STOP and len(w) > 2]


class SopIndex:
    def __init__(self, sections: list[Section], k1: float = 1.4, b: float = 0.75):
        self.sections = sections
        self.by_id = {s.id: s for s in sections}
        self.docs = [tokenize(f"{s.sop_title} {s.heading} {s.text}") for s in sections]
        self.avgdl = sum(map(len, self.docs)) / len(self.docs)
        df = Counter(w for d in self.docs for w in set(d))
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.tf = [Counter(d) for d in self.docs]
        self.k1, self.b = k1, b

    def scores(self, query: str) -> list[float]:
        q = set(tokenize(query))
        out = []
        for tf, d in zip(self.tf, self.docs, strict=True):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * len(d) / self.avgdl)
            for w in q:
                f = tf.get(w, 0)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out

    def fused_scores(self, queries: list[str], rrf_k: int = 60, depth: int = 5) -> list[float]:
        """Reciprocal rank fusion: each query (one fact or signal) votes for its top
        `depth` sections, so a single specific fact (e.g. night-time) is not drowned out
        by the rest of the case."""
        fused = [0.0] * len(self.sections)
        for q in queries:
            sc = self.scores(q)
            order = sorted(range(len(sc)), key=lambda i: -sc[i])
            for rank, i in enumerate(order[:depth]):
                if sc[i] > 0:
                    fused[i] += 1 / (rrf_k + rank)
        return fused

    def search(
        self,
        query: str | list[str],
        k: int = 6,
        exclude: set[str] | None = None,
        max_per_sop: int = 2,
    ) -> list[Section]:
        """Top-k sections for one query or (fused) several, at most `max_per_sop` from one
        SOP so one long procedure cannot crowd out the others."""
        exclude = exclude or set()
        sc = self.fused_scores(query) if isinstance(query, list) else self.scores(query)
        ranked = sorted(zip(sc, range(len(self.sections)), strict=True), reverse=True)
        if isinstance(query, list):
            # Coverage first: every fact's best-matching section is a candidate before
            # sections that are merely popular across many facts.
            tops = set()
            for q in query:
                qs = self.scores(q)
                best = max(range(len(qs)), key=lambda i: qs[i])
                if qs[best] > 0:
                    tops.add(best)
            ranked = [x for x in ranked if x[1] in tops] + [x for x in ranked if x[1] not in tops]
        hits, per = [], Counter()
        for s, i in ranked:
            sec = self.sections[i]
            if s <= 0 or sec.id in exclude or per[sec.sop_id] >= max_per_sop:
                continue
            hits.append(sec)
            per[sec.sop_id] += 1
            if len(hits) == k:
                break
        return hits

    def context_for(self, query: str | list[str], k: int, pinned: list[str]) -> list[Section]:
        """Pinned policy sections first (always shown), then the top-k retrieved."""
        pins = [self.by_id[p] for p in pinned if p in self.by_id]
        return pins + self.search(query, k, exclude={p.id for p in pins})
