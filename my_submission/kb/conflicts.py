"""Find places where the site contradicts itself and decide which value is current."""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import date
from typing import Callable

from . import authority
from . import text as T
from .store import ChunkStore, Record

_TIER_ONLY = frozenset({"month", "year", "model", "or", "less", "and", "older", "newer", "up", "over"})
_YEAR_SPAN_RE = re.compile(r"\b(?:19|20)\d{2}\s*-\s*\d{2,4}\b|\b(?:19|20)\d{2}\b")
_RATE_HEADER = re.compile(r"\b(rate|apr|apy|dividend)\b", re.I)
_ROW_CLASSES = frozenset({authority.OFFICIAL, authority.CORE, authority.MARKETING,
                          authority.CALCULATOR, authority.NEWS})

ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")

# (topic, pattern, keywords) for facts stated in prose, typically press-release boilerplate.
CLAIMS: list[tuple[str, re.Pattern, str]] = [
    ("Number of Dupaco branch locations",
     re.compile(r"\b(\d{1,3})\s+(?:full-service\s+)?branch(?:es|\s+offices?|\s+office\s+locations|\s+locations)\b", re.I),
     "branch locations"),
    ("Times Forbes has recognized Dupaco",
     re.compile(r"\b(" + "|".join(ORDINALS) + r")\s+time\s+Forbes", re.I), "forbes recognition"),
    ("Dupaco membership count",
     re.compile(r"membership has grown to (?:more than|over|nearly)?\s*([\d,]{5,})", re.I), "membership members"),
    ("Dupaco total assets",
     re.compile(r"assets (?:exceeding|of more than|of over|of)\s+\$([\d.]+)\s+billion", re.I), "assets billion"),
    ("Dupaco employee count",
     re.compile(r"(?:nearly|more than|over|about)\s+([\d,]{3,})\s+employees", re.I), "employees"),
    ("Dupaco routing number", re.compile(r"routing number:?\s*(\d{9})\b", re.I), "routing number"),
]


class ConflictFinder:
    def __init__(self, store: ChunkStore, make_conflict: Callable, make_source: Callable) -> None:
        self.store = store
        self.make_conflict = make_conflict
        self.make_source = make_source

    def find(self) -> list:
        out = self._row_conflicts() + self._claim_conflicts()
        out.sort(key=lambda c: c.topic.lower())
        return out

    # ---------- table rows ----------

    def _row_conflicts(self) -> list:
        groups: dict[tuple, list[Record]] = defaultdict(list)
        spanned: dict[tuple, list[Record]] = defaultdict(list)
        for rec in self.store.records:
            if not self._is_rate_row(rec):
                continue
            label = rec.chunk.label
            if T.year_span(label):
                spanned[self._row_key(rec, strip_years=True)].append(rec)
            else:
                groups[self._row_key(rec)].append(rec)
        for key, recs in spanned.items():
            for i, cluster in enumerate(_cluster_by_span(recs)):
                groups[key + (i,)] = cluster
        conflicts = []
        for key, recs in groups.items():
            conflict = self._conflict_from_rows(recs)
            if conflict:
                conflicts.append(conflict)
        return conflicts

    @staticmethod
    def _is_rate_row(rec: Record) -> bool:
        c = rec.chunk
        if rec.kind != "row" or rec.page_class not in _ROW_CLASSES:
            return False
        if not c.rates or len(c.label) > 120 or len(c.rates) > 4:
            return False
        return bool(_RATE_HEADER.search(c.header)) and "intro" not in {k for k, _ in c.rates}

    @staticmethod
    def _row_key(rec: Record, strip_years: bool = False) -> tuple:
        label = rec.chunk.label
        if strip_years:
            label = _YEAR_SPAN_RE.sub("", label)
        norm = re.sub(r"\s+", " ", T.normalize(label).lower().replace(",", "")).strip(" |")
        norm = re.sub(r"\bmonths\b", "month", norm)
        words = {t for t in T.tokenize(norm) if not t.isdigit()}
        if not words - _TIER_ONLY:
            norm = f"{rec.chunk.caption.lower()} | {norm}"
        return (norm, tuple(sorted({k for k, _ in rec.chunk.rates})))

    def _conflict_from_rows(self, recs: list[Record]):
        by_value: dict[tuple, list[Record]] = defaultdict(list)
        for r in recs:
            by_value[r.chunk.rates].append(r)
        if len(by_value) < 2 or self._coexisting_products(by_value):
            return None
        latest_idx = len(self.store.snapshots) - 1
        today = self.store.snapshots[latest_idx].taken_on
        live = [r for r in recs if not (r.chunk.ends and r.chunk.ends < today)] or recs
        current = [r for r in live if r.in_snapshot(latest_idx)] or live
        winner = max(current, key=lambda r: (r.page_weight, r.chunk.effective or date.min, r.snapshots()[-1],
                                             "business" not in r.url or "business" in r.chunk.label.lower(), -r.id))
        ordered = sorted(by_value.items(), key=lambda kv: min(r.snapshots()[0] for r in kv[1]))
        values, sources = [], []
        for _, members in ordered:
            first = min(members, key=lambda r: (r.snapshots()[0], -r.page_weight, r.id))
            values.append(_value_text(first))
            sources.append(self._source(first))
        topic = _row_topic(winner)
        reason = self._row_reason(winner, ordered, latest_idx)
        return self.make_conflict(topic=topic, values=values, sources=sources,
                                  resolution=_value_text(winner), reason=reason)

    @staticmethod
    def _coexisting_products(by_value: dict) -> bool:
        """Different values side by side on one page in one snapshot are different products, not a contradiction."""
        seen = [{(r.url, i) for r in members for i in r.snapshots()} for members in by_value.values()]
        return all(a & b for i, a in enumerate(seen) for b in seen[i + 1:])

    def _row_reason(self, winner: Record, ordered, latest_idx: int) -> str:
        urls = {r.url for _, members in ordered for r in members}
        changed_over_time = any(
            len({r.chunk.rates for _, ms in ordered for r in ms if r.url == u}) > 1
            for u in urls)
        parts = []
        if changed_over_time:
            parts.append(f"the newest snapshot ({self.store.snapshots[latest_idx].taken_on}) supersedes earlier ones")
        if len(urls) > 1:
            eff = winner.chunk.effective
            parts.append(f"{authority.classify(winner.url)} page {winner.url}"
                         + (f" (rates effective {eff})" if eff else "")
                         + " outranks calculators, promos, news and pages with older effective dates")
        return "; ".join(parts) or "most recent authoritative source"

    def _source(self, rec: Record):
        sid = self.store.snapshots[rec.snapshots()[0]].snapshot_id
        return self.make_source(url=rec.url, snapshot_id=sid, quote=rec.text[:200])

    # ---------- prose claims ----------

    def _claim_conflicts(self) -> list:
        out = []
        for topic, pattern, keywords in CLAIMS:
            seen: dict[str, list[Record]] = defaultdict(list)
            for rec in self.store.records:
                if rec.kind != "text" or rec.page_class == authority.JUNK:
                    continue
                for m in pattern.finditer(rec.text):
                    seen[m.group(1).lower().replace(",", "")].append(rec)
            if len(seen) < 2:
                continue
            winner_value = max(seen, key=lambda v: max(_claim_rank(r, self.store) for r in seen[v]))
            ordered = sorted(seen.items(), key=lambda kv: min(_recency(r) for r in kv[1]))
            values = [_claim_text(v, topic) for v, _ in ordered]
            sources = [self._source(max(rs, key=lambda r: _recency(r))) for _, rs in ordered]
            best = max(seen[winner_value], key=lambda r: _claim_rank(r, self.store))
            when = best.page_date or self.store.snapshots[best.snapshots()[-1]].taken_on
            out.append(self.make_conflict(
                topic=f"{topic} ({keywords})", values=values, sources=sources,
                resolution=_claim_text(winner_value, topic),
                reason=f"most recent statement in the current site ({best.url}, dated {when}) wins over older pages"))
        return out


def _cluster_by_span(recs: list[Record]) -> list[list[Record]]:
    spans = sorted({T.year_span(r.chunk.label) for r in recs})
    parent = {s: s for s in spans}

    def find(s):
        while parent[s] != s:
            s = parent[s]
        return s

    for i, a in enumerate(spans):
        for b in spans[i + 1:]:
            inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
            union = max(a[1], b[1]) - min(a[0], b[0]) + 1
            if union and inter / union >= 0.5:
                parent[find(b)] = find(a)
    clusters: dict = defaultdict(list)
    for r in recs:
        clusters[find(T.year_span(r.chunk.label))].append(r)
    return [clusters[k] for k in sorted(clusters)]


def _value_text(rec: Record) -> str:
    return ", ".join(f"{_pct(value)}% {kind}" for kind, value in rec.chunk.rates)


def _pct(value: float) -> str:
    text = f"{value:.3f}"
    return text[:-1] if text.endswith("0") else text


def _row_topic(rec: Record) -> str:
    c = rec.chunk
    label = c.label.replace(" | ", ", ")
    caption = c.caption if c.caption and c.caption.lower() not in label.lower() else ""
    hyphenated = ", ".join(f"{n}-{u}" for n, u in sorted(T.terms(c.label)))
    page = rec.title.split("|")[0].strip()
    bits = [b for b in (caption, label) if b]
    topic = " ".join(bits)
    if hyphenated:
        topic += f" ({hyphenated})"
    return f"{topic} rate [{page}]" if page else f"{topic} rate"


def _recency(rec: Record) -> date:
    return rec.page_date or rec.chunk.effective or date.min


def _claim_rank(rec: Record, store: ChunkStore) -> tuple:
    return (_recency(rec), rec.snapshots()[-1], rec.page_weight)


def _claim_text(value: str, topic: str) -> str:
    if value.isdigit() and len(value) >= 4 and "routing" not in topic.lower():
        return f"{int(value):,}"
    return value
