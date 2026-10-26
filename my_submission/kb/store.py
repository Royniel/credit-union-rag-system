"""Versioned chunk store with a BM25 index.

Every distinct chunk is stored once and carries a bitmask of the snapshots it appeared in,
so one index serves both the latest view and any "as of" view of the site.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Optional

from . import authority
from .chunker import Chunk, chunk_page
from .text import tokenize

K1, B = 1.2, 0.6


@dataclass(eq=False)
class Record:
    id: int
    url: str
    title: str
    chunk: Chunk
    page_class: str
    page_weight: float
    page_date: Optional[date]
    context_tokens: frozenset
    length: int
    mask: int = 0                   # snapshots whose crawl contained this chunk
    view: int = 0                   # snapshots whose "as of" view shows it (latest known page versions)
    text_key: str = ""
    lead: bool = False              # opening paragraph of an article

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def kind(self) -> str:
        return self.chunk.kind

    def in_snapshot(self, idx: int) -> bool:
        return bool(self.view >> idx & 1)

    def crawled_in(self, idx: int) -> bool:
        return bool(self.mask >> idx & 1)

    def snapshots(self) -> list[int]:
        return [i for i in range(self.mask.bit_length()) if self.mask >> i & 1]


@dataclass
class SnapshotInfo:
    index: int
    snapshot_id: str
    taken_on: date
    pages: dict = field(default_factory=dict)   # url -> title


class ChunkStore:
    def __init__(self) -> None:
        self.records: list[Record] = []
        self.snapshots: list[SnapshotInfo] = []
        self._by_key: dict[tuple, int] = {}
        self._page_cache: dict[str, list[int]] = {}       # markdown hash -> record ids
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._df_cache: dict[str, float] = {}
        self._avg_len = 1.0
        self._tables: Optional[dict] = None
        self._pages_by_text: dict[str, set] = defaultdict(set)
        self._by_text: dict[str, list[int]] = defaultdict(list)
        self._page_ids: dict[str, list[int]] = defaultdict(list)
        self._title_brands: dict[str, int] = defaultdict(int)
        self._latest_ids: dict[str, list[int]] = {}       # url -> record ids of its newest version

    # ---------- ingest ----------

    def add_snapshot(self, snapshot_id: str, taken_on: date, pages: Iterable) -> SnapshotInfo:
        info = SnapshotInfo(len(self.snapshots), snapshot_id, taken_on)
        self.snapshots.append(info)
        bit = 1 << info.index
        for page in pages:
            info.pages[page.url] = page.title
            if "|" in page.title:
                self._title_brands[page.title.rsplit("|", 1)[1].strip().lower()] += 1
            digest = hashlib.sha1(f"{page.url}\n{page.markdown}".encode("utf-8")).hexdigest()
            ids = self._page_cache.get(digest)
            if ids is None:
                ids = self._index_page(page.url, page.title, page.markdown)
                self._page_cache[digest] = ids
            for rid in ids:
                self.records[rid].mask |= bit
            self._latest_ids[page.url] = ids
        # A page missing from this crawl keeps its last known version in the view of the site.
        for ids in self._latest_ids.values():
            for rid in ids:
                self.records[rid].view |= bit
        self._df_cache.clear()
        self._tables = None
        self._avg_len = sum(r.length for r in self.records) / max(1, len(self.records))
        return info

    def _index_page(self, url: str, title: str, markdown: str) -> list[int]:
        page_class = authority.classify(url)
        page_weight = authority.WEIGHTS[page_class]
        published = authority.page_date(url, markdown)
        title_tokens = tokenize(title.split("|")[0])
        url_tokens = tokenize(url.split("dupaco.com", 1)[-1].replace("-", " ").replace("/", " "))
        ids = []
        lead_open = page_class in (authority.NEWS, authority.BLOG)
        for chunk in chunk_page(markdown):
            key = (url, chunk.kind, chunk.heading, chunk.text)
            rid = self._by_key.get(key)
            if rid is None:
                rid = self._add_record(url, title, chunk, page_class, page_weight, published,
                                       title_tokens, url_tokens)
                self._by_key[key] = rid
            if lead_open and chunk.kind == "text" and len(chunk.text) >= 150:
                self.records[rid].lead = True
                lead_open = False
            ids.append(rid)
        return list(dict.fromkeys(ids))

    def _add_record(self, url, title, chunk, page_class, page_weight, published,
                    title_tokens, url_tokens) -> int:
        body = tokenize(chunk.text)
        context = tokenize(f"{chunk.heading} {chunk.caption} {chunk.header}")
        # Context terms count once each so long headings cannot swamp the body.
        bag = body + list(dict.fromkeys(context + title_tokens + url_tokens))
        rid = len(self.records)
        text_key = hashlib.sha1(chunk.text.lower().encode("utf-8")).hexdigest()
        self._pages_by_text[text_key].add(url)
        self._by_text[text_key].append(rid)
        self._page_ids[url].append(rid)
        tf: dict[str, int] = defaultdict(int)
        for tok in bag:
            tf[tok] += 1
        for tok, n in tf.items():
            self._postings[tok].append((rid, n))
        self.records.append(Record(
            rid, url, title, chunk, page_class, page_weight, published,
            frozenset(bag), max(1, len(bag)), text_key=text_key))
        return rid

    def same_text(self, rec: Record, snapshot_idx: int) -> list[Record]:
        """Identical chunks on other pages of the same snapshot (mirrored rate tables, shared footers)."""
        return [self.records[i] for i in self._by_text.get(rec.text_key, ())
                if i != rec.id and self.records[i].in_snapshot(snapshot_idx)]

    def page_records(self, url: str, snapshot_idx: int) -> list[Record]:
        return [self.records[i] for i in self._page_ids.get(url, ()) if self.records[i].in_snapshot(snapshot_idx)]

    def page_spread(self, rec: Record) -> int:
        """How many pages carry this exact text; site-wide headers and footers score high."""
        return len(self._pages_by_text.get(rec.text_key, ()))

    # ---------- query ----------

    def snapshot_for(self, as_of: Optional[date]) -> Optional[SnapshotInfo]:
        if not self.snapshots:
            return None
        if as_of is None:
            return self.snapshots[-1]
        eligible = [s for s in self.snapshots if s.taken_on <= as_of]
        return eligible[-1] if eligible else None

    @property
    def org_token(self) -> str:
        """The site's own name, read from the '| Brand' suffix most page titles share."""
        if not self._title_brands:
            return ""
        return max(self._title_brands, key=self._title_brands.get)

    def idf(self, term: str) -> float:
        cached = self._df_cache.get(term)
        if cached is None:
            df = len(self._postings.get(term, ()))
            n = len(self.records)
            cached = math.log(1 + (n - df + 0.5) / (df + 0.5)) if df else 0.0
            self._df_cache[term] = cached
        return cached

    def known(self, term: str) -> bool:
        return term in self._postings

    def search(self, weighted_terms: dict[str, float], snapshot_idx: int, limit: int = 300) -> list[tuple[float, Record]]:
        bit = 1 << snapshot_idx
        scores: dict[int, float] = defaultdict(float)
        records = self.records
        for term, qweight in weighted_terms.items():
            idf = self.idf(term)
            if not idf:
                continue
            for rid, tf in self._postings[term]:
                rec = records[rid]
                if not rec.view & bit:
                    continue
                norm = K1 * (1 - B + B * rec.length / self._avg_len)
                scores[rid] += qweight * idf * tf * (K1 + 1) / (tf + norm)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
        return [(score, records[rid]) for rid, score in ranked]

    def rows_of_table(self, record: Record, snapshot_idx: int) -> list[Record]:
        table = record.chunk.table
        return [r for r in self._table_index().get((record.url, table), []) if r.in_snapshot(snapshot_idx)]

    def _table_index(self) -> dict:
        if self._tables is None:
            tables = defaultdict(list)
            for rec in self.records:
                if rec.kind == "row":
                    tables[(rec.url, rec.chunk.table)].append(rec)
            self._tables = tables
        return self._tables
