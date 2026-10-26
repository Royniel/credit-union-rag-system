"""Answer composition: rerank retrieved chunks, enforce evidence and qualifiers, and decline when unsupported."""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from datetime import date
from typing import Callable, Optional

from . import authority
from . import text as T
from .query import QUALIFIER_OF, RATE_WORDS, SOFT_QUALIFIERS, URL_WORDS, Query, analyze
from .store import ChunkStore, Record

REFUSAL = ("I don't know. The website does not list that information, so I can't confirm it. "
           "Please contact the credit union directly for help.")

_ADDRESS = re.compile(r"\b\d{2,6}\s+[A-Z0-9][\w.']*(?:\s+[\w.']+){0,4}\s+(?:St|Street|Rd|Road|Ave|Avenue|Ln|Lane|Dr|Drive|"
                      r"Blvd|Way|Hwy|Highway|Ct|Court|Pkwy|Pl|Plaza|Cir)\b\.?", re.I)
_RATE_STATEMENT = re.compile(
    r"\d+(?:\.\d+)?%\s*(?:-\s*\d+(?:\.\d+)?%\s*)?(?:intro(?:ductory)?\s+)?(?:apr|apy|annual percentage|dividend|interest|fixed|variable)"
    r"|\b(?:apr|apy|rate)\)?\s*(?:of|is|:|at)?\s*\d+(?:\.\d+)?%", re.I)
_DATE_TEXT = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:,\s*\d{4})?|\b\d{1,2}/\d{1,2}/\d{2,4}\b", re.I)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\s*\n\s*")
_NUMBER = re.compile(r"\d")
_PHONE = re.compile(r"\b\d{3}-\d{3}-\d{4}\b|\(\d{3}\)\s?\d{3}-\d{4}")
_ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth")
_GENERIC = frozenset(RATE_WORDS | set(T.stem_all((
    "account", "loan", "much", "cost", "fee", "offer", "program", "service", "branch", "member", "people",
    "use", "hour", "time", "keep", "same", "go", "new", "still", "way", "help", "make", "sure", "option",
    "pay", "earn", "charge", "receive", "tier"))))
BOILERPLATE_PAGES = 8


@dataclass
class Scored:
    score: float
    record: Record


@dataclass
class Composed:
    text: str
    records: list[Record]
    note: Optional[str] = None


class Answerer:
    def __init__(self, store: ChunkStore) -> None:
        self.store = store
        self._qual_cache: dict[int, frozenset] = {}

    # ---------- public ----------

    def answer(self, question: str, as_of: Optional[date]) -> tuple[Optional[Composed], Optional[str]]:
        q = analyze(question)
        latest = self.store.snapshots[-1].taken_on if self.store.snapshots else None
        if as_of is None and q.as_of and latest and q.as_of < latest:
            as_of = q.as_of
        snap = self.store.snapshot_for(as_of)
        if snap is None:
            return None, None
        if any(not self.store.known(a) for a in q.anchors):
            return None, snap.snapshot_id

        composed = self._compose(q, self._rank(q, snap.index), snap.index)
        if composed is None and q.wants_hours and re.search(r"\b(open|close)", q.raw, re.I):
            # "When does the X branch close?" can mean a permanent closure rather than daily hours.
            q = dataclasses.replace(q, wants_hours=False, wants_date=True)
            composed = self._compose(q, self._rank(q, snap.index), snap.index)
        if composed and snap.index != len(self.store.snapshots) - 1:
            composed.text = f"As of {long_date(snap.taken_on)}: {composed.text}"
        return composed, snap.snapshot_id

    # ---------- ranking ----------

    def _rank(self, q: Query, snap_idx: int) -> list[Scored]:
        hits = self.store.search(q.weights, snap_idx, limit=400)
        today = self.store.snapshots[snap_idx].taken_on
        out = []
        for bm25, rec in hits:
            factor = rec.page_weight * self._constraint_factor(q, rec) * self._evidence_factor(q, rec)
            if rec.chunk.ends and rec.chunk.ends < today:
                factor *= 0.05      # the page itself says the offer is over
            if not rec.crawled_in(snap_idx):
                factor *= 0.85      # carried forward from an earlier crawl
            if rec.page_date and rec.page_class in (authority.NEWS, authority.BLOG):
                factor *= 0.8 + 0.2 * min(1.0, max(0.0, (rec.page_date.year - 2015) / 11))
            if self.store.page_spread(rec) >= BOILERPLATE_PAGES:
                factor *= 0.6
            if rec.lead:
                factor *= 1.3
            elif rec.kind == "text" and len(rec.text) < 140 and not (q.wants_hours or q.wants_money):
                factor *= 0.85
            # Chunks missing the question's rarest terms are demoted even when frequent terms pile up.
            factor *= 0.6 + 0.4 * self._coverage(q, rec)
            factor *= 1.0 + 0.3 * self._title_match(q, rec)
            out.append(Scored(bm25 * factor, rec))
        out.sort(key=lambda s: (-s.score, s.record.id))
        return out

    def _qualifiers_of(self, rec: Record) -> frozenset:
        cached = self._qual_cache.get(rec.id)
        if cached is None:
            c = rec.chunk
            source = f"{c.label} {c.caption} {c.heading} {rec.title.split('|')[0]} {rec.url.split('dupaco.com', 1)[-1]}"
            if rec.kind == "text":
                source += " " + c.text[:160]
            toks = T.tokenize(source.replace("-", " ").replace("/", " "))
            quals = {QUALIFIER_OF[t] for t in toks if t in QUALIFIER_OF}
            if "money" in toks and "market" in toks:
                quals.add("money market")
            if "market" in toks and "money" not in toks:
                quals.discard("money market")
            cached = frozenset(quals)
            self._qual_cache[rec.id] = cached
        return cached

    def _constraint_factor(self, q: Query, rec: Record) -> float:
        factor = 1.0
        quals = self._qualifiers_of(rec)
        for missing in q.qualifiers - quals:
            factor *= 0.6 if missing in SOFT_QUALIFIERS else 0.08
        for extra in quals - q.qualifiers:
            if extra in SOFT_QUALIFIERS:
                factor *= 0.9
            else:
                factor *= 0.75 if q.wants_list else 0.4
        if rec.kind != "row":
            return factor
        label = f"{rec.chunk.label} {rec.chunk.caption}"
        if q.terms:
            units = {u for _, u in q.terms}
            same_unit = {t for t in T.terms(label) if t[1] in units}
            if same_unit:
                factor *= 1.6 if q.terms & same_unit else 0.1
        if q.amounts:
            span = T.amount_range(rec.chunk.label)
            if span:
                lo, hi = span
                if all(lo - 1 <= a <= hi + 1 for a in q.amounts):
                    factor *= 1.5 + (0.3 if hi != float("inf") else 0)
                else:
                    factor *= 0.1
        if q.years:
            span = T.year_span(label)
            if span:
                factor *= 1.6 if all(span[0] <= y <= span[1] for y in q.years) else 0.1
        return factor

    def _evidence_factor(self, q: Query, rec: Record) -> float:
        text = rec.text
        factor = 1.0
        if q.wants_rate:
            if rec.kind == "row" and "%" in text:
                factor *= 1.5
            elif not _RATE_STATEMENT.search(text):
                factor *= 0.2
            if rec.page_class not in authority.RATE_AUTHORITIES:
                factor *= 0.3
        if q.wants_hours:
            factor *= 1.4 if T.has_time(text) else 0.15
        if q.wants_address:
            factor *= 1.4 if _ADDRESS.search(text) else 0.3
        if q.wants_date:
            factor *= 1.3 if _DATE_TEXT.search(text) else 0.5
        if q.wants_count:
            factor *= 2.0 if _counts(text, q.count_noun) else 0.3
        if q.wants_money:
            factor *= 1.3 if re.search(r"\$\s?\d", text) else 0.5
        if q.wants_phone:
            factor *= 1.5 if _PHONE.search(text) else 0.4
        if q.wants_routing:
            factor *= 1.6 if re.search(r"\b\d{9}\b", text) else 0.3
        if (q.wants_address or q.wants_hours) and q.mentions_branch:
            factor *= 1.0 if T.stem("branch") in rec.context_tokens else 0.5
        if rec.kind == "text" and not q.years:
            mentioned = [y for y in T.years(text) if 1990 <= y <= 2030]
            if mentioned and max(mentioned) <= self._latest_year() - 3:
                factor *= 0.7
        if q.wants_url and (rec.chunk.links or rec.page_class in (authority.CORE, authority.OFFICIAL)):
            factor *= 1.2
        return factor

    def _title_match(self, q: Query, rec: Record) -> float:
        content = {t for t in q.tokens if t not in _GENERIC and not t[0].isdigit()}
        if len(content) < 2:
            return 0.0
        title = set(T.tokenize(rec.title.split("|")[0]))
        return len(content & title) / len(content)

    def _latest_year(self) -> int:
        return self.store.snapshots[-1].taken_on.year if self.store.snapshots else date.today().year

    # ---------- composition ----------

    def _compose(self, q: Query, ranked: list[Scored], snap_idx: int) -> Optional[Composed]:
        pool = ranked
        if q.wants_rate:
            pool = [s for s in ranked if s.record.page_class in authority.RATE_AUTHORITIES
                    and not self._missing_hard_qualifier(q, s.record)
                    and (("%" in s.record.text and s.record.kind == "row") or _RATE_STATEMENT.search(s.record.text))]
        elif q.wants_hours:
            pool = [s for s in ranked if T.has_time(s.record.text)]
        elif q.wants_address:
            pool = [s for s in ranked if _ADDRESS.search(s.record.text)]
        if q.anchors:
            pool = [s for s in pool if self._covers(s.record, q.anchors)]
        if not pool:
            return None
        top = self._prefer_newest(pool)
        evidence_backed = q.wants_rate or q.wants_hours or q.wants_address
        if self._coverage(q, top.record) < (0.25 if evidence_backed else 0.3):
            return None

        if q.wants_change and top.record.kind == "row":
            history = self._history(top.record)
            if history:
                return history
        if q.wants_rate and self._contradicts_constraints(q, top.record):
            return None
        if q.wants_rate and top.record.kind == "row":
            return self._compose_rows(q, top, pool, snap_idx)
        if q.wants_rate and q.wants_list:
            return self._compose_list(q, top, pool)
        if q.wants_url:
            return self._compose_url(q, top, pool)
        if q.wants_count and q.count_noun:
            counted = self._compose_count(q, pool)
            if counted:
                return counted
        return self._compose_passage(q, top, pool, snap_idx)

    @staticmethod
    def _contradicts_constraints(q: Query, rec: Record) -> bool:
        """The best evidence is for a different term, tier or model year than asked: the asked one is not published."""
        label = f"{rec.chunk.label} {rec.chunk.caption}" if rec.kind == "row" else rec.text
        if q.terms:
            units = {u for _, u in q.terms}
            found = {t for t in T.terms(label) if t[1] in units}
            if not q.terms & found and (found or rec.kind == "text"):
                return True
        if rec.kind != "row":
            return False
        span = T.amount_range(rec.chunk.label) if q.amounts else None
        if span and not all(span[0] - 1 <= a <= span[1] + 1 for a in q.amounts):
            return True
        years = T.year_span(label) if q.years else None
        return bool(years and not all(years[0] <= y <= years[1] for y in q.years))

    def _missing_hard_qualifier(self, q: Query, rec: Record) -> bool:
        return any(m not in SOFT_QUALIFIERS for m in q.qualifiers - self._qualifiers_of(rec))

    @staticmethod
    def _covers(rec: Record, anchors: list[str]) -> bool:
        return all(a in rec.context_tokens for a in anchors)

    def _coverage(self, q: Query, rec: Record) -> float:
        ignore = _GENERIC | URL_WORDS if q.wants_url else _GENERIC
        content = [t for t in dict.fromkeys(q.tokens) if t not in ignore and not t[0].isdigit()]
        if not content:
            return 1.0
        total = sum(self.store.idf(t) for t in content) or 1.0
        got = sum(self.store.idf(t) for t in content if t in rec.context_tokens
                  or any(s in rec.context_tokens for s in T.SYNONYMS.get(t, ())))
        return got / total

    @staticmethod
    def _prefer_newest(pool: list[Scored]) -> Scored:
        """Among near-tied, near-duplicate candidates, the most recently published one wins."""
        top = pool[0]
        top_tokens = top.record.context_tokens
        best = top
        for cand in pool[1:12]:
            if cand.score < 0.8 * top.score:
                break
            union = top_tokens | cand.record.context_tokens
            overlap = len(top_tokens & cand.record.context_tokens) / max(1, len(union))
            if overlap >= 0.35 and _recency(cand.record) > _recency(best.record):
                best = cand
        return best

    def _compose_rows(self, q: Query, top: Scored, pool: list[Scored], snap_idx: int) -> Composed:
        scores = {s.record.id: s.score for s in pool}
        same_table = [r for r in self.store.rows_of_table(top.record, snap_idx) if r.id in scores]
        same_table.sort(key=lambda r: (-scores[r.id], r.id))
        if q.superlative:
            eligible = [r for r in same_table if scores[r.id] >= 0.5 * top.score] or [top.record]
            pick = max if q.superlative == "max" else min
            chosen = [pick(eligible, key=lambda r: (_primary_value(r), -r.id))]
        elif q.percents:
            chosen = [r for r in same_table if scores[r.id] >= 0.4 * top.score][:8] or [top.record]
        elif q.wants_list:
            chosen = self._list_rows(top, pool, snap_idx)
        else:
            chosen = [r for r in same_table if scores[r.id] >= 0.8 * top.score][:8] or [top.record]
        chosen, dropped = _dedupe_facts(chosen)
        body = "; ".join(r.text for r in chosen) + "."
        if q.percents:
            asked = q.percents[0]
            found = any(asked in T.percents(r.chunk.values) for r in chosen)
            body = (f"Yes, {asked:g}% is listed. " if found else f"No, {asked:g}% is not one of the listed rates. ") + body
        eff = top.record.chunk.effective
        if eff:
            body += f" (Rates effective {eff:%m/%d/%Y}.)"
        return Composed(body, self._with_corroboration(chosen, pool, snap_idx), dropped)

    def _list_rows(self, top: Scored, pool: list[Scored], snap_idx: int, limit: int = 12) -> list[Record]:
        """Whole tables, in ranking order, for questions that ask for every rate of a product family."""
        scores = {s.record.id: s.score for s in pool}
        out, seen_tables = [], set()
        for cand in pool[:60]:
            rec = cand.record
            if cand.score < 0.55 * top.score or len(out) >= limit:
                break
            if rec.kind != "row" or (rec.url, rec.chunk.table) in seen_tables:
                continue
            seen_tables.add((rec.url, rec.chunk.table))
            rows = sorted((r for r in self.store.rows_of_table(rec, snap_idx)
                           if scores.get(r.id, 0) >= 0.55 * top.score), key=lambda r: r.id)
            out.extend(rows[: limit - len(out)])
        return out or [top.record]

    def _compose_list(self, q: Query, top: Scored, pool: list[Scored]) -> Composed:
        picked, headings = [], set()
        for cand in pool[:30]:
            if cand.score < 0.35 * top.score or len(picked) >= 6:
                break
            key = cand.record.chunk.heading or cand.record.chunk.label
            if key in headings:
                continue
            headings.add(key)
            picked.append(cand.record)
        parts = [self._labelled(r, self._best_sentences(q, r, need="rate")) for r in picked]
        return Composed(" ".join(parts), picked)

    def _compose_url(self, q: Query, top: Scored, pool: list[Scored]) -> Composed:
        qtoks = set(q.tokens) | {s for t in q.tokens for s in T.SYNONYMS.get(t, ())}
        best_link, best_score = None, 0.0
        for cand in pool[:20]:
            for label, url in cand.record.chunk.links:
                if not url.startswith("http"):
                    continue
                host = re.sub(r"^https?://", "", url).split("/")[0]
                overlap = len(qtoks & set(T.tokenize(label))) + 2 * len(qtoks & set(T.tokenize(url.replace(".", " ").replace("/", " "))))
                overlap += 3 * len(qtoks & set(T.tokenize(host.replace(".", " "))))
                score = overlap * (cand.score / top.score) ** 0.5
                if score > best_score:
                    best_link, best_score = (url, cand.record), score
        rec = top.record
        urls = [rec.url.split("#")[0]]
        for cand in pool[1:4]:
            page = cand.record.url.split("#")[0]
            if page not in urls and cand.record.page_class in (authority.OFFICIAL, authority.CORE) \
                    and cand.score >= 0.7 * top.score:
                urls.append(page)
                break
        if best_link:
            link_url, link_rec = best_link
            own_overlap = 2 * len(qtoks & set(T.tokenize(rec.url.replace("/", " ").replace("-", " ")))) + 1
            if best_score >= own_overlap:
                urls.insert(0, link_url)
                rec = link_rec
            elif link_url.split("#")[0] not in urls:
                urls.append(link_url)
        links = " (also " + ", ".join(urls[1:3]) + ")" if len(urls) > 1 else ""
        return Composed(f"{self._best_sentences(q, rec)} Link: {urls[0]}{links}", [rec])

    def _compose_passage(self, q: Query, top: Scored, pool: list[Scored], snap_idx: int) -> Composed:
        need = ("time" if q.wants_hours else "date" if q.wants_date else "number" if q.wants_count
                else "money" if q.wants_money else "phone" if q.wants_phone else None)
        if q.wants_hours and (q.wants_list or re.search(r"\b(same|differ|vary|varies|each|every)\b", q.raw, re.I)):
            return self._compose_schedules(top, pool)
        if q.wants_hours or q.wants_address:
            return Composed(self._labelled(top.record, top.record.text), self._with_corroboration([top.record], pool, snap_idx))
        floor = min(0.4, self._coverage(q, top.record))
        peers = [top] + [c for c in pool[:6] if c is not top and c.score >= 0.35 * top.score
                         and self._coverage(q, c.record) >= floor]
        if top.record.page_class in (authority.NEWS, authority.BLOG):
            # An article tells one story; its key sentence may sit in any paragraph.
            seen = {c.record.id for c in peers}
            peers += [Scored(top.score, r) for r in self.store.page_records(top.record.url, snap_idx)
                      if r.kind == "text" and r.id not in seen and self.store.page_spread(r) < BOILERPLATE_PAGES]
        best_key, best = None, (top.record, top.record.text)
        for cand in peers:
            weight = (cand.score / top.score) ** 0.5
            for key, text in self._sentence_options(q, cand.record, need):
                key = (key[0], key[1] * weight)
                if best_key is None or key > best_key:
                    best_key, best = key, (cand.record, text)
        rec, body = best
        return Composed(self._labelled(rec, body), self._with_corroboration([rec], pool, snap_idx))

    @staticmethod
    def _compose_schedules(top: Scored, pool: list[Scored], limit: int = 3) -> Composed:
        """Distinct published schedules, for questions about whether hours differ between branches."""
        picked, seen = [], set()
        for cand in pool[:20]:
            if cand.score < 0.5 * top.score or len(picked) >= limit:
                break
            times = tuple(re.findall(r"\d{1,2}(?::\d{2})?\s?[ap]\.?m\.?|noon", cand.record.text, re.I))
            if times and times not in seen:
                seen.add(times)
                picked.append(cand.record)
        picked = picked or [top.record]
        body = " | ".join(r.text for r in picked)
        if len(picked) > 1:
            body = f"Hours vary by branch; {len(picked)} published schedules: " + body
        return Composed(body, picked)

    def _compose_count(self, q: Query, pool: list[Scored]) -> Optional[Composed]:
        """Counts drift over time (branches open and close), so the newest relevant statement wins."""
        found = [(cand, value) for cand in pool[:80] for value in _count_values(cand.record.text, q.count_noun)]
        if not found:
            return None
        floor = 0.35 * max(c.score for c, _ in found)
        current = self.store.snapshots[-1].taken_on
        cand, value = max(((c, v) for c, v in found if c.score >= floor),
                          key=lambda cv: (cv[0].record.page_date is not None, cv[0].record.page_date or current,
                                          cv[0].score, -cv[0].record.id))
        sentence = next((x.strip() for x in _SENTENCE_SPLIT.split(cand.record.text)
                         if value in _count_values(x, q.count_noun)), cand.record.text)
        return Composed(sentence, [cand.record])

    # ---------- helpers ----------

    def _best_sentences(self, q: Query, rec: Record, need: Optional[str] = None) -> str:
        options = self._sentence_options(q, rec, need)
        return max(options, key=lambda o: o[0])[1] if options else rec.text

    def _sentence_options(self, q: Query, rec: Record, need: Optional[str]) -> list[tuple[tuple, str]]:
        """Each sentence of a chunk, expanded with its follow-on sentences, with a ranking key."""
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(rec.text) if len(s.strip()) > 2]
        if len(sentences) <= 1:
            return [((True, 1.0), rec.text)]
        check: Callable[[str], bool] = {
            "rate": lambda s: bool(_RATE_STATEMENT.search(s)), "time": T.has_time,
            "date": lambda s: bool(_DATE_TEXT.search(s)), "number": lambda s: _counts(s, q.count_noun),
            "money": lambda s: bool(re.search(r"\$\s?\d", s)), "phone": lambda s: bool(_PHONE.search(s)),
        }.get(need, lambda s: True)
        qtoks = set(q.tokens) | {s for t in q.tokens for s in T.SYNONYMS.get(t, ())}
        lines = _is_list_like(rec.text)
        definitional = bool(re.match(r"\s*(what is|what are|what's|tell me about|explain|describe)\b", q.raw, re.I))
        options = []
        for i, sentence in enumerate(sentences):
            toks = set(T.tokenize(sentence, keep_stopwords=True))
            base = sum(self.store.idf(t) for t in qtoks & toks)
            if self.store.org_token in toks:
                base += 0.5
            chosen = [sentence]
            nxt = i + 1
            # FAQ questions, lead-ins and fragments carry their answer in the following sentences.
            while nxt < len(sentences) and len(chosen) < 3 and (
                    chosen[-1].endswith(("?", ":")) or (not lines and len(" ".join(chosen)) < 90)
                    or not check(" ".join(chosen)) or (definitional and len(chosen) < 2)):
                chosen.append(sentences[nxt])
                nxt += 1
            text = " ".join(chosen)
            options.append(((check(text), base - 0.01 * i), text))
        return options

    @staticmethod
    def _labelled(rec: Record, body: str) -> str:
        heading = rec.chunk.heading.split(" > ")[-1] if rec.chunk.heading else ""
        title = re.split(r"\s[|:]\s?|:\s", rec.title)[0].strip()
        if heading and len(heading) <= 60 and heading.lower() not in body.lower():
            body = f"{heading}: {body}"
        if (rec.page_class in (authority.OFFICIAL, authority.CORE) and title and len(title) <= 50
                and title.lower() not in body.lower() and not re.search(r"\d", title)):
            body = f"{title} - {body}"
        return body

    def _with_corroboration(self, chosen: list[Record], pool: list[Scored], snap_idx: Optional[int] = None) -> list[Record]:
        """Cite independent current pages that state the same values, then exact mirrors of the answer."""
        out, urls = [], set()
        for r in chosen:
            if r.url not in urls:
                out.append(r)
                urls.add(r.url)
        key_values = set(re.findall(r"\d[\d,.]*%|\b\d{6,}\b", chosen[0].text)) if chosen else set()
        trusted = (authority.OFFICIAL, authority.CORE)
        if key_values:
            texts = {r.text_key for r in chosen}
            for cand in pool[:80]:
                rec = cand.record
                if len(urls) >= 4:
                    break
                if rec.url in urls or rec.page_class not in trusted or rec.text_key in texts:
                    continue
                if key_values <= set(re.findall(r"\d[\d,.]*%|\b\d{6,}\b", rec.text)):
                    out.append(rec)
                    urls.add(rec.url)
        if snap_idx is not None:
            for r in chosen[:1]:
                for mirror in self.store.same_text(r, snap_idx):
                    if len(urls) >= 4:
                        break
                    if mirror.url not in urls and mirror.page_class in trusted:
                        out.append(mirror)
                        urls.add(mirror.url)
        return out

    def _history(self, rec: Record) -> Optional[Composed]:
        """How one table row's rates moved across snapshots of the same page."""
        label = _norm_label(rec.chunk.label)
        caption = set(T.tokenize(rec.chunk.caption))
        versions = {r.id: r for i in range(len(self.store.snapshots)) for r in self.store.page_records(rec.url, i)
                    if r.kind == "row" and _norm_label(r.chunk.label) == label
                    and (not caption or caption & set(T.tokenize(r.chunk.caption)))}
        versions = sorted(versions.values(), key=lambda r: r.snapshots()[0])
        steps: list[Record] = []
        for r in versions:
            if not steps or r.chunk.rates != steps[-1].chunk.rates:
                steps.append(r)
        if len(steps) < 2:
            return None
        parts = [f"{r.chunk.values} (from {long_date(self.store.snapshots[r.snapshots()[0]].taken_on)})" for r in steps]
        name = rec.chunk.label if rec.chunk.caption.lower() in rec.chunk.label.lower() else f"{rec.chunk.caption} {rec.chunk.label}"
        return Composed(f"{name}: " + ", then ".join(parts) + f". Current: {steps[-1].chunk.values}.", [rec])


def _counts(text: str, noun: Optional[str]) -> bool:
    """True when a number directly quantifies the noun, as in '22 branch office locations'."""
    if not noun:
        return bool(_NUMBER.search(text))
    return bool(_count_values(text, noun))


def _count_values(text: str, noun: Optional[str]) -> list[str]:
    if not noun:
        return []
    quantity = r"\d{1,4}(?:,\d{3})?|" + "|".join(_ORDINALS)
    pattern = rf"\b({quantity})[ \t]+(?:[A-Za-z-]+[ \t]+){{0,2}}{re.escape(noun)}"
    return [m.group(1).replace(",", "").lower() for m in re.finditer(pattern, text, re.I)]


def long_date(d: date) -> str:
    return f"{d:%B} {d.day}, {d.year}"


def _is_list_like(text: str) -> bool:
    lines = [ln for ln in text.split("\n") if ln.strip()]
    return len(lines) >= 3 and sum(len(ln) for ln in lines) / len(lines) < 70


def _recency(rec: Record) -> date:
    return rec.chunk.effective or rec.page_date or date.min


def _primary_value(rec: Record) -> float:
    vals = T.percents(rec.chunk.values)
    return vals[-1] if vals else 0.0


def _norm_label(label: str) -> str:
    label = re.sub(r"\s+", " ", T.normalize(label).lower().replace(",", ""))
    return re.sub(r"\bmonths\b", "month", label).strip()


def _dedupe_facts(rows: list[Record]) -> tuple[list[Record], Optional[str]]:
    """Keep one row per fact; when two sources disagree keep the newest, most authoritative one."""
    best: dict[str, Record] = {}
    order: list[str] = []
    notes = []
    for r in rows:
        key = _norm_label(re.sub(r"\b(19|20)\d{2}\s*-\s*\d{2,4}\b", "", r.chunk.label))
        cur = best.get(key)
        if cur is None:
            best[key] = r
            order.append(key)
            continue
        if r.chunk.values == cur.chunk.values:
            continue
        winner, loser = (r, cur) if (r.page_weight, _recency(r)) > (cur.page_weight, _recency(cur)) else (cur, r)
        best[key] = winner
        notes.append(f"Ignored '{loser.chunk.values}' from {loser.url} in favour of {winner.url} "
                     "(newer or more authoritative).")
    return [best[k] for k in order], (" ".join(notes) or None)
