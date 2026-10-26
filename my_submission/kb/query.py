"""Question analysis: intent, qualifiers and numeric constraints."""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from . import text as T

# Qualifiers that make a product a different fact. A chunk carrying one the member did not ask
# for is another product (jumbo vs standard, business vs personal, refinance vs purchase).
QUALIFIERS = {
    "jumbo": ("jumbo",), "business": ("business", "commercial"), "refinance": ("refinance", "refi"),
    "roth": ("roth",), "traditional": ("traditional",), "ira": ("ira", "retirement"),
    "kids": ("kid", "dopack", "youth"), "student": ("student",), "money market": ("market",),
    "high-yield": ("high",), "holiday": ("holiday",), "launchpad": ("launchpad",),
    "bright": ("bright",), "you name it": ("name",), "investor": ("investor",), "achieve": ("achieve",),
    "recreational": ("recreational", "boat", "rv", "camper", "atv", "snowmobile", "ski", "watercraft"),
    "home equity": ("equity", "heloc"), "share-secured": ("secured",), "credit coach": ("coach",),
    "mini max": ("mini", "max"), "rewards": ("reward",), "intro": ("intro", "introductory", "transfer"),
    "adjustable": ("adjustable", "arm"), "construction": ("construction",), "land": ("land", "lot"),
    "hsa": ("hsa", "health"), "rate relief": ("relief",),
}
QUALIFIER_OF = {T.stem(tok): name for name, toks in QUALIFIERS.items() for tok in toks}
# Qualifiers whose absence is a product's default reading and never disqualifies it.
SOFT_QUALIFIERS = frozenset({"rewards", "intro", "ira"})

RATE_WORDS = frozenset(T.stem_all(("rate", "apr", "apy", "dividend", "interest", "yield", "percent")))
LIST_WORDS = frozenset(T.stem_all(("offer", "option", "type", "kind", "all", "list", "available", "choice", "every")))
SUPERLATIVE_MAX = frozenset(T.stem_all(("highest", "best", "top", "most", "largest")))
SUPERLATIVE_MIN = frozenset(T.stem_all(("lowest", "cheapest", "least", "smallest")))
CHANGE_WORDS = ("change", "changed", "previous", "previously", "used to", "go up", "went up",
                "go down", "went down", "increase", "decrease", "raised", "lowered", "before")
URL_WORDS = frozenset(T.stem_all(("link", "url", "website", "webpage", "page", "site", "online")))


@dataclass
class Query:
    raw: str
    tokens: list[str]
    weights: dict[str, float]
    qualifiers: set[str]
    terms: set[tuple[int, str]]
    amounts: list[float]
    years: list[int]
    percents: list[float]
    anchors: list[str]
    as_of: Optional[date] = None
    wants_rate: bool = False
    wants_hours: bool = False
    wants_address: bool = False
    wants_url: bool = False
    wants_count: bool = False
    wants_list: bool = False
    wants_change: bool = False
    wants_date: bool = False
    wants_money: bool = False
    wants_phone: bool = False
    wants_routing: bool = False
    mentions_branch: bool = False
    superlative: Optional[str] = None
    count_noun: Optional[str] = None
    flags: set[str] = field(default_factory=set)


_CAPITAL = re.compile(r"\b([A-Z][a-zA-Z$'-]+(?:\s+[A-Z][a-zA-Z$'-]+)*)")
_GENERIC_CAPS = frozenset("""
i dupaco dupacos what how when where which who why can could do does is are visa apr apy
cd cds ira atm usa us october november december january february march april may june july
august september monday tuesday wednesday thursday friday saturday sunday url faq pin id ssn ach
fdic ncua hsa llc usa
""".split())
_GENERIC_CAPS = frozenset(_GENERIC_CAPS | set(T.stem_all(_GENERIC_CAPS)))


def _with_equivalent_terms(terms: set) -> set:
    """'7-year' and '84-month' name the same loan term."""
    out = set(terms)
    for n, unit in terms:
        if unit == "year" and n <= 30:
            out.add((n * 12, "month"))
        elif unit == "month" and n % 12 == 0:
            out.add((n // 12, "year"))
    return out


def parse_as_of(question: str) -> Optional[date]:
    """Explicit 'as of May 2026' style dates in the question text."""
    month = "|".join(sorted(T.MONTHS, key=len, reverse=True))
    m = re.search(rf"\b(?:as of|in|during|back in|on)\s+({month})\.?\s+(?:(\d{{1,2}}),?\s+)?((?:19|20)\d{{2}})\b",
                  question, re.I)
    if not m:
        return None
    mo, yr = T.MONTHS[m.group(1).lower()[:3]], int(m.group(3))
    day = int(m.group(2)) if m.group(2) else calendar.monthrange(yr, mo)[1]
    try:
        return date(yr, mo, day)
    except ValueError:
        return None


def analyze(question: str) -> Query:
    q = T.normalize(question)
    plain = T.strip_dates(q)
    digits = T.words_to_digits(plain)
    toks = T.tokenize(digits)
    low = f" {plain.lower()} "

    weights: dict[str, float] = {}
    for tok in toks:
        weights[tok] = weights.get(tok, 0.0) + 1.0
        for syn in T.SYNONYMS.get(tok, ()):
            weights.setdefault(syn, 0.4)
    for tok in list(weights):
        if tok in RATE_WORDS:
            weights[tok] = min(weights[tok], 0.6)
    for pct in T.percents(q):
        weights.pop(T.norm_number(f"{pct}"), None)

    quals = {QUALIFIER_OF[t] for t in toks if t in QUALIFIER_OF}
    if "money" in toks and "market" in toks:
        quals.add("money market")
    if re.search(r"\b\d/\d\b", q):
        quals.add("adjustable")

    pcts = T.percents(q)
    anchors = []
    for m in _CAPITAL.finditer(q):
        if m.start() == 0 or re.search(r"[.?!]\s*$", q[:m.start()]):
            continue
        phrase_tokens = [t for t in T.tokenize(m.group(1)) if t not in _GENERIC_CAPS and not t.isdigit()]
        anchors.extend(phrase_tokens)

    tokset = set(toks)
    query = Query(
        raw=question, tokens=toks, weights=weights, qualifiers=quals,
        terms=_with_equivalent_terms(T.terms(digits)), amounts=[a for a in T.amounts(plain)],
        years=[y for y in T.years(plain) if 1990 <= y <= 2030], percents=pcts,
        anchors=list(dict.fromkeys(anchors)), as_of=parse_as_of(q),
    )
    query.wants_url = (bool(tokset & URL_WORDS) and bool(re.search(r"\b(link|url|website|web page|site|where can i|online)\b", low))
                       ) or bool(re.search(r"\bwhere (?:can|do|should) i (?:find|see|look|go|check)\b", low))
    query.wants_hours = "hour" in tokset or bool(re.search(r"\bwhat time\b|\bwhen\b.*\b(open|close)s?\b(?!.*\b(permanently|for good)\b)", low)
                                                 and re.search(r"\b(lobby|lobbies|drive|branch|branches|office)\b", low))
    query.wants_address = T.stem("address") in tokset or bool(re.search(r"\bwhere is\b|\blocated\b", low))
    query.wants_count = bool(re.search(r"\bhow many\b|\bwhat age\b|\bhow old\b|\bhow long\b", low))
    m = re.search(r"how many ([a-z-]+)(?: ([a-z-]+))?", low)
    if m:
        nouns = [t for t in T.tokenize(f"{m.group(1)} {m.group(2) or ''}") if t]
        query.count_noun = nouns[0] if nouns else None
    deposit_pays = bool(re.search(r"\b(pay|pays|paying|earn|earns|earning)\b", low)) and bool(
        re.search(r"\b(certificate|cd|savings|checking|money market|ira|account|share)s?\b", low))
    query.wants_rate = (bool(tokset & RATE_WORDS) or bool(pcts) or deposit_pays) and not query.wants_url
    query.wants_date = bool(re.search(
        r"^\s*when\b|\bwhat date\b|\bstill (open|available|offered)\b|\bexpir|"
        r"\b(is|are|will)\b[^,]*\bclos(e|es|ed|ing)\b[^,]*\?\s*$|\bclos(e|es|ing) (on|date)\b", low)) and not query.wants_hours
    query.wants_routing = bool(re.search(r"\brouting\b|\baba\b", low))
    query.mentions_branch = bool(re.search(r"\b(branch|branches|location|office)\b", low))
    query.wants_phone = bool(re.search(r"\bphone\b|\bcall\b|\btelephone\b|\bnumber to (call|reach)\b", low)) and not query.wants_rate
    query.wants_money = bool(re.search(r"\bhow much\b|\bfees?\b|\bcosts?\b|\bcharge[sd]?\b|\bprice\b", low)) and not query.wants_rate
    query.wants_change = any(f" {w}" in low for w in CHANGE_WORDS)
    if tokset & SUPERLATIVE_MAX:
        query.superlative = "max"
    elif tokset & SUPERLATIVE_MIN:
        query.superlative = "min"
    specific = query.terms or query.amounts or query.years
    plural_rates = bool(re.search(r"\b(rates|aprs|apys|cards|accounts|loans|certificates|cds)\b", low))
    query.wants_list = bool(tokset & LIST_WORDS) or (plural_rates and not specific and not query.superlative)
    return query
