"""Source authority: how much a page should be trusted for current facts."""
from __future__ import annotations

import re
from datetime import date
from typing import Optional
from urllib.parse import urlparse

from .text import parse_date

OFFICIAL, CORE, MARKETING, NEWS, DOCUMENT, BLOG, CALCULATOR, JUNK = (
    "official", "core", "marketing", "news", "document", "blog", "calculator", "junk")

WEIGHTS = {
    OFFICIAL: 1.0, CORE: 0.92, MARKETING: 0.62, NEWS: 0.58, DOCUMENT: 0.5,
    BLOG: 0.45, CALCULATOR: 0.3, JUNK: 0.15,
}

# Page classes allowed to state a current rate. Calculators, blogs and news repeat stale numbers.
RATE_AUTHORITIES = frozenset({OFFICIAL, CORE, MARKETING, DOCUMENT})

_CORE_PREFIXES = ("/contact", "/about", "/membership", "/connect", "/online-and-mobile-banking",
                  "/mobile", "/shine", "/disclosures", "/careers", "/foundation", "/community",
                  "/scholarships", "/financial-services")
_OFFICIAL_PREFIXES = ("/rates", "/business-rates", "/personal", "/business", "/savings-account",
                      "/loans", "/checking")
_URL_DATE = re.compile(r"/(20\d{2}|19\d{2})/(\d{2})(?:/(\d{2}))?/")


def classify(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/").lower() or "/"
    if parsed.query and "campaignid" in parsed.query.lower():
        return JUNK
    if "privacy-policy" in path or "subscription" in path or "photo-release" in path:
        return JUNK
    if "calculator" in path:
        return CALCULATOR
    if path.startswith("/newsroom"):
        return NEWS
    if path.startswith(("/learn", "/dupacodaily", "/resources", "/business/resources")) or _URL_DATE.search(path + "/"):
        return BLOG
    if path.startswith("/hubfs"):
        return OFFICIAL if re.search(r"fee-schedule|rate-sheet|rate-and-fee|truth-in-savings", path) else DOCUMENT
    if path == "/" or path.startswith(_CORE_PREFIXES):
        return CORE
    if path.startswith(_OFFICIAL_PREFIXES):
        return OFFICIAL
    return MARKETING


def weight(url: str) -> float:
    return WEIGHTS[classify(url)]


def page_date(url: str, markdown: str) -> Optional[date]:
    """Publication date of dated content (news, blog posts), from the URL or the byline."""
    m = _URL_DATE.search(urlparse(url).path + "/")
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3) or 1))
        except ValueError:
            pass
    if classify(url) in (NEWS, BLOG):
        return parse_date(markdown[:1500])
    return None
