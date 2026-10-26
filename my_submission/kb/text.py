"""Text normalization, tokenization and value parsing shared by every stage."""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

_TRANSLATE = str.maketrans({
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\xa0": " ", "\u200b": "",
})

_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_DATA_URI = re.compile(r"data:[a-z]+/[^)\s]*")
_LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]*)[^)]*\)")
_ESCAPE = re.compile(r"\\([\\*_#\[\]()$|+\-.!~`>])")
_LITERAL_NEWLINES = re.compile(r"(\\\\?r)?\\\\?n")
_EMPHASIS = re.compile(r"\*\*|__")
_FOOTNOTE = re.compile(r"(%|\bAPR|\bAPY|\))(\d)\b")
_SPACES = re.compile(r"[ \t]+")

MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
MONTHS["sept"] = 9

NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "fifteen": "15",
    "eighteen": "18", "twenty": "20", "thirty": "30", "forty": "40", "sixty": "60",
}

STOPWORDS = frozenset("""
a an and are as at be been but by can could do does for from had has have how i if in into is it
its me my no not of on or our please so that the their them then there these they this to us was
we were what when where which who why will with would you your yours dupaco dupacos credit union
tell know like get give want need right now today currently current also any some just about
much many am id im let see find show provide hi hello thank thanks there s t m ve ll re d going
""".split())

_RAW_SYNONYMS = {
    "cd": ("certificate",), "certificate": ("cd",), "mortgage": ("home", "conventional"),
    "interest": ("rate",), "car": ("auto", "vehicle"), "vehicle": ("auto",), "auto": ("vehicle",),
    "locator": ("location",), "atm": ("location",), "phone": ("call",),
    "thru": ("up",), "through": ("up",), "card": ("visa",),
    "website": ("online",), "cost": ("fee", "charge"), "minimum": ("min",), "maximum": ("limit", "max"), "age": ("under", "older"), "member": ("join",), "college": ("student",), "university": ("student",), "price": ("fee",), "charge": ("fee",), "join": ("membership", "member"), "eligible": ("membership",),
}

_TOKEN = re.compile(r"[a-z]+|\d+(?:[.,]\d+)*")



def normalize(text: str) -> str:
    return (text or "").translate(_TRANSLATE)


def clean_markdown(md: str) -> tuple[str, list[tuple[str, str]]]:
    """Strip images and markup noise, keep link text inline and return the links separately."""
    text = normalize(md)
    text = _DATA_URI.sub("", text)
    text = _IMAGE.sub("", text)
    links: list[tuple[str, str]] = []

    def _link(m: re.Match) -> str:
        label, url = m.group(1).strip(), m.group(2).strip()
        if url and not url.startswith(("tel:", "mailto:", "#", "javascript")):
            links.append((label, url))
        return label

    text = _LINK.sub(_link, text)
    text = _LITERAL_NEWLINES.sub(" ", text)
    text = _ESCAPE.sub(r"\1", text)
    return text, links


def tidy(text: str) -> str:
    """Display form: no emphasis markers, footnote digits or doubled spaces."""
    text = _EMPHASIS.sub("", text)
    text = _FOOTNOTE.sub(r"\1", text)
    return _SPACES.sub(" ", text).strip()


def stem(word: str) -> str:
    """Plural and -ing stripping, then truncation so 'recognized' and 'recognition' meet."""
    if len(word) <= 3:
        return word
    for suffix, repl in (("ies", "y"), ("sses", "ss"), ("ches", "ch"), ("shes", "sh"), ("xes", "x")):
        if word.endswith(suffix):
            word = word[: -len(suffix)] + repl
            break
    else:
        if word.endswith("s") and not word.endswith(("ss", "us", "is")):
            word = word[:-1]
    if word.endswith("ing") and len(word) >= 7:
        word = word[:-3]
    return word[:6] if len(word) >= 7 else word


def norm_number(tok: str) -> str:
    tok = tok.replace(",", "")
    if "." in tok:
        whole, _, frac = tok.partition(".")
        frac = frac.rstrip("0")
        tok = f"{whole}.{frac}" if frac else whole
    return tok


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    out = []
    for tok in _TOKEN.findall(normalize(text).lower()):
        if tok[0].isdigit():
            tok = norm_number(tok)
        elif tok in STOPWORDS and not keep_stopwords:
            continue
        else:
            tok = stem(NUMBER_WORDS.get(tok, tok))
        if keep_stopwords or tok not in STOPWORDS:
            out.append(tok)
    return out


def words_to_digits(text: str) -> str:
    return re.sub(r"\b(" + "|".join(NUMBER_WORDS) + r")\b",
                  lambda m: NUMBER_WORDS[m.group(1).lower()], text, flags=re.I)


# ---------- value parsing ----------

_TERM = re.compile(r"\b(\d{1,3})\s*-?\s*(month|mo|year|yr)s?\b", re.I)
_AMOUNT = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(k|m|million|thousand)?\b|\b(\d+(?:\.\d+)?)\s?(k)\b", re.I)
_YEAR_SPAN = re.compile(r"\b((?:19|20)\d{2})\s*-\s*(\d{2}|\d{4})\b")
_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s?%")
_TIME = re.compile(r"\b\d{1,2}(?::\d{2})?\s?(?:a\.?m\.?|p\.?m\.?)|\bnoon\b", re.I)
_MDY = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")
_ISO = re.compile(r"\b(20\d{2}|19\d{2})-(\d{2})-(\d{2})\b")
_LONG_DATE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+(\d{1,2}),?\s+(\d{4})\b", re.I)


def terms(text: str) -> set[tuple[int, str]]:
    out = set()
    for n, unit in _TERM.findall(words_to_digits(text)):
        out.add((int(n), "month" if unit.lower().startswith("mo") else "year"))
    return out


def amounts(text: str) -> list[float]:
    vals = []
    for m in _AMOUNT.finditer(text):
        raw, mult = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        val = float(raw.replace(",", ""))
        mult = (mult or "").lower()
        if mult in ("k", "thousand"):
            val *= 1_000
        elif mult in ("m", "million"):
            val *= 1_000_000
        vals.append(val)
    return vals


def amount_range(label: str) -> Optional[tuple[float, float]]:
    """Balance or loan-amount tier of a table label, e.g. '$10,000 - $24,999.99' or '$300,000 or less'."""
    vals = amounts(label)
    if not vals:
        return None
    low = label.lower()
    if len(vals) >= 2:
        return min(vals[:2]), max(vals[:2])
    if re.search(r"or less|and under|up to|below|under", low):
        return 0.0, vals[0]
    if "+" in label or re.search(r"and (up|over|above)|or more|over", low):
        return vals[0], float("inf")
    m = re.search(r"\$\s?[\d,.]+\s*-\s*([\d,.]+)", label)
    if m:
        return vals[0], float(m.group(1).replace(",", ""))
    return vals[0], vals[0]


def year_span(label: str) -> Optional[tuple[int, int]]:
    """Model-year tier, e.g. '2024-27 Model Year', 'Model Year 2026-23', '2016 Model Year (and older)'."""
    m = _YEAR_SPAN.search(label)
    if m:
        a = int(m.group(1))
        b = int(m.group(2)) if len(m.group(2)) == 4 else int(m.group(1)[:2] + m.group(2))
        return min(a, b), max(a, b)
    m = _YEAR.search(label)
    if m and "model" in label.lower():
        y = int(m.group(1))
        if "older" in label.lower() or "earlier" in label.lower():
            return 1900, y
        if "newer" in label.lower():
            return y, 2100
        return y, y
    return None


def years(text: str) -> list[int]:
    return [int(y) for y in _YEAR.findall(text)]


def percents(text: str) -> list[float]:
    return [float(p) for p in _PERCENT.findall(text)]


def has_time(text: str) -> bool:
    return bool(_TIME.search(text))


def parse_date(text: str) -> Optional[date]:
    m = _ISO.search(text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    m = _MDY.search(text)
    if m:
        mo, d, y = (int(g) for g in m.groups())
        y = y + 2000 if y < 100 else y
        try:
            return date(y, mo, d)
        except ValueError:
            return None
    m = _LONG_DATE.search(text)
    if m:
        try:
            return date(int(m.group(3)), MONTHS[m.group(1).lower()[:3]], int(m.group(2)))
        except (ValueError, KeyError):
            return None
    return None


def strip_dates(text: str) -> str:
    """Remove calendar expressions so their years are not read as product qualifiers."""
    text = _LONG_DATE.sub(" ", text)
    text = _MDY.sub(" ", text)
    month = "|".join(sorted(MONTHS, key=len, reverse=True))
    return re.sub(rf"\b(?:{month})\.?,?\s+(?:of\s+)?(?:19|20)\d{{2}}\b", " ", text, flags=re.I)


def stem_all(words) -> tuple:
    return tuple(stem(w) for w in words)


SYNONYMS = {stem(k): stem_all(v) for k, v in _RAW_SYNONYMS.items()}
