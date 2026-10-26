"""Split a page into retrievable chunks: one per table row, plus short passages of prose."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from .text import clean_markdown, parse_date, tidy

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_SEP = re.compile(r"^\|?\s*:?-{3,}")
_EFFECTIVE = re.compile(r"effective (?:as of )?(\d{1,2}/\d{1,2}/\d{2,4})", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_NOISE = re.compile(r"^(?:\* \* \*|✕|Continue Cancel|Cancel|Loading.*|Go|Zoom in|Zoom out|\d+ Results)$")

_HEADER_ALIASES = (
    (re.compile(r"annual percentage yield\s*\(apy\)", re.I), "APY"),
    (re.compile(r"annual percentage rate\s*\(apr\)", re.I), "APR"),
)

_SKIP_COLUMNS = re.compile(r"payment example|^points$", re.I)
_RATE_COLUMN = re.compile(r"\b(rate|apr|apy|dividend|yield)\b", re.I)
_DECIMAL = re.compile(r"\d+\.\d+|\d+(?=\s?%)")

_INLINE_FACT = re.compile(r"^(?!\d{1,2}:\d)([A-Za-z0-9][^:|]{2,80}?):\s*(.{1,80})$")
_EXPIRED = re.compile(r"\b(?:ended|expired|ends|expires|valid (?:through|until)|through)\s+(?:on\s+)?"
                      r"(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|[A-Z][a-z]{2,8}\.? \d{1,2},? \d{4})", re.I)

MAX_PASSAGE = 520
SHORT_BLOCK = 140


@dataclass
class Chunk:
    kind: str                      # "row" or "text"
    text: str                      # display text
    heading: str = ""
    caption: str = ""
    label: str = ""                # row identity (non-value cells)
    values: str = ""               # row value cells
    table: str = ""                # table identity within the page
    header: str = ""
    rates: tuple = ()              # (kind, value) for each rate, APR, APY or dividend cell
    effective: Optional[date] = None
    ends: Optional[date] = None    # stated end date of an offer
    links: list[tuple[str, str]] = field(default_factory=list)


def _header_name(cell: str) -> str:
    cell = tidy(cell)
    for pattern, alias in _HEADER_ALIASES:
        cell = pattern.sub(alias, cell)
    return cell


def _is_value_cell(cell: str) -> bool:
    return bool(re.search(r"\d\s?%|^\$\s?[\d,.]+$|^[\d.]+$", cell))


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _sentences(text: str, limit: int) -> list[str]:
    parts, buf = [], ""
    for sent in _SENTENCE.split(text):
        if buf and len(buf) + len(sent) > limit:
            parts.append(buf)
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf:
        parts.append(buf)
    return parts


def chunk_page(markdown: str) -> list[Chunk]:
    text, links = clean_markdown(markdown)
    lines = text.splitlines()
    chunks: list[Chunk] = []
    headings: list[str] = []
    recent_lines: list[str] = []      # short lines seen since the last table, used as table captions
    buffer: list[str] = []

    def heading_path() -> str:
        return " > ".join(headings[-2:])

    def flush() -> None:
        if not buffer:
            return
        lines = [tidy(b) for b in buffer]
        body = tidy(" \n".join(buffer))
        buffer.clear()
        if len(body) < 3:
            return
        facts = [_inline_fact(ln, heading_path()) for ln in lines]
        if lines and all(facts):
            chunks.extend(facts)
            return
        if _is_price_list(lines):
            # Fee schedules: every "Item $X" line is its own fact, titled by the nearest sub-header.
            group = heading_path()
            for line in lines:
                if "$" not in line and len(line) < 60:
                    group = line
                    continue
                chunks.append(Chunk("text", line, heading=group, links=_links_in(line, links)))
            return
        pieces = [body] if len(body) <= MAX_PASSAGE else _sentences(body.replace(" \n", " "), MAX_PASSAGE)
        for piece in pieces:
            chunks.append(Chunk("text", piece, heading=heading_path(), links=_links_in(piece, links),
                                ends=offer_end(piece)))

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        heading = _HEADING.match(line)
        if heading:
            flush()
            level, title = len(heading.group(1)), tidy(heading.group(2))
            del headings[max(0, level - 1):]
            headings.extend([""] * (level - 1 - len(headings)))
            headings.append(title)
            headings[:] = [h for h in headings if h]
            recent_lines.clear()
            i += 1
            continue

        if line.startswith("|") and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1].strip()):
            flush()
            header = [_header_name(c) for c in _split_row(line)]
            caption = " ".join(recent_lines[-1:]) or (headings[-1] if headings else "")
            table_id = f"{heading_path()}:{caption}:{'|'.join(header)}"
            j, rows = i + 2, []
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append(_split_row(lines[j]))
                j += 1
            effective = None
            for k in range(j, min(j + 8, len(lines))):
                m = _EFFECTIVE.search(lines[k])
                if m:
                    effective = parse_date(m.group(1))
                    break
                if lines[k].strip().startswith("|") or _HEADING.match(lines[k].strip()):
                    break
            for cells in rows:
                chunks.append(_row_chunk(cells, header, caption, heading_path(), table_id, effective))
            recent_lines.clear()
            i = j
            continue

        if not line or _NOISE.match(line):
            if not line and buffer and sum(len(b) for b in buffer) >= SHORT_BLOCK * 2:
                flush()
            i += 1
            continue

        if len(line) <= 60:
            recent_lines.append(tidy(line))
        if len(line) > MAX_PASSAGE:
            flush()
        buffer.append(line)
        if sum(len(b) for b in buffer) >= MAX_PASSAGE:
            flush()
        i += 1
    flush()
    return chunks


def _row_chunk(cells, header, caption, heading, table_id, effective) -> Chunk:
    cells = [tidy(c) for c in cells]
    width = max(len(header), len(cells))
    header = header + [""] * (width - len(header))
    cells = cells + [""] * (width - len(cells))
    label_parts, value_parts, rates = [], [], []
    for name, cell in zip(header, cells):
        if not cell or cell in ("-", "\u2014") or _SKIP_COLUMNS.search(name):
            continue
        if _is_value_cell(cell) and label_parts:
            value_parts.append(f"{name} {cell}".strip())
            if _RATE_COLUMN.search(name) or "%" in cell:
                first = _DECIMAL.search(cell)
                if first:
                    rates.append((rate_kind(name), float(first.group())))
        else:
            label_parts.append(cell)
    label = " | ".join(label_parts)
    if caption and caption.lower() not in label.lower():
        shown = f"{caption}: {label}"
    else:
        shown = label
    values = ", ".join(value_parts)
    display = f"{shown}: {values}" if values else shown
    return Chunk("row", display, heading=heading, caption=caption, label=label, values=values,
                 table=table_id, effective=effective, header=" ".join(header), rates=tuple(rates))


def _is_price_list(lines: list[str]) -> bool:
    priced = sum(1 for ln in lines if re.search(r"\$\s?\d", ln) and len(ln) < 160)
    return len(lines) >= 4 and priced >= 0.5 * len(lines)


def _inline_fact(line: str, heading: str) -> Optional[Chunk]:
    """'6-month certificate APY: 3.50%' style statements become single-row facts."""
    m = _INLINE_FACT.match(line)
    if not m or not re.search(r"\d\s?%|\$\s?\d", m.group(2)):
        return None
    label, value = m.group(1).strip(), m.group(2).strip()
    first = _DECIMAL.search(value)
    rates = ((rate_kind(label + " " + value), float(first.group())),) if first and "%" in value else ()
    return Chunk("row", line, heading=heading, caption=heading.split(" > ")[-1] if heading else "",
                 label=label, values=value, table=f"{heading}:inline", header=label, rates=rates,
                 ends=offer_end(line))


def offer_end(text: str) -> Optional[date]:
    m = _EXPIRED.search(text)
    return parse_date(m.group(1)) if m else None


def rate_kind(header: str) -> str:
    """Canonical name of a rate column so header rewording across snapshots does not split a fact."""
    low = header.lower()
    if re.search(r"intro|1st \d+ months|first \d+ months", low):
        return "intro"
    if "apy" in low or "yield" in low:
        return "APY"
    if "apr" in low or "annual percentage rate" in low:
        return "APR"
    if "dividend" in low:
        return "dividend rate"
    return "rate"


def _links_in(piece: str, links: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(label, url) for label, url in links if label and label in piece][:6]
