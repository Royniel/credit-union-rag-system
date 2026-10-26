"""Temporal knowledge system for credit-union website snapshots.

build() returns an object with ingest(), answer() and conflicts() as defined in harness/contract.py.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from datetime import date
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _locate_contract() -> None:
    """The scorer puts harness/ on sys.path; fall back to searching for it when run standalone."""
    if importlib.util.find_spec("contract") is not None:
        return
    candidates = [os.environ.get("HACKATHON_HARNESS", "")]
    candidates += [str(d / "harness") for d in _HERE.parents]
    for path in candidates:
        if path and (Path(path) / "contract.py").exists():
            sys.path.insert(0, path)
            return
    raise ImportError("harness/contract.py not found; set HACKATHON_HARNESS to the kit's harness folder")


_locate_contract()

from contract import Answer, Conflict, Snapshot, Source  # noqa: E402

from kb.answerer import REFUSAL, Answerer  # noqa: E402
from kb.conflicts import ConflictFinder  # noqa: E402
from kb.store import ChunkStore  # noqa: E402


def _to_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


class TemporalKnowledgeSystem:
    def __init__(self) -> None:
        self._store = ChunkStore()
        self._answerer = Answerer(self._store)
        self._conflicts: Optional[list[Conflict]] = None

    def ingest(self, snapshot: Snapshot) -> None:
        taken_on = _to_date(snapshot.taken_on) or date.min
        self._store.add_snapshot(snapshot.snapshot_id, taken_on, snapshot.pages)
        self._conflicts = None

    def answer(self, question: str, as_of: Optional[str] = None) -> Answer:
        try:
            composed, snapshot_id = self._answerer.answer(question, _to_date(as_of))
        except Exception:  # an unexpected failure must never surface as a confident answer
            composed, snapshot_id = None, None
        if composed is None:
            return Answer(text=REFUSAL, sources=[], confident=False)
        sources = [Source(url=r.url, snapshot_id=snapshot_id, quote=r.text[:200]) for r in composed.records[:4]]
        return Answer(text=composed.text, sources=sources, confident=True, conflict_note=composed.note)

    def conflicts(self) -> list[Conflict]:
        if self._conflicts is None:
            finder = ConflictFinder(self._store, Conflict, Source)
            self._conflicts = finder.find()
        return list(self._conflicts)


def build() -> TemporalKnowledgeSystem:
    return TemporalKnowledgeSystem()
