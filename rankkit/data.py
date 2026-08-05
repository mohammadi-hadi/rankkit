"""Reading ranked lists and click logs from JSONL.

Two record shapes are supported, both one JSON object per line:

    labelled run   {"qid": "q1", "ranking": ["d3", "d1"], "relevance": {"d1": 2}}
    click log      {"qid": "q1", "ranking": ["d3", "d1"], "clicks": ["d1"]}

`ranking` is always the order the documents were placed in, best first. For a
click log it is specifically the order that was *shown to the user*, which is
what position-bias correction needs. Extra fields are ignored.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Run:
    """One query's ranking together with its relevance labels."""

    qid: str
    ranking: tuple[str, ...]
    relevance: dict[str, float]

    def gain(self, doc: str) -> float:
        """Graded relevance of `doc`; unlabelled documents count as 0."""
        return self.relevance.get(doc, 0.0)

    @property
    def n_relevant(self) -> int:
        return sum(1 for grade in self.relevance.values() if grade > 0)


@dataclass(frozen=True)
class ClickQuery:
    """One query's *shown* ranking together with the documents that were clicked."""

    qid: str
    ranking: tuple[str, ...]
    clicks: tuple[str, ...]

    def shown_rank(self, doc: str) -> int:
        """1-based position `doc` occupied in the ranking that was shown."""
        return self.ranking.index(doc) + 1


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Parse a JSONL file into a list of objects, reporting the offending line."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {lineno}: not valid JSON ({exc.msg})") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path} line {lineno}: expected a JSON object")
            records.append(obj)
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def _as_id(value: Any, where: str) -> str:
    """Normalise a document or query id to a string, rejecting booleans."""
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{where}: id must be a string or a number, got {value!r}")
    if isinstance(value, (str, int)):
        text = str(value)
        if not text:
            raise ValueError(f"{where}: id must not be empty")
        return text
    raise ValueError(f"{where}: id must be a string or a number, got {value!r}")


def _as_ranking(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{where}: ranking must be a list of document ids")
    if not value:
        raise ValueError(f"{where}: ranking must not be empty")
    docs = tuple(_as_id(doc, where) for doc in value)
    if len(set(docs)) != len(docs):
        raise ValueError(f"{where}: ranking contains the same document twice")
    return docs


def _as_relevance(value: Any, where: str) -> dict[str, float]:
    """Accept either {doc: grade} or a flat list of relevant document ids."""
    if isinstance(value, list):
        return {_as_id(doc, where): 1.0 for doc in value}
    if not isinstance(value, dict):
        raise ValueError(f"{where}: relevance must be an object of grades or a list of ids")
    grades: dict[str, float] = {}
    for doc, grade in value.items():
        if isinstance(grade, bool) or not isinstance(grade, (int, float)):
            raise ValueError(f"{where}: relevance grade for {doc!r} must be a number")
        if grade < 0:
            raise ValueError(f"{where}: relevance grade for {doc!r} must not be negative")
        grades[_as_id(doc, where)] = float(grade)
    return grades


def extract_runs(
    records: Iterable[dict[str, Any]],
    *,
    qid_key: str = "qid",
    rank_key: str = "ranking",
    rel_key: str = "relevance",
) -> list[Run]:
    """Turn raw JSONL objects into `Run`s, rejecting duplicate query ids."""
    runs: list[Run] = []
    seen: set[str] = set()
    for i, rec in enumerate(records, start=1):
        where = f"record {i}"
        for key in (qid_key, rank_key):
            if key not in rec:
                raise ValueError(f"{where}: missing {key!r}")
        qid = _as_id(rec[qid_key], where)
        if qid in seen:
            raise ValueError(f"{where}: query id {qid!r} appears more than once")
        seen.add(qid)
        ranking = _as_ranking(rec[rank_key], f"{where} (qid {qid})")
        relevance = _as_relevance(rec.get(rel_key, {}), f"{where} (qid {qid})")
        runs.append(Run(qid=qid, ranking=ranking, relevance=relevance))
    return runs


def extract_clicks(
    records: Iterable[dict[str, Any]],
    *,
    qid_key: str = "qid",
    rank_key: str = "ranking",
    click_key: str = "clicks",
) -> list[ClickQuery]:
    """Turn raw JSONL objects into `ClickQuery`s.

    A click on a document that was not in the shown ranking means the log and
    the ranking disagree about what the user saw, which would silently corrupt
    every propensity weight, so it is an error rather than a dropped row.
    """
    queries: list[ClickQuery] = []
    seen: set[str] = set()
    for i, rec in enumerate(records, start=1):
        where = f"record {i}"
        for key in (qid_key, rank_key):
            if key not in rec:
                raise ValueError(f"{where}: missing {key!r}")
        qid = _as_id(rec[qid_key], where)
        if qid in seen:
            raise ValueError(f"{where}: query id {qid!r} appears more than once")
        seen.add(qid)
        ranking = _as_ranking(rec[rank_key], f"{where} (qid {qid})")
        raw_clicks = rec.get(click_key, [])
        if not isinstance(raw_clicks, list):
            raise ValueError(f"{where} (qid {qid}): clicks must be a list of document ids")
        clicks = tuple(_as_id(doc, f"{where} (qid {qid})") for doc in raw_clicks)
        shown = set(ranking)
        for doc in clicks:
            if doc not in shown:
                raise ValueError(
                    f"{where} (qid {qid}): clicked document {doc!r} is not in the shown ranking"
                )
        queries.append(ClickQuery(qid=qid, ranking=ranking, clicks=clicks))
    return queries


def align(
    left: Sequence[Run] | Sequence[ClickQuery],
    right: Sequence[Run] | Sequence[ClickQuery],
) -> tuple[list[Any], list[Any], list[str]]:
    """Restrict two sets of queries to the ids they share, in a stable order.

    Returns the two aligned lists plus the ids that were dropped, so callers can
    report them instead of silently comparing different query sets.
    """
    by_left = {q.qid: q for q in left}
    by_right = {q.qid: q for q in right}
    shared = [qid for qid in by_left if qid in by_right]
    if not shared:
        raise ValueError("the two files share no query ids")
    dropped = sorted(set(by_left) ^ set(by_right))
    return [by_left[q] for q in shared], [by_right[q] for q in shared], dropped
