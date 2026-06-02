"""Own the step-level trace - the reliable core attendees inspect.

Each step emits a structured record (query, mode, candidates + scores, signal
readings, decision, action, latency, tokens). This is what the notebook renders as
a table/timeline.

trace.py is dependency-light and duck-typed (it reads attributes off retrieval /
signals / policy objects) so it never hard-imports them.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class StepRecord:
    step: int
    query_text: str
    sub_queries: list
    mode: str
    candidates: list   # [{rank, doc_id, title, score, is_gold}]
    signals: list      # [{name, value, threshold, fired}]
    decision: str
    action: str
    reason: str
    latency_ms: float
    tokens: int = 0
    note: str = ""

    @classmethod
    def from_objects(cls, step, result, report, decision, tokens=0, gold_doc_ids=None, note=""):
        """Build a record from a RetrievalResult, a SignalReport, and a Decision
        (all read via duck typing)."""
        gold = set(gold_doc_ids or [])
        candidates = [
            {
                "rank": i + 1,
                "doc_id": c.doc_id,
                "title": c.title,
                "score": round(float(c.score), 4),
                "is_gold": c.doc_id in gold,
            }
            for i, c in enumerate(result.candidates)
        ]
        signals = [
            {"name": r.name, "value": round(float(r.value), 4), "threshold": round(float(r.threshold), 4), "fired": bool(r.fired)}
            for r in report.readings
        ]
        action = getattr(decision.action, "value", str(decision.action))
        return cls(
            step=step,
            query_text=result.query,
            sub_queries=list(result.sub_queries),
            mode=result.mode,
            candidates=candidates,
            signals=signals,
            decision=action,
            action=action,
            reason=decision.reason,
            latency_ms=round(float(result.latency_ms), 1),
            tokens=tokens,
            note=note,
        )


@dataclass
class Trace:
    question: str
    question_id: str = ""
    answerable: bool | None = None
    gold_doc_ids: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    answer: str = ""
    stopped: bool = False

    def add(self, record: StepRecord) -> None:
        self.steps.append(record)

    # --- summaries (for the wrap / eval) ---
    @property
    def tool_calls(self) -> int:
        return len(self.steps)

    @property
    def total_latency_ms(self) -> float:
        return round(sum(s.latency_ms for s in self.steps), 1)

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens for s in self.steps)

    @property
    def final_doc_ids(self) -> list:
        return [c["doc_id"] for c in self.steps[-1].candidates] if self.steps else []

    def gold_hits(self) -> int:
        return len(set(self.final_doc_ids) & set(self.gold_doc_ids))

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "question_id": self.question_id,
            "answerable": self.answerable,
            "gold_doc_ids": self.gold_doc_ids,
            "answer": self.answer,
            "stopped": self.stopped,
            "tool_calls": self.tool_calls,
            "total_latency_ms": self.total_latency_ms,
            "total_tokens": self.total_tokens,
            "steps": [asdict(s) for s in self.steps],
        }

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kw)
