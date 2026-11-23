from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class AgentResult:
    ok: bool
    artifacts: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


@dataclass
class QAResult:
    ok: bool
    score: float
    issues: list = field(default_factory=list)
    details: dict = field(default_factory=dict)


class Agent:
    name = "agent"
    role = ""

    def run(self, context: dict) -> AgentResult:
        raise NotImplementedError


class QualityAgent:
    name = "qualite"
    role = ""
    threshold = 0.7

    def check(self, context: dict) -> QAResult:
        raise NotImplementedError


def qa_summary(results: list) -> tuple:
    issues = [f"[{r.name}] {i}" for r in results for i in r.issues]
    score = min(r.score for r in results) if results else 1.0
    ok = all(r.ok for r in results)
    return ok, score, issues


def retry_stage(runner: Callable, attempts: int = 2):
    last = None
    for _ in range(attempts):
        last = runner()
        if last is not None and last.ok:
            return last
    return last
