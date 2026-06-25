"""Tests for targeted evidence collection.

Run standalone:
    uv run python tests/test_evidence_node.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent.graph as graph  # noqa: E402
from agent.execution import ExecutionResult  # noqa: E402
from agent.graph import AgentState, evidence_node  # noqa: E402


class _Resp:
    def __init__(self, content: str):
        self.content = content


class _LLM:
    def invoke(self, _messages):
        return _Resp("SELECT DISTINCT gender FROM client; SELECT COUNT(*) FROM client GROUP BY gender")


def test_evidence_node_accumulates_and_dedupes_queries():
    state = AgentState(
        question="How many male clients?",
        db_id="financial",
        schema="CREATE TABLE client(gender TEXT)",
        sql="SELECT COUNT(*) FROM client WHERE gender = 'male'",
        verify_issue="literal may use display value instead of stored code",
        needs_evidence=True,
        evidence_questions=["Check distinct client.gender values"],
        findings={},
        evidence=[{
            "issue": "prior",
            "questions": ["Check distinct client.gender values"],
            "findings": {"SELECT DISTINCT gender FROM client": "columns: gender\nM\nF"},
        }],
    )
    state.execution = ExecutionResult(ok=True, rows=[(0,)], columns=["COUNT(*)"], row_count=1)

    original_llm = graph.llm
    original_execute_sql = graph.execute_sql
    graph.llm = lambda *a, **k: _LLM()
    graph.execute_sql = lambda _db, sql: ExecutionResult(
        ok=True,
        rows=[("M", 339), ("F", 197)],
        columns=["gender", "count"],
        row_count=2,
    )
    try:
        out = evidence_node(state)
    finally:
        graph.llm = original_llm
        graph.execute_sql = original_execute_sql

    assert out["needs_evidence"] is False
    assert out["evidence_questions"] == []
    assert len(out["evidence"]) == 2
    new_findings = out["evidence"][-1]["findings"]
    assert "SELECT DISTINCT gender FROM client" not in new_findings
    assert "SELECT COUNT(*) FROM client GROUP BY gender" in new_findings
    assert out["history"][-1]["node"] == "evidence"


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
