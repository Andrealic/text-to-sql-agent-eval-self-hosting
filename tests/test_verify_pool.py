"""Tests for the pooled verifier (3 independent voters, majority vote).

The LLM is stubbed with scripted replies (no network). We check the 2-of-3
majority logic, that a single prose / parse-fail reply can no longer sink a
correct answer, and that on rejection the issues of the dissenting voters are
returned to revise.

Run standalone (no pytest needed):
    uv run python tests/test_verify_pool.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent.graph as graph  # noqa: E402
from agent.execution import ExecutionResult  # noqa: E402
from agent.graph import AgentState, verify_node  # noqa: E402

OK = '{"ok": true, "issue": ""}'
NO = '{"ok": false, "issue": "wrong columns"}'
NO2 = '{"ok": false, "issue": "zero rows"}'
NO_EVIDENCE = (
    '{"ok": false, "issue": "literal may use display value instead of stored code", '
    '"needs_evidence": true, '
    '"evidence_questions": ["Check distinct client.gender values", "Count rows by gender"]}'
)
NO_EVIDENCE_DUP = (
    '{"ok": false, "issue": "stored code is unverified", '
    '"needs_evidence": true, '
    '"evidence_questions": ["Check distinct client.gender values"]}'
)
PROSE = "The query returns **339** male clients in the 'Hl.m. Praha' district."  # parse-fail


class _Resp:
    def __init__(self, content): self.content = content


class _SeqLLM:
    """Fake client: each invoke() returns the next scripted reply."""
    def __init__(self, replies): self._replies = list(replies); self._i = 0
    def invoke(self, _messages):
        r = self._replies[self._i]
        self._i += 1
        return _Resp(r)


def _verify_with(replies: list[str]) -> dict:
    state = AgentState(question="q", db_id="toxicology", schema="", findings={}, sql="SELECT 1")
    state.execution = ExecutionResult(ok=True, rows=[[339]], columns=["c"], row_count=1)
    original = graph.llm
    graph.llm = lambda *a, **k: _SeqLLM(replies)
    try:
        return verify_node(state)
    finally:
        graph.llm = original


def test_unanimous_ok_passes():
    out = _verify_with([OK, OK, OK])
    assert out["verify_ok"] is True
    assert out["verify_issue"] == "none"


def test_majority_ok_passes_2_1():
    out = _verify_with([OK, NO, OK])
    assert out["verify_ok"] is True


def test_majority_reject_fails_and_returns_both_reasons():
    out = _verify_with([NO, OK, NO2])
    assert out["verify_ok"] is False
    # reasons from BOTH rejecting voters are handed to revise
    assert "wrong columns" in out["verify_issue"]
    assert "zero rows" in out["verify_issue"]


def test_single_prose_vote_cannot_sink_correct_answer():
    """The Praha bug: one natural-language (parse-fail) reply is just 1 'not ok'
    vote, so 2 proper ok votes still pass."""
    out = _verify_with([OK, PROSE, OK])
    assert out["verify_ok"] is True


def test_two_parse_fails_reject_with_reason():
    out = _verify_with([PROSE, PROSE, OK])
    assert out["verify_ok"] is False
    assert "could not parse" in out["verify_issue"]


def test_votes_recorded_in_history():
    out = _verify_with([OK, NO, OK])
    entry = out["history"][-1]
    assert entry["node"] == "verify"
    assert len(entry["votes"]) == 3


def test_rejected_votes_can_request_deduped_evidence():
    out = _verify_with([NO_EVIDENCE, OK, NO_EVIDENCE_DUP])
    assert out["verify_ok"] is False
    assert out["needs_evidence"] is True
    assert out["evidence_questions"] == [
        "Check distinct client.gender values",
        "Count rows by gender",
    ]
    entry = out["history"][-1]
    assert entry["needs_evidence"] is True


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
