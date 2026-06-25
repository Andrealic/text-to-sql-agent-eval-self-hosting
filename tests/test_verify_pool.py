"""Tests for the verifier node.

The LLM is stubbed with scripted replies (no network). We check JSON parsing,
ok/reject routing, parse-fail handling, and targeted evidence requests.

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
NO_EVIDENCE = (
    '{"ok": false, "issue": "literal may use display value instead of stored code", '
    '"needs_evidence": true, '
    '"evidence_questions": ["Check distinct client.gender values", "Count rows by gender"]}'
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


def test_ok_passes():
    out = _verify_with([OK])
    assert out["verify_ok"] is True
    assert out["verify_issue"] == "none"


def test_reject_fails_and_returns_reason():
    out = _verify_with([NO])
    assert out["verify_ok"] is False
    assert "wrong columns" in out["verify_issue"]


def test_parse_fail_rejects_with_reason():
    out = _verify_with([PROSE])
    assert out["verify_ok"] is False
    assert "could not parse" in out["verify_issue"]


def test_votes_recorded_in_history():
    out = _verify_with([OK])
    entry = out["history"][-1]
    assert entry["node"] == "verify"
    assert len(entry["votes"]) == 1


def test_rejected_vote_can_request_evidence():
    out = _verify_with([NO_EVIDENCE])
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
