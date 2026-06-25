"""Tests for the explore (data-analyst) step that runs up front, before generate.

The explore node asks the model for ';'-separated read-only exploration queries,
runs them against the DB, and hands the findings to generate_sql (and later revise).
Here we STUB the LLM call (no network) but run the proposed queries against the real
toxicology DB, so the test proves the parse -> execute -> format pipeline surfaces the
real values the model would otherwise guess wrong: the atom.element codes (e.g. 'ca')
and the molecule.label values ('+'/'-').

Run standalone (no pytest needed):
    uv run python tests/test_explore.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent.graph as graph  # noqa: E402
from agent.execution import execute_sql  # noqa: E402
from agent.graph import AgentState, _render_rows, explore_node  # noqa: E402
from agent.schema import render_schema  # noqa: E402
from agent.sql_utils import extract_statements  # noqa: E402


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    """Stand-in for the chat client: invoke() returns a fixed reply."""

    def __init__(self, content: str):
        self._content = content

    def invoke(self, _messages):
        return _FakeResponse(self._content)


def _run_explore_with_reply(reply: str) -> dict:
    """Run explore_node against the real toxicology DB with a stubbed LLM reply."""
    state = AgentState(
        question="Among molecules with element Calcium, are they carcinogenic?",
        db_id="toxicology",
        schema=render_schema("toxicology"),
    )
    original = graph.llm
    graph.llm = lambda *a, **k: _FakeLLM(reply)
    try:
        return explore_node(state)
    finally:
        graph.llm = original


def test_extract_statements_multiple():
    out = extract_statements("SELECT DISTINCT element FROM atom; SELECT label FROM molecule")
    assert out == ["SELECT DISTINCT element FROM atom", "SELECT label FROM molecule"]


def test_extract_statements_fenced():
    assert extract_statements("```sql\nSELECT 1;\nSELECT 2;\n```") == ["SELECT 1", "SELECT 2"]


def test_explore_surfaces_real_element_and_label_values():
    """The findings dict must contain the real values the agent was missing."""
    reply = (
        "SELECT DISTINCT element FROM atom LIMIT 15; "
        "SELECT label, COUNT(*) FROM molecule GROUP BY label"
    )
    out = _run_explore_with_reply(reply)
    findings = out["findings"]
    assert isinstance(findings, dict), f"findings should be a dict, got {type(findings)}"
    blob = "\n".join(findings.values())  # join the per-query results
    assert "ca" in blob, f"real element code 'ca' not surfaced:\n{blob}"
    assert "+" in blob, f"real label '+' not surfaced:\n{blob}"


def test_explore_findings_keyed_by_query():
    """findings is a {query: result} dict keyed by the exploration queries."""
    out = _run_explore_with_reply("SELECT DISTINCT element FROM atom; SELECT label FROM molecule")
    assert set(out["findings"].keys()) == {
        "SELECT DISTINCT element FROM atom",
        "SELECT label FROM molecule",
    }


def test_explore_records_history_and_bounds_queries():
    out = _run_explore_with_reply("SELECT DISTINCT element FROM atom; SELECT 1; SELECT 2")
    entry = out["history"][-1]
    assert entry["node"] == "explore"
    assert len(entry["findings"]) == 3


def test_explore_handles_no_queries():
    out = _run_explore_with_reply("sorry, no idea")
    # "sorry, no idea" has no ';' so it parses as one bogus statement that errors;
    # the point is explore must not crash and must still produce a findings dict.
    assert isinstance(out["findings"], dict) and out["findings"]


def test_render_rows_truncates_long_cells():
    from agent.execution import ExecutionResult
    long = "x" * 200
    ex = ExecutionResult(ok=True, rows=[(long,)], columns=["c"], row_count=1)
    rendered = _render_rows(ex)
    assert "…" in rendered and len(rendered) < 200


# --- standalone runner (works without pytest) --------------------------------

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

    # Monitoring view: what the explorer actually gathered.
    out = _run_explore_with_reply(
        "SELECT DISTINCT element FROM atom LIMIT 15; "
        "SELECT label, COUNT(*) FROM molecule GROUP BY label"
    )
    print("\n--- explore findings (toxicology) ---")
    for q, r in out["findings"].items():
        print(f"-- {q}\n{r}\n")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
