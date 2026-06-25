"""Regression test for the SQL-extraction fix (agent/sql_utils.py).

Pins the exact failure we hit in the baseline eval: the "Coldsnap" question.
The model returned the query followed by a stray ``` fence on its own line and
NO opening fence, so the old _extract_sql returned the whole reply - trailing
fence included - and sqlite raised:

    ProgrammingError: You can only execute one statement at a time.

These tests assert the extractor now returns a single clean statement, and -
the end-to-end "monitoring" part - that the extracted SQL actually executes
against the real card_games DB instead of erroring.

Run standalone (no pytest needed):
    uv run python tests/test_extract_sql.py
Or, if pytest is installed:
    uv run pytest tests/test_extract_sql.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.execution import execute_sql  # noqa: E402
from agent.sql_utils import extract_sql, first_statement, split_statements  # noqa: E402

# The exact iteration-0 reply for the Coldsnap question: a valid query, then a
# stray closing fence on its own line, with no opening fence. This is what broke.
COLDSNAP_REPLY = (
    'SELECT c."name"\n'
    'FROM "cards" c\n'
    'JOIN "sets" s ON c."setCode" = s."code"\n'
    'WHERE s."name" = \'Coldsnap\'\n'
    'AND c."convertedManaCost" = (\n'
    '  SELECT MAX(c2."convertedManaCost")\n'
    '  FROM "cards" c2\n'
    '  JOIN "sets" s2 ON c2."setCode" = s2."code"\n'
    '  WHERE s2."name" = \'Coldsnap\'\n'
    ');\n'
    '```'
)


def test_coldsnap_no_fence_leaks_into_sql():
    """The stray ``` must not survive into the extracted SQL."""
    sql = extract_sql(COLDSNAP_REPLY)
    assert "`" not in sql, f"backtick leaked into SQL: {sql!r}"


def test_coldsnap_is_single_statement():
    """No second statement after a ';' (that was the sqlite error)."""
    sql = extract_sql(COLDSNAP_REPLY)
    # Only trailing content after a quote-safe ';' would be a 2nd statement.
    assert first_statement(sql) == sql, f"more than one statement: {sql!r}"


def test_coldsnap_executes_against_real_db():
    """End-to-end: the extracted SQL runs on card_games without the
    'one statement at a time' error and returns a row."""
    sql = extract_sql(COLDSNAP_REPLY)
    result = execute_sql("card_games", sql)
    assert result.ok, f"execution failed: {result.error}"
    assert result.row_count >= 1, "expected at least one card back"


def test_semicolon_inside_quotes_is_preserved():
    """A legitimate ';' inside a string literal must not truncate the query."""
    sql = extract_sql("SELECT * FROM t WHERE x = 'a;b';")
    assert sql == "SELECT * FROM t WHERE x = 'a;b'"


def test_plain_fenced_block_still_works():
    """The normal happy path (closed ```sql block) is unchanged."""
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"


def test_trailing_second_statement_dropped():
    assert extract_sql("SELECT 1; DROP TABLE t;") == "SELECT 1"


def test_split_statements_returns_all():
    """The general splitter keeps every statement (agent only takes the first)."""
    assert split_statements("SELECT 1; SELECT 2; SELECT 3") == ["SELECT 1", "SELECT 2", "SELECT 3"]


def test_split_statements_ignores_semicolon_in_quotes():
    assert split_statements("SELECT 'a;b'; SELECT 2") == ["SELECT 'a;b'", "SELECT 2"]


def test_split_statements_skips_empty_fragments():
    assert split_statements(";; SELECT 1 ;;") == ["SELECT 1"]


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

    # Monitoring view: show what the extractor produced and the DB answer.
    sql = extract_sql(COLDSNAP_REPLY)
    res = execute_sql("card_games", sql)
    print("\n--- Coldsnap monitoring ---")
    print("extracted SQL ends with:", repr(sql[-30:]))
    print("execute ok:", res.ok, "| rows:", res.row_count)
    if res.ok and res.rows:
        print("answer:", res.rows[0])

    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
