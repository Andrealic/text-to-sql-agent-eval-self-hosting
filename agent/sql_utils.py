"""Small SQL text helpers used by the agent."""
from __future__ import annotations

import re


def split_statements(sql: str) -> list[str]:
    """Split SQL into statements, ignoring semicolons inside quoted strings.

    Returns the non-empty statements, each stripped of surrounding whitespace and
    the separating semicolon. A plain str.split(";") would break a query that
    legitimately contains ";" inside a literal, e.g. WHERE x = 'a;b'.
    """
    statements: list[str] = []
    in_single = in_double = False
    start = 0
    for i, ch in enumerate(sql):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == ";" and not in_single and not in_double:
            stmt = sql[start:i].strip()
            if stmt:
                statements.append(stmt)
            start = i + 1
    tail = sql[start:].strip()
    if tail:
        statements.append(tail)
    return statements


def first_statement(sql: str) -> str:
    """Return the first runnable statement, or an empty string if there is none."""
    statements = split_statements(sql)
    return statements[0] if statements else ""


def extract_sql(text: str) -> str:
    """Pull a single runnable SQL statement out of an LLM reply.

    If there is a closed ```sql block, take its contents. Otherwise drop lines
    that are just stray fence markers. In both cases, keep only the first
    statement so trailing prose/fences cannot trip sqlite's one-statement rule.
    """
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        body = fenced.group(1)
    else:
        body = "\n".join(
            line for line in text.splitlines()
            if not re.fullmatch(r"\s*```(?:sql)?\s*", line, re.IGNORECASE)
        )
    return first_statement(body.strip())
