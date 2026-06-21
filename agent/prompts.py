"""Prompt templates for the agent nodes.

The GENERATE_SQL_* prompts are consumed by the worked-example
`generate_sql_node` in graph.py via `.format(schema=..., question=...)`, so
keep those placeholders intact. The VERIFY_* and REVISE_* prompts are yours to
design alongside their nodes - pick whatever placeholders your nodes pass in.

Filling these in is part of Phase 3.
"""

GENERATE_SQL_SYSTEM = """You are a SQLite expert. Given a schema and a question, write ONE read-only SELECT query.
Rules:
- Output ONLY the SQL inside a ```sql fenced block.
- Use double-quoted identifiers when names have spaces or reserved words.
- No INSERT/UPDATE/DELETE/DROP or other DML and non-read-only statements.
"""


# Available placeholders: {schema}, {question}
GENERATE_SQL_USER = """Schema:
{schema}
Question: {question}
"""

VERIFY_SYSTEM = """You verify whether SQL results answer the user's question.
Reply with ONLY JSON: {"ok": true/false, "issue": "short explanation if not ok or error message, otherwise "none"}
Mark ok=false if: SQL error, 0 rows when the question expects for sure data, wrong columns, clearly wrong answer, unexpected nulls, ."""

VERIFY_USER = """Question: {question}
SQL executed:
{sql}
Execution result:
{execution}
"""

REVISE_SYSTEM = """Fix the SQL query based on the verifier feedback. Output ONLY corrected SQL in ```sql block."""

REVISE_USER = """Schema:
{schema}
Question: {question}
Previous SQL:
{sql}
Execution result:
{execution}
Verifier ok:
{verify_ok}
Verifier issue: 
{verify_issue} 
Write a corrected SELECT."""