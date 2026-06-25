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
- Sample values and metrics gathered from the database are provided.
- When filtering or matching on a text column, compare case-insensitively with
  UPPER() on both sides, e.g. WHERE UPPER("col") = UPPER('value'), because the
  stored capitalization often differs from how the question phrases it.
"""


# Available placeholders: {schema}, {findings}, {question}
GENERATE_SQL_USER = """Schema:
{schema}

Data exploration (each block is an exploration query and its real result):
{findings}

Question: {question}
"""

VERIFY_SYSTEM = """You are a strict SQLite result verifier: decide whether the executed SQL result answers the question using ONLY the provided schema, SQL, execution result, and data exploration.

Output format - THIS IS MANDATORY:
- Reply with a SINGLE raw JSON object and NOTHING else. No prose, no explanation, no markdown, no ``` fences.
- Exact shape: {"ok": <true|false>, "issue": "<empty string if ok, otherwise a short reason>"}
- "ok" must be a JSON boolean (true/false), not a sentence.

Decision rules:
- Do NOT use outside/world knowledge to reject a result. If the database says a race, school, label, code, or
  location has a certain value, treat that as the source of truth unless the provided exploration contradicts it.
- Be strict about the shape of the answer. If the question asks for specific columns/fields, the result must return
  exactly those fields, with no extra explanatory/id/name columns unless the question asked for them.
- Column order matters when the question lists fields in an order (for example Street, City, Zip, State). Reject if
  the SQL returns the right fields in a different order.
- If the question asks for one scalar such as a count, average, percentage, or difference, reject multi-column answers
  that include intermediate totals or helper values.
- Reject successful SQL whose result is NULL/None/empty for a requested scalar unless the schema/exploration makes
  that NULL clearly expected.
- Reject if the SQL errored; returned 0 rows when the question or exploration implies rows exist; uses filters,
  joins, inclusive/exclusive bounds, date handling, or literals that do not match the question and explored data.
- Otherwise set ok=true.

Do NOT describe the result in words. Output ONLY the JSON object."""

VERIFY_USER = """Question: {question}
SQL executed:
{sql}

Execution result:
{execution}

Data exploration (each block is an exploration query and its real result):
{findings}
"""

EXPLORE_SYSTEM = """You are a data analyst. BEFORE the final query is written, you explore the
database to understand the data needed to answer the question - exactly as an analyst would
poke at the tables first. Propose READ-ONLY exploration queries that reveal: the distinct
values actually stored in the relevant columns, their exact format (date strings, codes), a
few sample rows, and counts.

Rules:
- Output ONLY SQL SELECT statements separated by ';'. No prose, no markdown.
- Each must be a single read-only SELECT. Keep them small with LIMIT.
- A handful of focused queries is enough.
- When filtering or matching on a text column, compare case-insensitively with
  UPPER() on both sides, e.g. WHERE UPPER("col") = UPPER('value'), because the
  stored capitalization often differs from how the question phrases it.
"""

EXPLORE_USER = """Schema:
{schema}
Question: {question}
Return read-only SELECT exploration queries (separated by ';') to understand the data needed to answer it."""

REVISE_SYSTEM = """Fix the SQL query based on the verifier feedback and the gathered information. Output ONLY corrected SQL in ```sql block.
When filtering or matching on a text column, compare case-insensitively with UPPER() on both sides
(e.g. WHERE UPPER("col") = UPPER('value')) - stored capitalization often differs from the question."""

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
Information gathered from the database (each block is an exploration query and its real result; use the real values/formats shown here):
{findings}

Write a corrected SELECT."""