"""LangGraph agent: text-to-SQL with verify+revise loop.

Graph shape:

    START -> attach_schema -> generate_sql -> execute -> verify
                                                          |
                                              ok=true ----+----> END
                                                          |
                                              ok=false ---+----> revise -> execute -> verify (loop)

Loop is capped at MAX_ITERATIONS total generate/revise calls.

The execute node and the graph wiring are provided. `generate_sql_node` is
filled in as a worked example; you implement `verify`, `revise`, and the
conditional router following the same shape.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_openrouter import ChatOpenRouter

from langgraph.graph import END, START, StateGraph

from agent import prompts
from agent.execution import ExecutionResult, execute_sql
from agent.schema import render_schema
from agent.sql_utils import extract_sql, extract_statements
from pydantic import BaseModel, ValidationError

# Total generate + revise calls before the loop is forced to stop.
# 3-5 is a reasonable range; tune it as part of Phase 3.
MAX_ITERATIONS = 3

#VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
#VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "https://openrouter.ai/api/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "minimax/minimax-m2.5")
# vLLM ignores the key, but a hosted OpenAI-compatible provider needs a real one.
# Lets you point the agent at e.g. OpenAI while iterating without a running vLLM.
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")

class VerifyResult(BaseModel):
    ok: bool = False
    issue: str = ""

@dataclass
class AgentState:
    """State threaded through the graph. Extend with fields you need."""

    question: str
    db_id: str
    schema: str = ""
    sql: str = ""
    execution: ExecutionResult | None = None
    verify_ok: bool = False
    verify_issue: str = ""
    exploration_queries: list[str] = field(default_factory=list)
    findings: dict[str, str] = field(default_factory=dict)
    revise_result: str = ""
    iteration: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


def llm() -> ChatOpenRouter:
    """Chat client pointed at VLLM_BASE_URL (your local vLLM by default)."""
    return ChatOpenRouter(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0.0,
    )


# ---- Nodes ------------------------------------------------------------

def _attach_schema(state: AgentState) -> dict:
    """Provided. Render the DB schema once at the start of the run."""
    return {"schema": render_schema(state.db_id)}


def _parse_verify_json(text: str) -> tuple[bool, str]:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    raw = (fenced.group(1) if fenced else text).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start : end + 1]
    try:
        result = VerifyResult.model_validate_json(raw)
        return result.ok, result.issue
    except ValidationError:
        return False, "could not parse verifier output"

# Bounds for the explore step: keep the LLM-proposed exploration cheap and its
# output small enough to fit the revise prompt.
MAX_EXPLORE_QUERIES = 6
EXPLORE_MAX_ROWS = 15
EXPLORE_MAX_CELL = 80


def _render_rows(execution: ExecutionResult) -> str:
    """Compact rendering of an exploration query result, with long cells truncated."""
    if not execution.ok:
        return f"ERROR: {execution.error}"
    rows = execution.rows or []
    if not rows:
        return "(0 rows)"

    def cell(c: Any) -> str:
        s = "" if c is None else str(c)
        return s if len(s) <= EXPLORE_MAX_CELL else s[:EXPLORE_MAX_CELL] + "…"

    cols = ", ".join(execution.columns or [])
    body = "\n".join(" | ".join(cell(c) for c in r) for r in rows[:EXPLORE_MAX_ROWS])
    more = f"\n... (+{len(rows) - EXPLORE_MAX_ROWS} more rows)" if len(rows) > EXPLORE_MAX_ROWS else ""
    return f"columns: {cols}\n{body}{more}"


def _findings_text(findings: dict[str, str]) -> str:
    """Render the {query: result} findings dict into readable prompt text."""
    if not findings:
        return "(no exploration queries returned)"
    return "\n\n".join(f"-- {query}\n{result}" for query, result in findings.items())


def explore_node(state: AgentState) -> dict:
    """Explore the data up front, like a data analyst, before any SQL is written.

    Runs once after attach_schema. One LLM call proposes read-only exploration
    queries from the question + schema (distinct values, formats, sample rows,
    counts). We parse them with our ';' splitter, run each read-only (bounded +
    truncated), and store the findings so generate_sql (and later revise) filter
    on the real literals/formats instead of guessing.
    """
    response = llm().invoke([
        ("system", prompts.EXPLORE_SYSTEM),
        ("user", prompts.EXPLORE_USER.format(
            schema=state.schema,
            question=state.question,
        )),
    ])
    exploration_queries = extract_statements(response.content)[:MAX_EXPLORE_QUERIES]
    findings = {q: _render_rows(execute_sql(state.db_id, q)) for q in exploration_queries}
    return {
        "exploration_queries": exploration_queries,
        "findings": findings,
        "history": state.history + [{"node": "explore", "exploration_queries": exploration_queries, "findings": findings}],
    }

def generate_sql_node(state: AgentState) -> dict:
    """Worked example - the other LLM nodes follow this same shape.

    Build messages from the prompts, call the shared llm(), extract the SQL,
    and return only the state fields you changed. `iteration` is bumped here
    (and in revise) so route_after_verify can enforce MAX_ITERATIONS.

    This node is wired and ready; fill in GENERATE_SQL_SYSTEM / GENERATE_SQL_USER
    in prompts.py to make it produce real queries.
    """
    response = llm().invoke([
        ("system", prompts.GENERATE_SQL_SYSTEM),
        ("user", prompts.GENERATE_SQL_USER.format(
            schema=state.schema,
            exploration_queries="\n".join(state.exploration_queries),
            findings=_findings_text(state.findings),
            question=state.question,
        )),
    ])
    sql = extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "generate_sql", "sql": sql}],
    }


def execute_node(state: AgentState) -> dict:
    """Provided. Runs the SQL and stores the result."""
    return {"execution": execute_sql(state.db_id, state.sql)}


def verify_node(state: AgentState) -> dict:
    """Decide whether state.execution plausibly answers state.question.

    Follow the generate_sql_node pattern: build messages from the VERIFY_*
    prompts, call llm(), parse the reply. Ask the model for a small JSON object
    like {"ok": bool, "issue": str} and parse it defensively - the model may
    wrap it in prose or fences. state.execution.render() gives you a compact
    view of the rows or error to feed into the prompt.

    Return: {"verify_ok": <str>, "verify_issue": <str>}.
    What counts as "not plausible" is yours to define - see the Phase 3 targets
    in the README.
    """
    response = llm().invoke([
        ("system", prompts.VERIFY_SYSTEM),
        ("user", prompts.VERIFY_USER.format(
            schema=state.schema,
            question=state.question,
            exploration_queries="\n".join(state.exploration_queries),
            findings=_findings_text(state.findings),
            sql=state.sql,
            execution=state.execution.render(),
        )),
    ])
    verify_result = _parse_verify_json(response.content)
    print(f"verify_result: {verify_result}")
    return {
        "verify_ok": verify_result[0],
        "verify_issue": verify_result[1],
        "history": state.history + [{"node": "verify", "verify_ok": verify_result[0], "verify_issue": verify_result[1]}],
        
    }


def revise_node(state: AgentState) -> dict:
    """Produce a revised SQL query given state.verify_issue and the prior attempt.

    Same shape as generate_sql_node, but the prompt also includes the failing
    SQL, its execution result, the verifier's complaint, and the findings the
    explore node gathered. Bump the iteration counter the same way
    generate_sql_node does so the loop terminates.

    Return: {"sql": <str>, "iteration": state.iteration + 1, ...}.
    """
    response = llm().invoke([
        ("system", prompts.REVISE_SYSTEM),
        ("user", prompts.REVISE_USER.format(
            schema=state.schema,
            question=state.question,
            sql=state.sql,
            execution=state.execution.render(),
            verify_ok=state.verify_ok,
            verify_issue=state.verify_issue,
            exploration_queries="\n".join(state.exploration_queries),
            findings=_findings_text(state.findings),
        )),
    ])
    revise_result = extract_sql(response.content)
    return {
    "sql": revise_result,
    "iteration": state.iteration + 1,
    "history": state.history + [{"node": "revise", "sql": revise_result}],
}


def route_after_verify(state: AgentState) -> str:
    """Conditional router: return "revise" to loop, "end" to terminate.

    Two reasons to end: the verifier was happy (state.verify_ok), or you've hit
    the iteration cap (state.iteration >= MAX_ITERATIONS). Otherwise revise -
    the data exploration already happened up front, so revise reuses it.
    """
    if state.verify_ok:
        return "end"
    elif state.iteration >= MAX_ITERATIONS:
        return "end"
    else:
        return "revise"


# ---- Graph wiring -----------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("attach_schema", _attach_schema)
    g.add_node("explore", explore_node)
    g.add_node("generate_sql", generate_sql_node)
    g.add_node("execute", execute_node)
    g.add_node("verify", verify_node)
    g.add_node("revise", revise_node)

    g.add_edge(START, "attach_schema")
    g.add_edge("attach_schema", "explore")
    g.add_edge("explore", "generate_sql")
    g.add_edge("generate_sql", "execute")
    g.add_edge("execute", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"revise": "revise", "end": END},
    )
    g.add_edge("revise", "execute")
    return g.compile()


graph = build_graph()
