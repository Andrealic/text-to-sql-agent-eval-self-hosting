"""FastAPI wrapper exposing the agent over HTTP.

Run:
    uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001

The /answer endpoint accepts {question, db, tags?} and returns the
agent's final SQL, the result rows, and per-iteration history.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

from agent.graph import AgentState, graph  # noqa: E402

# Use uvicorn's configured logger so request summaries show up with the server logs.
logger = logging.getLogger("uvicorn.error")

# Langfuse callback handler. If keys are set we initialize it; failures
# are NOT swallowed - a misconfigured Langfuse should not silently
# produce zero traces.
_lf_handler: Any = None
if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
    from langfuse import get_client

    from agent.langfuse_callbacks import OpenRouterCostCallbackHandler

    _lf_handler = OpenRouterCostCallbackHandler()


app = FastAPI()


def _short_question(question: str, max_len: int = 120) -> str:
    """Keep request logs useful without dumping long prompts."""
    compact = " ".join(question.split())
    return compact if len(compact) <= max_len else compact[: max_len - 1] + "..."


def _last_verify(history: list[dict[str, Any]]) -> tuple[bool | None, str | None]:
    for entry in reversed(history):
        if entry.get("node") == "verify":
            return entry.get("verify_ok"), entry.get("verify_issue")
    return None, None


class AnswerRequest(BaseModel):
    question: str
    db: str
    tags: dict[str, str] = {}


class AnswerResponse(BaseModel):
    sql: str
    rows: list[list[Any]] | None
    iterations: int
    ok: bool
    error: str | None = None
    history: list[dict[str, Any]] = []


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest) -> AnswerResponse:
    started = time.perf_counter()
    state = AgentState(question=req.question, db_id=req.db)
    DEFAULT_TAGS = {
        "agent_version": "v0.2.0"
    }
    metadata = {**DEFAULT_TAGS, **req.tags}
    logger.info(
        "answer_start db=%s tags=%s question=%r",
        req.db,
        metadata,
        _short_question(req.question),
    )
    config: dict[str, Any] = {
        "callbacks": [_lf_handler] if _lf_handler is not None else [],
        "metadata": metadata,
    }
    try:
        final = graph.invoke(state, config=config)
    except Exception as e:  # noqa: BLE001
        elapsed = time.perf_counter() - started
        logger.exception(
            "answer_error db=%s elapsed_seconds=%.3f error=%s",
            req.db,
            elapsed,
            f"{type(e).__name__}: {e}",
        )
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    finally:
        if _lf_handler is not None:
            get_client().flush()

    sql = final.get("sql", "")
    iteration = final.get("iteration", 0)
    history = final.get("history", [])
    execution = final.get("execution")
    verify_ok, verify_issue = _last_verify(history)
    elapsed = time.perf_counter() - started

    if execution is None:
        logger.warning(
            "answer_complete ok=false db=%s elapsed_seconds=%.3f iterations=%s "
            "verify_ok=%s verify_issue=%r error=%r",
            req.db,
            elapsed,
            iteration,
            verify_ok,
            verify_issue,
            "agent produced no execution result",
        )
        return AnswerResponse(
            sql=sql,
            rows=None,
            iterations=iteration,
            ok=False,
            error="agent produced no execution result",
            history=history,
        )
    if not execution.ok:
        logger.warning(
            "answer_complete ok=false db=%s elapsed_seconds=%.3f iterations=%s "
            "verify_ok=%s verify_issue=%r error=%r",
            req.db,
            elapsed,
            iteration,
            verify_ok,
            verify_issue,
            execution.error,
        )
        return AnswerResponse(
            sql=sql,
            rows=None,
            iterations=iteration,
            ok=False,
            error=execution.error,
            history=history,
        )

    logger.info(
        "answer_complete ok=true db=%s elapsed_seconds=%.3f iterations=%s "
        "verify_ok=%s verify_issue=%r row_count=%s",
        req.db,
        elapsed,
        iteration,
        verify_ok,
        verify_issue,
        execution.row_count,
    )
    return AnswerResponse(
        sql=sql,
        rows=[list(r) for r in (execution.rows or [])],
        iterations=iteration,
        ok=True,
        history=history,
    )
