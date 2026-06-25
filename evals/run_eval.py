"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def _attempts_from_history(history: list[dict], fallback_sql: str) -> list[dict]:
    """Rebuild SQL attempts (generate_sql / revise) with verify_ok attached."""
    attempts: list[dict] = []
    for entry in history:
        node = entry.get("node")
        if node in ("generate_sql", "revise"):
            attempts.append({
                "node": node,
                "sql": entry.get("sql", ""),
                "verify_ok": None,
            })
        elif node == "verify" and attempts:
            attempts[-1]["verify_ok"] = entry.get("verify_ok")

    if not attempts and fallback_sql:
        attempts.append({"node": "generate_sql", "sql": fallback_sql, "verify_ok": None})
    return attempts


async def eval_one(question: dict, client: httpx.AsyncClient, agent_url: str, timeout: float = 120.0) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness."""
    db_id = question["db_id"]
    gold_ok, gold_rows, gold_err = await asyncio.to_thread(run_sql, db_id, question["gold_sql"])

    base = {
        "question": question["question"],
        "db_id": db_id,
        "gold_sql": question["gold_sql"],
        "gold_exec_ok": gold_ok,
        "gold_error": gold_err,
    }

    try:
        resp = await client.post(
            agent_url,
            json={"question": question["question"], "db": db_id},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        return {
            **base,
            "transport_error": f"{type(e).__name__}: {e}",
            "final_sql": "",
            "iterations": 0,
            "agent_ok": False,
            "agent_error": None,
            "final_correct": False,
            "per_iteration": [],
        }

    attempts = _attempts_from_history(data.get("history") or [], data.get("sql", ""))
    per_iteration: list[dict] = []
    for i, attempt in enumerate(attempts):
        pred_ok, pred_rows, pred_err = await asyncio.to_thread(run_sql, db_id, attempt["sql"])
        correct = gold_ok and pred_ok and matches(gold_rows, pred_rows)
        per_iteration.append({
            "iteration": i,
            "node": attempt["node"],
            "sql": attempt["sql"],
            "correct": correct,
            "verify_ok": attempt["verify_ok"],
            "exec_ok": pred_ok,
            "error": pred_err,
        })

    final_correct = per_iteration[-1]["correct"] if per_iteration else False

    return {
        **base,
        "final_sql": data.get("sql", ""),
        "iterations": data.get("iterations", len(per_iteration)),
        "agent_ok": data.get("ok", False),
        "agent_error": data.get("error"),
        "final_correct": final_correct,
        "per_iteration": per_iteration,
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    n = len(results)
    if n == 0:
        return {
            "n": 0,
            "overall_pass_rate": 0.0,
            "per_iteration_pass_rate": [],
            "avg_iterations": 0.0,
            "n_agent_errors": 0,
            "n_transport_errors": 0,
            "n_gold_errors": 0,
        }

    max_iters = max(len(r["per_iteration"]) for r in results)
    per_iteration_pass_rate: list[float] = []

    for k in range(max_iters):
        correct_at_k = 0
        for r in results:
            per_iter = r["per_iteration"]
            if k < len(per_iter):
                correct_at_k += int(per_iter[k]["correct"])
            elif per_iter:
                correct_at_k += int(per_iter[-1]["correct"])
        per_iteration_pass_rate.append(correct_at_k / n)

    overall_pass_rate = sum(int(r["final_correct"]) for r in results) / n
    avg_iterations = sum(r.get("iterations", 0) for r in results) / n

    return {
        "n": n,
        "overall_pass_rate": overall_pass_rate,
        "per_iteration_pass_rate": per_iteration_pass_rate,
        "avg_iterations": avg_iterations,
        "n_agent_errors": sum(1 for r in results if not r.get("agent_ok", True)),
        "n_transport_errors": sum(1 for r in results if r.get("transport_error")),
        "n_gold_errors": sum(1 for r in results if not r.get("gold_exec_ok", True)),
    }


# ---------- Main (provided) --------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output file. Default: results/eval_<run_id>.json (so runs never overwrite each other).",
    )
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max agent requests in flight at once (keep low to avoid rate limits / SLO distortion).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Identifier for this run (default: <UTC timestamp>-<short uuid>). Lets results be tracked/compared.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout (s) for the agent HTTP call. Raise it when a slow backend needs more time per run.",
    )
    args = parser.parse_args()

    created_at = datetime.now(timezone.utc).isoformat()
    run_id = args.run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    out_path = args.out or (ROOT / "results" / f"eval_{run_id}.json")

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    sem = asyncio.Semaphore(args.concurrency)
    done = 0

    async def worker(q: dict, client: httpx.AsyncClient) -> dict:
        nonlocal done
        async with sem:
            result = await eval_one(q, client, args.agent_url, timeout=args.timeout)
        done += 1
        print(f"[{done}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        return result

    t0 = time.monotonic()
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*(worker(q, client) for q in questions))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "run_id": run_id,
        "created_at": created_at,
        "config": {
            "eval_set": str(args.eval_set),
            "agent_url": args.agent_url,
            "concurrency": args.concurrency,
            "timeout": args.timeout,
            "n_questions": len(questions),
            "model": os.environ.get("VLLM_MODEL"),
            "base_url": os.environ.get("VLLM_BASE_URL"),
        },
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path} (run_id={run_id})")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
