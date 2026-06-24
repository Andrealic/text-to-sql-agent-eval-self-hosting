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
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
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


def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness."""
    db_id = question["db_id"]
    gold_ok, gold_rows, gold_err = run_sql(db_id, question["gold_sql"])

    base = {
        "question": question["question"],
        "db_id": db_id,
        "gold_sql": question["gold_sql"],
        "gold_exec_ok": gold_ok,
        "gold_error": gold_err,
    }

    try:
        resp = httpx.post(
            agent_url,
            json={"question": question["question"], "db": db_id},
            timeout=120.0,
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
        pred_ok, pred_rows, pred_err = run_sql(db_id, attempt["sql"])
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

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
