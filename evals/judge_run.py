#!/usr/bin/env python3
"""LLM-as-judge over an eval run: lenient score + per-node assessment + free text.

Two metrics live side by side: the deterministic BIRD execution accuracy
(computed by run_eval.py) and a LENIENT score (0 / 0.5 / 1) produced here a
posteriori — "would this answer do as a chatbot reply vs gold", tolerant of
extra columns / non-exhaustive results. The judge also assesses each agent node
(dynamic to whatever nodes appear in the extracted trace) and writes a free-text
reason for each failure.

Judge != agent: use a strong model different from the agent (default Claude via
OpenRouter, reusing OPENAI_API_KEY). Override with JUDGE_MODEL / --model.

NOTE: the primary judge in day-to-day use is Claude Code itself (via the
sql-agent-eval skill). This script is the headless/automatable path and emits the
SAME results/eval_<run_id>.judged.json schema.

Usage:
    uv run python evals/judge_run.py results/eval_<run_id>.json [--model anthropic/claude-3.5-sonnet]
    uv run python evals/judge_run.py results/eval_<run_id>.json --limit 5   # subset for a smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from run_eval import run_sql  # type: ignore  # noqa: E402  (same dir)

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

_PREVIEW_ROWS = 8


def run_sql_rows_preview(db_id: str, sql: str) -> str:
    """Compact preview of a SQL result for the judge prompt."""
    if not sql:
        return "(no SQL)"
    ok, rows, err = run_sql(db_id, sql)
    if not ok:
        return f"ERROR: {err}"
    rows = rows or []
    head = "; ".join(str(tuple(r)) for r in rows[:_PREVIEW_ROWS])
    more = f" ... (+{len(rows) - _PREVIEW_ROWS} more rows)" if len(rows) > _PREVIEW_ROWS else ""
    return f"{len(rows)} rows: {head}{more}" if rows else "0 rows"

DEFAULT_JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "anthropic/claude-sonnet-4.5")
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "https://openrouter.ai/api/v1")
JUDGE_API_KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or "not-needed"

JUDGE_SYSTEM = """You grade a text-to-SQL agent against a gold answer, and you assess each agent node.
You are a DIFFERENT, stronger reviewer than the agent — do not rubber-stamp.

Return ONE raw JSON object, no prose, no markdown, exact shape:
{
  "lenient_score": 0 | 0.5 | 1,
  "failure_reason": "one or two sentences on what the agent misunderstood (empty string if score==1)",
  "nodes": {
    "explore":  {"verdict": "good|partial|poor|na", "suggested_right_queries": true|false, "note": ""},
    "generate": {"error_despite_context": true|false, "ignored_info": "", "note": ""},
    "revise":   {"helpful": "improved|neutral|worse|na", "used_evidence": true|false, "note": ""},
    "verify":   {"verdict_justified": true|false, "note": ""}
  }
}
Add a key under "nodes" for any OTHER node name present in the trace with {"did_its_job": true|false, "note": ""}.

Scoring the LENIENT metric (vs the gold result):
- 1   = same answer as gold for a chatbot user (exact, or only cosmetic differences: column order, extra
        label/helper columns, formatting), and not missing requested rows.
- 0.5 = partially acceptable: right idea but non-exhaustive, missing/extra rows, wrong granularity, or an
        extra/different column that changes meaning slightly yet a user could still get value.
- 0   = wrong: different entity/number, empty when data exists, or answers a different question.

Node rubric (assess only nodes present; be concrete in notes):
- explore: did its exploration queries target the columns/values the question actually needs?
- generate/revise: did it make an error DESPITE the findings/evidence already in context containing the
  fact it got wrong (set error_despite_context=true and name the ignored fact in ignored_info)?
- revise: did it improve over the prior SQL and actually use the gathered evidence?
- verify: was its accept/reject justified given the gold answer?"""


def _judge_user(rec: dict) -> str:
    steps = json.dumps(rec["steps"], ensure_ascii=False)[:6000]
    return f"""Question: {rec['question']}
DB: {rec['db_id']}

GOLD SQL: {rec['gold_sql']}
GOLD result (preview): {rec['gold_preview']}

AGENT final SQL: {rec['final_sql']}
AGENT result (preview): {rec['agent_preview']}

Ground-truth facts (already computed, trust these):
- bird_correct (exact row-set match vs gold): {rec['bird_correct']}
- a gold-correct query WAS produced at some step (maybe rejected by verify): {rec['correct_at_some_step']}

Agent node trace (ordered steps):
{steps}

Grade per the system instructions. Output only the JSON object."""


def call_judge(model: str, system: str, user: str) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": 1200,
    }).encode()
    req = urllib.request.Request(
        f"{JUDGE_BASE_URL.rstrip('/')}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {JUDGE_API_KEY}", "Content-Type": "application/json"},
    )
    data = json.load(urllib.request.urlopen(req, timeout=120))
    return data["choices"][0]["message"]["content"] or ""


def _parse_json(text: str) -> dict:
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1:
        try:
            return json.loads(text[s:e + 1])
        except json.JSONDecodeError:
            pass
    return {"lenient_score": None, "failure_reason": "judge output unparseable", "nodes": {}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path)
    ap.add_argument("--traces", type=Path, default=None, help="default: <results>.traces.json")
    ap.add_argument("--out", type=Path, default=None, help="default: <results>.judged.json")
    ap.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--limit", type=int, default=None, help="judge only the first N questions")
    args = ap.parse_args()

    d = json.loads(args.results.read_text())
    traces_path = args.traces or args.results.with_suffix(".traces.json")
    steps_by_q = {t["question"]: t["steps"] for t in json.loads(traces_path.read_text())["traces"]}

    judged = []
    results = d["results"][: args.limit] if args.limit else d["results"]
    for i, r in enumerate(results, 1):
        q = r["question"]
        per_iter = r.get("per_iteration", [])
        rec = {
            "question": q,
            "db_id": r["db_id"],
            "gold_sql": r["gold_sql"],
            "final_sql": r.get("final_sql", ""),
            "bird_correct": r["final_correct"],
            "correct_at_some_step": any(it.get("correct") for it in per_iter),
            "gold_preview": run_sql_rows_preview(r["db_id"], r["gold_sql"]),
            "agent_preview": run_sql_rows_preview(r["db_id"], r.get("final_sql", "")),
            "steps": steps_by_q.get(q, []),
        }
        try:
            verdict = _parse_json(call_judge(args.model, JUDGE_SYSTEM, _judge_user(rec)))
        except Exception as e:  # noqa: BLE001
            verdict = {"lenient_score": None, "failure_reason": f"judge error: {e}", "nodes": {}}
        # carry deterministic facts into the record
        verdict.setdefault("nodes", {}).setdefault("generate", {})["correct_at_some_step"] = rec["correct_at_some_step"]
        judged.append({"question": q, "db_id": r["db_id"], "bird_correct": r["final_correct"], **verdict})
        print(f"[{i}/{len(results)}] {r['db_id']}: lenient={verdict.get('lenient_score')}", flush=True)

    out_path = args.out or args.results.with_suffix(".judged.json")
    out_path.write_text(json.dumps({"run_id": d.get("run_id"), "judge_model": args.model, "results": judged}, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
