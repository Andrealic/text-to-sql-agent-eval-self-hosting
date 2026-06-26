#!/usr/bin/env python3
"""Extract per-question agent node traces from Langfuse for one eval run.

The agent's intermediate reasoning is NOT exposed by the /answer API on purpose;
it lives in Langfuse. This script reconstructs, per question, the ordered node
steps (explore / generate_sql / execute / verify / evidence / revise / ...) from
the Langfuse spans of the run, so the judge can assess each node offline.

It is architecture-dynamic: it records whatever node names appear, pulling the
fields each node returns. Join to the eval question is by the trace input text;
the time window is taken from the results JSON (created_at + wall_clock).

Usage:
    uv run python evals/extract_traces.py results/eval_<run_id>.json
    # -> writes results/eval_<run_id>.traces.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from langfuse_utils import client  # type: ignore  # noqa: E402  (same dir)

# Nodes whose output we summarize specially; anything else is captured generically.
ROW_PREVIEW = 8


def _question_of(trace: dict) -> str | None:
    inp = trace.get("input")
    if isinstance(inp, dict):
        if "question" in inp:
            return inp["question"]
        # langgraph sometimes nests the state
        for v in inp.values():
            if isinstance(v, dict) and "question" in v:
                return v["question"]
    blob = json.dumps(inp)
    return blob if inp else None


def _step_from_obs(o: dict) -> dict | None:
    name = o.get("name")
    out = o.get("output") if isinstance(o.get("output"), dict) else {}
    if name not in ("explore", "generate_sql", "execute", "verify", "evidence", "revise"):
        # Unknown / future node: capture generically so the judge can still see it.
        if name in ("attach_schema", "route_after_verify", "ChatOpenRouter", "LangGraph"):
            return None
        return {"node": name, "output": out}

    if name == "explore":
        f = out.get("findings") or {}
        return {"node": "explore", "exploration_queries": list(f.keys()), "findings": f}
    if name in ("generate_sql", "revise"):
        return {"node": name, "sql": out.get("sql", "")}
    if name == "execute":
        ex = (out or {}).get("execution") or {}
        return {
            "node": "execute",
            "ok": ex.get("ok"),
            "row_count": ex.get("row_count"),
            "error": ex.get("error"),
        }
    if name == "verify":
        return {
            "node": "verify",
            "verify_ok": out.get("verify_ok"),
            "verify_issue": out.get("verify_issue"),
            "needs_evidence": out.get("needs_evidence"),
            "evidence_questions": out.get("evidence_questions") or [],
        }
    if name == "evidence":
        # The per-call evidence is the last 'evidence' entry in the returned history.
        entry = {}
        for h in reversed(out.get("history", []) or []):
            if h.get("node") == "evidence":
                entry = h
                break
        return {
            "node": "evidence",
            "questions": entry.get("questions") or [],
            "findings": entry.get("findings") or {},
        }
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path, help="results/eval_<run_id>.json")
    ap.add_argument("--out", type=Path, default=None, help="default: <results>.traces.json")
    ap.add_argument("--buffer", type=float, default=120.0, help="window padding seconds")
    args = ap.parse_args()

    d = json.loads(args.results.read_text())
    run_id = d.get("run_id")
    db_by_q = {r["question"]: r["db_id"] for r in d["results"]}

    lf = client()
    if lf is None:
        raise SystemExit("Langfuse not configured (.env keys missing) — cannot extract traces.")

    start = dt.datetime.fromisoformat(d["created_at"])
    end = start + dt.timedelta(seconds=d.get("wall_clock_seconds", 0) + args.buffer)
    traces = lf.traces_in_window(start, end)

    by_question: dict[str, dict] = {}
    for t in sorted(traces, key=lambda x: x["timestamp"]):
        q = _question_of(t)
        if not q:
            continue
        steps = [s for o in lf.observations(t["id"]) if (s := _step_from_obs(o))]
        if not steps:
            continue
        # If a question recurs (shouldn't within one run), keep the richest trace.
        if q not in by_question or len(steps) > len(by_question[q]["steps"]):
            by_question[q] = {
                "question": q,
                "db_id": db_by_q.get(q),
                "trace_id": t["id"],
                "steps": steps,
            }

    # Preserve the eval-set order; warn on any question with no trace.
    out_records = []
    missing = []
    for r in d["results"]:
        rec = by_question.get(r["question"])
        if rec is None:
            missing.append(r["question"][:50])
        else:
            out_records.append(rec)

    out_path = args.out or args.results.with_suffix(".traces.json")
    out_path.write_text(json.dumps({"run_id": run_id, "traces": out_records}, indent=2))
    print(f"Wrote {out_path}: {len(out_records)} question traces "
          f"({len(traces)} Langfuse traces in window)")
    if missing:
        print(f"  WARNING: no trace matched {len(missing)} question(s): {missing[:3]}...")


if __name__ == "__main__":
    main()
