#!/usr/bin/env python3
"""Append/update one row in docs/eval-history.md for a run — historicized by version+timestamp.

Idempotent by run_id: re-running replaces that run's row. Reads the strict BIRD metric from the
results JSON and the lenient metric from the sibling .judged.json if present.

Usage:
    uv run python evals/update_history.py results/eval_<run_id>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HISTORY = ROOT / "docs" / "eval-history.md"
HEADER = (
    "# Eval history\n\n"
    "One row per eval run, newest first. Strict = BIRD execution accuracy; "
    "Lenient = LLM-judge 0/0.5/1 mean. Per-run detail in `docs/eval-report_<run_id>.md`.\n\n"
    "| created (UTC) | agent | git | run_id | model | strict | iter0→final | lenient | n |\n"
    "|---|---|---|---|---|---|---|---|---|\n"
)


def _row(d: dict) -> tuple[str, str]:
    s = d["summary"]
    c = d.get("config", {})
    git = (d.get("git_sha") or "?") + ("-dirty" if d.get("git_dirty") else "")
    pir = s.get("per_iteration_pass_rate") or []
    loop = f"{pir[0]:.2f}→{pir[-1]:.2f}" if pir else "-"
    lenient = "-"
    judged = Path(str(d.get("_path"))).with_suffix(".judged.json")
    if judged.exists():
        recs = json.loads(judged.read_text()).get("results", [])
        scores = [r["lenient_score"] for r in recs if r.get("lenient_score") is not None]
        if scores:
            lenient = f"{sum(scores)/len(scores):.2f}"
    created = (d.get("created_at") or "")[:19]
    run_id = d.get("run_id") or "?"
    row = (f"| {created} | {d.get('agent_version') or '?'} | {git} | {run_id} | "
           f"{c.get('model') or '?'} | {s['overall_pass_rate']:.3f} | {loop} | {lenient} | {s.get('n','?')} |")
    return run_id, row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path)
    args = ap.parse_args()
    d = json.loads(args.results.read_text())
    d["_path"] = str(args.results)
    run_id, row = _row(d)

    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if HISTORY.exists():
        for ln in HISTORY.read_text().splitlines():
            t = ln.strip()
            is_data = t.startswith("| ") and not t.startswith("| created") and not t.startswith("|---")
            if is_data and f"| {run_id} |" not in t:
                existing.append(t)
    HISTORY.write_text(HEADER + "\n".join([row] + existing) + "\n")  # newest first
    print(f"Updated {HISTORY} (run_id={run_id})")


if __name__ == "__main__":
    main()
