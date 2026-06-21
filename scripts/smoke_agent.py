#!/usr/bin/env python3
"""Fire a few random eval questions at the agent (useful for Langfuse smoke tests).

Run (agent server must be up on :8001):
    uv run python scripts/smoke_agent.py
    uv run python scripts/smoke_agent.py --n 5 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


def load_questions(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        sys.exit(f"No questions found in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Call /answer with random eval questions.")
    parser.add_argument("--n", type=int, default=5, help="Number of requests (default: 5)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    parser.add_argument("--eval-set", type=Path, default=EVAL_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = load_questions(args.eval_set)
    n = min(args.n, len(questions))

    rng = random.Random(args.seed)
    sample = rng.sample(questions, n)

    print(f"Agent: {args.agent_url}")
    print(f"Questions: {n} random from {args.eval_set} (seed={args.seed})\n")

    with httpx.Client(timeout=120.0) as client:
        for i, q in enumerate(sample, 1):
            payload = {
                "question": q["question"],
                "db": q["db_id"],
                "tags": {
                    "run_type": "smoke",
                },
            }
            t0 = time.monotonic()
            try:
                resp = client.post(args.agent_url, json=payload)
                elapsed = time.monotonic() - t0
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as e:
                print(f"[{i}/{n}] ERROR {q['db_id']}: {e}")
                continue

            preview = q["question"][:70] + ("..." if len(q["question"]) > 70 else "")
            nodes = [h.get("node") for h in data.get("history", [])]
            print(
                f"[{i}/{n}] {q['db_id']} | ok={data.get('ok')} "
                f"iter={data.get('iterations')} | {elapsed:.1f}s"
            )
            print(f"  Q: {preview}")
            print(f"  history: {' → '.join(nodes) or '(empty)'}")
            if data.get("error"):
                print(f"  error: {data['error']}")
            print()


if __name__ == "__main__":
    main()
