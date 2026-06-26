#!/usr/bin/env python3
"""Analyze one eval run against the gold set, plus per-node health from Langfuse.

Reads a results JSON written by run_eval.py (which already scored every attempt
vs gold) and prints:
  - headline result (overall + per-iteration pass rate, avg iterations, errors)
  - verifier confusion matrix vs gold (TP / TN / FP / FN)
  - what revise did (fixed x->C, broke C->x, regressions)
  - per-database breakdown and the failure list with correctness/verify sequences
  - (optional, --langfuse) per-node health: explore / verify / evidence / revise / execute,
    using the run's created_at + wall_clock to scope the Langfuse trace window.

Usage:
    uv run python evals/analyze_run.py results/eval_<run_id>.json
    uv run python evals/analyze_run.py results/eval_<run_id>.json --langfuse
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter, defaultdict
from pathlib import Path

from langfuse_utils import client  # type: ignore  # noqa: E402  (same dir)

ROOT = Path(__file__).resolve().parent.parent


def _bool(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return None


def analyze_gold(d: dict) -> None:
    s = d["summary"]
    c = d.get("config", {})
    git = d.get("git_sha") or "?"
    git += "-dirty" if d.get("git_dirty") else ""
    print("=" * 70)
    print(f"RUN {d.get('run_id')}  |  agent {d.get('agent_version') or '?'}  |  git {git}")
    print(f"model: {c.get('model') or '(env default)'}  |  created: {d.get('created_at')}  |  wall: {round(d.get('wall_clock_seconds', 0))}s")
    print("=" * 70)
    print(f"overall_pass_rate      = {s['overall_pass_rate']}")
    print(f"per_iteration_pass_rate= {[round(x, 3) for x in s['per_iteration_pass_rate']]}")
    print(f"avg_iterations         = {round(s['avg_iterations'], 2)}")
    print(f"errors: agent={s['n_agent_errors']} transport={s['n_transport_errors']} gold={s['n_gold_errors']}")

    tp = fp = fn = tn = 0
    fixed = broke = regress = 0
    by_db: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in d["results"]:
        pis = r.get("per_iteration", [])
        by_db[r["db_id"]][0] += int(r["final_correct"])
        by_db[r["db_id"]][1] += 1
        if pis:
            if not pis[0]["correct"] and pis[-1]["correct"]:
                fixed += 1
            if pis[0]["correct"] and not pis[-1]["correct"]:
                broke += 1
            if any(it["correct"] for it in pis[:-1]) and not pis[-1]["correct"]:
                regress += 1
        for it in pis:
            vok = _bool(it.get("verify_ok"))
            co = it.get("correct")
            if vok is None or co is None:
                continue
            if vok and co:
                tp += 1
            elif vok and not co:
                fp += 1
            elif (not vok) and co:
                fn += 1
            else:
                tn += 1

    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total else float("nan")
    print("\nVERIFIER vs GOLD (per scored iteration):")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}  accuracy={acc:.2f}")
    print(f"  FP (accepts wrong -> ends loop on a bad answer) is the quality ceiling.")
    print(f"\nREVISE: fixed(x->C)={fixed}  broke(C->x)={broke}  regressions={regress}")

    print("\nPER-DB (final correct / total):")
    for db, (cc, tt) in sorted(by_db.items()):
        print(f"  {db:24} {cc}/{tt}")

    print("\nFAILURES (db | iters | corr-seq | verify-seq | question):")
    for r in d["results"]:
        if r["final_correct"]:
            continue
        pis = r.get("per_iteration", [])
        cs = "".join("C" if it["correct"] else "x" for it in pis) or "-"
        vs = "".join("T" if _bool(it.get("verify_ok")) else "F" for it in pis) or "-"
        err = r.get("transport_error") or r.get("agent_error") or ""
        print(f"  {r['db_id']:22} it={r.get('iterations')} [{cs:3}] vfy[{vs:3}] | {r['question'][:46]}{' | ERR:'+err[:30] if err else ''}")


def analyze_langfuse(d: dict) -> None:
    lf = client()
    if lf is None:
        print("\n(langfuse: not configured, skipping node health)")
        return
    # Window from created_at .. created_at + wall + buffer
    start = dt.datetime.fromisoformat(d["created_at"])
    end = start + dt.timedelta(seconds=d.get("wall_clock_seconds", 0) + 120)
    try:
        traces = lf.traces_in_window(start, end)
    except Exception as e:  # noqa: BLE001
        print(f"\n(langfuse: query failed: {e})")
        return

    explore_q, explore_upper, explore_empty = [], 0, 0
    vspans = vparsefail = 0
    vote_dist = Counter()
    revise_n = evidence_n = 0
    exec_err = Counter()
    for t in traces:
        for o in lf.observations(t["id"]):
            n = o.get("name")
            out = o.get("output") if isinstance(o.get("output"), dict) else {}
            if n == "explore":
                f = out.get("findings") or {}
                explore_q.append(len(f))
                explore_empty += int(not f)
                explore_upper += int(any("UPPER(" in q.upper() for q in f))
            elif n == "verify":
                vspans += 1
                votes = None
                for h in reversed(out.get("history", []) or []):
                    if h.get("node") == "verify" and "votes" in h:
                        votes = h["votes"]
                        break
                if votes:
                    vote_dist[f"{sum(1 for x in votes if x.get('ok'))}/{len(votes)}"] += 1
                    if any("could not parse" in (x.get("issue") or "") for x in votes):
                        vparsefail += 1
            elif n == "evidence":
                evidence_n += 1
            elif n == "revise":
                revise_n += 1
            elif n == "execute":
                ex = (out or {}).get("execution") or {}
                if ex and not ex.get("ok"):
                    exec_err[(ex.get("error") or "").split(":")[0]] += 1

    nq = len(explore_q) or 1
    print("\n" + "=" * 70)
    print(f"NODE HEALTH (Langfuse, {len(traces)} traces in run window)")
    print("=" * 70)
    print(f"EXPLORE : runs={len(explore_q)} avg_q/run={sum(explore_q)/nq:.1f} empty={explore_empty} using_UPPER={explore_upper}")
    print(f"VERIFY  : spans={vspans} parse_fails={vparsefail} vote_dist={dict(vote_dist)}")
    print(f"EVIDENCE: node invocations={evidence_n}")
    print(f"REVISE  : calls={revise_n}")
    print(f"EXECUTE : sql_errors={dict(exec_err) or 'none'}")


def analyze_judged(judged: dict) -> None:
    """Aggregate the LLM-judge output: lenient metric + per-node verdicts + failure reasons."""
    recs = judged.get("results", [])
    if not recs:
        return
    n = len(recs)
    scores = [r.get("lenient_score") for r in recs if r.get("lenient_score") is not None]
    dist = Counter(scores)
    mean = sum(scores) / len(scores) if scores else float("nan")

    print("\n" + "=" * 70)
    print(f"LENIENT METRIC (LLM judge = {judged.get('judge_model')}, {n} questions)")
    print("=" * 70)
    print(f"  mean lenient score = {mean:.3f}   (BIRD accuracy is the strict counterpart)")
    # Note: in Python 1 == 1.0 and 0 == 0.0 hash-collide, so each get() already covers both.
    print(f"  distribution: 1.0={dist.get(1, 0)}  0.5={dist.get(0.5, 0)}  0.0={dist.get(0, 0)}  (unscored={n-len(scores)})")

    # Per-node aggregates (dynamic: whatever node keys the judge returned)
    explore_v = Counter()
    gen_err_ctx = 0
    gen_correct_step = 0
    revise_help = Counter()
    revise_used_ev = 0
    verify_just = 0
    verify_seen = 0
    for r in recs:
        nodes = r.get("nodes") or {}
        if "explore" in nodes:
            explore_v[nodes["explore"].get("verdict")] += 1
        g = nodes.get("generate") or {}
        gen_err_ctx += int(bool(g.get("error_despite_context")))
        gen_correct_step += int(bool(g.get("correct_at_some_step")))
        rv = nodes.get("revise") or {}
        if rv.get("helpful"):
            revise_help[rv["helpful"]] += 1
        revise_used_ev += int(bool(rv.get("used_evidence")))
        if "verify" in nodes:
            verify_seen += 1
            verify_just += int(bool(nodes["verify"].get("verdict_justified")))

    print("\n  PER-NODE (judge):")
    print(f"    explore verdicts      = {dict(explore_v)}")
    print(f"    generate: correct-at-some-step={gen_correct_step}/{n}  error-despite-context={gen_err_ctx}/{n}")
    print(f"    revise helpfulness    = {dict(revise_help)}  used_evidence={revise_used_ev}/{n}")
    print(f"    verify verdict-justified = {verify_just}/{verify_seen}")

    print("\n  FAILURE REASONS (lenient < 1):")
    for r in recs:
        if (r.get("lenient_score") or 0) < 1 and r.get("failure_reason"):
            print(f"    [{r['db_id']}] {r['failure_reason'][:140]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path, help="Path to results/eval_<run_id>.json")
    ap.add_argument("--langfuse", action="store_true", help="Also pull per-node health from Langfuse")
    ap.add_argument("--judged", action="store_true", help="Also aggregate the LLM-judge .judged.json (lenient metric + node verdicts)")
    args = ap.parse_args()
    d = json.loads(args.results.read_text())
    analyze_gold(d)
    if args.langfuse:
        analyze_langfuse(d)
    if args.judged:
        judged_path = args.results.with_suffix(".judged.json")
        if judged_path.exists():
            analyze_judged(json.loads(judged_path.read_text()))
        else:
            print(f"\n(--judged: {judged_path.name} not found; run evals/judge_run.py or the skill judge step first)")


if __name__ == "__main__":
    main()
