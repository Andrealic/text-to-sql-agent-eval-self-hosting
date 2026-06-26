---
name: sql-agent-eval
description: Run and report a text-to-SQL agent eval against the BIRD gold set. Use when asked to "run the eval", test the agent on N questions / all 30, compare to gold, analyze FP/FN/TP/TN of the verifier, check node health (explore/verify/evidence/revise), or produce an eval report. Wraps evals/run_eval.py + evals/analyze_run.py and writes docs/eval-report_<run_id>.md.
---

# SQL agent eval + report

Repeatable workflow to evaluate the LangGraph text-to-SQL agent against the BIRD gold set and
write a structured report. Everything is scored vs gold by **executed row sets** (canonicalized);
node behaviour comes from **Langfuse** traces.

## 0. Preconditions
- Agent server running on `:8001` (the user starts it: `uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001`). Check: `curl -s -o /dev/null -w "%{http_code}" http://localhost:8001/health` → 200. If it's not the current code, ask the user to restart (uvicorn does not auto-reload).
- **Model matters:** off-GPU use a fast **non-reasoning instruct** model (e.g. `qwen/qwen3-30b-a3b-instruct-2507` via OpenRouter in `.env`). A reasoning model (e.g. minimax) makes calls slow and causes hangs — the agent makes 5-13 LLM calls/question. Real SLO/pass-rate numbers must come from `Qwen3-30B-A3B` on the H100.
- BIRD DBs present under `data/bird/`.

## 1. Pick the question set
- Full set: `evals/eval_set.jsonl` (30 questions).
- A slice: `sed -n '11,20p' evals/eval_set.jsonl > evals/eval_set_10b.jsonl` (questions 11-20), etc.

## 2. Smoke-probe one question first
Confirm the server runs the current graph and the flow is healthy before a full run:
```
curl -s -m 250 -X POST http://localhost:8001/answer -H "Content-Type: application/json" \
  -d '{"question":"...","db":"..."}' | python3 -m json.tool
```
Check the `history` shows the expected node flow (explore → generate_sql → verify → [evidence → revise → ...]).

## 3. Run the eval
Always pass a meaningful `--run-id` so results never overwrite (saved to `results/eval_<run_id>.json`,
which records run_id / created_at / config / model). Concurrency 1 keeps it clean; raise `--timeout`
for slow backends. Long runs: launch in the background and poll the output for `[n/N]` / `Wrote`.
```
uv run python evals/run_eval.py --eval-set evals/eval_set.jsonl \
  --concurrency 1 --timeout 300 --run-id <descriptive-id>
```

## 4. Analyze vs gold + node health
```
uv run python evals/analyze_run.py results/eval_<run_id>.json --langfuse
```
This prints: overall + per-iteration pass rate; **verifier confusion matrix vs gold (TP/TN/FP/FN)**;
what revise did (fixed x→C, broke C→x, regressions); per-DB breakdown; the failure list with
correctness/verify sequences; and per-node health (explore avg queries + UPPER usage, verify
parse-fails + vote distribution, evidence invocations, revise calls, execute errors).

## 5. Write the report
Save to `docs/eval-report_<run_id>.md`, referencing the run_id. Structure (see existing reports
in `docs/` for the template):
1. **Header** — run_id, model, set, config, wall clock, errors.
2. **Result vs gold** — overall + per-iteration (does the loop earn its keep? iter0 → final).
3. **Failure taxonomy** — group fails: domain-knowledge gaps (missing BIRD `evidence`),
   verifier false-positives (accepted wrong), interpretation/shape, output-format/label.
4. **Node-by-node** — for explore / generate / execute / verify / evidence / revise / router:
   is it doing its job, and well? Give the verifier confusion matrix and the role of evidence + revise.
5. **Verdict & prioritized levers** — be honest if the machinery isn't beating a prior baseline;
   compare run-to-run (results files are kept per run_id).

## Key things to look for (learned)
- **Verifier FP is usually the ceiling**: it accepts plausible-but-wrong answers, ending the loop
  before revise can help. FN (rejecting correct) wastes iterations / passes become luck.
- **`could not parse` verify outputs** = the model answered in prose, not JSON (reinforce VERIFY_SYSTEM).
- **Latency/hangs** are LLM-side, never the DB (sub-ms). A reasoning model + many calls/question is
  the usual cause; `request_timeout` on `llm()` is in **milliseconds** in langchain_openrouter.
- **~1/3 of BIRD fails need external `evidence`** (clinical ranges, status-code semantics, label
  encodings) that the data alone can't reveal — a hard ceiling for the explorer.
