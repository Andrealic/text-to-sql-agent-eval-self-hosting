---
name: sql-agent-eval
description: Evaluate and report on the text-to-SQL agent each implementation cycle. Use when asked to "run the eval", test the agent on N / all 30 BIRD questions, measure if we improved, judge node-level behaviour (explore/generate/verify/evidence/revise), score the lenient chatbot metric, explain why a question fails, or produce an eval report. Two total metrics: strict BIRD execution accuracy + a lenient 0/0.5/1 LLM-judge. Wraps run_eval.py → extract_traces.py → judge → analyze_run.py and writes docs/eval-report_<run_id>.md.
---

# SQL agent eval + report (run at the end of every agent change)

Repeatable framework to answer "did this change improve the agent?" at the **total** and **per-node**
level. Two total metrics live side by side:
- **BIRD execution accuracy** — strict, programmatic (executed row sets vs gold). From `run_eval.py`.
- **Lenient score 0 / 0.5 / 1** — LLM-judge, computed a posteriori: "would this do as a chatbot reply
  vs gold", tolerant of extra columns / non-exhaustive rows. **Judge ≠ agent** (use a stronger, different model).

Per-node assessment is **dynamic**: it judges whatever nodes appear in the trace (role rubric below +
generic fallback), so it survives architecture changes. Node data comes from **Langfuse** (the `/answer`
API intentionally does not expose intermediate reasoning).

## 0. Preconditions
- Agent server on `:8001` running the **current** code (uvicorn does not auto-reload — if the user just
  changed the agent, ask them to restart). Check `curl -s -o /dev/null -w "%{http_code}" :8001/health`.
- Off-GPU model = a fast **non-reasoning instruct** (e.g. `qwen/qwen3-30b-a3b-instruct-2507` in `.env`).
- Langfuse up (for trace extraction); BIRD DBs under `data/bird/`.

## 1. Pick the set & smoke-probe
`evals/eval_set.jsonl` = 30 questions; a slice e.g. `sed -n '11,20p' evals/eval_set.jsonl > evals/eval_set_10b.jsonl`.
Probe one question first and check `history` shows the expected node flow before a full run.

## 2. Run the eval (strict metric + capture)
First, if the agent changed, **bump `AGENT_VERSION` in `agent/__init__.py`** (single source; `server.py`
imports it). Then run — always pass a descriptive `--run-id` (results saved to `results/eval_<run_id>.json`,
never overwriting):
```
uv run python evals/run_eval.py --eval-set evals/eval_set.jsonl --concurrency 1 --timeout 300 --run-id <id>
```
The results JSON is **historicized**: it records `agent_version` (from code), `git_sha` + `git_dirty`,
`model`/`base_url` (from `.env`), and `created_at` (UTC). Long runs: launch in background, poll for `[n/N]`.

## 3. Extract per-node traces from Langfuse
```
uv run python evals/extract_traces.py results/eval_<id>.json   # -> results/eval_<id>.traces.json
```

## 4. Judge (lenient metric + per-node + free-text)  →  results/eval_<id>.judged.json
**Primary path — Claude-as-judge (you, when running this skill).** For each question read: the question,
gold SQL + executed gold rows, the agent's final SQL + executed rows, `final_correct`, whether any step
was gold-correct (`per_iteration[*].correct`), and the node `steps` from the traces file. Emit one record
per question into `results/eval_<id>.judged.json` with this exact schema:
```
{ "run_id": "<id>", "judge_model": "claude-code", "results": [
  { "question", "db_id", "bird_correct": bool,
    "lenient_score": 0 | 0.5 | 1,
    "failure_reason": "1-2 sentences on what it misunderstood (\"\" if score==1)",
    "nodes": {
      "explore":  {"verdict":"good|partial|poor|na","suggested_right_queries":bool,"note":""},
      "generate": {"correct_at_some_step":bool,"error_despite_context":bool,"ignored_info":"","note":""},
      "revise":   {"helpful":"improved|neutral|worse|na","used_evidence":bool,"note":""},
      "verify":   {"verdict_justified":bool,"note":""},
      "<any other node present>": {"did_its_job":bool,"note":""}   // generic => dynamic
    } } ] }
```
Lenient scoring: **1** = same answer for a user (cosmetic diffs/extra cols ok); **0.5** = partially
acceptable (non-exhaustive, extra/missing rows, slightly off granularity); **0** = wrong entity/number/empty.
Node rubric (be concrete in `note`): explore = did its queries target the columns/values the question
needs? generate/revise = did any step produce a gold-correct query (even if verify rejected it)? did it
err **despite** findings/evidence in context that contained the right fact (name it in `ignored_info`)?
revise = did it improve and use the gathered evidence? verify = was accept/reject justified vs gold?

**Headless path — script** (for CI / no Claude session). Judge model defaults to a strong model ≠ agent
(`anthropic/claude-sonnet-4.5` via the OpenRouter key; override `--model` / `JUDGE_MODEL`):
```
uv run python evals/judge_run.py results/eval_<id>.json            # all questions
uv run python evals/judge_run.py results/eval_<id>.json --limit 5  # smoke subset
```

## 5. Aggregate both metrics + node verdicts
```
uv run python evals/analyze_run.py results/eval_<id>.json --langfuse --judged
```
Prints: BIRD overall + per-iteration (does the loop earn its keep, iter0→final); verifier confusion
matrix vs gold (TP/TN/FP/FN); revise fixed/broke/regressions; per-DB; failure list; Langfuse node-health
(explore queries+UPPER, verify parse-fails+votes, evidence/revise calls, execute errors); and the
**lenient mean + distribution** plus **per-node judge aggregates** + per-question failure reasons.

## 6. Update the history index
Append/refresh this run's row in `docs/eval-history.md` (idempotent by run_id, newest first):
```
uv run python evals/update_history.py results/eval_<id>.json
```
This is the across-cycles view: created (UTC) · agent_version · git · run_id · model · strict · iter0→final · lenient · n.

## 7. Write the report
Filename convention (historicized): **`docs/eval-report_<YYYYMMDD-HHMMSS>_<agent_version>_<run_id>.md`**
(timestamp from `created_at`, `agent_version` from the run; for past runs not stamped in the JSON, read it
from the Langfuse trace `metadata.agent_version`). Sections: header (run_id, **agent_version,
git_sha(+dirty), model, created_at (UTC)**, config); **two headline metrics** (BIRD + lenient); loop value
(iter0→final); failure taxonomy
(domain-knowledge gap / verifier FP / interpretation / output-format); **node-by-node** (is each doing its
job, with the judge verdicts + the deterministic confusion matrix + the evidence/revise roles); honest
run-to-run comparison; prioritized next levers.

## Recurring lessons (what to look for)
- **Verifier FP is usually the ceiling** (accepts plausible-but-wrong → loop stops). FN wastes iterations.
- **Domain-knowledge cluster (~1/3 of BIRD fails)** needs external `evidence` (clinical ranges, status
  codes, label encodings) absent from the data — a hard ceiling; the evidence node helps only with
  data-derivable facts.
- **Latency/hangs are LLM-side**, never the DB. A reasoning model + many calls/question causes timeouts.
- Keep the per-node rubric in step 4 in sync if you add/rename nodes; the framework still scores unknown
  nodes via the generic fallback.
