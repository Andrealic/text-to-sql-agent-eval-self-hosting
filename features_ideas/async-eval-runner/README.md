# Feature idea: async eval runner

**File:** `evals/run_eval.py`
**Status:** ✅ implemented (see Implementation note below)
**Motivation:** speed up `run_eval` by overlapping in-flight agent requests.

## Metadata

| Field | Value |
| --- | --- |
| Created | 2026-06-24 19:14:08 +0200 |
| Author | Andrealic (andrea.licata@hiop.io) |
| Branch | `agent-mode` |
| Commit | `de4b73f` — "feat: eval" |
| Commit (full) | `de4b73f40188490807bcac377ae43789728078fa` |

> Snapshot of the repo at idea-capture time. Use the commit above to diff
> against the code this proposal was written for.

## Problem

`run_eval.py` is fully synchronous:

- `main()` loops over questions one at a time (`for i, q in enumerate(questions, 1)`).
- Each `eval_one` blocks on `httpx.post(..., timeout=120.0)`.

The bottleneck is the agent call — network I/O sitting idle while waiting for
the LLM. With 30 questions that means 30 sequential round-trips. Async lets us
overlap the in-flight requests and cut wall-clock time dramatically.

The local sqlite queries (`run_sql`) are *not* the bottleneck — they run in
milliseconds against a local file.

## Proposed changes

### 1. Make the agent call async
`eval_one` uses blocking `httpx.post`. Switch to a shared `httpx.AsyncClient`:

```python
async def eval_one(question: dict, client: httpx.AsyncClient, agent_url: str) -> dict:
    ...
    resp = await client.post(agent_url, json={...}, timeout=120.0)
```

Create the `AsyncClient` once in `main` and pass it in, so connections are pooled
(avoid a client-per-call).

### 2. Handle the blocking `run_sql` calls
`run_sql` uses `sqlite3` synchronously; inside an async function it blocks the
event loop. Two options:

- **Simplest:** leave it sync. sqlite queries are local and fast (ms), so briefly
  blocking the loop is acceptable for a toy eval.
- **Cleaner:** wrap with `await asyncio.to_thread(run_sql, db_id, sql)` so they
  don't stall concurrent agent requests.

Start with sync unless the loop is observed stalling.

### 3. Replace the sequential loop with bounded concurrency
The real win. Use a semaphore to cap how many agents we hit at once (so we don't
overwhelm vLLM):

```python
async def main() -> None:
    ...
    sem = asyncio.Semaphore(args.concurrency)  # e.g. default 5

    async def worker(i, q):
        async with sem:
            print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
            return await eval_one(q, client, args.agent_url)

    async with httpx.AsyncClient() as client:
        t0 = time.monotonic()
        results = await asyncio.gather(*(worker(i, q) for i, q in enumerate(questions, 1)))
        elapsed = time.monotonic() - t0
```

### 4. Wire up the entrypoint
- `import asyncio`
- Add a `--concurrency` CLI arg (default 5).
- Change `if __name__ == "__main__": main()` → `asyncio.run(main())`.

## What does NOT change
`summarize`, `canonicalize`, `matches`, `_attempts_from_history` are pure/CPU —
leave them sync. Output ordering is preserved because `asyncio.gather` returns in
input order, so the result JSON stays deterministic.

## Current backend reality (as of this branch)

On the `agent-mode` branch **vLLM is not running**. The agent still calls
**OpenRouter** (the default `VLLM_BASE_URL` in `graph.py` points at a hosted API,
not the real H100 endpoint). This matters for how async should be tuned:

- **Latency numbers from this branch are meaningless for the SLO.** OpenRouter
  is a shared remote API with its own queueing — it is not our single H100. Per
  the README / AGENTS.md, pass rate, latency and the final SLO must come from the
  real `Qwen3-30B-A3B` on the H100. Async here only buys faster *iteration while
  developing*, not SLO evidence.
- **Watch OpenRouter rate limits.** Firing many concurrent requests at a hosted
  API can trip 429s. Keep the semaphore low (≈3–5) and make sure the transport
  error path in `eval_one` already handles non-2xx gracefully (it does — the
  `try/except` records `transport_error`). Consider light retry/backoff on 429 if
  it becomes a problem.

So the async work is worth doing now for dev-loop speed, but the SLO-relevant
re-run still has to happen later against vLLM on the H100.

## Caveat: concurrency vs. SLO measurement

The concurrency cap matters for the eval's meaning. Once we are on vLLM/H100,
firing all 30 questions at once effectively load-tests the server, and
per-question latency will inflate — which muddies the Phase 5 pass-rate numbers.
A modest semaphore (≈5) gets the wall-clock speedup without distorting latency
too much.

The README treats load testing (Phase 6) as a separate concern from eval accuracy
(Phase 5), so keep eval concurrency low and leave dedicated latency/RPS
measurement to the load test driver.

## Implementation note

Implemented as proposed. Changes landed in `evals/run_eval.py`:

- `import asyncio` added.
- `eval_one` → `async def`, takes a shared `httpx.AsyncClient`, awaits
  `client.post(...)`. Both `run_sql` calls (gold + per-iteration) wrapped in
  `asyncio.to_thread` so sqlite doesn't block the event loop (went with the
  "cleaner" option from §2 rather than leaving them sync).
- `main` → `async def`; sequential loop replaced with `asyncio.gather` over
  per-question workers sharing one `AsyncClient`. New `--concurrency` arg
  (default 5) feeds an `asyncio.Semaphore`. Progress print moved into the worker
  (counts completions). Result ordering preserved by `gather`, so output JSON
  stays deterministic.
- Entrypoint → `asyncio.run(main())`.

Not yet done: not smoke-tested end-to-end against a running agent; no 429
retry/backoff added (still just records `transport_error`).
