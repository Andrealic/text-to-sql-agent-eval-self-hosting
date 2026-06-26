# AGENTS.md

Guida operativa per agenti AI (Claude Code & co.) che lavorano su questo repository.

## Cos'è questo progetto

Home assignment MLOps: un PoC **text-to-SQL** con focus su **inference + observability**, non sul prodotto.
Un analista fa una domanda in inglese → un agente LangGraph genera SQL → lo esegue su uno
sqlite di [BIRD-bench](https://bird-bench.github.io/) → verifica la plausibilità → eventualmente
rivede. Sotto c'è vLLM che serve `Qwen/Qwen3-30B-A3B-Instruct-2507` su 1× H100, con Prometheus +
Grafana per il serving e Langfuse per le trace dell'agente.

L'assignment è strutturato in **7 fasi** (vedi `README.md`): 0 setup, 1 vLLM, 2 Grafana,
3 agente, 4 tracing, 5 eval, 6 SLO, 7 report. Il deliverable finale è `REPORT.md` + screenshot +
risultati eval. **L'obiettivo del lavoro è il ragionamento e la diagnosi, non la spunta verde.**

> SLO target (Fasi 1 e 6): **P95 latenza end-to-end < 5 s, ≥ 10 RPS** (1 RPS = 1 run completo
> dell'agente al secondo) su finestra di 5 minuti.

## Layout del repo

```
agent/          Agente LangGraph + server FastAPI
  graph.py        Grafo: attach_schema → generate_sql → execute → verify →(revise loop)
  prompts.py      Template di prompt per i nodi LLM
  server.py       Wrapper FastAPI: POST /answer, GET /health (porta 8001)
  execution.py    Esecuzione SQL read-only su sqlite (FORNITO, non modificare)
  schema.py       Rendering dello schema DB per i prompt (FORNITO, non modificare)
evals/
  run_eval.py     Runner eval: execution accuracy + pass rate per-iterazione
  eval_set.jsonl  30 domande curate (gitignored; generato)
load_test/
  driver.py       Load test: --rps N --duration 300
infra/
  prometheus.yml                              Scrape di /metrics di vLLM
  grafana/provisioning/dashboards/serving.json  Dashboard (deliverable Fase 2)
scripts/
  start_vllm.sh   Launch di vLLM (da configurare, Fase 1)
  load_data.py    Scarica il subset BIRD in data/bird/
docker-compose.yml  o11y stack: Prometheus, Grafana, Langfuse (+ postgres/clickhouse/redis/minio)
README.md           La consegna completa, fase per fase — fonte di verità
REPORT.md           Deliverable finale (writeup ≤ 3 pagine)
```

## Comandi

```bash
uv sync                                   # installa dipendenze (usa SEMPRE uv, non pip)
uv run python scripts/load_data.py        # carica i dati BIRD in data/bird/
docker compose up -d                      # avvia lo stack di observability
bash scripts/start_vllm.sh                # avvia vLLM (porta 8000)

# Avvia il server dell'agente (porta 8001)
uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001

# Interroga l'agente
curl -X POST http://localhost:8001/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "...", "db": "..."}'

uv run python evals/run_eval.py                          # eval (Fase 5)
uv run python load_test/driver.py --rps 10 --duration 300  # load test (Fase 6)
```

UI (via port-forward SSH): Grafana `:3000` (admin/admin) · Prometheus `:9090` · Langfuse `:3001`
· vLLM `:8000` · agente `:8001`.

## Convenzioni e vincoli

- **Gestione dipendenze:** `uv` esclusivamente. Le dipendenze stanno in `pyproject.toml`,
  pin esatti in `uv.lock`. Python ≥ 3.12.
- **Cosa è "fornito" e NON va toccato salvo richiesta esplicita:** `agent/execution.py`,
  `agent/schema.py`, il nodo `generate_sql_node` (esempio già completo), e il wiring del grafo.
  Il lavoro vero sta nei nodi `verify`/`revise`, nel router `route_after_verify`, nei prompt
  (`agent/prompts.py`), nella dashboard, nell'eval runner.
- **Config LLM:** `agent/graph.py` legge `VLLM_BASE_URL` / `VLLM_MODEL` / `OPENAI_API_KEY` da env.
  Si può sviluppare contro un backend leggero (API hosted o vLLM CPU con `Qwen3-0.6B`) e passare
  all'H100 solo quando i numeri contano. **Pass rate, latenza e l'SLO finale devono venire dal vero
  `Qwen3-30B-A3B` sull'H100** — vedi tabella "off-GPU" nel README.
- **Segreti:** `.env` è gitignored. Non committare mai chiavi (HF_TOKEN, Langfuse, OpenAI).
  Partire da `.env.example`.
- **Artefatti gitignored:** `data/bird/`, `results/*.json`, `screenshots/*.png`,
  `evals/eval_set.jsonl`, `load_test/perf_pool.jsonl` sono generati e non versionati.
- **Stato attuale del codice (work in progress):** `MAX_ITERATIONS = 1` in `graph.py` di fatto
  disabilita il loop di revise — va alzato (3–5) per far lavorare l'architettura. `verify_ok` è
  trattato come stringa (`"true"`), non bool: maneggiare con cura quando si tocca `route_after_verify`
  o `VerifyResult`. Il default in `graph.py` punta a OpenRouter, non al vero endpoint vLLM.

## Stile di lavoro atteso

- Questo è materiale d'esame/apprendimento: privilegiare cambiamenti **comprensibili e motivati**.
  Quando si modifica config di serving, prompt o grafo, spiegare il *perché* (è esattamente ciò
  che viene valutato).
- Per le fasi di tuning (1 e 6) il pattern richiesto è: *"visto X → ipotizzato Y → cambiato Z →
  risultato W"*, una riga in `REPORT.md` per iterazione, con screenshot Grafana a supporto.
- Non productionizzare oltre il livello toy (niente Kubernetes, niente frontend). Vedi disclaimer
  nel README.
- `README.md` è la fonte di verità sulla consegna: in caso di dubbio su cosa fare in una fase,
  consultarlo prima di assumere.
- **Eval ad ogni giro:** dopo ogni modifica all'agente invoca la skill **`sql-agent-eval`**
  (`.claude/skills/sql-agent-eval/SKILL.md`). Misura due metriche totali — accuracy BIRD (stretta,
  programmatica) e una metrica **lasca 0/0.5/1** via LLM-judge (giudice ≠ agente) — più una valutazione
  **per-nodo dinamica** (explore/generate/verify/evidence/revise, con fallback per nodi nuovi) e un
  commento testo libero sul perché ogni domanda fallisce. Pipeline:
  `run_eval.py → extract_traces.py (da Langfuse) → judge (Claude o evals/judge_run.py) → analyze_run.py`,
  con report in `docs/eval-report_<run_id>.md`. Il `/answer` NON espone gli step intermedi: i dati
  per-nodo vengono da Langfuse.
