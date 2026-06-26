# CLAUDE.md

Le istruzioni per gli agenti che lavorano su questo repository sono in **[AGENTS.md](./AGENTS.md)**.

Leggi quel file: contiene panoramica del progetto, layout, comandi, convenzioni e stile di lavoro atteso.

## Eval ad ogni giro
Alla fine di **ogni** giro di implementazione che tocca l'agente (`agent/`, prompt, grafo), invoca la
skill **`sql-agent-eval`** per rimisurare: accuracy BIRD (stretta) + metrica lasca 0/0.5/1 (LLM judge) +
salute per-nodo, e aggiornare un report in `docs/eval-report_<run_id>.md`. È il modo per sapere se stiamo
migliorando. Dettagli in `.claude/skills/sql-agent-eval/SKILL.md`.
