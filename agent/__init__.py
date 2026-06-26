"""Agent package.

AGENT_VERSION is the single source of truth for the agent version. Bump it on every
agent-architecture change — eval runs/reports record it (see the sql-agent-eval skill)
so results are historicized by version + timestamp.
"""

AGENT_VERSION = "v0.2.0"
