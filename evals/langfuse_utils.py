"""Tiny read-only Langfuse client shared by the eval tooling.

Reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST from .env and
exposes the two calls the eval scripts need. Returns None from `client()` if
Langfuse is not configured, so callers can degrade gracefully.
"""
from __future__ import annotations

import base64
import datetime as _dt
import json
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _parse_ts(ts: str) -> _dt.datetime:
    """Parse a Langfuse/ISO timestamp (handles trailing 'Z')."""
    return _dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))


class LangfuseClient:
    def __init__(self, public_key: str, secret_key: str, host: str):
        self.host = host.rstrip("/")
        self._auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()

    def _get(self, path: str, **params):
        q = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{self.host}/api/public/{path}" + (f"?{q}" if q else "")
        req = urllib.request.Request(url, headers={"Authorization": f"Basic {self._auth}"})
        return json.load(urllib.request.urlopen(req, timeout=30))

    def traces_in_window(self, lo: _dt.datetime, hi: _dt.datetime, limit: int = 100) -> list[dict]:
        """Traces whose timestamp falls within [lo, hi] (timezone-aware datetimes)."""
        data = self._get("traces", limit=limit).get("data", [])
        return [t for t in data if lo <= _parse_ts(t["timestamp"]) <= hi]

    def observations(self, trace_id: str, limit: int = 100) -> list[dict]:
        obs = self._get("observations", traceId=trace_id, limit=limit).get("data", [])
        return sorted(obs, key=lambda o: o.get("startTime") or "")


def client(env_path: Path | None = None) -> LangfuseClient | None:
    """Build a client from .env, or None if keys are missing / dotenv absent."""
    try:
        from dotenv import dotenv_values
    except Exception:
        return None
    v = dotenv_values(env_path or (ROOT / ".env"))
    pk, sk = v.get("LANGFUSE_PUBLIC_KEY"), v.get("LANGFUSE_SECRET_KEY")
    if not (pk and sk):
        return None
    return LangfuseClient(pk, sk, v.get("LANGFUSE_HOST") or "http://localhost:3001")
