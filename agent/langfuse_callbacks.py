"""Langfuse LangChain callbacks with OpenRouter cost forwarding.

Langfuse's stock CallbackHandler records token usage but ignores OpenRouter's
``response_metadata["cost"]``. OpenRouter returns the billed USD amount on every
completion; we attach it as ``cost_details`` so per-generation and trace totals
are correct without manual model pricing in the Langfuse UI.
"""
from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from langchain_core.outputs import ChatGeneration, LLMResult
from langfuse.langchain.CallbackHandler import (
    LangchainCallbackHandler,
    _extract_raw_response,
    _parse_model,
    _parse_usage,
)
from langfuse.logger import langfuse_logger


def _extract_openrouter_cost_details(response: LLMResult) -> dict[str, float] | None:
    """Read OpenRouter usage.cost / cost_details from LangChain LLMResult."""
    for generation in response.generations:
        for chunk in generation:
            message = getattr(chunk, "message", None)
            if message is None:
                continue
            metadata = getattr(message, "response_metadata", None)
            if not isinstance(metadata, dict):
                continue

            cost = metadata.get("cost")
            if not isinstance(cost, (int, float)):
                continue

            details: dict[str, float] = {"total": float(cost)}
            raw_breakdown = metadata.get("cost_details")
            if isinstance(raw_breakdown, dict):
                for key, value in raw_breakdown.items():
                    if isinstance(value, (int, float)):
                        details[str(key)] = float(value)
            return details

    return None


def _parse_model_from_response(response: LLMResult) -> str | None:
    """Prefer OpenRouter model_name in response_metadata over llm_output."""
    model = _parse_model(response)
    if model:
        return str(model)

    for generation in response.generations:
        for chunk in generation:
            message = getattr(chunk, "message", None)
            if message is None:
                continue
            metadata = getattr(message, "response_metadata", None)
            if not isinstance(metadata, dict):
                continue
            name = metadata.get("model_name") or metadata.get("model")
            if name:
                return str(name)
    return None


class OpenRouterCostCallbackHandler(LangchainCallbackHandler):
    """Langfuse handler that forwards OpenRouter-reported cost on each LLM call."""

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            self._log_debug_event(
                "on_llm_end", run_id, parent_run_id, response=response, kwargs=kwargs
            )
            response_generation = response.generations[-1][-1]
            extracted_response = (
                self._convert_message_to_dict(response_generation.message)
                if isinstance(response_generation, ChatGeneration)
                else _extract_raw_response(response_generation)
            )

            llm_usage = _parse_usage(response)
            model = _parse_model_from_response(response)
            cost_details = _extract_openrouter_cost_details(response)

            generation = self._detach_observation(run_id)

            if generation is not None:
                update_kwargs: dict[str, Any] = {
                    "output": extracted_response,
                    "usage": llm_usage,
                    "usage_details": llm_usage,
                    "input": kwargs.get("inputs"),
                    "model": model,
                }
                if cost_details is not None:
                    update_kwargs["cost_details"] = cost_details

                generation.update(**update_kwargs).end()

        except Exception as e:  # noqa: BLE001
            langfuse_logger.exception(e)

        finally:
            self._updated_completion_start_time_memo.discard(run_id)

            if parent_run_id is None:
                self._clear_root_run_resume_key(run_id)
                self._reset(run_id)
