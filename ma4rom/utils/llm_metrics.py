from __future__ import annotations

from copy import deepcopy
from typing import Any


class LLMBudgetExceeded(RuntimeError):
    """Raised before a paid request would exceed the configured hard budget."""


_METRICS = {
    "api_attempts": 0,
    "llm_calls": 0,
    "failed_attempts": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "estimated_input_token_calls": 0,
    "models": {},
}


def reset_llm_metrics() -> None:
    _METRICS.clear()
    _METRICS.update(
        {
            "api_attempts": 0,
            "llm_calls": 0,
            "failed_attempts": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_input_token_calls": 0,
            "models": {},
        }
    )


def record_llm_attempt() -> None:
    _METRICS["api_attempts"] += 1


def record_llm_failure() -> None:
    _METRICS["failed_attempts"] += 1


def ensure_llm_budget(
    *,
    max_api_attempts: int,
    max_total_tokens: int,
    estimated_prompt_tokens: int,
    max_output_tokens: int,
) -> None:
    """Check spend limits before a request leaves the process.

    Usage is not known until the provider responds, so reserve the configured
    maximum output plus a conservative prompt estimate.  This prevents a
    long-context request from starting when it cannot fit inside the budget.
    """
    attempts = int(_METRICS.get("api_attempts", 0))
    total = int(_METRICS.get("total_tokens", 0))
    prompt_estimate = max(1, int(estimated_prompt_tokens))
    output_reserve = max(0, int(max_output_tokens))
    if max_api_attempts > 0 and attempts >= max_api_attempts:
        raise LLMBudgetExceeded(
            f"LLM API attempt budget exhausted ({attempts}/{max_api_attempts})"
        )
    if max_total_tokens > 0 and total + prompt_estimate + output_reserve > max_total_tokens:
        raise LLMBudgetExceeded(
            "LLM token budget would be exceeded: "
            f"used={total}, reserve={prompt_estimate + output_reserve}, "
            f"limit={max_total_tokens}"
        )


def _usage_value(usage: Any, name: str) -> int | None:
    value = getattr(usage, name, None)
    if value is None and isinstance(usage, dict):
        value = usage.get(name)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def record_llm_success(
    response: Any,
    *,
    model: str,
    prompt: str,
    system: str,
) -> None:
    usage = getattr(response, "usage", None)
    input_tokens = _usage_value(usage, "prompt_tokens")
    output_tokens = _usage_value(usage, "completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens")

    if input_tokens is None:
        # Only used when an OpenAI-compatible endpoint omits usage metadata.
        input_tokens = max(1, (len(prompt) + len(system)) // 4)
        _METRICS["estimated_input_token_calls"] += 1
    if output_tokens is None:
        output_tokens = 0
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens

    _METRICS["llm_calls"] += 1
    _METRICS["input_tokens"] += input_tokens
    _METRICS["output_tokens"] += output_tokens
    _METRICS["total_tokens"] += total_tokens

    model_metrics = _METRICS["models"].setdefault(
        model,
        {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    )
    model_metrics["llm_calls"] += 1
    model_metrics["input_tokens"] += input_tokens
    model_metrics["output_tokens"] += output_tokens
    model_metrics["total_tokens"] += total_tokens


def snapshot_llm_metrics() -> dict:
    return deepcopy(_METRICS)


def diff_llm_metrics(before: dict, after: dict) -> dict:
    keys = (
        "api_attempts",
        "llm_calls",
        "failed_attempts",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "estimated_input_token_calls",
    )
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in keys}
