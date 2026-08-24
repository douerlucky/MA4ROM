import json
import multiprocessing as mp
import os
import re
import time
from json import JSONDecoder
from types import SimpleNamespace

from openai import BadRequestError

from utils.llm_set import client
from config import (
    LLM_MODEL,
    LLM_FALLBACK_MODELS,
    LLM_HARD_TIMEOUT_SECONDS,
    LLM_TIMEOUT_SECONDS,
    LLM_MAX_RETRIES,
    LLM_THINKING_ENABLED,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_MAX_API_ATTEMPTS,
    LLM_MAX_TOTAL_TOKENS,
)
from utils.llm_metrics import (
    LLMBudgetExceeded,
    ensure_llm_budget,
    record_llm_attempt,
    record_llm_failure,
    record_llm_success,
)

_DEFAULT_SYSTEM = (
    "你是本体对齐专家（OBDA）。"
    "从给定候选列表中选出最符合的一项。"
    "严格输出 JSON，不含任何 Markdown 包裹或注释。"
)


class LLMHardTimeoutError(TimeoutError):
    """One model request exceeded the process-level deadline."""


def _api_call_worker(
    send_conn,
    model,
    system_prompt,
    prompt,
    request_timeout,
    max_output_tokens,
    thinking_enabled,
):
    """Execute one API request in a disposable child process."""
    try:
        request = dict(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            timeout=request_timeout,
            max_tokens=max_output_tokens,
            response_format={"type": "json_object"},
        )
        if not thinking_enabled:
            # DeepSeek's OpenAI-compatible API accepts provider-specific
            # controls through extra_body.  This explicitly selects the
            # non-thinking V4-Flash path and prevents hidden reasoning tokens.
            request["extra_body"] = {"thinking": {"type": "disabled"}}
        response = client.chat.completions.create(**request)
        usage = getattr(response, "usage", None)
        send_conn.send(
            {
                "ok": True,
                "content": (response.choices[0].message.content or "").strip(),
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                },
            }
        )
    except BaseException as exc:
        send_conn.send({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})
    finally:
        send_conn.close()


def _call_api_with_deadline(model, system_prompt, prompt):
    """Return content and usage, forcibly recovering only a hung request."""
    if os.name != "posix":
        request = dict(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            timeout=LLM_TIMEOUT_SECONDS,
            max_tokens=LLM_MAX_OUTPUT_TOKENS,
            response_format={"type": "json_object"},
        )
        if not LLM_THINKING_ENABLED:
            request["extra_body"] = {"thinking": {"type": "disabled"}}
        response = client.chat.completions.create(**request)
        return (response.choices[0].message.content or "").strip(), response

    context = mp.get_context("fork")
    recv_conn, send_conn = context.Pipe(duplex=False)
    worker = context.Process(
        target=_api_call_worker,
        args=(
            send_conn,
            model,
            system_prompt,
            prompt,
            LLM_TIMEOUT_SECONDS,
            LLM_MAX_OUTPUT_TOKENS,
            LLM_THINKING_ENABLED,
        ),
        daemon=True,
    )
    worker.start()
    send_conn.close()
    try:
        if not recv_conn.poll(LLM_HARD_TIMEOUT_SECONDS):
            worker.terminate()
            worker.join(timeout=2)
            raise LLMHardTimeoutError(
                f"LLM call exceeded the {LLM_HARD_TIMEOUT_SECONDS}-second hard deadline."
            )
        payload = recv_conn.recv()
    finally:
        recv_conn.close()
        if worker.is_alive():
            worker.join(timeout=1)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=1)

    if not payload.get("ok"):
        raise RuntimeError(f"{payload.get('error_type', 'LLMError')}: {payload.get('error', '')}")
    return payload["content"], SimpleNamespace(
        usage=SimpleNamespace(**payload.get("usage", {}))
    )


def call_llm(prompt: str, system: str = None, prefer_fast: bool = True) -> dict:
    """
    调用 LLM，返回解析后的 dict。
    模型名称从 config.LLM_MODEL 读取。
    """
    # Deterministic replay is useful when validating a structural mapping fix
    # separately from a provider/model change.  It is opt-in and raises before
    # any network request; callers already have conservative local fallbacks.
    if os.environ.get("MAMG_LLM_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError("LLM offline mode is enabled for this replay")

    def _extract_json_dict(raw_text: str) -> dict:
        text = (raw_text or "").strip()
        # 去掉 markdown 包裹 / 思维链标签
        text = re.sub(r"^```json|^```|```$", "", text, flags=re.MULTILINE).strip()
        text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()

        decoder = JSONDecoder()
        for i, ch in enumerate(text):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
        raise json.JSONDecodeError("No valid JSON object found in model output", text, 0)

    # The camera-ready policy is intentionally single-model.  Keep the loop
    # shape for compatibility with older callers, but the configured fallback
    # list is empty and is never populated from a provider alias.
    fallback_models = [m for m in (LLM_FALLBACK_MODELS or []) if m != LLM_MODEL]
    models = [LLM_MODEL] + fallback_models
    last_err = None

    for model in models:
        for attempt in range(1, max(1, int(LLM_MAX_RETRIES)) + 1):
            try:
                system_prompt = system or _DEFAULT_SYSTEM
                ensure_llm_budget(
                    max_api_attempts=LLM_MAX_API_ATTEMPTS,
                    max_total_tokens=LLM_MAX_TOTAL_TOKENS,
                    estimated_prompt_tokens=(
                        len(prompt) + len(system_prompt) + 3
                    ) // 4,
                    max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
                )
                record_llm_attempt()
                raw, response = _call_api_with_deadline(
                    model, system_prompt, prompt
                )
                record_llm_success(
                    response,
                    model=model,
                    prompt=prompt,
                    system=system_prompt,
                )
                return _extract_json_dict(raw)
            except LLMBudgetExceeded:
                # A hard budget is a terminal run condition.  Retrying or
                # falling through to another model would defeat the guard.
                raise
            except BadRequestError as e:
                record_llm_failure()
                last_err = e
                msg = str(e).lower()
                # 模型不存在时，直接切到下一候选模型
                if "model not exist" in msg or "invalid_request_error" in msg:
                    break
                if attempt >= max(1, int(LLM_MAX_RETRIES)):
                    break
                time.sleep(min(2 ** (attempt - 1), 4))
            except Exception as e:
                record_llm_failure()
                last_err = e
                if attempt >= max(1, int(LLM_MAX_RETRIES)):
                    break
                time.sleep(min(2 ** (attempt - 1), 4))

    if last_err:
        raise last_err
    raise RuntimeError("LLM call failed with unknown error")
