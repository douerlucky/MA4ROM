import httpx

from openai import OpenAI
from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_TIMEOUT_SECONDS,
    LLM_MAX_RETRIES,
)   # 统一从 config.py 读取

# Do not inherit the desktop system proxy: it can leave a local proxy socket
# readable forever without yielding a model response.
_http_client = httpx.Client(
    timeout=httpx.Timeout(
        LLM_TIMEOUT_SECONDS,
        connect=min(10.0, float(LLM_TIMEOUT_SECONDS)),
        read=LLM_TIMEOUT_SECONDS,
        write=LLM_TIMEOUT_SECONDS,
        pool=min(10.0, float(LLM_TIMEOUT_SECONDS)),
    ),
    trust_env=False,
)

client = OpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
    http_client=_http_client,
    timeout=LLM_TIMEOUT_SECONDS,
    # Explicit retries are measured by utils.llm_client.
    max_retries=0,
)
