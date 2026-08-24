"""Local guards for the paid-model boundary.

These tests never create an HTTP client request.  They verify that the repair
path is single-model/non-thinking and that a hard budget is checked before the
request function is entered.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "ma4rom"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import llm_client  # noqa: E402
from utils.llm_metrics import (  # noqa: E402
    LLMBudgetExceeded,
    reset_llm_metrics,
    snapshot_llm_metrics,
)


class LLMGuardTests(unittest.TestCase):
    def setUp(self):
        reset_llm_metrics()

    def tearDown(self):
        reset_llm_metrics()

    def test_camera_ready_model_has_no_fallback(self):
        from config import LLM_FALLBACK_MODELS, LLM_MODEL, LLM_THINKING_ENABLED

        self.assertEqual(LLM_MODEL, "deepseek-v4-flash")
        self.assertEqual(LLM_FALLBACK_MODELS, [])
        self.assertFalse(LLM_THINKING_ENABLED)

    def test_budget_is_checked_before_request(self):
        with patch.object(
            llm_client,
            "_call_api_with_deadline",
            side_effect=AssertionError("request must not be entered"),
        ):
            with patch.object(llm_client, "LLM_MAX_API_ATTEMPTS", 1):
                # One prior attempt is enough to exhaust the local cap.
                from utils.llm_metrics import record_llm_attempt

                record_llm_attempt()
                with self.assertRaises(LLMBudgetExceeded):
                    llm_client.call_llm("return {}")

    def test_success_path_records_only_one_call(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=12,
                completion_tokens=8,
                total_tokens=20,
            )
        )
        with patch.object(
            llm_client,
            "_call_api_with_deadline",
            return_value=("{\"ok\": true}", response),
        ):
            result = llm_client.call_llm("return JSON")
        self.assertEqual(result, {"ok": True})
        metrics = snapshot_llm_metrics()
        self.assertEqual(metrics["api_attempts"], 1)
        self.assertEqual(metrics["llm_calls"], 1)
        self.assertEqual(metrics["total_tokens"], 20)

    def test_worker_request_disables_thinking_and_caps_output(self):
        captured = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))],
                    usage=SimpleNamespace(
                        prompt_tokens=1,
                        completion_tokens=1,
                        total_tokens=2,
                    ),
                )

        class FakeClient:
            chat = SimpleNamespace(completions=FakeCompletions())

        class Pipe:
            def __init__(self):
                self.payload = None

            def send(self, payload):
                self.payload = payload

            def close(self):
                pass

        pipe = Pipe()
        with patch.object(llm_client, "client", FakeClient()):
            llm_client._api_call_worker(
                pipe,
                "deepseek-v4-flash",
                "system",
                "user",
                3,
                512,
                False,
            )
        self.assertEqual(captured["model"], "deepseek-v4-flash")
        self.assertEqual(captured["max_tokens"], 512)
        self.assertEqual(captured["response_format"], {"type": "json_object"})
        self.assertEqual(
            captured["extra_body"],
            {"thinking": {"type": "disabled"}},
        )


if __name__ == "__main__":
    unittest.main()
