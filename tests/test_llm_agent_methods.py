import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


def _load_llm_module():
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda: None

    fake_litellm = types.ModuleType("litellm")

    class FakeRouter:
        def __init__(self, model_list):
            self.model_list = model_list

        async def acompletion(self, model, messages):
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=f"{model}:{messages[-1]['content']}")
                    )
                ]
            )

    fake_litellm.Router = FakeRouter

    with patch.dict(sys.modules, {"dotenv": fake_dotenv, "litellm": fake_litellm}):
        if "agenteconomy.llm.llm" in sys.modules:
            del sys.modules["agenteconomy.llm.llm"]
        return importlib.import_module("agenteconomy.llm.llm")


llm = _load_llm_module()


class CallLLMAgentMethodTests(unittest.IsolatedAsyncioTestCase):
    async def test_call_llm_uses_configured_agent_method_when_valid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text("agent_name: cot\n", encoding="utf-8")

            agent_mock = AsyncMock(return_value="agent-result")
            direct_mock = AsyncMock(return_value="direct-result")

            with patch.object(llm, "CONFIG_PATH", config_path), patch.object(
                llm, "_call_agent_method", agent_mock
            ), patch.object(llm, "_call_direct_llm_raw", direct_mock):
                result = await llm.call_llm("user prompt", "system prompt")

            self.assertEqual(result, "agent-result")
            agent_mock.assert_awaited_once()
            direct_mock.assert_not_awaited()

    async def test_call_llm_falls_back_to_direct_mode_when_agent_name_invalid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text("agent_name: invalid_method\n", encoding="utf-8")

            agent_mock = AsyncMock(return_value="agent-result")
            direct_mock = AsyncMock(return_value="direct-result")

            with patch.object(llm, "CONFIG_PATH", config_path), patch.object(
                llm, "_call_agent_method", agent_mock
            ), patch.object(llm, "_call_direct_llm_raw", direct_mock):
                result = await llm.call_llm("user prompt", "system prompt")

            self.assertEqual(result, "direct-result")
            agent_mock.assert_not_awaited()
            direct_mock.assert_awaited_once()

    async def test_call_llm_simple_delegates_with_simple_model_type(self):
        call_mock = AsyncMock(return_value="simple-result")

        with patch.object(llm, "call_llm", call_mock):
            result = await llm.call_llm_simple("hello", "sys")

        self.assertEqual(result, "simple-result")
        call_mock.assert_awaited_once_with(
            "hello",
            system_prompt="sys",
            model_type="simple",
        )

    async def test_call_llm_strong_delegates_with_strong_model_type(self):
        call_mock = AsyncMock(return_value="strong-result")

        with patch.object(llm, "call_llm", call_mock):
            result = await llm.call_llm_strong("hello", "sys")

        self.assertEqual(result, "strong-result")
        call_mock.assert_awaited_once_with(
            "hello",
            system_prompt="sys",
            model_type="strong",
        )
