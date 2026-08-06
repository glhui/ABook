"""Agent 运行时实时观察入口的测试。"""

import unittest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.test import TestModel

from agent_profiles import AgentModelConfig
from agent_runtime import compact_history, run_observed, summarize_and_compact_history


# 验证观察入口既实时转发模型响应，也保留最终结构化运行结果。
class ObservedAgentRunTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_observed_returns_output_and_reports_response(self: "ObservedAgentRunTests") -> None:
        responses: list[ModelResponse] = []
        agent = Agent(TestModel(custom_output_text="completed"))

        result, messages = await run_observed(agent, "perform task", on_response=responses.append)

        self.assertEqual(result.output, "completed")
        self.assertTrue(responses)
        self.assertTrue(messages)

    # 达到阈值时运行入口应丢弃最早完整轮次，而不是把超长历史直接交给模型。
    async def test_run_observed_compacts_history_at_configured_threshold(self: "ObservedAgentRunTests") -> None:
        agent = Agent(TestModel(custom_output_text="completed"))
        _, history = await run_observed(agent, "first task")
        config = AgentModelConfig(context_window_tokens=100, max_output_tokens=1, compaction_threshold=0.5)

        compacted = compact_history(history, config)
        result, messages = await run_observed(agent, "second task", history, model_config=config)

        self.assertEqual(result.output, "completed")
        self.assertLessEqual(len(compacted), len(history))
        self.assertTrue(messages)

    # 压缩结果应以模型生成的检查点替换最早对话，并保留最近完整轮次。
    async def test_summarize_and_compact_history_adds_summary_checkpoint(self: "ObservedAgentRunTests") -> None:
        agent = Agent(TestModel(custom_output_text="history summary"))
        _, history = await run_observed(agent, "first task")
        config = AgentModelConfig(context_window_tokens=100, max_output_tokens=1, compaction_threshold=0.5)

        compacted = await summarize_and_compact_history(agent, history, config)

        self.assertGreaterEqual(len(compacted), 2)
        self.assertIn("history summary", str(compacted[0]))
