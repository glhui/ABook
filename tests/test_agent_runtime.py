"""Agent 运行时实时观察入口的测试。"""

import unittest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.test import TestModel

from agent_runtime import run_observed


# 验证观察入口既实时转发模型响应，也保留最终结构化运行结果。
class ObservedAgentRunTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_observed_returns_output_and_reports_response(self: "ObservedAgentRunTests") -> None:
        responses: list[ModelResponse] = []
        agent = Agent(TestModel(custom_output_text="completed"))

        result, messages = await run_observed(agent, "perform task", on_response=responses.append)

        self.assertEqual(result.output, "completed")
        self.assertTrue(responses)
        self.assertTrue(messages)
