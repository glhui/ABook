"""测试工具型 Agent Loop，不访问真实模型接口。"""

import unittest

from pydantic_ai import UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from experiments.agent_loop.main import (
    Skill,
    SkillMetadata,
    TextInspection,
    create_agent,
    inspect_text,
    run_agent_loop,
)


class InspectTextTests(unittest.TestCase):
    """验证只读工具提供稳定且明确的文本指标。"""

    def test_inspect_text_counts_unicode_lines_and_paragraphs(self) -> None:
        """空白行分隔段落，字符数使用 Python Unicode 语义。"""
        text = "请帖 A\n\n第二段\n第三行"

        inspection = inspect_text(text)

        self.assertEqual(len(text), inspection.total_characters)
        self.assertEqual(
            sum(not character.isspace() for character in text),
            inspection.non_whitespace_characters,
        )
        self.assertEqual(4, inspection.line_count)
        self.assertEqual(2, inspection.paragraph_count)

    def test_inspect_text_handles_whitespace_only_text(self) -> None:
        """只有空白的文本没有非空白字符和段落。"""
        inspection = inspect_text(" \n")

        self.assertEqual(
            TextInspection(
                total_characters=2,
                non_whitespace_characters=0,
                line_count=2,
                paragraph_count=0,
            ),
            inspection,
        )


class AgentLoopTests(unittest.TestCase):
    """验证模型调用、工具执行、观察回填和继续生成的完整消息序列。"""

    def setUp(self) -> None:
        self.skill = Skill(
            metadata=SkillMetadata(
                name="general",
                description="用于离线测试的通用 Skill",
            ),
            content="回答用户请求，并使用工具验证客观限制。",
        )

    def test_agent_observes_tool_result_before_final_output(self) -> None:
        """工具返回值应进入同一次运行历史并触发第二次模型请求。"""
        model = TestModel(
            call_tools=["inspect_text"],
            custom_output_text="最终答案",
        )
        agent = create_agent(model, self.skill)

        result = run_agent_loop(agent, "生成一段短文本")

        parts = [
            part
            for message in result.all_messages()
            for part in message.parts
        ]
        tool_calls = [
            part for part in parts if isinstance(part, ToolCallPart)
        ]
        tool_returns = [
            part for part in parts if isinstance(part, ToolReturnPart)
        ]

        self.assertEqual("最终答案", result.output)
        self.assertEqual(["inspect_text"], [
            part.tool_name for part in tool_calls
        ])
        self.assertEqual(["inspect_text"], [
            part.tool_name for part in tool_returns
        ])
        self.assertEqual(
            TextInspection(
                total_characters=1,
                non_whitespace_characters=1,
                line_count=1,
                paragraph_count=1,
            ),
            tool_returns[0].content,
        )
        self.assertEqual(2, result.usage.requests)
        self.assertEqual(1, result.usage.tool_calls)

    def test_request_limit_stops_an_unfinished_tool_loop(self) -> None:
        """工具观察后若没有剩余模型预算，运行必须明确失败。"""
        model = TestModel(
            call_tools=["inspect_text"],
            custom_output_text="不会到达",
        )
        agent = create_agent(model, self.skill)

        with self.assertRaises(UsageLimitExceeded):
            run_agent_loop(
                agent,
                "生成一段短文本",
                usage_limits=UsageLimits(
                    request_limit=1,
                    tool_calls_limit=1,
                ),
            )

    def test_blank_request_is_rejected_before_model_call(self) -> None:
        """空白请求不应消耗模型或工具预算。"""
        agent = create_agent(TestModel(), self.skill)

        with self.assertRaisesRegex(ValueError, "请求不能为空"):
            run_agent_loop(agent, "   ")


if __name__ == "__main__":
    unittest.main()
