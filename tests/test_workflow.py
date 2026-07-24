"""轻量串行 Workflow 的可发现离线行为测试。"""

import unittest
from dataclasses import dataclass
from typing import Any, cast

from pydantic_ai import Agent

from experiments.agent_loop.context import (
    AgentContext,
    AgentDependencies,
    ContextRuntime,
)
from experiments.workflow import Workflow, WorkflowStep, run_workflow
from experiments.workflow.main import build_demo_workflow


@dataclass(frozen=True)
class _Turn:
    output: str


class _RecordingRunner:
    def __init__(self) -> None:
        self.requests: list[str] = []
        self.contexts: list[object] = []

    async def run_turn(
        self,
        agent: object,
        runtime: object,
        agent_context: object,
        request: str,
    ) -> _Turn:
        self.requests.append(request)
        self.contexts.append(agent_context)
        return _Turn(output=f"完成：{request}")


class _Runtime:
    def __init__(self, runner: _RecordingRunner) -> None:
        self.runner = runner

    def get_agent_runner(self) -> _RecordingRunner:
        return self.runner


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    def test_demo_workflow_has_fixed_serial_steps(self) -> None:
        """命令行示例只声明分析、执行和验证三个固定步骤。"""
        workflow = build_demo_workflow("修改代码并运行测试")

        self.assertEqual(
            ["分析", "执行", "验证总结"],
            [step.name for step in workflow.steps],
        )
        self.assertIn("修改代码并运行测试", workflow.steps[0].request)
        self.assertTrue(
            all("不要调用 assign_tasks" in step.request for step in workflow.steps)
        )

    async def test_workflow_runs_steps_in_order_with_one_context(self) -> None:
        """所有步骤按声明顺序运行，并复用同一个 AgentContext。"""
        runner = _RecordingRunner()
        runtime = cast(ContextRuntime, _Runtime(runner))
        agent = cast(Agent[AgentDependencies, str], object())
        agent_context = cast(AgentContext, object())
        workflow = Workflow(
            name="实现流程",
            steps=(
                WorkflowStep("分析", "分析需求"),
                WorkflowStep("实现", "实现修改"),
                WorkflowStep("验证", "运行验证"),
            ),
        )

        result = await run_workflow(
            workflow, agent, runtime, agent_context
        )

        self.assertEqual(
            ["分析需求", "实现修改", "运行验证"], runner.requests
        )
        self.assertEqual([agent_context] * 3, runner.contexts)
        self.assertEqual("完成：运行验证", result.output)

    async def test_workflow_stops_when_a_step_fails(self) -> None:
        """步骤异常直接向上传播，后续步骤不会继续执行。"""

        class FailingRunner(_RecordingRunner):
            async def run_turn(
                self,
                agent: object,
                runtime: object,
                agent_context: object,
                request: str,
            ) -> _Turn:
                self.requests.append(request)
                if request == "失败步骤":
                    raise RuntimeError("step failed")
                return _Turn(output=request)

        runner = FailingRunner()
        runtime = cast(ContextRuntime, _Runtime(runner))
        workflow = Workflow(
            name="失败流程",
            steps=(
                WorkflowStep("第一步", "成功步骤"),
                WorkflowStep("第二步", "失败步骤"),
                WorkflowStep("第三步", "不应执行"),
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "step failed"):
            await run_workflow(
                workflow,
                cast(Agent[AgentDependencies, Any], object()),
                runtime,
                cast(AgentContext, object()),
            )

        self.assertEqual(["成功步骤", "失败步骤"], runner.requests)

    def test_workflow_rejects_duplicate_step_names(self) -> None:
        """重复名称会破坏步骤识别，因此在执行前拒绝。"""
        with self.assertRaisesRegex(ValueError, "步骤名称不能重复"):
            Workflow(
                name="重复步骤",
                steps=(
                    WorkflowStep("分析", "第一次分析"),
                    WorkflowStep("分析", "第二次分析"),
                ),
            )


if __name__ == "__main__":
    unittest.main()
