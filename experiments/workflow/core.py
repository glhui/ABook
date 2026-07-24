"""在单个 AgentContext 中按固定顺序执行声明式步骤。

该模块只描述和驱动固定 Workflow，不重新实现 Agent、Runtime、工具或消息历史。
所有模型调用都交给 ``agent_loop`` 已有的 ``AgentRunner``，因此每一步自然复用
同一个 ``ContextRuntime`` 和 ``AgentContext``，并延续前面步骤的消息历史。
"""

from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic_ai import Agent

from experiments.agent_loop.context import (
    AgentContext,
    AgentDependencies,
    ContextRuntime,
)
from experiments.agent_loop.runner import AgentTurnResult


OutputT = TypeVar("OutputT")


@dataclass(frozen=True)
class WorkflowStep:
    """一个由同一 Agent 顺序执行的固定 Workflow 步骤。

    ``request`` 会作为该步骤的新请求交给现有 ``AgentRunner``。前序步骤的模型
    消息已经保存在共享的 ``AgentContext`` 中，因此无需复制或拼接前序输出。
    """

    name: str
    request: str

    def __post_init__(self) -> None:
        normalized_name = self.name.strip()
        normalized_request = self.request.strip()
        if not normalized_name:
            raise ValueError("Workflow 步骤名称不能为空")
        if not normalized_request:
            raise ValueError("Workflow 步骤请求不能为空")
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "request", normalized_request)


@dataclass(frozen=True)
class Workflow:
    """固定名称和执行顺序的 Workflow 定义。"""

    name: str
    steps: tuple[WorkflowStep, ...]

    def __post_init__(self) -> None:
        normalized_name = self.name.strip()
        if not normalized_name:
            raise ValueError("Workflow 名称不能为空")
        if not self.steps:
            raise ValueError("Workflow 至少需要一个步骤")
        step_names = [step.name for step in self.steps]
        if len(step_names) != len(set(step_names)):
            raise ValueError("同一 Workflow 中的步骤名称不能重复")
        object.__setattr__(self, "name", normalized_name)


@dataclass(frozen=True)
class WorkflowStepResult(Generic[OutputT]):
    """一个 Workflow 步骤及其现有 AgentRunner 返回结果。"""

    step: WorkflowStep
    turn: AgentTurnResult[OutputT]


@dataclass(frozen=True)
class WorkflowResult(Generic[OutputT]):
    """按执行顺序保存全部步骤结果的 Workflow 返回值。"""

    workflow: Workflow
    steps: tuple[WorkflowStepResult[OutputT], ...]

    @property
    def output(self) -> OutputT:
        """返回最后一个步骤的模型输出。"""
        return self.steps[-1].turn.output


async def run_workflow(
    workflow: Workflow,
    agent: Agent[AgentDependencies, OutputT],
    runtime: ContextRuntime,
    agent_context: AgentContext,
) -> WorkflowResult[OutputT]:
    """使用同一 Runtime、Agent 和 AgentContext 串行执行所有步骤。

    步骤严格按定义顺序执行。函数不捕获模型或工具异常：任一步失败都会立即终止
    Workflow，并把原异常交给调用边界决定是否重试或展示。首版不提供并发、依赖
    调度和持久化执行游标；需要这些语义时应使用现有任务分配调度器。
    """
    step_results: list[WorkflowStepResult[OutputT]] = []
    runner = runtime.get_agent_runner()
    for step in workflow.steps:
        turn = await runner.run_turn(
            agent,
            runtime,
            agent_context,
            step.request,
        )
        step_results.append(WorkflowStepResult(step=step, turn=turn))
    return WorkflowResult(workflow=workflow, steps=tuple(step_results))
