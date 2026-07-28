"""运行复用 agent_loop 组件的三步串行 Workflow 示例。"""

import asyncio
from pathlib import Path
import sys

from experiments.agent_loop.core.agents.agent_runtime import create_coordinator_agent
from experiments.agent_loop.core.context import (
    ContextRuntime,
    TaskState,
    WorkspaceContextBuilder,
)
from experiments.agent_loop.core.main import create_model

from .core import Workflow, WorkflowStep, run_workflow


def build_demo_workflow(user_request: str) -> Workflow:
    """根据用户目标创建不使用任务分配调度器的固定三步流程。

    三个步骤由同一个 Agent 和 AgentContext 顺序执行，因此后一步能够通过现有
    消息历史看到前一步的分析和操作。步骤请求明确禁止分配子任务，确保该示例只
    展示 Workflow Runner，而不会进入 ``AssignmentScheduler``。
    """
    normalized_request = user_request.strip()
    if not normalized_request:
        raise ValueError("Workflow 用户请求不能为空")
    return Workflow(
        name="分析、执行与验证",
        steps=(
            WorkflowStep(
                name="分析",
                request=(
                    "这是固定 Workflow 的分析步骤。不要调用 assign_tasks 或分配"
                    "子任务，也不要修改文件。分析下面的用户目标，检查必要的工作区"
                    "事实，并给出本次执行步骤的明确计划。\n\n"
                    f"用户目标：{normalized_request}"
                ),
            ),
            WorkflowStep(
                name="执行",
                request=(
                    "这是固定 Workflow 的执行步骤。不要调用 assign_tasks 或分配"
                    "子任务。根据上一分析步骤和原始用户目标完成必要工作；只有原始"
                    "用户目标明确授权修改时才允许写文件。"
                ),
            ),
            WorkflowStep(
                name="验证总结",
                request=(
                    "这是固定 Workflow 的验证总结步骤。不要调用 assign_tasks 或"
                    "分配子任务。检查前一步结果；如果修改了代码，运行适当的离线"
                    "验证。最后总结完成内容、实际验证结果和仍存在的风险。"
                ),
            ),
        ),
    )


async def run_demo(user_request: str) -> None:
    """创建一个 Runtime 和 AgentContext，执行示例 Workflow 并逐步输出。"""
    experiment_root = Path(__file__).parent.parent
    workspace_context = WorkspaceContextBuilder(
        workspace_root=Path.cwd(),
        skills_root=experiment_root / "agent_loop" / "skills",
    ).build()
    runtime = ContextRuntime(
        workspace_context,
        TaskState(goal=user_request.strip()),
    )
    agent_context = runtime.create_agent_context(
        agent_id="workflow-agent",
        task=user_request,
        skill_id="general",
        role="coordinator",
    )
    result = await run_workflow(
        build_demo_workflow(user_request),
        create_coordinator_agent(create_model()),
        runtime,
        agent_context,
    )
    for step_result in result.steps:
        print(f"\n[{step_result.step.name}]\n{step_result.turn.output}")


def main() -> None:
    """从命令行或标准输入读取目标并运行串行 Workflow 示例。"""
    user_request = " ".join(sys.argv[1:]).strip()
    if not user_request:
        try:
            user_request = input("Workflow goal> ").strip()
        except EOFError:
            return
    if not user_request:
        return
    asyncio.run(run_demo(user_request))


if __name__ == "__main__":
    main()
