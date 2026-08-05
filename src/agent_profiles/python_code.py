"""创建具备固定开发规范的 Python 代码 Agent。"""

from typing import Final

from pydantic_ai import Agent
from pydantic_ai.models import Model

from agent_profiles.model_settings import AgentModelConfig
from agent_profiles.profiles import PYTHON_CODE_PROFILE, create_profile_agent, create_profile_context
from tool_execution import ToolApproval, ToolCapability, ToolExecutionContext, WorkspaceToolExecutor


PYTHON_CODE_AGENT_ID: Final[str] = PYTHON_CODE_PROFILE.agent_id
PYTHON_CODE_CAPABILITIES: Final[frozenset[ToolCapability]] = PYTHON_CODE_PROFILE.capabilities
PYTHON_CODE_APPROVALS: Final[frozenset[ToolApproval]] = PYTHON_CODE_PROFILE.approvals
PYTHON_CODE_INSTRUCTIONS: Final[str] = PYTHON_CODE_PROFILE.base_instructions


# 为 Python 代码任务创建固定角色的最小授权上下文，用户确认由调用方显式传入。
def create_python_code_context(
    task_id: str,
    approvals: frozenset[ToolApproval] = frozenset(),
) -> ToolExecutionContext:
    return create_profile_context(PYTHON_CODE_PROFILE, task_id, approvals)


# 按固定角色设计创建本次任务专属的 Agent，避免工具上下文跨任务复用。
def create_python_code_agent(
    model: Model,
    executor: WorkspaceToolExecutor,
    context: ToolExecutionContext,
    skill_instructions: str | None = None,
    model_config: AgentModelConfig | None = None,
) -> Agent[None, str]:
    return create_profile_agent(
        PYTHON_CODE_PROFILE,
        model,
        executor,
        context,
        skill_instructions,
        model_config,
    )
