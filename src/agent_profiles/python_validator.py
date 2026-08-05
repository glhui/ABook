"""创建只负责黑盒标准输入输出验收的 Python 验证 Agent。"""

from typing import Final

from pydantic_ai import Agent
from pydantic_ai.models import Model

from agent_profiles.model_settings import AgentModelConfig
from agent_profiles.profiles import PYTHON_VALIDATOR_PROFILE, create_profile_agent, create_profile_context
from tool_execution import ToolApproval, ToolCapability, ToolExecutionContext, WorkspaceToolExecutor


PYTHON_VALIDATOR_AGENT_ID: Final[str] = PYTHON_VALIDATOR_PROFILE.agent_id
PYTHON_VALIDATOR_CAPABILITIES: Final[frozenset[ToolCapability]] = PYTHON_VALIDATOR_PROFILE.capabilities
PYTHON_VALIDATOR_APPROVALS: Final[frozenset[ToolApproval]] = PYTHON_VALIDATOR_PROFILE.approvals
PYTHON_VALIDATOR_INSTRUCTIONS: Final[str] = PYTHON_VALIDATOR_PROFILE.base_instructions


# 创建验证 Agent 的固定授权上下文，禁止读取工作区以保持与编写 Agent 独立。
def create_python_validator_context(
    task_id: str,
    approvals: frozenset[ToolApproval] = frozenset(),
) -> ToolExecutionContext:
    return create_profile_context(PYTHON_VALIDATOR_PROFILE, task_id, approvals)


# 创建只可写入验证脚本并运行验证命令的 Agent。
def create_python_validator_agent(
    model: Model,
    executor: WorkspaceToolExecutor,
    context: ToolExecutionContext,
    model_config: AgentModelConfig | None = None,
) -> Agent[None, str]:
    return create_profile_agent(PYTHON_VALIDATOR_PROFILE, model, executor, context, model_config=model_config)
