"""创建具备固定测试规范的 Python 测试编写 Agent。"""

from typing import Final

from pydantic_ai import Agent
from pydantic_ai.models import Model

from agent_profiles.model_settings import create_agent_model_settings
from tool_execution import (
    AuthorizedWorkspaceTools,
    ToolApproval,
    ToolCapability,
    ToolExecutionContext,
    WorkspaceToolExecutor,
)


PYTHON_TEST_AGENT_ID: Final[str] = "python-test"
PYTHON_TEST_CAPABILITIES: Final[frozenset[ToolCapability]] = frozenset(
    {
        ToolCapability.FILE_READ,
        ToolCapability.FILE_WRITE,
        ToolCapability.FILE_EDIT,
        ToolCapability.BASH_EXECUTE,
    }
)
PYTHON_TEST_APPROVALS: Final[frozenset[ToolApproval]] = frozenset(
    {ToolApproval.OVERWRITE_FILE, ToolApproval.RUN_BASH}
)
PYTHON_TEST_INSTRUCTIONS: Final[str] = (
    "你是一名 Python 测试工程师。遵循项目 AGENTS.md 中的编码规范；"
    "先阅读现有实现和测试，再为指定行为编写独立、可重复的自动化测试；"
    "测试应定义期望行为，不因当前实现尚未完成而削弱断言；"
    "使用 bash 编译或收集新增测试，确认测试代码可被正常加载。"
    "当前生产代码尚未实现时，测试断言失败是允许的，不得为了使测试通过而修改生产代码。"
)


# 为 Python 测试任务创建固定角色的最小授权上下文，用户确认由调用方显式传入。
def create_python_test_context(
    task_id: str,
    approvals: frozenset[ToolApproval] = frozenset(),
) -> ToolExecutionContext:
    unsupported_approvals = approvals - PYTHON_TEST_APPROVALS
    if unsupported_approvals:
        approval_names = ", ".join(sorted(approval.value for approval in unsupported_approvals))
        raise ValueError(f"Python 测试 Agent 不支持以下确认项：{approval_names}")
    return ToolExecutionContext(
        agent_id=PYTHON_TEST_AGENT_ID,
        task_id=task_id,
        capabilities=PYTHON_TEST_CAPABILITIES,
        approvals=approvals,
    )


# 按固定测试角色设计创建本次任务专属的 Agent，避免工具上下文跨任务复用。
def create_python_test_agent(
    model: Model,
    executor: WorkspaceToolExecutor,
    context: ToolExecutionContext,
) -> Agent[None, str]:
    _validate_context(context)
    authorized_tools = AuthorizedWorkspaceTools(executor, context)
    return Agent(
        model,
        instructions=PYTHON_TEST_INSTRUCTIONS,
        model_settings=create_agent_model_settings(),
        tools=authorized_tools.as_pydantic_tools(),
    )


# 拒绝角色设计之外的身份、工具能力和高风险确认，防止手动组装扩大权限。
def _validate_context(context: ToolExecutionContext) -> None:
    if context.agent_id != PYTHON_TEST_AGENT_ID:
        raise ValueError(f"Python 测试 Agent 的 agent_id 必须为：{PYTHON_TEST_AGENT_ID}")
    unsupported_capabilities = context.capabilities - PYTHON_TEST_CAPABILITIES
    if unsupported_capabilities:
        capability_names = ", ".join(sorted(capability.value for capability in unsupported_capabilities))
        raise ValueError(f"Python 测试 Agent 不支持以下能力：{capability_names}")
    unsupported_approvals = context.approvals - PYTHON_TEST_APPROVALS
    if unsupported_approvals:
        approval_names = ", ".join(sorted(approval.value for approval in unsupported_approvals))
        raise ValueError(f"Python 测试 Agent 不支持以下确认项：{approval_names}")
