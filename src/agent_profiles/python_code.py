"""创建具备固定开发规范的 Python 代码 Agent。"""

from typing import Final

from pydantic_ai import Agent
from pydantic_ai.models import Model

from agent_profiles.model_settings import create_agent_model_settings
from agent_profiles.role_instructions import load_role_instructions
from tool_execution import (
    AuthorizedWorkspaceTools,
    ToolApproval,
    ToolCapability,
    ToolExecutionContext,
    WorkspaceToolExecutor,
)


PYTHON_CODE_AGENT_ID: Final[str] = "python-code"
PYTHON_CODE_CAPABILITIES: Final[frozenset[ToolCapability]] = frozenset(
    {
        ToolCapability.FILE_READ,
        ToolCapability.FILE_WRITE,
        ToolCapability.FILE_EDIT,
        ToolCapability.BASH_EXECUTE,
    }
)
PYTHON_CODE_APPROVALS: Final[frozenset[ToolApproval]] = frozenset(
    {ToolApproval.OVERWRITE_FILE, ToolApproval.RUN_BASH}
)
PYTHON_CODE_INSTRUCTIONS: Final[str] = (
    "你是一名 Python 工程师。先用 read_file 读取工作区根目录的 AGENTS.md 并遵循其中的编码规范；"
    "先读取相关代码和测试，再进行最小范围的修改；"
    "修改行为时必须补充或更新测试；"
    "不得修改测试以回避失败；完成后使用 bash 运行相关测试，只有测试通过才可声称任务完成。"
    "若调用任务明确说明由宿主程序在并行阶段后统一验证，则不得自行宣称最终测试结论，并用中文简洁说明结果。"
)


# 为 Python 代码任务创建固定角色的最小授权上下文，用户确认由调用方显式传入。
def create_python_code_context(
    task_id: str,
    approvals: frozenset[ToolApproval] = frozenset(),
) -> ToolExecutionContext:
    unsupported_approvals = approvals - PYTHON_CODE_APPROVALS
    if unsupported_approvals:
        approval_names = ", ".join(sorted(approval.value for approval in unsupported_approvals))
        raise ValueError(f"Python 代码 Agent 不支持以下确认项：{approval_names}")
    return ToolExecutionContext(
        agent_id=PYTHON_CODE_AGENT_ID,
        task_id=task_id,
        capabilities=PYTHON_CODE_CAPABILITIES,
        approvals=approvals,
    )


# 按固定角色设计创建本次任务专属的 Agent，避免工具上下文跨任务复用。
def create_python_code_agent(
    model: Model,
    executor: WorkspaceToolExecutor,
    context: ToolExecutionContext,
) -> Agent[None, str]:
    _validate_context(context)
    authorized_tools = AuthorizedWorkspaceTools(executor, context)
    return Agent(
        model,
        instructions=f"{PYTHON_CODE_INSTRUCTIONS}\n\n{load_role_instructions('python_code')}",
        model_settings=create_agent_model_settings(),
        tools=authorized_tools.as_pydantic_tools(),
    )


# 拒绝角色设计之外的身份、工具能力和高风险确认，防止手动组装扩大权限。
def _validate_context(context: ToolExecutionContext) -> None:
    if context.agent_id != PYTHON_CODE_AGENT_ID:
        raise ValueError(f"Python 代码 Agent 的 agent_id 必须为：{PYTHON_CODE_AGENT_ID}")
    unsupported_capabilities = context.capabilities - PYTHON_CODE_CAPABILITIES
    if unsupported_capabilities:
        capability_names = ", ".join(sorted(capability.value for capability in unsupported_capabilities))
        raise ValueError(f"Python 代码 Agent 不支持以下能力：{capability_names}")
    unsupported_approvals = context.approvals - PYTHON_CODE_APPROVALS
    if unsupported_approvals:
        approval_names = ", ".join(sorted(approval.value for approval in unsupported_approvals))
        raise ValueError(f"Python 代码 Agent 不支持以下确认项：{approval_names}")
