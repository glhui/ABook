"""创建只负责黑盒标准输入输出验收的 Python 验证 Agent。"""

from typing import Final

from pydantic_ai import Agent
from pydantic_ai.models import Model

from tool_execution import AuthorizedWorkspaceTools, ToolApproval, ToolCapability, ToolExecutionContext, WorkspaceToolExecutor


PYTHON_VALIDATOR_AGENT_ID: Final[str] = "python-validator"
PYTHON_VALIDATOR_CAPABILITIES: Final[frozenset[ToolCapability]] = frozenset(
    {ToolCapability.FILE_WRITE, ToolCapability.BASH_EXECUTE}
)
PYTHON_VALIDATOR_APPROVALS: Final[frozenset[ToolApproval]] = frozenset(
    {ToolApproval.OVERWRITE_FILE, ToolApproval.RUN_BASH}
)
PYTHON_VALIDATOR_INSTRUCTIONS: Final[str] = (
    "你是独立的 Python 黑盒验证工程师。只根据用户需求设计验收，不读取生产代码和 tests/。"
    "请在指定路径创建 validator.py。validator.py 必须先通过 subprocess 启动 case_generator.py 获取 JSON 案例，"
    "再逐例启动 solution.py，使用 stdin 写入 UTF-8 JSON，读取 stdout，并将输出解析为 JSON 后与你推导的正确 oracle 比较。"
    "不得 import solution、读取其源代码或复用 tests/；必须覆盖正常、边界和错误输入。"
    "验证脚本失败时返回非零退出码，全部用例通过时返回 0；不要输出与验收无关的内容。"
)


# 创建验证 Agent 的固定授权上下文，禁止读取工作区以保持与编写 Agent 独立。
def create_python_validator_context(
    task_id: str,
    approvals: frozenset[ToolApproval] = frozenset(),
) -> ToolExecutionContext:
    unsupported_approvals = approvals - PYTHON_VALIDATOR_APPROVALS
    if unsupported_approvals:
        names = ", ".join(sorted(approval.value for approval in unsupported_approvals))
        raise ValueError(f"Python 验证 Agent 不支持以下确认项：{names}")
    return ToolExecutionContext(
        agent_id=PYTHON_VALIDATOR_AGENT_ID,
        task_id=task_id,
        capabilities=PYTHON_VALIDATOR_CAPABILITIES,
        approvals=approvals,
    )


# 创建只可写入验证脚本并运行验证命令的 Agent。
def create_python_validator_agent(
    model: Model,
    executor: WorkspaceToolExecutor,
    context: ToolExecutionContext,
) -> Agent[None, str]:
    if context.agent_id != PYTHON_VALIDATOR_AGENT_ID:
        raise ValueError(f"验证 Agent 的 agent_id 必须为：{PYTHON_VALIDATOR_AGENT_ID}")
    if context.capabilities - PYTHON_VALIDATOR_CAPABILITIES:
        raise ValueError("验证 Agent 获得了不支持的能力")
    return Agent(model, instructions=PYTHON_VALIDATOR_INSTRUCTIONS, tools=AuthorizedWorkspaceTools(executor, context).as_pydantic_tools())
