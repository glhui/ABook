"""集中声明角色 Profile，并提供通用上下文校验和 Agent 创建逻辑。"""

from dataclasses import dataclass
from typing import Final

from pydantic_ai import Agent
from pydantic_ai.models import Model

from agent_profiles.model_settings import AgentModelConfig, DEFAULT_AGENT_MODEL_CONFIG, create_agent_model_settings
from agent_profiles.role_instructions import RoleInstructionName, get_role_instruction_definition, load_role_instructions
from tool_execution import (
    AuthorizedWorkspaceTools,
    ToolApproval,
    ToolCapability,
    ToolExecutionContext,
    WorkspaceToolExecutor,
)


# 描述一个工具型 Agent 的固定身份、授权上限、角色说明与默认模型参数。
@dataclass(frozen=True)
class AgentProfile:
    agent_id: str
    role: RoleInstructionName
    capabilities: frozenset[ToolCapability]
    approvals: frozenset[ToolApproval]
    base_instructions: str
    model_config: AgentModelConfig = DEFAULT_AGENT_MODEL_CONFIG


PYTHON_CODE_PROFILE: Final[AgentProfile] = AgentProfile(
    agent_id="python-code",
    role="python_code",
    capabilities=frozenset(
        {
            ToolCapability.FILE_READ,
            ToolCapability.FILE_WRITE,
            ToolCapability.FILE_EDIT,
            ToolCapability.BASH_EXECUTE,
        }
    ),
    approvals=frozenset({ToolApproval.OVERWRITE_FILE, ToolApproval.RUN_BASH}),
    base_instructions=(
        "你是一名 Python 工程师。先用 read_file 读取工作区根目录的 AGENTS.md 并遵循其中的编码规范；"
        "先读取相关代码和测试，再进行最小范围的修改；"
        "修改行为时必须补充或更新测试；"
        "不得修改测试以回避失败；完成后使用 bash 运行相关测试，只有测试通过才可声称任务完成。"
        "若调用任务明确说明由宿主程序在并行阶段后统一验证，则不得自行宣称最终测试结论，并用中文简洁说明结果。"
    ),
)
PYTHON_TEST_PROFILE: Final[AgentProfile] = AgentProfile(
    agent_id="python-test",
    role="python_test",
    capabilities=PYTHON_CODE_PROFILE.capabilities,
    approvals=PYTHON_CODE_PROFILE.approvals,
    base_instructions=(
        "你是一名 Python 测试工程师。先用 read_file 读取工作区根目录的 AGENTS.md 并遵循其中的编码规范；"
        "先阅读现有实现和测试，再为指定行为编写独立、可重复的自动化测试；"
        "测试应定义期望行为，不因当前实现尚未完成而削弱断言；"
        "使用 bash 编译或收集新增测试，确认测试代码可被正常加载。"
        "当前生产代码尚未实现时，测试断言失败是允许的，不得为了使测试通过而修改生产代码。"
    ),
)
PYTHON_VALIDATOR_PROFILE: Final[AgentProfile] = AgentProfile(
    agent_id="python-validator",
    role="python_validator",
    capabilities=frozenset({ToolCapability.FILE_WRITE, ToolCapability.BASH_EXECUTE}),
    approvals=frozenset({ToolApproval.OVERWRITE_FILE, ToolApproval.RUN_BASH}),
    base_instructions=(
        "你是独立的 Python 黑盒验证工程师。只根据用户需求设计验收，不读取生产代码和 tests/。"
        "请在指定路径创建 validator.py。validator.py 必须先通过 subprocess 启动 case_generator.py 获取 JSON 案例，"
        "再逐例启动 solution.py，使用 stdin 写入 UTF-8 JSON，读取 stdout，并将输出解析为 JSON 后与你推导的正确 oracle 比较。"
        "不得 import solution、读取其源代码或复用 tests/；必须覆盖正常、边界和错误输入。"
        "验证脚本失败时返回非零退出码，全部用例通过时返回 0；不要输出与验收无关的内容。"
    ),
)


# 为固定角色创建最小上下文，并拒绝调用方请求 Profile 未声明的确认项。
def create_profile_context(
    profile: AgentProfile,
    task_id: str,
    approvals: frozenset[ToolApproval] = frozenset(),
) -> ToolExecutionContext:
    unsupported_approvals = approvals - profile.approvals
    if unsupported_approvals:
        approval_names = ", ".join(sorted(approval.value for approval in unsupported_approvals))
        raise ValueError(f"{profile.agent_id} 不支持以下确认项：{approval_names}")
    return ToolExecutionContext(
        agent_id=profile.agent_id,
        task_id=task_id,
        capabilities=profile.capabilities,
        approvals=approvals,
    )


# 根据 Profile 创建工具型 Agent；每次调用仍使用独立上下文，避免跨任务复用状态。
def create_profile_agent(
    profile: AgentProfile,
    model: Model,
    executor: WorkspaceToolExecutor,
    context: ToolExecutionContext,
    skill_instructions: str | None = None,
    model_config: AgentModelConfig | None = None,
) -> Agent[None, str]:
    _validate_profile_context(profile, context)
    role_definition = get_role_instruction_definition(profile.role)
    if skill_instructions and not role_definition.accepts_skill_instructions:
        raise ValueError(f"{profile.agent_id} 不支持 Skill 指令。")
    instructions = f"{profile.base_instructions}\n\n{load_role_instructions(profile.role)}"
    if skill_instructions:
        instructions = f"{instructions}\n\n{skill_instructions}"
    return Agent(
        model,
        instructions=instructions,
        model_settings=create_agent_model_settings(model_config or profile.model_config),
        tools=AuthorizedWorkspaceTools(executor, context).as_pydantic_tools(),
    )


# 拒绝角色设计之外的身份、工具能力和高风险确认，防止手动组装扩大角色范围。
def _validate_profile_context(profile: AgentProfile, context: ToolExecutionContext) -> None:
    if context.agent_id != profile.agent_id:
        raise ValueError(f"{profile.agent_id} 的 agent_id 必须为：{profile.agent_id}")
    unsupported_capabilities = context.capabilities - profile.capabilities
    if unsupported_capabilities:
        capability_names = ", ".join(sorted(capability.value for capability in unsupported_capabilities))
        raise ValueError(f"{profile.agent_id} 不支持以下能力：{capability_names}")
    unsupported_approvals = context.approvals - profile.approvals
    if unsupported_approvals:
        approval_names = ", ".join(sorted(approval.value for approval in unsupported_approvals))
        raise ValueError(f"{profile.agent_id} 不支持以下确认项：{approval_names}")
