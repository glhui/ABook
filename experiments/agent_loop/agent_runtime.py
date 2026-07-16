"""创建执行 Agent，并处理 Skill 选择和可继续的子 Agent 委派。"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, AgentRunResult, RunContext
from pydantic_ai.models import Model
from pydantic_ai.toolsets import FunctionToolset

from .context import (
    AgentContext,
    AgentDependencies,
    AgentSession,
    ContextRuntime,
    SKILL_ID_PATTERN,
    SkillRuntime,
)
from .workspace_tools import (
    RecoverableToolError,
    RetryToolset,
    create_workspace_toolset,
)


class AgentTemplate(BaseModel):
    """父 Agent 可用于创建子 Agent 会话的固定模板。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    instructions: str
    can_write: bool


class DelegationResult(BaseModel):
    """子 Agent 完成一轮任务后返回给父 Agent 的摘要。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    template: str
    skill: str
    output: str
    model_requests: int
    tool_calls: int


AGENT_TEMPLATES = {
    template.name: template
    for template in (
        AgentTemplate(
            name="explorer",
            description="只读探索代码、定位文件并汇总证据",
            instructions=(
                "只读取和搜索工作区，不修改文件。先收集证据，再给出简洁结论。"
            ),
            can_write=False,
        ),
        AgentTemplate(
            name="worker",
            description="实现用户已经明确授权的代码修改并运行验证",
            instructions=(
                "完成一个范围明确的实现任务。只有任务明确要求修改时才写文件，"
                "修改后运行相关验证并报告结果。"
            ),
            can_write=True,
        ),
        AgentTemplate(
            name="reviewer",
            description="只读审查实现、测试和潜在风险",
            instructions=(
                "审查现有代码和修改，不写文件。结论必须引用工具获得的证据。"
            ),
            can_write=False,
        ),
    )
}


def _normalize_task_items(name: str, items: list[str]) -> list[str]:
    """规范化模型提供的任务状态列表，并拒绝空白条目。"""
    normalized_items = [item.strip() for item in items]
    if any(not item for item in normalized_items):
        raise RecoverableToolError(f"{name} 不能包含空白条目")
    return normalized_items


def update_task_state(
    ctx: RunContext[AgentDependencies],
    plan: Annotated[
        list[str] | None,
        Field(max_length=20, description="替换当前待执行计划；省略则保持不变"),
    ] = None,
    completed_steps: Annotated[
        list[str] | None,
        Field(max_length=20, description="替换当前已完成步骤；省略则保持不变"),
    ] = None,
    important_facts: Annotated[
        list[str] | None,
        Field(
            max_length=20,
            description="替换工具已经证实的重要事实；省略则保持不变",
        ),
    ] = None,
    completion_criteria: Annotated[
        list[str] | None,
        Field(max_length=20, description="替换任务完成条件；省略则保持不变"),
    ] = None,
    status: Annotated[
        Literal["in_progress", "complete", "blocked"] | None,
        Field(description="任务状态；省略则保持不变"),
    ] = None,
) -> str:
    """更新 root Agent 的计划、事实、完成条件和整体状态。

    修改文件与验证结果不能由模型填写，它们只由实际工具调用记录。只有完成条件
    已满足且必要验证成功时才可将状态设为 ``complete``。
    """
    task_state = ctx.deps.runtime.task_state
    if plan is not None:
        task_state.plan = _normalize_task_items("plan", plan)
    if completed_steps is not None:
        task_state.completed_steps = _normalize_task_items(
            "completed_steps", completed_steps
        )
    if important_facts is not None:
        task_state.important_facts = _normalize_task_items(
            "important_facts", important_facts
        )
    if completion_criteria is not None:
        task_state.completion_criteria = _normalize_task_items(
            "completion_criteria", completion_criteria
        )
    if status is not None:
        task_state.status = status
    return task_state.render()


def select_skill(
    ctx: RunContext[AgentDependencies],
    skill_id: Annotated[
        str,
        Field(
            pattern=SKILL_ID_PATTERN.pattern,
            description="从上下文 Skill 目录中选择的 Skill ID",
        ),
    ],
) -> str:
    """按需加载 Skill，并更新当前 Agent 私有上下文。"""
    try:
        skill = SkillRuntime(ctx.deps.skills_root).load(skill_id)
    except (FileNotFoundError, ValueError) as error:
        raise RecoverableToolError(
            f"无法选择 Skill {skill_id!r}：{error}"
        ) from error
    ctx.deps.agent_context.skill = skill
    return f"# Skill: {skill.metadata.name}\n\n{skill.content}"


async def delegate_task(
    ctx: RunContext[AgentDependencies],
    template: Annotated[
        str,
        Field(
            min_length=1,
            max_length=50,
            description="子 Agent 模板：explorer、worker 或 reviewer",
        ),
    ],
    task: Annotated[
        str,
        Field(
            min_length=1,
            max_length=4_000,
            description="交给子 Agent 的单一、范围明确的任务",
        ),
    ],
    skill_id: Annotated[
        str,
        Field(
            pattern=SKILL_ID_PATTERN.pattern,
            description="子 Agent 使用的 Skill ID",
        ),
    ] = "general",
) -> DelegationResult:
    """创建不可递归委派的子 Agent，并保存可继续交互的内存会话。"""
    agent_template = AGENT_TEMPLATES.get(template)
    if agent_template is None:
        raise RecoverableToolError(
            f"未知 Agent 模板 {template!r}；可用模板为 "
            f"{', '.join(AGENT_TEMPLATES)}"
        )
    runtime = ctx.deps.runtime
    try:
        session_id = runtime.next_subagent_id(agent_template.name)
        child_context = runtime.create_agent_context(
            session_id, task, skill_id
        )
    except (FileNotFoundError, ValueError) as error:
        raise RecoverableToolError(
            f"无法为子 Agent 加载 Skill {skill_id!r}：{error}"
        ) from error

    child_agent = Agent(
        ctx.model,
        deps_type=AgentDependencies,
        instructions=(
            f"你是由父 Agent 创建的 {agent_template.name} 子 Agent。"
            "只完成委派任务，不扩展任务范围，也不能创建其他 Agent。\n\n"
            f"## 模板指令\n{agent_template.instructions}"
        ),
        toolsets=[create_workspace_toolset(agent_template.can_write)],
    )

    @child_agent.instructions
    def child_runtime_context(
        run_context: RunContext[AgentDependencies],
    ) -> str:
        """在子 Agent 每次运行前组合共享上下文与其私有上下文。"""
        return run_context.deps.runtime.render_agent_instructions(
            run_context.deps.agent_context
        )

    child_result = await child_agent.run(
        child_context.task,
        deps=AgentDependencies(runtime, child_context),
    )
    child_context.message_history = child_result.all_messages()
    runtime.subagents[session_id] = AgentSession(
        template=agent_template.name,
        agent=child_agent,
    )
    return DelegationResult(
        session_id=session_id,
        template=agent_template.name,
        skill=child_context.skill.metadata.name,
        output=child_result.output,
        model_requests=child_result.usage.requests,
        tool_calls=child_result.usage.tool_calls,
    )


async def continue_subagent(
    ctx: RunContext[AgentDependencies],
    session_id: Annotated[
        str,
        Field(
            min_length=1,
            max_length=100,
            description="delegate_task 返回的子 Agent 会话 ID",
        ),
    ],
    task: Annotated[
        str,
        Field(
            min_length=1,
            max_length=4_000,
            description="给同一子 Agent 的验证反馈或后续任务",
        ),
    ],
) -> DelegationResult:
    """把反馈交回同一个子 Agent，并显式传入它之前的消息历史。"""
    runtime = ctx.deps.runtime
    session = runtime.subagents.get(session_id)
    if session is None:
        raise RecoverableToolError(
            f"未知子 Agent 会话 {session_id!r}；请先调用 delegate_task"
        )

    child_context = runtime.agent_contexts[session_id]
    child_result = await session.agent.run(
        task,
        deps=AgentDependencies(runtime, child_context),
        message_history=child_context.message_history,
    )
    child_context.message_history = child_result.all_messages()
    return DelegationResult(
        session_id=session_id,
        template=session.template,
        skill=child_context.skill.metadata.name,
        output=child_result.output,
        model_requests=child_result.usage.requests,
        tool_calls=child_result.usage.tool_calls,
    )


def create_agent(model: Model) -> Agent:
    """创建使用工作区工具和编排工具的执行 Agent。"""
    orchestration_tools = RetryToolset(
        FunctionToolset[AgentDependencies](
            tools=[
                update_task_state,
                select_skill,
                delegate_task,
                continue_subagent,
            ],
            max_retries=2,
        )
    )
    template_catalog = "\n".join(
        f"- {template.name}: {template.description}"
        for template in AGENT_TEMPLATES.values()
    )
    agent = Agent(
        model,
        deps_type=AgentDependencies,
        instructions=(
            "你是一个在本地工作区中协助用户完成任务的执行 Agent。"
            "只把已提供的项目指令和 Skill 当作持久上下文。"
            "需要工作区事实时，使用工具列出文件、读取文件或搜索文本。"
            "需要 Git 查询、PowerShell 查询或运行本地测试时，使用受限命令工具。"
            "任务开始、计划变化或验证完成后，调用 update_task_state 更新工作状态。"
            "需要其他 Skill 时先调用 select_skill。"
            "只有任务能被拆成范围明确的子任务时才调用 delegate_task；"
            "该工具会返回 session_id。验证失败或需要补充修改时，"
            "调用 continue_subagent 把反馈交回同一个子 Agent。"
            "简单任务由你直接完成。"
            "只有用户明确要求修改代码或文件时，才允许调用精确文本替换工具。"
            "不得把模型记忆或猜测描述为工作区内容。"
            "当前不能安装依赖、访问网络、执行破坏性命令或提交 Git 变更。"
            "超出能力时应明确说明。\n\n"
            f"## 子 Agent 模板\n{template_catalog}"
        ),
        toolsets=[
            create_workspace_toolset(can_write=True),
            orchestration_tools,
        ],
    )

    @agent.instructions
    def runtime_context(
        run_context: RunContext[AgentDependencies],
    ) -> str:
        """在当前 Agent 每次运行前组合共享上下文与其私有上下文。"""
        return run_context.deps.runtime.render_agent_instructions(
            run_context.deps.agent_context
        )

    return agent


def run_agent(
    agent: Agent,
    runtime: ContextRuntime,
    agent_context: AgentContext,
    request: str | None = None,
) -> AgentRunResult[str]:
    """使用 Agent 自己的历史运行一轮，并把新历史写回其上下文。"""
    current_request = (request or agent_context.task).strip()
    if not current_request:
        raise ValueError("请求不能为空")
    result = agent.run_sync(
        current_request,
        deps=AgentDependencies(runtime, agent_context),
        message_history=agent_context.message_history,
    )
    agent_context.message_history = result.all_messages()
    return result
