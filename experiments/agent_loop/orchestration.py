"""定义协调 Agent 的任务分配工具、执行模板和结构化交接协议。"""

from typing import Annotated, Literal

from pydantic import Field
from pydantic_ai import RunContext
from pydantic_ai.models import Model

from .assignment_models import (
    AssignmentSnapshot,
    TaskAssignmentReceipt,
    TaskAssignmentRequest,
    TaskReport,
)
from .context import (
    AgentDependencies,
    AssignmentCompletionEvent,
    AssignmentSession,
    ContextRuntime,
    FactClaim,
    SKILL_ID_PATTERN,
    SkillRuntime,
    TaskHandoff,
)
from .runner import AgentTurnResult
from .scheduler import AssignmentScheduler
from .task_agents import AGENT_TEMPLATES, create_task_agent
from .workspace_tools import RecoverableToolError


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
        list[FactClaim] | None,
        Field(
            max_length=20,
            description=(
                "合并重要事实；每项必须引用 evidence_id 并逐字摘录支持原文"
            ),
        ),
    ] = None,
    unresolved_issues: Annotated[
        list[str] | None,
        Field(max_length=20, description="替换当前未决事项；省略则保持不变"),
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
    """更新整体目标的计划、证据事实、未决事项和状态。"""
    task_state = ctx.deps.runtime.task_state
    if plan is not None:
        task_state.plan = _normalize_task_items("plan", plan)
    if completed_steps is not None:
        task_state.completed_steps = _normalize_task_items(
            "completed_steps", completed_steps
        )
    if important_facts is not None:
        try:
            task_state.merge_facts(
                ctx.deps.runtime.resolve_fact_claims(important_facts)
            )
        except ValueError as error:
            raise RecoverableToolError(str(error)) from error
    if unresolved_issues is not None:
        task_state.unresolved_issues = _normalize_task_items(
            "unresolved_issues", unresolved_issues
        )
    if completion_criteria is not None:
        task_state.completion_criteria = _normalize_task_items(
            "completion_criteria", completion_criteria
        )
    if status is not None:
        task_state.status = status
    ctx.deps.runtime.persist()
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
    """按需加载 Skill，并只更新当前 Agent 的私有上下文。"""
    try:
        skill = SkillRuntime(ctx.deps.skills_root).load(skill_id)
    except (FileNotFoundError, ValueError) as error:
        raise RecoverableToolError(
            f"无法选择 Skill {skill_id!r}：{error}"
        ) from error
    ctx.deps.agent_context.skill = skill
    ctx.deps.runtime.persist()
    return f"# Skill: {skill.metadata.name}\n\n{skill.content}"


def _require_coordinator(ctx: RunContext[AgentDependencies]) -> None:
    """只允许承担协调职责的 Agent 管理任务分配。"""
    if ctx.deps.agent_context.role != "coordinator":
        raise RecoverableToolError("当前 Agent 没有任务协调职责")


async def assign_tasks(
    ctx: RunContext[AgentDependencies],
    tasks: Annotated[
        list[TaskAssignmentRequest],
        Field(
            min_length=1,
            max_length=8,
            description="立即交给模板 Agent 的一个到八个独立工作包",
        ),
    ],
) -> tuple[TaskAssignmentReceipt, ...]:
    """分配工作包并立即返回回执，不等待执行 Agent 完成。

    每个工作包先进入统一调度队列；调度器按优先级、依赖和全局并发上限启动，
    最终完成后由 Runtime 保存结构化交接并通知协调 Agent 重新安排。
    """
    _require_coordinator(ctx)
    _validate_assignment_batch(ctx, tasks)

    scheduler = _ensure_scheduler(ctx)
    assignment_ids_by_key: dict[str, str] = {}
    created_assignments: list[tuple[TaskAssignmentRequest, AssignmentSession]] = []
    for request in tasks:
        assignment = _create_assignment_session(ctx, request)
        if request.task_key is not None:
            assignment_ids_by_key[request.task_key] = assignment.assignment_id
        created_assignments.append((request, assignment))

    receipts: list[TaskAssignmentReceipt] = []
    for request, assignment in created_assignments:
        assignment.depends_on = tuple(
            assignment_ids_by_key.get(dependency, dependency)
            for dependency in request.depends_on
        )
    # 只将已解析为实际 ID 的依赖关系写入可恢复快照，避免进程中断后把批内
    # 临时别名当作未知的持久化依赖。
    ctx.deps.runtime.persist()

    for request, assignment in created_assignments:
        task_context = ctx.deps.runtime.agent_contexts[assignment.agent_id]
        receipts.append(
            _schedule_assignment_turn(scheduler, assignment, task_context.task)
        )
    return tuple(receipts)


async def send_task_feedback(
    ctx: RunContext[AgentDependencies],
    assignment_id: Annotated[
        str,
        Field(
            min_length=1,
            max_length=100,
            description="assign_tasks 返回的任务分配 ID",
        ),
    ],
    feedback: Annotated[
        str,
        Field(
            min_length=1,
            max_length=4_000,
            description="对原任务 Agent 的修正要求或下一步工作",
        ),
    ],
) -> TaskAssignmentReceipt:
    """向原任务 Agent 提交反馈，并立即返回新的运行回执。"""
    _require_coordinator(ctx)
    runtime = ctx.deps.runtime
    assignment = runtime.assignments.get(assignment_id)
    if assignment is None:
        raise RecoverableToolError(
            f"未知任务分配 {assignment_id!r}；请先调用 assign_tasks"
        )
    if assignment.coordinator_id != ctx.deps.agent_context.agent_id:
        raise RecoverableToolError("只有发出该任务的协调 Agent 可以提交反馈")
    if assignment.status in {"queued", "running"}:
        raise RecoverableToolError("该任务仍在排队或运行，不能重复提交反馈")

    previous = assignment.latest_handoff
    if previous is None:
        previous_context = "无"
    else:
        unresolved = "、".join(previous.unresolved_issues) or "无"
        recommended = "、".join(previous.recommended_next_actions) or "无"
        previous_context = (
            f"状态：{previous.status}\n"
            f"摘要：{previous.summary}\n"
            f"未决事项：{unresolved}\n"
            f"建议下一步：{recommended}"
        )
    request = (
        "协调 Agent 后续反馈：\n"
        f"{feedback.strip()}\n\n"
        "上一轮结构化交接：\n"
        f"{previous_context}"
    )
    return _schedule_assignment_turn(_ensure_scheduler(ctx), assignment, request)


async def cancel_assignment(
    ctx: RunContext[AgentDependencies],
    assignment_id: Annotated[
        str, Field(min_length=1, max_length=100, description="要取消的任务分配 ID")
    ],
    reason: Annotated[
        str, Field(min_length=1, max_length=1_000, description="取消原因")
    ],
) -> str:
    """取消当前协调 Agent 发出的排队中或运行中的任务。"""
    _require_coordinator(ctx)
    assignment = ctx.deps.runtime.assignments.get(assignment_id)
    if (
        assignment is None
        or assignment.coordinator_id != ctx.deps.agent_context.agent_id
    ):
        raise RecoverableToolError(f"未知任务分配 {assignment_id!r}")
    if not await _ensure_scheduler(ctx).cancel(assignment_id, reason):
        raise RecoverableToolError("任务已经结束，不能取消")
    return f"任务分配 {assignment_id} 已取消"


def _validate_assignment_batch(
    ctx: RunContext[AgentDependencies], tasks: list[TaskAssignmentRequest]
) -> None:
    """在创建后台会话前校验批内别名、依赖引用和整个依赖图。

    已存在的任务 ID 可以作为外部前置条件；批内 ``task_key`` 只能引用同一次
    调用的请求。由于新任务不会成为已有任务的前置条件，只需检测批内环即可。
    """
    task_keys = [request.task_key for request in tasks if request.task_key]
    duplicate_keys = sorted(
        {task_key for task_key in task_keys if task_keys.count(task_key) > 1}
    )
    if duplicate_keys:
        raise RecoverableToolError(
            "同一批任务的 task_key 必须唯一：" + ", ".join(duplicate_keys)
        )
    batch_keys = set(task_keys)
    conflicting_keys = sorted(
        batch_keys.intersection(ctx.deps.runtime.assignments)
    )
    if conflicting_keys:
        raise RecoverableToolError(
            "task_key 不能与已有任务 ID 相同：" + ", ".join(conflicting_keys)
        )
    for request in tasks:
        if request.template not in AGENT_TEMPLATES:
            raise RecoverableToolError(
                f"未知 Agent 模板 {request.template!r}；可用模板为 "
                f"{', '.join(AGENT_TEMPLATES)}"
            )
        try:
            SkillRuntime(ctx.deps.skills_root).load(request.skill_id)
        except (FileNotFoundError, ValueError) as error:
            raise RecoverableToolError(
                f"无法加载任务 Agent Skill {request.skill_id!r}：{error}"
            ) from error
        unknown_dependencies = [
            dependency
            for dependency in request.depends_on
            if (
                dependency not in ctx.deps.runtime.assignments
                and dependency not in batch_keys
            )
        ]
        if unknown_dependencies:
            raise RecoverableToolError(
                "未知依赖任务：" + ", ".join(unknown_dependencies)
            )

    dependencies_by_key = {
        request.task_key: {
            dependency
            for dependency in request.depends_on
            if dependency in batch_keys
        }
        for request in tasks
        if request.task_key is not None
    }
    _reject_cyclic_batch_dependencies(dependencies_by_key)


def _reject_cyclic_batch_dependencies(
    dependencies_by_key: dict[str, set[str]],
) -> None:
    """拒绝会使调度器永久等待的批内循环依赖。"""
    visiting: set[str] = set()
    completed: set[str] = set()

    def visit(task_key: str) -> None:
        if task_key in completed:
            return
        if task_key in visiting:
            raise RecoverableToolError(
                f"批内任务依赖存在循环：{task_key}"
            )
        visiting.add(task_key)
        for dependency in dependencies_by_key[task_key]:
            visit(dependency)
        visiting.remove(task_key)
        completed.add(task_key)

    for task_key in dependencies_by_key:
        visit(task_key)


def _create_assignment_session(
    ctx: RunContext[AgentDependencies], request: TaskAssignmentRequest
) -> AssignmentSession:
    """创建已登记但尚未开始模型调用的任务分配会话。"""
    runtime = ctx.deps.runtime
    coordinator_context = ctx.deps.agent_context
    template = AGENT_TEMPLATES[request.template]
    assignment_id = runtime.next_assignment_id(template.name)
    try:
        task_context = runtime.create_agent_context(
            assignment_id,
            request.task,
            request.skill_id,
            role="task",
            coordinator_id=coordinator_context.agent_id,
        )
    except (FileNotFoundError, ValueError) as error:
        raise RecoverableToolError(
            f"无法创建任务 Agent 上下文：{error}"
        ) from error
    assignment = AssignmentSession(
        assignment_id=assignment_id,
        coordinator_id=coordinator_context.agent_id,
        agent_id=task_context.agent_id,
        template=template.name,
        agent=create_task_agent(ctx.model, template),
        priority=request.priority,
        depends_on=tuple(dict.fromkeys(request.depends_on)),
        max_attempts=request.max_attempts,
    )
    runtime.assignments[assignment_id] = assignment
    return assignment


def _schedule_assignment_turn(
    scheduler: AssignmentScheduler,
    assignment: AssignmentSession,
    request: str,
) -> TaskAssignmentReceipt:
    """把一轮任务 Agent 调用放入调度队列并返回即时回执。"""
    scheduler.enqueue(assignment, request)
    task_context = scheduler.runtime.agent_contexts[assignment.agent_id]
    return TaskAssignmentReceipt(
        assignment_id=assignment.assignment_id,
        template=assignment.template,
        task=task_context.task,
        status=assignment.status,
    )


async def _run_assignment_turn(
    runtime: ContextRuntime,
    assignment: AssignmentSession,
    request: str,
) -> AssignmentCompletionEvent:
    """在后台完成工作包，保存交接并通知协调 Agent 重新安排。"""
    task_context = runtime.agent_contexts[assignment.agent_id]
    modified_files_before = len(
        runtime.modified_files_by_agent.get(task_context.agent_id, [])
    )
    validation_count_before = len(
        runtime.validation_results_by_agent.get(task_context.agent_id, [])
    )
    event: AssignmentCompletionEvent
    try:
        if assignment.agent is None:
            raise RuntimeError("任务 Agent 尚未恢复")
        turn = await runtime.get_agent_runner().run_turn(
            assignment.agent, runtime, task_context, request
        )
        handoff = _build_handoff(
            runtime,
            task_context.agent_id,
            assignment.coordinator_id,
            assignment.assignment_id,
            assignment.template,
            task_context.skill.metadata.name,
            turn,
            modified_files_before,
            validation_count_before,
        )
        runtime.record_handoff(handoff)
        event = AssignmentCompletionEvent(
            assignment_id=assignment.assignment_id,
            coordinator_id=assignment.coordinator_id,
            status=handoff.status,
            handoff=handoff,
        )
    except Exception as error:
        assignment.status = "failed"
        assignment.error = f"{type(error).__name__}: {error}"
        event = AssignmentCompletionEvent(
            assignment_id=assignment.assignment_id,
            coordinator_id=assignment.coordinator_id,
            status="failed",
            error=assignment.error,
        )
        runtime.persist()
    return event


def _ensure_scheduler(
    ctx: RunContext[AgentDependencies],
) -> AssignmentScheduler:
    """按当前模型创建调度器，并重建快照中不可序列化的任务 Agent。"""
    return initialize_assignment_scheduler(ctx.deps.runtime, ctx.model)


def initialize_assignment_scheduler(
    runtime: ContextRuntime,
    model: Model,
) -> AssignmentScheduler:
    """在宿主事件循环启动后重建调度器并恢复未完成任务。"""
    if runtime.scheduler is not None:
        return runtime.scheduler
    for assignment in runtime.assignments.values():
        if assignment.agent is None:
            assignment.agent = create_task_agent(
                model, AGENT_TEMPLATES[assignment.template]
            )

    async def execute(
        assignment: AssignmentSession, request: str
    ) -> AssignmentCompletionEvent:
        return await _run_assignment_turn(runtime, assignment, request)

    scheduler = AssignmentScheduler(runtime, execute)
    runtime.scheduler = scheduler
    scheduler.restore_queued()
    return scheduler


def list_assignments(
    ctx: RunContext[AgentDependencies],
) -> tuple[AssignmentSnapshot, ...]:
    """列出当前协调 Agent 发出的任务及其最新结构化状态。"""
    _require_coordinator(ctx)
    coordinator_id = ctx.deps.agent_context.agent_id
    snapshots: list[AssignmentSnapshot] = []
    for assignment in ctx.deps.runtime.assignments.values():
        if assignment.coordinator_id != coordinator_id:
            continue
        latest = assignment.latest_handoff
        snapshots.append(
            AssignmentSnapshot(
                assignment_id=assignment.assignment_id,
                template=assignment.template,
                turn_count=(latest.turn_index if latest else 0),
                status=assignment.status,
                latest_summary=(latest.summary if latest else None),
                error=assignment.error,
                unresolved_issues=(
                    latest.unresolved_issues if latest else ()
                ),
                recommended_next_actions=(
                    latest.recommended_next_actions if latest else ()
                ),
            )
        )
    return tuple(snapshots)


def inspect_assignment(
    ctx: RunContext[AgentDependencies],
    assignment_id: Annotated[
        str,
        Field(min_length=1, max_length=100, description="要检查的任务分配 ID"),
    ],
) -> TaskHandoff:
    """返回指定任务最近一次完整交接，供协调 Agent 重新安排。"""
    _require_coordinator(ctx)
    assignment = ctx.deps.runtime.assignments.get(assignment_id)
    if (
        assignment is None
        or assignment.coordinator_id != ctx.deps.agent_context.agent_id
    ):
        raise RecoverableToolError(f"未知任务分配 {assignment_id!r}")
    if assignment.latest_handoff is None:
        if assignment.status == "failed":
            raise RecoverableToolError(
                f"任务分配 {assignment_id!r} 执行失败：{assignment.error}"
            )
        raise RecoverableToolError(
            f"任务分配 {assignment_id!r} 仍在排队或运行，尚无交接"
        )
    return assignment.latest_handoff


def _build_handoff(
    runtime: ContextRuntime,
    agent_id: str,
    coordinator_id: str,
    assignment_id: str,
    template: str,
    skill: str,
    turn: AgentTurnResult[TaskReport],
    modified_files_before: int,
    validation_count_before: int,
) -> TaskHandoff:
    """合并任务 Agent 报告与 Runtime 观察到的本轮副作用。"""
    report = turn.output
    agent_modified_files = runtime.modified_files_by_agent.get(
        agent_id, []
    )
    modified_files = agent_modified_files[modified_files_before:]
    agent_validation_results = runtime.validation_results_by_agent.get(
        agent_id, []
    )
    validation_results = agent_validation_results[validation_count_before:]
    return TaskHandoff(
        assignment_id=assignment_id,
        coordinator_id=coordinator_id,
        agent_id=agent_id,
        template=template,
        skill=skill,
        turn_index=turn.turn_index,
        status=report.status,
        summary=report.summary,
        facts=tuple(runtime.resolve_fact_claims(report.facts)),
        evidence=tuple(
            runtime.evidence_records[evidence_id]
            for evidence_id in report.evidence_ids
        ),
        modified_files=tuple(dict.fromkeys(modified_files)),
        validation_results=tuple(validation_results),
        unresolved_issues=tuple(report.unresolved_issues),
        recommended_next_actions=tuple(report.recommended_next_actions),
        compacted=turn.compacted,
        model_requests=turn.requests,
        tool_calls=turn.tool_calls,
    )
