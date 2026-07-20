"""把 Agent Runtime 的可恢复状态保存为版本化 JSON 快照。"""

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.messages import ModelMessagesTypeAdapter

from .context import (
    AgentContext,
    AssignmentCompletionEvent,
    AssignmentSession,
    ContextRuntime,
    EvidenceRecord,
    SkillRuntime,
    TaskFact,
    TaskHandoff,
    TaskState,
    ValidationResult,
    WorkspaceContext,
)


SNAPSHOT_VERSION = 1


class TaskStateSnapshot(BaseModel):
    """整体目标中需要跨进程保留的结构化状态。"""

    model_config = ConfigDict(extra="forbid")

    goal: str
    plan: list[str]
    completed_steps: list[str]
    important_facts: list[TaskFact]
    unresolved_issues: list[str]
    completion_criteria: list[str]
    modified_files: list[str]
    validation_results: list[ValidationResult]
    status: Literal["in_progress", "complete", "blocked"]


class AgentContextSnapshot(BaseModel):
    """Agent 私有历史的序列化表示，不包含锁或模型对象。"""

    model_config = ConfigDict(extra="forbid")

    agent_id: str
    task: str
    skill_id: str
    role: Literal["coordinator", "task"]
    coordinator_id: str | None
    message_history: list[dict[str, Any]]
    conversation_summary: str | None
    compaction_count: int
    turn_count: int


class AssignmentRecord(BaseModel):
    """任务分配的持久字段；运行句柄和 Agent 实例在恢复时重建。"""

    model_config = ConfigDict(extra="forbid")

    assignment_id: str
    coordinator_id: str
    agent_id: str
    template: str
    status: Literal[
        "queued",
        "running",
        "completed",
        "needs_follow_up",
        "blocked",
        "failed",
        "cancelled",
    ]
    pending_request: str
    priority: int
    depends_on: tuple[str, ...]
    max_attempts: int
    attempts: int
    latest_handoff: TaskHandoff | None
    error: str | None


class RuntimeSnapshot(BaseModel):
    """一个完整且带版本号的 Runtime 恢复点。"""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=SNAPSHOT_VERSION)
    task_state: TaskStateSnapshot
    agent_contexts: list[AgentContextSnapshot]
    assignments: list[AssignmentRecord]
    evidence_records: list[EvidenceRecord]
    assignment_history: list[TaskHandoff]
    pending_completion_events: list[AssignmentCompletionEvent]
    modified_files_by_agent: dict[str, list[str]]
    validation_results_by_agent: dict[str, list[ValidationResult]]
    next_call_id: int
    max_concurrent_assignments: int


class RuntimeStateStore:
    """以同目录临时文件和原子替换维护单个 Runtime 快照。"""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    def save(self, runtime: ContextRuntime) -> None:
        """保存 Runtime；不会序列化模型、回调、锁或 asyncio 任务。"""
        snapshot = self._create_snapshot(runtime)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_name(f"{self.path.name}.tmp")
        temporary_path.write_text(
            snapshot.model_dump_json(indent=2), encoding="utf-8"
        )
        temporary_path.replace(self.path)

    def load(self, workspace: WorkspaceContext) -> ContextRuntime | None:
        """读取快照并重建纯状态对象；不存在快照时返回 ``None``。

        上次退出时仍为 ``running`` 的任务不能恢复原协程，因此会转换成
        ``queued`` 并保留请求和尝试次数，等待调度器重新执行。
        """
        if not self.path.is_file():
            return None
        snapshot = RuntimeSnapshot.model_validate_json(
            self.path.read_text(encoding="utf-8")
        )
        if snapshot.version != SNAPSHOT_VERSION:
            raise ValueError(
                f"不支持的 Runtime 快照版本：{snapshot.version}"
            )
        task = snapshot.task_state
        task_state = TaskState(
            goal=task.goal,
            plan=list(task.plan),
            completed_steps=list(task.completed_steps),
            important_facts=list(task.important_facts),
            unresolved_issues=list(task.unresolved_issues),
            completion_criteria=list(task.completion_criteria),
            modified_files=list(task.modified_files),
            validation_results=list(task.validation_results),
            status=task.status,
        )
        runtime = ContextRuntime(
            workspace,
            task_state,
            max_concurrent_assignments=snapshot.max_concurrent_assignments,
        )
        skill_runtime = SkillRuntime(Path(workspace.skills_root))
        for record in snapshot.agent_contexts:
            runtime.agent_contexts[record.agent_id] = AgentContext(
                agent_id=record.agent_id,
                task=record.task,
                skill=skill_runtime.load(record.skill_id),
                role=record.role,
                coordinator_id=record.coordinator_id,
                message_history=ModelMessagesTypeAdapter.validate_python(
                    record.message_history
                ),
                conversation_summary=record.conversation_summary,
                compaction_count=record.compaction_count,
                turn_count=record.turn_count,
            )
        for record in snapshot.assignments:
            status = "queued" if record.status == "running" else record.status
            error = record.error
            if record.status == "running":
                error = "上次进程在任务运行期间结束；任务已重新排队"
            runtime.assignments[record.assignment_id] = AssignmentSession(
                **record.model_dump(
                    exclude={"status", "latest_handoff", "error"}
                ),
                agent=None,
                status=status,
                latest_handoff=record.latest_handoff,
                error=error,
            )
        runtime.evidence_records = {
            record.evidence_id: record for record in snapshot.evidence_records
        }
        runtime.assignment_history = list(snapshot.assignment_history)
        runtime.pending_completion_events = list(
            snapshot.pending_completion_events
        )
        runtime.modified_files_by_agent = snapshot.modified_files_by_agent
        runtime.validation_results_by_agent = (
            snapshot.validation_results_by_agent
        )
        runtime._next_call_id = snapshot.next_call_id
        return runtime

    @staticmethod
    def _create_snapshot(runtime: ContextRuntime) -> RuntimeSnapshot:
        """从混合 Runtime 中提取可持久化字段。"""
        contexts = []
        for context in runtime.agent_contexts.values():
            serialized = ModelMessagesTypeAdapter.dump_json(
                context.message_history
            )
            contexts.append(
                AgentContextSnapshot(
                    agent_id=context.agent_id,
                    task=context.task,
                    skill_id=context.skill.metadata.name,
                    role=context.role,
                    coordinator_id=context.coordinator_id,
                    message_history=json.loads(serialized),
                    conversation_summary=context.conversation_summary,
                    compaction_count=context.compaction_count,
                    turn_count=context.turn_count,
                )
            )
        assignments = [
            AssignmentRecord(
                assignment_id=session.assignment_id,
                coordinator_id=session.coordinator_id,
                agent_id=session.agent_id,
                template=session.template,
                status=session.status,
                pending_request=session.pending_request,
                priority=session.priority,
                depends_on=session.depends_on,
                max_attempts=session.max_attempts,
                attempts=session.attempts,
                latest_handoff=session.latest_handoff,
                error=session.error,
            )
            for session in runtime.assignments.values()
        ]
        return RuntimeSnapshot(
            task_state=TaskStateSnapshot(**vars(runtime.task_state)),
            agent_contexts=contexts,
            assignments=assignments,
            evidence_records=list(runtime.evidence_records.values()),
            assignment_history=runtime.assignment_history,
            pending_completion_events=runtime.pending_completion_events,
            modified_files_by_agent=runtime.modified_files_by_agent,
            validation_results_by_agent=runtime.validation_results_by_agent,
            next_call_id=runtime._next_call_id,
            max_concurrent_assignments=runtime.max_concurrent_assignments,
        )
