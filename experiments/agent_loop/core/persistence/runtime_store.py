"""把 Agent Runtime 的可恢复状态保存为版本化 JSON 快照。"""

import json
import os
from pathlib import Path
import tempfile
from threading import Lock
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.messages import ModelMessagesTypeAdapter

from ..context import (
    AgentContext,
    AgentCompletionEvent,
    AssignmentSession,
    ContextRuntime,
    ToolCallRecord,
    SkillRuntime,
    Fact,
    TaskHandoff,
    TaskState,
    WorkspaceContext,
)


SNAPSHOT_VERSION = 5
WINDOWS_REPLACE_ATTEMPTS = 5
WINDOWS_REPLACE_RETRY_SECONDS = 0.02


class TaskStateSnapshot(BaseModel):
    """整体目标中需要跨进程保留的结构化状态。"""

    model_config = ConfigDict(extra="forbid")

    goal: str
    plan: list[str]
    completed_steps: list[str]
    global_facts: list[Fact]
    unresolved_issues: list[str]
    completion_criteria: list[str]
    modified_files: list[str]
    validation_results: list[ToolCallRecord]
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
    task_facts: list[Fact] = Field(default_factory=list)
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
    tool_call_records: list[ToolCallRecord]
    assignment_history: list[TaskHandoff]
    pending_completion_events: list[AgentCompletionEvent]
    modified_files_by_agent: dict[str, list[str]]
    validation_results_by_agent: dict[str, list[ToolCallRecord]]
    modification_revisions_by_agent: dict[str, int] = Field(
        default_factory=dict
    )
    validated_revisions_by_agent: dict[str, int] = Field(
        default_factory=dict
    )
    next_call_id: int
    max_concurrent_assignments: int

class RuntimeStateStore:
    """以同目录临时文件和原子替换维护单个 Runtime 快照。"""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self._write_lock = Lock()

    def save(self, runtime: ContextRuntime) -> None:
        """线程安全地原子保存 Runtime，并容忍 Windows 短暂文件占用。

        每次保存使用唯一的同目录临时文件，避免多个工具线程争用固定 ``.tmp``。
        文件关闭并刷新到磁盘后再替换目标。Windows 上杀毒软件或另一实例可能短暂
        持有目标文件，因此 ``PermissionError`` 会按很短的退避间隔有界重试。
        """
        with self._write_lock:
            snapshot = self._create_snapshot(runtime)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary_file = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                delete=False,
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
            )
            temporary_path = Path(temporary_file.name)
            try:
                with temporary_file:
                    temporary_file.write(snapshot.model_dump_json(indent=2))
                    temporary_file.flush()
                    os.fsync(temporary_file.fileno())
                self._replace_with_retry(temporary_path)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()

    def _replace_with_retry(self, temporary_path: Path) -> None:
        """在 Windows 短暂共享冲突后重试原子替换。"""
        for attempt in range(WINDOWS_REPLACE_ATTEMPTS):
            try:
                temporary_path.replace(self.path)
                return
            except PermissionError:
                if attempt == WINDOWS_REPLACE_ATTEMPTS - 1:
                    raise
                time.sleep(
                    WINDOWS_REPLACE_RETRY_SECONDS * (2 ** attempt)
                )

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
                f"仅支持当前 Runtime 快照版本：{SNAPSHOT_VERSION}"
            )
        task = snapshot.task_state
        task_state = TaskState(
            goal=task.goal,
            plan=list(task.plan),
            completed_steps=list(task.completed_steps),
            global_facts=list(task.global_facts),
            unresolved_issues=list(task.unresolved_issues),
            completion_criteria=list(task.completion_criteria),
            modified_files=list(task.modified_files),
            validation_results=list(task.validation_results),
            status=task.status,
        )
        runtime = ContextRuntime(workspace, task_state)
        runtime.max_concurrent_assignments = snapshot.max_concurrent_assignments
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
                task_facts=list(record.task_facts),
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
        runtime.tool_call_records = {
            record.tool_call_id: record
            for record in snapshot.tool_call_records
        }
        runtime.assignment_history = list(snapshot.assignment_history)
        runtime.pending_completion_events = list(
            snapshot.pending_completion_events
        )
        runtime.modified_files_by_agent = snapshot.modified_files_by_agent
        runtime.validation_results_by_agent = (
            snapshot.validation_results_by_agent
        )
        runtime.modification_revisions_by_agent = (
            snapshot.modification_revisions_by_agent
        )
        runtime.validated_revisions_by_agent = (
            snapshot.validated_revisions_by_agent
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
                    task_facts=list(context.task_facts),
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
            tool_call_records=list(runtime.tool_call_records.values()),
            assignment_history=runtime.assignment_history,
            pending_completion_events=runtime.pending_completion_events,
            modified_files_by_agent=runtime.modified_files_by_agent,
            validation_results_by_agent=runtime.validation_results_by_agent,
            modification_revisions_by_agent=(
                runtime.modification_revisions_by_agent
            ),
            validated_revisions_by_agent=(
                runtime.validated_revisions_by_agent
            ),
            next_call_id=runtime._next_call_id,
            max_concurrent_assignments=runtime.max_concurrent_assignments,
        )
