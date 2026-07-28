"""定义上下文系统共享的不可变记录、事实与交接模型。"""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class ToolCallRecord(BaseModel):
    """一次由宿主完成的工具调用及其可引用结果。

    ``request`` 保存宿主实际接受的结构化请求，``result_content`` 保存实际返回给
    模型的有界内容。事实仅保存对此记录的 ID 与逐字引文，完整结果不在事实和交接
    中重复嵌入。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_call_id: str = Field(min_length=1, max_length=100)
    agent_id: str = Field(min_length=1, max_length=100)
    tool_name: str = Field(min_length=1, max_length=200)
    status: Literal["succeeded", "timed_out"]
    request: dict[str, JsonValue] = Field(default_factory=dict)
    result_content: str
    result_truncated: bool


class EvidenceReference(BaseModel):
    """事实指向工具结果的轻量、可验证引用。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_call_id: str
    quote: str = Field(min_length=1, max_length=2_000)


class Fact(BaseModel):
    """可在任务或大任务作用域保存的、带证据的事实。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    statement: str = Field(min_length=1, max_length=2_000)
    citations: list[EvidenceReference] = Field(min_length=1, max_length=10)


class TaskHandoff(BaseModel):
    """任务 Agent 完成一轮工作后，由 Runtime 保存的结构化交接。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assignment_id: str
    coordinator_id: str
    agent_id: str
    template: str
    skill: str
    turn_index: int
    status: Literal["completed", "needs_follow_up", "blocked"]
    model_requests: int
    summary: str
    task_facts: tuple[Fact, ...]
    validation_results: tuple[ToolCallRecord, ...]
    unresolved_issues: tuple[str, ...]
    recommended_next_actions: tuple[str, ...]
    compacted: bool
    tool_calls: int
    modified_files: tuple[str, ...]


@dataclass(frozen=True)
class AgentCallEvent:
    """一次 Agent 或上下文压缩调用的生命周期通知。"""

    call_id: int
    agent_id: str
    kind: Literal["agent", "compaction"]
    phase: Literal["started", "completed", "failed"]
    turn_index: int
    detail: str | None = None


class AgentCompletionEvent(BaseModel):
    """后台任务 Agent 到达最终状态后发送给协调 Agent 的通知。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assignment_id: str
    coordinator_id: str
    status: Literal[
        "completed", "needs_follow_up", "blocked", "failed", "cancelled"
    ]
    handoff: TaskHandoff | None = None
    error: str | None = None
