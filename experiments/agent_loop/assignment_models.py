"""定义任务分配工具使用的稳定请求、回执和结构化报告模型。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .context import FactClaim, SKILL_ID_PATTERN


class AgentTemplate(BaseModel):
    """协调 Agent 分配工作时可选择的固定执行模板。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    instructions: str
    can_write: bool


class TaskReport(BaseModel):
    """任务 Agent 必须返回的语义报告；副作用字段由 Runtime 补充。"""

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "needs_follow_up", "blocked"]
    summary: str = Field(min_length=1, max_length=4_000)
    facts: list[FactClaim] = Field(default_factory=list, max_length=20)
    evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    unresolved_issues: list[str] = Field(default_factory=list, max_length=20)
    recommended_next_actions: list[str] = Field(
        default_factory=list, max_length=20
    )


class TaskAssignmentRequest(BaseModel):
    """协调 Agent 发出的一次非阻塞任务分配请求。

    ``task_key`` 仅在当前 ``assign_tasks`` 调用内有效。后续任务可在
    ``depends_on`` 中引用它，从而一次提交包含串行与并行分支的完整 DAG；
    Runtime 仍会将其解析为不可变的实际 ``assignment_id`` 后再持久化。
    """

    model_config = ConfigDict(extra="forbid")

    template: Literal["explorer", "worker", "reviewer"]
    task: str = Field(min_length=1, max_length=4_000)
    task_key: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_-]{0,63}$",
        description=(
            "本批任务内唯一的依赖别名；后续任务可通过 depends_on 引用。"
        ),
    )
    skill_id: str = Field(default="general", pattern=SKILL_ID_PATTERN.pattern)
    priority: int = Field(default=0, ge=-100, le=100)
    depends_on: list[str] = Field(default_factory=list, max_length=20)
    max_attempts: int = Field(default=1, ge=1, le=5)


class TaskAssignmentReceipt(BaseModel):
    """工作包已经交给模板 Agent 并进入运行状态的即时回执。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assignment_id: str
    template: str
    task: str
    status: Literal["queued", "running"] = "queued"


class AssignmentSnapshot(BaseModel):
    """协调 Agent 查询任务分配时返回的有界状态。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assignment_id: str
    template: str
    turn_count: int
    status: Literal[
        "queued", "running", "completed", "needs_follow_up", "blocked",
        "failed", "cancelled"
    ]
    latest_summary: str | None
    error: str | None
    unresolved_issues: tuple[str, ...]
    recommended_next_actions: tuple[str, ...]
