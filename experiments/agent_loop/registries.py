"""封装 Runtime 中任务分配集合及其可恢复生命周期数据。"""

from dataclasses import dataclass, field
from typing import Generic, TypeVar


SessionT = TypeVar("SessionT")
HandoffT = TypeVar("HandoffT")
CompletionEventT = TypeVar("CompletionEventT")


@dataclass
class AssignmentRegistry(Generic[SessionT, HandoffT, CompletionEventT]):
    """集中拥有任务会话、交接历史、待处理事件和并发配置。

    这些字段原先直接散落在 ``ContextRuntime``。注册表把任务域状态聚合到明确
    边界，同时仍由现有 Runtime 快照负责序列化，以保持磁盘格式兼容。
    """

    sessions: dict[str, SessionT] = field(default_factory=dict)
    history: list[HandoffT] = field(default_factory=list)
    pending_events: list[CompletionEventT] = field(default_factory=list)
    max_concurrent: int = 4

    def __post_init__(self) -> None:
        """拒绝无法执行任何任务的并发配置。"""
        if self.max_concurrent <= 0:
            raise ValueError("任务并发上限必须为正数")

    def get_required(self, assignment_id: str) -> SessionT:
        """返回指定任务；未知 ID 使用领域错误而不是泄漏 ``KeyError``。"""
        session = self.sessions.get(assignment_id)
        if session is None:
            raise ValueError(f"未知任务分配：{assignment_id}")
        return session

    def set_max_concurrent(self, value: int) -> None:
        """更新并发上限，并在进入调度器前验证取值。"""
        if value <= 0:
            raise ValueError("任务并发上限必须为正数")
        self.max_concurrent = value
