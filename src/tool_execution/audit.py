"""定义工具执行审计记录。"""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


# 表示一次经过策略层处理的工具调用，包含允许和拒绝两种结果。
@dataclass(frozen=True)
class ToolAuditEvent:
    timestamp: datetime
    agent_id: str
    task_id: str
    operation: str
    path: str
    allowed: bool
    reason: str | None


# 抽象审计存储，避免执行器绑定到特定日志或数据库实现。
class ToolAuditLog(Protocol):
    def record(self: "ToolAuditLog", event: ToolAuditEvent) -> None:
        ...


# 提供适合示例和测试的内存审计存储。
class InMemoryToolAuditLog:
    def __init__(self: "InMemoryToolAuditLog") -> None:
        self._events: list[ToolAuditEvent] = []

    # 保存一条工具调用审计事件。
    def record(self: "InMemoryToolAuditLog", event: ToolAuditEvent) -> None:
        self._events.append(event)

    # 返回不可变快照，防止调用方修改内部审计顺序。
    def events(self: "InMemoryToolAuditLog") -> tuple[ToolAuditEvent, ...]:
        return tuple(self._events)
