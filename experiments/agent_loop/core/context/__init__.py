"""工作区、任务与运行上下文接口。"""

from .records import (
    AgentCallEvent,
    AgentCompletionEvent,
    EvidenceReference,
    Fact,
    TaskHandoff,
    ToolCallRecord,
)
from .runtime import (
    AgentContext,
    AgentDependencies,
    AssignmentSession,
    ContextRuntime,
    ContextWindowPolicy,
    ProjectInstruction,
    SKILL_ID_PATTERN,
    Skill,
    SkillMetadata,
    SkillRuntime,
    TaskState,
    WorkspaceContext,
    WorkspaceContextBuilder,
)

__all__ = [
    "AgentCallEvent",
    "AgentCompletionEvent",
    "AgentContext",
    "AgentDependencies",
    "AssignmentSession",
    "ContextRuntime",
    "ContextWindowPolicy",
    "EvidenceReference",
    "Fact",
    "ProjectInstruction",
    "SKILL_ID_PATTERN",
    "Skill",
    "SkillMetadata",
    "SkillRuntime",
    "TaskHandoff",
    "TaskState",
    "ToolCallRecord",
    "WorkspaceContext",
    "WorkspaceContextBuilder",
]
