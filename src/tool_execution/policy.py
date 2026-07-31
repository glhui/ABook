"""定义工作区工具的逻辑隔离策略。"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


# 区分读取、整文件写入和精确编辑，供任务或 Agent 进行最小授权。
class ToolCapability(StrEnum):
    BASH_EXECUTE = "bash_execute"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_EDIT = "file_edit"


# 表示执行层会检查的显式用户授权。
class ToolApproval(StrEnum):
    OVERWRITE_FILE = "overwrite_file"
    RUN_BASH = "run_bash"


# 描述本次 Agent 调用拥有的能力和已取得的用户确认。
@dataclass(frozen=True)
class ToolExecutionContext:
    agent_id: str
    task_id: str
    capabilities: frozenset[ToolCapability]
    approvals: frozenset[ToolApproval] = frozenset() # 显式用户授权高风险操作。


# 集中保存工作区路径边界和受保护目录规则。
@dataclass(frozen=True)
class WorkspaceExecutionPolicy:
    workspace_root: Path
    readable_roots: frozenset[Path] = frozenset()
    writable_roots: frozenset[Path] = frozenset()
    protected_path_parts: frozenset[str] = frozenset({".git", ".venv", "__pycache__"})
    protected_file_names: frozenset[str] = frozenset({".env"})

    # 规范化工作区和访问根目录；省略白名单时仅允许访问工作区。
    def __post_init__(self: "WorkspaceExecutionPolicy") -> None:
        resolved_root = self.workspace_root.resolve()
        if not resolved_root.is_dir():
            raise ValueError(f"工作区根目录不存在或不是目录：{self.workspace_root}")
        object.__setattr__(self, "workspace_root", resolved_root)
        object.__setattr__(self, "readable_roots", self._resolve_roots(self.readable_roots, resolved_root))
        object.__setattr__(self, "writable_roots", self._resolve_roots(self.writable_roots, resolved_root))

    # 每个白名单根目录必须存在，避免拼写错误扩大为不可预期的路径规则。
    def _resolve_roots(self: "WorkspaceExecutionPolicy", roots: frozenset[Path], default_root: Path) -> frozenset[Path]:
        configured_roots = roots or frozenset({default_root})
        resolved_roots = frozenset(root.resolve() for root in configured_roots)
        invalid_roots = [root for root in resolved_roots if not root.is_dir()]
        if invalid_roots:
            raise ValueError(f"允许访问的根目录不存在或不是目录：{invalid_roots[0]}")
        return resolved_roots
