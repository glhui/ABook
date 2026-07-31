"""在策略校验后执行工作区文件工具。"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, NoReturn

from pydantic import Field
from pydantic_ai import ModelRetry
from pydantic_ai.tools import Tool

from tool_execution.audit import ToolAuditEvent, ToolAuditLog
from tool_execution.policy import ToolApproval, ToolCapability, ToolExecutionContext, WorkspaceExecutionPolicy
from workspace_tools import BashResult, EditFileResult, ReadFileResult, WorkspaceBashTool, WorkspaceFileTools, WriteFileResult


AbsoluteFilePath = Annotated[str, Field(description="位于受控白名单根目录内的 UTF-8 文本文件绝对路径。")]
StartLine = Annotated[int, Field(description="从 1 开始的首行行号。")]
EndLine = Annotated[int | None, Field(description="从 1 开始的末行行号，省略时读取至文件末尾。")]
ExpectedReplacements = Annotated[int, Field(description="旧文本必须出现的次数。")]
BashCommand = Annotated[str, Field(description="在受控工作区根目录中执行的完整 Bash 命令。")]
TimeoutSeconds = Annotated[float | None, Field(description="命令超时秒数；省略时由 Bash 后端决定。")]


# 表示执行层拒绝了当前工具调用，包装层会将其转换为模型可纠正的工具反馈。
class ToolExecutionDenied(PermissionError):
    pass


# 先执行策略校验、审计，再委托无策略的绝对路径文件工具完成文件操作。
class WorkspaceToolExecutor:
    def __init__(
        self: "WorkspaceToolExecutor",
        policy: WorkspaceExecutionPolicy,
        tools: WorkspaceFileTools,
        audit_log: ToolAuditLog,
        bash_tool: WorkspaceBashTool | None = None,
    ) -> None:
        self._policy = policy
        self._tools = tools
        self._audit_log = audit_log
        self._bash_tool = bash_tool

    # 指示当前执行器是否已配置可调用的 Bash 后端。
    @property
    def has_bash_tool(self: "WorkspaceToolExecutor") -> bool:
        return self._bash_tool is not None

    # 校验读取授权和路径后读取文件。
    def read_file(
        self: "WorkspaceToolExecutor",
        context: ToolExecutionContext,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> ReadFileResult:
        target_path = self._authorize(context, "read_file", ToolCapability.FILE_READ, path)
        result = self._tools.read_file(str(target_path), start_line, end_line)
        self._record(context, "read_file", target_path, allowed=True, reason=None)
        return result

    # 校验写入授权、路径和覆盖确认后写入完整文件。
    def write_file(
        self: "WorkspaceToolExecutor",
        context: ToolExecutionContext,
        path: str,
        content: str,
    ) -> WriteFileResult:
        target_path = self._authorize(context, "write_file", ToolCapability.FILE_WRITE, path)
        if target_path.exists() and ToolApproval.OVERWRITE_FILE not in context.approvals:
            self._deny(context, "write_file", target_path, "覆盖已有文件需要用户确认。")
        result = self._tools.write_file(str(target_path), content)
        self._record(context, "write_file", target_path, allowed=True, reason=None)
        return result

    # 校验编辑授权、路径和精确匹配次数后替换文本。
    def replace_text(
        self: "WorkspaceToolExecutor",
        context: ToolExecutionContext,
        path: str,
        old_text: str,
        new_text: str,
        expected_replacements: int = 1,
    ) -> EditFileResult:
        target_path = self._authorize(context, "replace_text", ToolCapability.FILE_EDIT, path)
        if not old_text:
            self._deny(context, "replace_text", target_path, "old_text 不能为空。")
        if expected_replacements < 1:
            self._deny(context, "replace_text", target_path, "expected_replacements 必须至少为 1。")

        current_content = self._tools.read_file(str(target_path)).content
        actual_replacements = current_content.count(old_text)
        if actual_replacements != expected_replacements:
            self._deny(
                context,
                "replace_text",
                target_path,
                f"旧文本出现 {actual_replacements} 次，期望出现 {expected_replacements} 次。",
            )
        result = self._tools.replace_text(str(target_path), old_text, new_text)
        self._record(context, "replace_text", target_path, allowed=True, reason=None)
        return result

    # 在取得显式用户确认后运行 Bash；任意 Shell 命令不能仅凭能力标签自动放行。
    def run_bash(
        self: "WorkspaceToolExecutor",
        context: ToolExecutionContext,
        command: str,
        timeout_seconds: float | None = None,
    ) -> BashResult:
        if ToolCapability.BASH_EXECUTE not in context.capabilities:
            self._deny(context, "run_bash", Path("."), "当前 Agent 缺少 bash_execute 能力。")
        if ToolApproval.RUN_BASH not in context.approvals:
            self._deny(context, "run_bash", Path("."), "执行 Bash 命令需要用户确认。")
        if self._bash_tool is None:
            self._deny(context, "run_bash", Path("."), "当前执行器未配置 Bash 后端。")
        result = self._bash_tool.run(command, timeout_seconds)
        self._record(context, "run_bash", Path("."), allowed=True, reason=None)
        return result

    # 根据调用上下文、读写根目录白名单和保护规则确定允许访问的真实文件路径。
    def _authorize(
        self: "WorkspaceToolExecutor",
        context: ToolExecutionContext,
        operation: str,
        capability: ToolCapability,
        path: str,
    ) -> Path:
        if capability not in context.capabilities:
            self._deny(context, operation, Path(path), f"当前 Agent 缺少 {capability.value} 能力。")

        requested_path = Path(path)
        if not requested_path.is_absolute():
            self._deny(context, operation, requested_path, "路径必须是绝对路径。")
        target_path = requested_path.resolve()
        allowed_roots = self._allowed_roots(capability)
        relative_paths = self._relative_to_allowed_roots(target_path, allowed_roots)
        if not relative_paths:
            self._deny(context, operation, target_path, "路径不在允许访问的根目录中。")
        if any(not relative_path.parts for relative_path in relative_paths):
            self._deny(context, operation, target_path, "路径必须指向允许根目录中的文件。")
        if self._policy.protected_path_parts.intersection(target_path.parts):
            self._deny(context, operation, target_path, "路径位于受保护目录中。")
        if target_path.name in self._policy.protected_file_names:
            self._deny(context, operation, target_path, "路径指向受保护文件。")
        return target_path

    # 读取使用只读白名单，修改使用可写白名单，避免外部依赖目录被意外改写。
    def _allowed_roots(self: "WorkspaceToolExecutor", capability: ToolCapability) -> frozenset[Path]:
        if capability == ToolCapability.FILE_READ:
            return self._policy.readable_roots
        return self._policy.writable_roots

    # 仅保留能包含目标路径的根目录；解析后的路径可阻止符号链接逃逸白名单。
    def _relative_to_allowed_roots(
        self: "WorkspaceToolExecutor", target_path: Path, allowed_roots: frozenset[Path]
    ) -> list[Path]:
        relative_paths: list[Path] = []
        for allowed_root in allowed_roots:
            try:
                relative_paths.append(target_path.relative_to(allowed_root))
            except ValueError:
                continue
        return relative_paths

    # 记录拒绝原因并抛出执行层异常，禁止底层工具开始产生副作用。
    def _deny(
        self: "WorkspaceToolExecutor",
        context: ToolExecutionContext,
        operation: str,
        path: Path,
        reason: str,
    ) -> NoReturn:
        self._record(context, operation, path, allowed=False, reason=reason)
        raise ToolExecutionDenied(reason)

    # 使用 UTC 时间记录审计事件，避免依赖本机时区。
    def _record(
        self: "WorkspaceToolExecutor",
        context: ToolExecutionContext,
        operation: str,
        path: Path,
        allowed: bool,
        reason: str | None,
    ) -> None:
        self._audit_log.record(
            ToolAuditEvent(
                timestamp=datetime.now(timezone.utc),
                agent_id=context.agent_id,
                task_id=context.task_id,
                operation=operation,
                path=str(path),
                allowed=allowed,
                reason=reason,
            )
        )


# 为一轮 Agent 调用固定授权上下文，并暴露可直接注册的 Pydantic AI 工具。
class AuthorizedWorkspaceTools:
    def __init__(self: "AuthorizedWorkspaceTools", executor: WorkspaceToolExecutor, context: ToolExecutionContext) -> None:
        self._executor = executor
        self._context = context

    # 在本轮固定的调用上下文中执行受控读取。
    def read_file(
        self: "AuthorizedWorkspaceTools",
        path: AbsoluteFilePath,
        start_line: StartLine = 1,
        end_line: EndLine = None,
    ) -> ReadFileResult:
        try:
            return self._executor.read_file(self._context, path, start_line, end_line)
        except ToolExecutionDenied as error:
            raise ModelRetry(str(error)) from error

    # 在本轮固定的调用上下文中执行受控整文件写入。
    def write_file(self: "AuthorizedWorkspaceTools", path: AbsoluteFilePath, content: str) -> WriteFileResult:
        try:
            return self._executor.write_file(self._context, path, content)
        except ToolExecutionDenied as error:
            raise ModelRetry(str(error)) from error

    # 在本轮固定的调用上下文中执行受控精确文本替换。
    def replace_text(
        self: "AuthorizedWorkspaceTools",
        path: AbsoluteFilePath,
        old_text: str,
        new_text: str,
        expected_replacements: ExpectedReplacements = 1,
    ) -> EditFileResult:
        try:
            return self._executor.replace_text(self._context, path, old_text, new_text, expected_replacements)
        except ToolExecutionDenied as error:
            raise ModelRetry(str(error)) from error

    # 在本轮固定的调用上下文中执行经确认的 Bash 命令。
    def run_bash(
        self: "AuthorizedWorkspaceTools",
        command: BashCommand,
        timeout_seconds: TimeoutSeconds = None,
    ) -> BashResult:
        try:
            return self._executor.run_bash(self._context, command, timeout_seconds)
        except ToolExecutionDenied as error:
            raise ModelRetry(str(error)) from error

    # 返回仅通过执行层访问底层文件工具的 Pydantic AI 工具定义。
    def as_pydantic_tools(self: "AuthorizedWorkspaceTools") -> list[Tool[None]]:
        tools: list[Tool[None]] = []
        if ToolCapability.FILE_READ in self._context.capabilities:
            tools.append(Tool(self.read_file, description="读取受控白名单根目录内的 UTF-8 文本文件绝对路径，可按行范围读取。"))
        if ToolCapability.FILE_WRITE in self._context.capabilities:
            tools.append(Tool(self.write_file, description="写入受控可写根目录内的 UTF-8 文本文件绝对路径；覆盖已有文件需用户确认。"))
        if ToolCapability.FILE_EDIT in self._context.capabilities:
            tools.append(
                Tool(
                    self.replace_text,
                    description="精确替换受控可写根目录中已有文件的文本绝对路径；旧文本出现次数必须与预期一致。",
                )
            )
        if ToolCapability.BASH_EXECUTE in self._context.capabilities and self._executor.has_bash_tool:
            tools.append(
                Tool(
                    self.run_bash,
                    name="bash",
                    description=(
                        "在受控工作区根目录执行平台 Shell 命令；Unix 使用 Bash，Windows 使用 PowerShell。"
                        "每次执行都必须已经获得用户确认。"
                    ),
                )
            )
        return tools
