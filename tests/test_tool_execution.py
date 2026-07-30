"""工作区工具执行层的独立测试。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from tool_execution import (
    AuthorizedWorkspaceTools,
    InMemoryToolAuditLog,
    ToolApproval,
    ToolCapability,
    ToolExecutionContext,
    ToolExecutionDenied,
    WorkspaceExecutionPolicy,
    WorkspaceToolExecutor,
)
from workspace_tools import BashResult, BashRunner, WorkspaceBashTool, create_workspace_file_tools


# 验证执行层会在调用纯工具之前强制执行能力、路径和确认策略。
class WorkspaceToolExecutorTests(unittest.TestCase):
    def test_read_file_requires_read_capability_and_records_denial(self: "WorkspaceToolExecutorTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            workspace_root = Path(temporary_directory)
            (workspace_root / "note.txt").write_text("content", encoding="utf-8")
            executor, audit_log = self._create_executor(workspace_root)
            context = ToolExecutionContext(agent_id="agent-1", task_id="task-1", capabilities=frozenset())

            with self.assertRaisesRegex(ToolExecutionDenied, "file_read"):
                executor.read_file(context, "note.txt")

            event = audit_log.events()[-1]
            self.assertFalse(event.allowed)
            self.assertEqual(event.operation, "read_file")

    def test_read_file_rejects_workspace_escape_and_protected_file(self: "WorkspaceToolExecutorTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            workspace_root = temporary_root / "workspace"
            workspace_root.mkdir()
            executor, audit_log = self._create_executor(workspace_root)
            context = self._context(ToolCapability.FILE_READ)

            with self.assertRaisesRegex(ToolExecutionDenied, "工作区之外"):
                executor.read_file(context, "../outside.txt")
            with self.assertRaisesRegex(ToolExecutionDenied, "受保护文件"):
                executor.read_file(context, ".env")

            self.assertEqual([event.allowed for event in audit_log.events()], [False, False])

    def test_write_file_requires_overwrite_approval(self: "WorkspaceToolExecutorTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            workspace_root = Path(temporary_directory)
            target_path = workspace_root / "note.txt"
            target_path.write_text("before", encoding="utf-8")
            executor, audit_log = self._create_executor(workspace_root)
            unapproved_context = self._context(ToolCapability.FILE_WRITE)

            with self.assertRaisesRegex(ToolExecutionDenied, "覆盖"):
                executor.write_file(unapproved_context, "note.txt", "after")

            approved_context = ToolExecutionContext(
                agent_id="agent-1",
                task_id="task-1",
                capabilities=frozenset({ToolCapability.FILE_WRITE}),
                approvals=frozenset({ToolApproval.OVERWRITE_FILE}),
            )
            result = executor.write_file(approved_context, "note.txt", "after")

            self.assertEqual(result.bytes_written, 5)
            self.assertEqual(target_path.read_text(encoding="utf-8"), "after")
            self.assertEqual([event.allowed for event in audit_log.events()], [False, True])

    def test_replace_text_requires_exact_match_before_mutation(self: "WorkspaceToolExecutorTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            workspace_root = Path(temporary_directory)
            target_path = workspace_root / "note.txt"
            target_path.write_text("value=one\nvalue=one\n", encoding="utf-8")
            executor, _ = self._create_executor(workspace_root)
            context = self._context(ToolCapability.FILE_EDIT)

            with self.assertRaisesRegex(ToolExecutionDenied, "出现 2 次"):
                executor.replace_text(context, "note.txt", "value=one", "value=two")

            self.assertEqual(target_path.read_text(encoding="utf-8"), "value=one\nvalue=one\n")

    def test_run_bash_requires_confirmation_and_records_success(self: "WorkspaceToolExecutorTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            workspace_root = Path(temporary_directory)
            bash_runner = RecordingBashRunner()
            bash_tool = WorkspaceBashTool(workspace_root, bash_runner)
            executor, audit_log = self._create_executor(workspace_root, bash_tool)
            unapproved_context = self._context(ToolCapability.BASH_EXECUTE)

            with self.assertRaisesRegex(ToolExecutionDenied, "用户确认"):
                executor.run_bash(unapproved_context, "rg --files")

            approved_context = ToolExecutionContext(
                agent_id="agent-1",
                task_id="task-1",
                capabilities=frozenset({ToolCapability.BASH_EXECUTE}),
                approvals=frozenset({ToolApproval.RUN_BASH}),
            )
            result = executor.run_bash(approved_context, "rg --files", timeout_seconds=10)

            self.assertEqual(result.stdout, "notes.txt\n")
            self.assertEqual(bash_runner.calls, [("rg --files", workspace_root.resolve(), 10)])
            self.assertEqual([event.allowed for event in audit_log.events()], [False, True])

    # 创建测试所需的执行层及其可检查的内存审计日志。
    def _create_executor(
        self: "WorkspaceToolExecutorTests",
        workspace_root: Path,
        bash_tool: WorkspaceBashTool | None = None,
    ) -> tuple[WorkspaceToolExecutor, InMemoryToolAuditLog]:
        audit_log = InMemoryToolAuditLog()
        policy = WorkspaceExecutionPolicy(workspace_root)
        tools = create_workspace_file_tools(workspace_root)
        return WorkspaceToolExecutor(policy, tools, audit_log, bash_tool), audit_log

    # 为测试构造具备最小能力集的固定调用上下文。
    def _context(self: "WorkspaceToolExecutorTests", capability: ToolCapability) -> ToolExecutionContext:
        return ToolExecutionContext(
            agent_id="agent-1",
            task_id="task-1",
            capabilities=frozenset({capability}),
        )


# 验证模型只会获得当前调用上下文已授权的受控工具。
class AuthorizedWorkspaceToolsTests(unittest.TestCase):
    def test_as_pydantic_tools_registers_controlled_tools(self: "AuthorizedWorkspaceToolsTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            workspace_root = Path(temporary_directory)
            executor, _ = WorkspaceToolExecutorTests()._create_executor(workspace_root)
            context = ToolExecutionContext(
                agent_id="agent-1",
                task_id="task-1",
                capabilities=frozenset({ToolCapability.FILE_READ}),
            )
            tools = AuthorizedWorkspaceTools(executor, context)

            agent = Agent(TestModel(), tools=tools.as_pydantic_tools())

            self.assertEqual(set(agent._function_toolset.tools), {"read_file"})

    def test_as_pydantic_tools_exposes_bash_when_backend_and_capability_exist(self: "AuthorizedWorkspaceToolsTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            workspace_root = Path(temporary_directory)
            bash_tool = WorkspaceBashTool(workspace_root, RecordingBashRunner())
            executor, _ = WorkspaceToolExecutorTests()._create_executor(workspace_root, bash_tool)
            context = ToolExecutionContext(
                agent_id="agent-1",
                task_id="task-1",
                capabilities=frozenset({ToolCapability.BASH_EXECUTE}),
                approvals=frozenset({ToolApproval.RUN_BASH}),
            )
            tools = AuthorizedWorkspaceTools(executor, context)

            agent = Agent(TestModel(), tools=tools.as_pydantic_tools())

            self.assertEqual(set(agent._function_toolset.tools), {"bash"})


# 提供确定性的 Bash 后端，避免测试依赖宿主是否安装 Bash 或 rg。
class RecordingBashRunner(BashRunner):
    def __init__(self: "RecordingBashRunner") -> None:
        self.calls: list[tuple[str, Path, float | None]] = []

    # 记录调用参数并返回固定命令输出。
    def run(self: "RecordingBashRunner", command: str, working_directory: Path, timeout_seconds: float | None) -> BashResult:
        self.calls.append((command, working_directory, timeout_seconds))
        return BashResult(command=command, stdout="notes.txt\n", stderr="", exit_code=0, timed_out=False)
