"""使用代码与测试角色上下文写入文件，并由宿主运行 pytest 的离线案例。"""

from pathlib import Path
import os
import sys

from agent_profiles import create_python_code_context, create_python_test_context
from tool_execution import InMemoryToolAuditLog, ToolApproval, WorkspaceExecutionPolicy, WorkspaceToolExecutor
from workspace_tools import BashResult, WorkspaceBashTool, create_workspace_file_tools


SOURCE_FILE = "word_normalizer.py"
TEST_FILE = "test_word_normalizer.py"
SOURCE_CONTENT = """\
def normalize_words(value: str) -> str:
    return " ".join(value.split()).lower()
"""
TEST_CONTENT = """\
from word_normalizer import normalize_words


def test_normalize_words_trims_collapses_and_lowercases() -> None:
    assert normalize_words("  Hello   WORLD  ") == "hello world"
"""


# 创建限定在案例临时目录内的执行器，使两个角色通过相同的授权边界写入各自文件。
def create_case_executor(workspace_root: Path) -> WorkspaceToolExecutor:
    return WorkspaceToolExecutor(
        WorkspaceExecutionPolicy(workspace_root=workspace_root),
        create_workspace_file_tools(),
        InMemoryToolAuditLog(),
        WorkspaceBashTool(workspace_root),
    )


# 依次模拟代码与测试角色的受控写入，再由宿主运行独立 pytest 验证生成的文件。
def run_case(workspace_root: Path) -> BashResult:
    executor = create_case_executor(workspace_root)
    write_approvals = frozenset({ToolApproval.OVERWRITE_FILE})
    code_context = create_python_code_context("write-source", write_approvals)
    test_context = create_python_test_context("write-test", write_approvals)

    executor.write_file(code_context, SOURCE_FILE, SOURCE_CONTENT)
    executor.write_file(test_context, TEST_FILE, TEST_CONTENT)

    command = (
        f"& '{sys.executable}' -m pytest {TEST_FILE}"
        if os.name == "nt"
        else f"'{sys.executable}' -m pytest {TEST_FILE}"
    )
    return WorkspaceBashTool(workspace_root).run(command, timeout_seconds=30)


# 直接运行时返回 pytest 的输出，并以 pytest 状态作为进程退出码。
def main() -> None:
    result = run_case(Path.cwd() / "tmp" / "profile-code-test-case")
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    raise SystemExit(result.exit_code or 0)


if __name__ == "__main__":
    main()
