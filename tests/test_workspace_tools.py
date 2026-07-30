"""工作区文件工具的独立测试。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from workspace_tools import create_workspace_file_tools


# 验证读取工具的分段语义和返回元数据。
class WorkspaceFileToolsReadTests(unittest.TestCase):
    def test_read_file_returns_requested_line_range(self: "WorkspaceFileToolsReadTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            workspace_root = Path(temporary_directory)
            with (workspace_root / "notes.txt").open("w", encoding="utf-8", newline="") as notes_file:
                notes_file.write("first\nsecond\nthird\n")
            tools = create_workspace_file_tools()

            result = tools.read_file(str(workspace_root / "notes.txt"), start_line=2, end_line=2)

            self.assertEqual(result.content, "second\n")
            self.assertEqual(result.start_line, 2)
            self.assertEqual(result.end_line, 2)
            self.assertEqual(result.total_lines, 3)
            self.assertTrue(result.truncated)

    def test_read_file_rejects_relative_path(self: "WorkspaceFileToolsReadTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            workspace_root = temporary_root / "workspace"
            workspace_root.mkdir()
            outside_path = temporary_root / "outside.txt"
            outside_path.write_text("outside", encoding="utf-8")
            tools = create_workspace_file_tools()

            with self.assertRaisesRegex(ValueError, "绝对路径"):
                tools.read_file("outside.txt")

            self.assertEqual(outside_path.read_text(encoding="utf-8"), "outside")


# 验证创建和编辑不会覆盖未经精确匹配确认的内容。
class WorkspaceFileToolsMutationTests(unittest.TestCase):
    def test_write_file_creates_parent_directories_and_overwrites_existing_file(self: "WorkspaceFileToolsMutationTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            workspace_root = Path(temporary_directory)
            tools = create_workspace_file_tools()

            result = tools.write_file(str(workspace_root / "nested" / "created.txt"), "hello")

            self.assertEqual(result.path, str((workspace_root / "nested" / "created.txt").resolve()))
            self.assertEqual(result.bytes_written, 5)
            self.assertEqual((workspace_root / "nested" / "created.txt").read_text(encoding="utf-8"), "hello")

            tools.write_file(str(workspace_root / "nested" / "created.txt"), "replacement")

            self.assertEqual((workspace_root / "nested" / "created.txt").read_text(encoding="utf-8"), "replacement")

    def test_replace_text_returns_actual_replacement_count(self: "WorkspaceFileToolsMutationTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            workspace_root = Path(temporary_directory)
            target_path = workspace_root / "settings.txt"
            target_path.write_text("enabled=true\nenabled=true\n", encoding="utf-8")
            tools = create_workspace_file_tools()

            result = tools.replace_text(str(target_path), "enabled=true", "enabled=false")

            self.assertEqual(result.replacements, 2)
            self.assertEqual(target_path.read_text(encoding="utf-8"), "enabled=false\nenabled=false\n")

    def test_as_pydantic_tools_exposes_all_file_operations(self: "WorkspaceFileToolsMutationTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            tools = create_workspace_file_tools()

            agent = Agent(TestModel(), tools=tools.as_pydantic_tools())

            self.assertEqual(
                set(agent._function_toolset.tools),
                {"read_file", "write_file", "replace_text"},
            )
