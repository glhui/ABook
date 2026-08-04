"""隔离代码生成示例的输出发布测试。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from examples.python_code_test_agents import create_isolated_run, publish_submission


# 验证临时运行目录删除前，最后一次提交会被复制到持久输出目录。
class PublishSubmissionTests(unittest.TestCase):
    def test_publish_submission_preserves_generated_code(self: "PublishSubmissionTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            root_directory = Path(temporary_directory)
            output_directory = root_directory / "output"
            isolated_run = create_isolated_run(root_directory, "src/solution.py", "tests/test_solution.py")
            isolated_run.solution_path.write_text("print(42)\n", encoding="utf-8")

            output_path = publish_submission(isolated_run, output_directory)

            self.assertEqual(output_path, output_directory / "solution.py")
            self.assertEqual(output_path.read_text(encoding="utf-8"), "print(42)\n")

    def test_create_isolated_run_prepares_pytest_files(self: "PublishSubmissionTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            isolated_run = create_isolated_run(
                Path(temporary_directory),
                "src/calculator.py",
                "tests/test_calculator.py",
            )

            self.assertTrue(isolated_run.source_file.is_file())
            self.assertTrue(isolated_run.test_file.is_file())
            self.assertTrue((isolated_run.root_directory / "AGENTS.md").is_file())

    def test_create_isolated_run_rejects_paths_outside_expected_directories(
        self: "PublishSubmissionTests",
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            with self.assertRaises(ValueError):
                create_isolated_run(
                    Path(temporary_directory),
                    "../solution.py",
                    "tests/test_solution.py",
                )
