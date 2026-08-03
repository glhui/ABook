"""隔离代码生成示例的输出发布测试。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from examples.python_code_test_agents import IsolatedRun, publish_submission


# 验证临时运行目录删除前，最后一次提交会被复制到持久输出目录。
class PublishSubmissionTests(unittest.TestCase):
    def test_publish_submission_preserves_generated_code(self: "PublishSubmissionTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            root_directory = Path(temporary_directory)
            submission_directory = root_directory / "submission"
            private_tests_directory = root_directory / "private-tests"
            output_directory = root_directory / "output"
            submission_directory.mkdir()
            private_tests_directory.mkdir()
            isolated_run = IsolatedRun(root_directory, submission_directory, private_tests_directory)
            isolated_run.solution_path.write_text("print(42)\n", encoding="utf-8")

            output_path = publish_submission(isolated_run, output_directory)

            self.assertEqual(output_path, output_directory / "solution.py")
            self.assertEqual(output_path.read_text(encoding="utf-8"), "print(42)\n")
