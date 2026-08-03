"""私有标准输入输出验证和有限修复闭环的测试。"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from submission_validation import (
    PrivateTestBundle,
    PrivateTestCase,
    SubmissionValidator,
    ValidationResult,
    ValidationStage,
    load_private_test_bundle,
    validate_with_repairs,
)
from tool_execution import (
    InMemoryToolAuditLog,
    ToolCapability,
    ToolExecutionContext,
    ToolExecutionDenied,
    WorkspaceExecutionPolicy,
    WorkspaceToolExecutor,
)
from workspace_tools import create_workspace_file_tools


# 验证宿主可以把真实编译错误交给修复回调，并在下一轮重新验证。
class SubmissionValidationLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_compile_failure_can_be_repaired_and_revalidated(
        self: "SubmissionValidationLoopTests",
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            submission_path = Path(temporary_directory) / "solution.py"
            submission_path.write_text("def broken(:\n", encoding="utf-8")
            validator = SubmissionValidator(
                submission_path,
                PrivateTestBundle((PrivateTestCase("one", {"value": 1}, 2),)),
            )
            feedback: list[str] = []

            async def repair(result: ValidationResult, attempt: int) -> None:
                feedback.append(result.feedback())
                self.assertEqual(attempt, 1)
                submission_path.write_text(
                    "import json\n"
                    "import sys\n"
                    "value = json.load(sys.stdin)['value']\n"
                    "print(json.dumps(value + 1))\n",
                    encoding="utf-8",
                )

            result = await validate_with_repairs(validator, repair, max_attempts=2)

            self.assertTrue(result.passed)
            self.assertIn("compile", feedback[0])
            self.assertFalse((submission_path.parent / "__pycache__").exists())

    async def test_retry_limit_returns_last_failure_without_looping_forever(
        self: "SubmissionValidationLoopTests",
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            submission_path = Path(temporary_directory) / "solution.py"
            submission_path.write_text("print('wrong')\n", encoding="utf-8")
            validator = SubmissionValidator(
                submission_path,
                PrivateTestBundle((PrivateTestCase("one", 1, 2),)),
            )
            attempts: list[int] = []

            async def repair(result: ValidationResult, attempt: int) -> None:
                attempts.append(attempt)

            result = await validate_with_repairs(validator, repair, max_attempts=3)

            self.assertFalse(result.passed)
            self.assertEqual(attempts, [1, 2])
            self.assertEqual(result.failure.stage, ValidationStage.OUTPUT)


# 验证私有 oracle 不会出现在面向代码 Agent 的修复反馈中。
class SubmissionValidatorTests(unittest.TestCase):
    def test_assertion_feedback_does_not_reveal_expected_value(self: "SubmissionValidatorTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            submission_path = Path(temporary_directory) / "solution.py"
            submission_path.write_text("print(3)\n", encoding="utf-8")
            result = SubmissionValidator(
                submission_path,
                PrivateTestBundle((PrivateTestCase("private-case", 1, 999),)),
            ).validate()

            self.assertFalse(result.passed)
            self.assertEqual(result.failure.stage, ValidationStage.ASSERTION)
            self.assertNotIn("999", result.feedback())

    def test_load_private_bundle_rejects_duplicate_case_ids(self: "SubmissionValidatorTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            bundle_path = Path(temporary_directory) / "test_bundle.json"
            bundle_path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {"id": "same", "input": 1, "expected": 1},
                            {"id": "same", "input": 2, "expected": 2},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "重复"):
                load_private_test_bundle(bundle_path)


# 验证提交 Agent 的执行器无法通过文件工具读取私有测试根目录。
class PrivateTestIsolationTests(unittest.TestCase):
    def test_submission_executor_cannot_read_private_test_bundle(self: "PrivateTestIsolationTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            root_directory = Path(temporary_directory)
            submission_directory = root_directory / "submission"
            private_tests_directory = root_directory / "private-tests"
            submission_directory.mkdir()
            private_tests_directory.mkdir()
            private_bundle_path = private_tests_directory / "test_bundle.json"
            private_bundle_path.write_text('{"cases": []}', encoding="utf-8")
            executor = WorkspaceToolExecutor(
                WorkspaceExecutionPolicy(submission_directory),
                create_workspace_file_tools(),
                InMemoryToolAuditLog(),
            )
            context = ToolExecutionContext(
                agent_id="python-code",
                task_id="implement-submission",
                capabilities=frozenset({ToolCapability.FILE_READ}),
            )

            with self.assertRaisesRegex(ToolExecutionDenied, "允许访问"):
                executor.read_file(context, str(private_bundle_path))
