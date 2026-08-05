"""隔离代码生成示例的输出发布测试。"""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pydantic_ai.messages import ModelResponse, TextPart

from examples.python_code_test_agents import (
    AgentRunLogger,
    ANSI_CYAN,
    create_file_only_context,
    create_isolated_executor,
    create_isolated_run,
    create_code_retry_prompt,
    print_log_block,
    publish_submission,
)


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


# 结构化 Agent 的原始响应可能包含 Schema 或失败重试内容，不应直接混入用户日志。
class AgentRunLoggerTests(unittest.TestCase):
    def test_print_log_block_uses_separator_and_color_by_default(self: "AgentRunLoggerTests") -> None:
        output = StringIO()

        with patch.dict("os.environ", {}, clear=True):
            print_log_block("工作流", "阶段", "内容", ANSI_CYAN, output)

        log = output.getvalue()
        self.assertIn("== [工作流][阶段]", log)
        self.assertIn(ANSI_CYAN, log)

    def test_structured_logger_hides_unvalidated_model_text(self: "AgentRunLoggerTests") -> None:
        output = StringIO()
        logger = AgentRunLogger("协调 Agent", show_model_text=False)

        with redirect_stdout(output):
            logger.on_response(ModelResponse(parts=[TextPart(content='{"properties": {}}')]))

        log = output.getvalue()
        self.assertIn("等待结构化校验", log)
        self.assertNotIn('"properties"', log)


# 回退到代码编写的提示必须携带失败诊断，同时保持代码 Agent 对测试文件的隔离约束。
class CodeRetryPromptTests(unittest.TestCase):
    def test_create_code_retry_prompt_includes_failure_and_preserves_test_boundary(self: "CodeRetryPromptTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            isolated_run = create_isolated_run(
                Path(temporary_directory),
                "src/newton.py",
                "tests/test_newton.py",
            )

            prompt = create_code_retry_prompt(
                isolated_run,
                "src/newton.py",
                "tests/test_newton.py",
                "newton_raphson(x: float) -> float",
                "导数为零时抛出 ValueError。",
                "FAILED tests/test_newton.py::test_zero_derivative_raises_value_error",
            )

        self.assertIn("FAILED tests/test_newton.py", prompt)
        self.assertIn(f"只读取 AGENTS.md 和 `{isolated_run.source_file}`", prompt)
        self.assertIn("只修改", prompt)
        self.assertIn("不要读取或修改测试文件", prompt)
        self.assertIn("不是独立的修复角色", prompt)
        self.assertIn("必须使用 bash", prompt)
        self.assertIn("不要使用 `cd` 或 `&&`", prompt)


# 两个编写 Agent 都需要 Bash 运行各自文件的本地编译检查。
class IsolatedAgentContextTests(unittest.TestCase):
    def test_create_file_only_context_grants_bash_when_requested(self: "IsolatedAgentContextTests") -> None:
        code_context = create_file_only_context("python-code", "compile-source", allow_bash=True)
        test_context = create_file_only_context("python-test", "compile-tests", allow_bash=True)

        self.assertIn("bash_execute", code_context.capabilities)
        self.assertIn("run_bash", code_context.approvals)
        self.assertIn("bash_execute", test_context.capabilities)
        self.assertIn("run_bash", test_context.approvals)

    def test_isolated_executor_configures_bash_backend(self: "IsolatedAgentContextTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            executor = create_isolated_executor(Path(temporary_directory))

        self.assertTrue(executor.has_bash_tool)
