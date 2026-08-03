"""隔离编写提交与私有测试，并由宿主进行有限次数的标准输入输出验证。"""

import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from typing import Final

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


ROOT_DIRECTORY: Final[Path] = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY: Final[Path] = ROOT_DIRECTORY / "src"
RUNS_DIRECTORY: Final[Path] = ROOT_DIRECTORY / "tmp" / "runs"
OUTPUT_DIRECTORY: Final[Path] = ROOT_DIRECTORY / "tmp" / "output"
MAX_VALIDATION_ATTEMPTS: Final[int] = 3
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from agent_profiles import (
    create_code_test_task_coordinator,
    create_python_code_agent,
    create_python_test_agent,
)
from submission_validation import SubmissionValidator, ValidationResult, load_private_test_bundle, validate_with_repairs
from tool_execution import (
    InMemoryToolAuditLog,
    ToolApproval,
    ToolCapability,
    ToolExecutionContext,
    WorkspaceExecutionPolicy,
    WorkspaceToolExecutor,
)
from workspace_tools import create_workspace_file_tools


# 保存本轮隔离目录；只有宿主同时持有提交与私有测试目录。
@dataclass(frozen=True)
class IsolatedRun:
    root_directory: Path
    submission_directory: Path
    private_tests_directory: Path

    @property
    def solution_path(self: "IsolatedRun") -> Path:
        return self.submission_directory / "solution.py"

    @property
    def test_bundle_path(self: "IsolatedRun") -> Path:
        return self.private_tests_directory / "test_bundle.json"


# 从 .env 创建 OpenAI 兼容模型，凭据仅由环境变量提供。
def create_model() -> OpenAIChatModel:
    load_dotenv(ROOT_DIRECTORY / ".env")
    model_name = os.getenv("ABOOK_MODEL", "deepseek-chat")
    api_key = os.getenv("ABOOK_API_KEY")
    base_url = os.getenv("ABOOK_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        raise ValueError("请先在项目根目录 .env 中设置 ABOOK_API_KEY。")
    return OpenAIChatModel(model_name, provider=OpenAIProvider(base_url=base_url, api_key=api_key))


# 创建一次性目录并向每个 Agent 的独立根目录复制不可修改的项目规范。
def create_isolated_run(root_directory: Path) -> IsolatedRun:
    submission_directory = root_directory / "submission"
    private_tests_directory = root_directory / "private-tests"
    submission_directory.mkdir()
    private_tests_directory.mkdir()
    shutil.copyfile(ROOT_DIRECTORY / "AGENTS.md", submission_directory / "AGENTS.md")
    shutil.copyfile(ROOT_DIRECTORY / "AGENTS.md", private_tests_directory / "AGENTS.md")
    isolated_run = IsolatedRun(root_directory, submission_directory, private_tests_directory)
    # 预置可读取模板，避免 Agent 因首次读取不存在文件而耗尽工具重试次数。
    isolated_run.solution_path.write_text("# 在此实现 stdin/stdout JSON 程序。\n", encoding="utf-8")
    isolated_run.test_bundle_path.write_text('{"cases": []}\n', encoding="utf-8")
    return isolated_run


# 仅授予所在目录的读写能力，不注册 Bash，避免 Agent 穿透目录读取另一侧的私有内容。
def create_isolated_executor(workspace_directory: Path) -> WorkspaceToolExecutor:
    policy = WorkspaceExecutionPolicy(
        workspace_directory,
        read_only_file_names=frozenset({"AGENTS.md"}),
    )
    return WorkspaceToolExecutor(policy, create_workspace_file_tools(), InMemoryToolAuditLog())


# 将最终或最后一次修复后的提交发布到固定目录，私有测试和过程文件仍保持临时。
def publish_submission(isolated_run: IsolatedRun, output_directory: Path = OUTPUT_DIRECTORY) -> Path | None:
    if not isolated_run.solution_path.is_file():
        return None
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "solution.py"
    shutil.copyfile(isolated_run.solution_path, output_path)
    return output_path


# 为隔离 Agent 构造无 Shell 的上下文，角色工厂仍会验证身份与最小文件能力。
def create_file_only_context(agent_id: str, task_id: str) -> ToolExecutionContext:
    return ToolExecutionContext(
        agent_id=agent_id,
        task_id=task_id,
        capabilities=frozenset(
            {
                ToolCapability.FILE_READ,
                ToolCapability.FILE_WRITE,
                ToolCapability.FILE_EDIT,
            }
        ),
        approvals=frozenset({ToolApproval.OVERWRITE_FILE}),
    )


# 将失败反馈交给代码 Agent；私有验证器不会泄露用例源码或 expected 值。
async def repair_submission(
    code_agent: Agent[None, str],
    result_feedback: str,
    attempt: int,
    submission_directory: Path,
) -> None:
    await code_agent.run(
        "这是宿主的第 "
        f"{attempt} 次私有验证反馈。只允许修改 `{submission_directory / 'solution.py'}`；"
        "不得试图读取测试或推测其文件路径。\n"
        f"{result_feedback}\n"
        "修复后停止，不要声称验证已经通过。"
    )


# 并行生成私有用例与提交实现，再由宿主执行最多三次的验证和修复闭环。
async def run_workflow(problem: str) -> bool:
    model = create_model()
    coordinator = create_code_test_task_coordinator(model)
    allocation = (await coordinator.run(problem)).output
    confirmation = input("允许 Agent 在隔离目录中修改文件吗？[y/N] ").strip().lower()
    if confirmation != "y":
        print("未取得确认，流程结束。")
        return False

    RUNS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="run-", dir=RUNS_DIRECTORY) as temporary_directory:
        isolated_run = create_isolated_run(Path(temporary_directory))
        try:
            test_executor = create_isolated_executor(isolated_run.private_tests_directory)
            code_executor = create_isolated_executor(isolated_run.submission_directory)
            test_agent = create_python_test_agent(
                model,
                test_executor,
                create_file_only_context("python-test", "write-private-test-bundle"),
            )
            code_agent = create_python_code_agent(
                model,
                code_executor,
                create_file_only_context("python-code", "implement-submission"),
            )
            test_prompt = (
                f"受控私有测试目录是 `{isolated_run.private_tests_directory}`。只允许读取此目录的 AGENTS.md 并修改 "
                f"`{isolated_run.test_bundle_path}`，不得访问或创建 solution.py。\n"
                f"任务：{allocation.testing_task}\n"
                "test_bundle.json 必须是 UTF-8 JSON，格式为 "
                '`{"cases":[{"id":"边界名称","input":<JSON>,"expected":<JSON>}]}`。'
                "用例必须独立、可重复，并包含正常、边界和错误输入。"
            )
            code_prompt = (
                f"受控提交目录是 `{isolated_run.submission_directory}`。只允许读取此目录的 AGENTS.md 并修改 "
                f"`{isolated_run.solution_path}`。\n"
                f"任务：{allocation.coding_task}\n"
                "solution.py 必须从 stdin 读取单个 UTF-8 JSON 值，向 stdout 输出单个 JSON 值；"
                "不得输出日志，不得创建测试、验证器或读取目录外文件。"
                "私有测试由另一个隔离 Agent 编写，宿主将只反馈有限失败摘要。"
            )
            await asyncio.gather(test_agent.run(test_prompt), code_agent.run(code_prompt))
            try:
                test_bundle = load_private_test_bundle(isolated_run.test_bundle_path)
            except (OSError, ValueError) as error:
                print(f"私有测试包无效：{error}", file=sys.stderr)
                return False

            validator = SubmissionValidator(isolated_run.solution_path, test_bundle)

            async def repair(result: ValidationResult, attempt: int) -> None:
                await repair_submission(code_agent, result.feedback(), attempt, isolated_run.submission_directory)

            validation_result = await validate_with_repairs(validator, repair, MAX_VALIDATION_ATTEMPTS)
            print(validation_result.feedback())
            return validation_result.passed
        finally:
            output_path = publish_submission(isolated_run)
            if output_path is not None:
                print(f"生成代码已保存到：{output_path}")


if __name__ == "__main__":
    try:
        completed = asyncio.run(run_workflow(input("请输入 Python 开发需求：").strip()))
    except UnexpectedModelBehavior as error:
        print(f"模型未能完成本轮生成：{error}", file=sys.stderr)
        completed = False
    raise SystemExit(0 if completed else 1)
