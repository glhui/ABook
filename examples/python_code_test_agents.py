"""并行编写 Python 实现与测试，并在两者结束后统一验证。"""

import asyncio
import os
import shutil
import sys
from pathlib import Path
from typing import Final

from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


ROOT_DIRECTORY: Final[Path] = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY: Final[Path] = ROOT_DIRECTORY / "src"
WORKSPACE_DIRECTORY: Final[Path] = ROOT_DIRECTORY / "tmp"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from agent_profiles import (
    create_code_test_task_coordinator,
    create_python_code_agent,
    create_python_code_context,
    create_python_test_agent,
    create_python_test_context,
    create_python_validator_agent,
    create_python_validator_context,
)
from tool_execution import (
    InMemoryToolAuditLog,
    ToolApproval,
    ToolCapability,
    ToolExecutionContext,
    WorkspaceExecutionPolicy,
    WorkspaceToolExecutor,
)
from workspace_tools import WorkspaceBashTool, create_workspace_file_tools


PYTHON_COMMAND: Final[str] = f'& "{sys.executable}"'
TEST_CHECK_COMMAND: Final[str] = f"{PYTHON_COMMAND} -m compileall case_generator.py solution.py validator.py"
VALIDATOR_COMMAND: Final[str] = f"{PYTHON_COMMAND} validator.py"
FINAL_VALIDATION_COMMAND: Final[str] = (
    f"{TEST_CHECK_COMMAND}; if ($LASTEXITCODE -ne 0) {{ exit $LASTEXITCODE }}; "
    f"{VALIDATOR_COMMAND}"
)
WORKSPACE_RULES: Final[str] = (
    f"受控工作区是 `{WORKSPACE_DIRECTORY}`。所有 read_file、write_file 和 replace_text 的 path 参数必须是"
    f"该目录内的绝对路径；禁止访问 `{ROOT_DIRECTORY}` 中的任何文件。"
    f"项目规范已复制到 `{WORKSPACE_DIRECTORY / 'AGENTS.md'}`，必须读取这一份，禁止读取项目根目录的 AGENTS.md。"
    "所有新建源代码和测试也必须只写入受控工作区。"
)


# 从 .env 创建 OpenAI 兼容模型，凭据仅由环境变量提供。
def create_model() -> OpenAIChatModel:
    load_dotenv(ROOT_DIRECTORY / ".env")
    model_name = os.getenv("ABOOK_MODEL", "deepseek-chat")
    api_key = os.getenv("ABOOK_API_KEY")
    base_url = os.getenv("ABOOK_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        raise ValueError("请先在项目根目录 .env 中设置 ABOOK_API_KEY。")
    return OpenAIChatModel(model_name, provider=OpenAIProvider(base_url=base_url, api_key=api_key))


# 创建位于 tmp 的隔离工作区，并同步 Agent 必须遵循的项目编码规范。
def create_executor() -> WorkspaceToolExecutor:
    WORKSPACE_DIRECTORY.mkdir(exist_ok=True)
    shutil.copyfile(ROOT_DIRECTORY / "AGENTS.md", WORKSPACE_DIRECTORY / "AGENTS.md")
    return WorkspaceToolExecutor(
        WorkspaceExecutionPolicy(WORKSPACE_DIRECTORY),
        create_workspace_file_tools(),
        InMemoryToolAuditLog(),
        WorkspaceBashTool(WORKSPACE_DIRECTORY),
    )


# 在代码、测试和黑盒验证脚本结束后，以宿主程序控制的标准输入输出验收提供唯一结论。
def validate_workspace(executor: WorkspaceToolExecutor) -> bool:
    context = ToolExecutionContext(
        agent_id="workflow-validator",
        task_id="validate-stdin-stdout-contract",
        capabilities=frozenset({ToolCapability.BASH_EXECUTE}),
        approvals=frozenset({ToolApproval.RUN_BASH}),
    )
    result = executor.run_bash(context, FINAL_VALIDATION_COMMAND)
    print(f"\n最终验证命令：{FINAL_VALIDATION_COMMAND}")
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.exit_code == 0 and not result.timed_out:
        print("最终验证通过：测试套件与独立标准输入输出验证均已通过。")
        return True
    print("最终验证失败：标准输入输出行为未满足独立验收条件。", file=sys.stderr)
    return False


# 并行执行测试和实现任务，并在两者完成后运行唯一的确定性验收步骤。
async def run_workflow(problem: str) -> bool:
    model = create_model()
    executor = create_executor()
    coordinator = create_code_test_task_coordinator(model)
    allocation = (await coordinator.run(problem)).output
    print(f"隔离工作区：{WORKSPACE_DIRECTORY}")
    print(f"测试任务：{allocation.testing_task}")
    print(f"编码任务：{allocation.coding_task}")

    confirmation = input("允许 Agent 修改文件并执行本地测试命令吗？[y/N] ").strip().lower()
    if confirmation != "y":
        print("未取得确认，流程结束。")
        return False
    approvals = frozenset({ToolApproval.OVERWRITE_FILE, ToolApproval.RUN_BASH})

    test_agent = create_python_test_agent(
        model,
        executor,
        create_python_test_context("write-tests", approvals),
    )
    test_prompt = (
        f"{WORKSPACE_RULES}\n"
        "你与代码 Agent 并行工作。只创建 case_generator.py，禁止修改生产代码和 validator.py。\n"
        f"{allocation.testing_task}\n"
        "case_generator.py 必须只向 stdout 输出 JSON 案例数组，不执行 solution.py，不包含断言或期望结果。"
    )
    code_prompt = (
        f"{WORKSPACE_RULES}\n"
        "你与测试 Agent 并行工作。你只可创建或修改 `tests` 目录之外的生产代码，禁止修改测试。\n"
        f"{allocation.coding_task}\n"
        "不以性能为目标，优先选择清晰、可证明正确且完整处理边界情况的实现。"
        "必须提供可执行的 solution.py 作为统一黑盒入口：从 stdin 读取 JSON，向 stdout 输出单个 JSON；"
        "不得把日志或调试信息写入 stdout，错误应通过非零退出码表示。"
        "测试 Agent 尚在并行写入测试；不要自行运行或宣称最终测试结论。"
        "两个任务结束后由宿主程序统一运行验收命令。"
    )

    code_agent = create_python_code_agent(
        model,
        executor,
        create_python_code_context("implement-code", approvals),
    )
    test_result, code_result = await asyncio.gather(
        test_agent.run(test_prompt),
        code_agent.run(code_prompt),
    )
    print(f"\n测试 Agent：\n{test_result.output}")
    print(f"\n代码 Agent：\n{code_result.output}")
    validator_agent = create_python_validator_agent(
        model,
        executor,
        create_python_validator_context("write-black-box-validator", approvals),
    )
    validator_prompt = (
        f"用户原始需求：{problem}\n"
        f"协调者分配的验收任务：{allocation.validation_task}\n"
        f"{WORKSPACE_RULES}\n"
        f"只允许创建 `{WORKSPACE_DIRECTORY / 'validator.py'}`，不得读取或修改其他文件。\n"
        "validator.py 必须先通过 subprocess 启动 case_generator.py，从其 stdout 解析 JSON 案例；"
        "再逐例启动 solution.py，向其 stdin 写入 JSON，从 stdout 解析单个 JSON 结果。"
        "禁止依赖 unittest、pytest 或现有 tests。\n"
        "请根据需求推导可证明正确的 oracle，生成覆盖边界情况的 validator.py，"
        "并用 subprocess 隔离运行 solution.py。"
    )
    validator_result = await validator_agent.run(validator_prompt)
    print(f"\n验证 Agent：\n{validator_result.output}")
    return validate_workspace(executor)


if __name__ == "__main__":
    completed = asyncio.run(run_workflow(input("请输入 Python 开发需求：").strip()))
    raise SystemExit(0 if completed else 1)
