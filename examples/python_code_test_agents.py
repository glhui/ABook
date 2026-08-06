"""隔离编写核心函数与 pytest 测试，并由宿主统一执行测试。"""

import asyncio
import atexit
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Final, TextIO

from dotenv import load_dotenv
from pydantic_ai import Agent, FunctionToolCallEvent, FunctionToolResultEvent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import HandleResponseEvent, ModelResponse, TextPart, ThinkingPart, ToolReturnPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


ROOT_DIRECTORY: Final[Path] = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY: Final[Path] = ROOT_DIRECTORY / "src"
RUNS_DIRECTORY: Final[Path] = ROOT_DIRECTORY / "tmp" / "runs"
OUTPUT_DIRECTORY: Final[Path] = ROOT_DIRECTORY / "tmp" / "output"
MAX_LOG_CHARACTERS: Final[int] = 4_000
MAX_VALIDATION_RETRIES: Final[int] = 2
LOG_SEPARATOR_WIDTH: Final[int] = 56
ANSI_RESET: Final[str] = "\033[0m"
ANSI_BLUE: Final[str] = "\033[34m"
ANSI_CYAN: Final[str] = "\033[36m"
ANSI_GREEN: Final[str] = "\033[32m"
ANSI_MAGENTA: Final[str] = "\033[35m"
ANSI_RED: Final[str] = "\033[31m"
ANSI_YELLOW: Final[str] = "\033[33m"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from agent_profiles import create_code_test_task_coordinator, create_python_code_agent, create_python_test_agent, normalize_task_plan
from agent_runtime import run_observed
from skill_loading import SkillCatalog, SkillSelector, SkillTaskContext, render_skill_instructions
from terminal import InteractiveLogRenderer, LogEntry, create_log_renderer
from tool_execution import (
    InMemoryToolAuditLog,
    ToolApproval,
    ToolCapability,
    ToolExecutionContext,
    WorkspaceExecutionPolicy,
    WorkspaceToolExecutor,
)
from workspace_tools import WorkspaceBashTool, create_workspace_file_tools


_LOG_RENDERER = create_log_renderer()
if isinstance(_LOG_RENDERER, InteractiveLogRenderer):
    atexit.register(_LOG_RENDERER.close)


def pause_interactive_logs() -> None:
    """在调用普通 input() 前释放交互式日志的键盘控制权。"""
    if isinstance(_LOG_RENDERER, InteractiveLogRenderer):
        _LOG_RENDERER.pause()


def resume_interactive_logs() -> None:
    """在普通 input() 完成后恢复交互式日志。"""
    if isinstance(_LOG_RENDERER, InteractiveLogRenderer):
        _LOG_RENDERER.resume()


# 保存本轮隔离项目及协调器指定的源码和测试文件。
@dataclass(frozen=True)
class IsolatedRun:
    root_directory: Path
    source_file: Path
    test_file: Path
    visualization_file: Path | None = None

    @property
    def artifacts_directory(self: "IsolatedRun") -> Path:
        return self.root_directory / "artifacts"

    @property
    def solution_path(self: "IsolatedRun") -> Path:
        return self.source_file


@dataclass(frozen=True)
class VisualizationResult:
    """保存可视化脚本执行和图片尺寸检查结果。"""

    script_exit_code: int | None
    image_path: Path | None
    width: int | None
    height: int | None
    error: str | None

# 按事件类型统一输出带颜色的分隔块；NO_COLOR 可用于日志采集或不支持 ANSI 的终端。
def print_log_block(scope: str, event: str, content: str, color: str, stream: TextIO | None = None) -> None:
    if stream is None and isinstance(_LOG_RENDERER, InteractiveLogRenderer):
        _LOG_RENDERER.emit(LogEntry(scope=scope, event=event, content=content, color=color))
        return
    heading = f"{'=' * 2} [{scope}][{event}] {'=' * (LOG_SEPARATOR_WIDTH - len(scope) - len(event) - 7)}"
    if os.getenv("NO_COLOR") is None:
        heading = f"{color}{heading}{ANSI_RESET}"
    target = sys.stdout if stream is None else stream
    print(f"{heading}\n{content}", file=target, flush=True)


# 实时打印一个 Agent 的每轮模型输出、工具调用开始和工具调用结果。
class AgentRunLogger:
    def __init__(self: "AgentRunLogger", agent_name: str, show_model_text: bool = True) -> None:
        self._agent_name = agent_name
        self._show_model_text = show_model_text
        self._response_number = 0
        self._pending_tool_arguments: dict[str, list[str]] = {}

    # 模型每完成一轮响应就记录轮次；仅对非结构化 Agent 显示原始文本，避免泄露未校验的 JSON/Schema。
    def on_response(self: "AgentRunLogger", response: ModelResponse) -> None:
        self._response_number += 1
        if not self._show_model_text:
            return
        contents = [
            part.content
            for part in response.parts
            if isinstance(part, TextPart | ThinkingPart) and part.content
        ]
        content = "\n".join(contents) if contents else "（本轮无文本输出，准备调用工具。）"
        print_log_block(self._agent_name, f"模型轮次 {self._response_number}", content, ANSI_BLUE)

    # 结构化输出在校验完成后记录，与对应模型轮次合并为一条用户可读日志。
    def log_structured_result(self: "AgentRunLogger", content: str) -> None:
        print_log_block(self._agent_name, f"模型轮次 {self._response_number}", content, ANSI_BLUE)

    # 工具开始和完成事件在发生时打印，单条内容有界以避免整文件写入淹没终端。
    def on_event(self: "AgentRunLogger", event: HandleResponseEvent) -> None:
        if isinstance(event, FunctionToolCallEvent):
            arguments = event.part.args
            serialized = json.dumps(arguments, ensure_ascii=False) if isinstance(arguments, dict) else str(arguments)
            self._pending_tool_arguments.setdefault(event.part.tool_name, []).append(self._bounded(serialized))
        elif isinstance(event, FunctionToolResultEvent):
            result_content = event.part.content if isinstance(event.part, ToolReturnPart) else event.content
            arguments = self._pending_tool_arguments.get(event.part.tool_name, [])
            serialized_arguments = arguments.pop(0) if arguments else "（参数未记录）"
            if not arguments:
                self._pending_tool_arguments.pop(event.part.tool_name, None)
            print_log_block(
                self._agent_name,
                f"工具：{event.part.tool_name} · {self._tool_argument_summary(serialized_arguments)}",
                f"参数：{serialized_arguments}\n结果：{self._bounded(str(result_content))}",
                ANSI_GREEN,
            )

    # 截断超长的单条日志，并明确标记截断位置。
    def _bounded(self: "AgentRunLogger", value: str) -> str:
        return bounded_log(value)

    # 将常见工具参数提炼到标题，折叠状态下也能区分连续的同名调用。
    def _tool_argument_summary(self: "AgentRunLogger", arguments: str) -> str:
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return self._bounded(arguments).replace("\n", " ")
        if isinstance(parsed, dict):
            for key in ("path", "command", "name"):
                value = parsed.get(key)
                if value is not None:
                    return self._bounded(str(value)).replace("\n", " ")
        return self._bounded(arguments).replace("\n", " ")


# 截断会反馈给 Agent 的命令输出，避免单次失败日志占满后续上下文。
def bounded_log(value: str) -> str:
    if len(value) <= MAX_LOG_CHARACTERS:
        return value
    return f"{value[:MAX_LOG_CHARACTERS]}\n……（日志已截断）"


# 从 .env 创建 OpenAI 兼容模型，凭据仅由环境变量提供。
def create_model() -> OpenAIChatModel:
    load_dotenv(ROOT_DIRECTORY / ".env")
    model_name = os.getenv("ABOOK_MODEL", "deepseek-chat")
    api_key = os.getenv("ABOOK_API_KEY")
    base_url = os.getenv("ABOOK_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        raise ValueError("请先在项目根目录 .env 中设置 ABOOK_API_KEY。")
    return OpenAIChatModel(model_name, provider=OpenAIProvider(base_url=base_url, api_key=api_key))


# 创建一次性 pytest 项目，并预置协调器指定的源码与测试文件。
def create_isolated_run(
    root_directory: Path,
    source_file: str,
    test_file: str,
    visualization_file: str | None = None,
) -> IsolatedRun:
    source_path = _resolve_generated_path(root_directory, source_file, "src")
    test_path = _resolve_generated_path(root_directory, test_file, "tests")
    visualization_path = (
        _resolve_generated_path(root_directory, visualization_file, "visualizations")
        if visualization_file is not None
        else None
    )
    source_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    if visualization_path is not None:
        visualization_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT_DIRECTORY / "AGENTS.md", root_directory / "AGENTS.md")
    source_path.write_text("# 在此实现核心函数。\n", encoding="utf-8")
    test_path.write_text("# 在此编写 pytest 测试。\n", encoding="utf-8")
    if visualization_path is not None:
        visualization_path.write_text("# 在此编写可视化程序。\n", encoding="utf-8")
        (root_directory / "artifacts").mkdir(parents=True, exist_ok=True)
    return IsolatedRun(root_directory, source_path, test_path, visualization_path)


# 执行可视化脚本，检查产物是否存在且尺寸处于安全范围。
def run_visualization(isolated_run: IsolatedRun) -> VisualizationResult:
    if isolated_run.visualization_file is None:
        return VisualizationResult(None, None, None, None, "任务未声明可视化脚本。")
    completed = subprocess.run(
        [sys.executable, str(isolated_run.visualization_file)],
        cwd=isolated_run.root_directory,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if completed.returncode != 0:
        return VisualizationResult(completed.returncode, None, None, None, completed.stderr[-2_000:])
    image_candidates = sorted(isolated_run.artifacts_directory.glob("*.png"))
    if not image_candidates:
        return VisualizationResult(completed.returncode, None, None, None, "可视化脚本未生成 PNG 图片。")
    image_path = image_candidates[-1]
    try:
        with image_path.open("rb") as image_file:
            header = image_file.read(24)
        if header[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError("不是有效的 PNG 文件。")
        width, height = struct.unpack(">II", header[16:24])
    except (OSError, ValueError, struct.error) as error:
        return VisualizationResult(completed.returncode, image_path, None, None, f"无法读取 PNG 尺寸：{error}")
    if not 32 <= width <= 4_096 or not 32 <= height <= 4_096:
        return VisualizationResult(completed.returncode, image_path, width, height, "图片尺寸必须位于 32 到 4096 像素之间。")
    return VisualizationResult(completed.returncode, image_path, width, height, None)


# 在支持终端图片协议时尝试内嵌预览，否则输出稳定的路径和尺寸信息。
def preview_visualization(result: VisualizationResult) -> None:
    if result.image_path is None:
        print_log_block("可视化", "检查失败", result.error or "没有图片产物。", ANSI_RED)
        return
    summary = f"图片：{result.image_path}\n尺寸：{result.width} × {result.height}"
    if result.error:
        print_log_block("可视化", "尺寸检查失败", f"{summary}\n{result.error}", ANSI_RED)
        return
    try:
        from rich.console import Console
        from rich.image import Image as RichImage

        Console().print(RichImage(str(result.image_path), width=60))
    except (ImportError, OSError, RuntimeError):
        print_log_block("可视化", "预览", summary, ANSI_GREEN)


# 将模型给出的相对路径限制在指定项目子目录，避免路径逃逸隔离工作区。
def _resolve_generated_path(root_directory: Path, relative_path: str, required_directory: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or not candidate.parts or candidate.parts[0] != required_directory:
        raise ValueError(f"生成路径必须位于 {required_directory}/：{relative_path}")
    resolved_root = root_directory.resolve()
    resolved_path = (resolved_root / candidate).resolve()
    if resolved_root not in resolved_path.parents:
        raise ValueError(f"生成路径不得离开隔离项目：{relative_path}")
    return resolved_path


# 仅在隔离项目内执行工具；代码 Agent 可额外使用 Bash 做编译检查，pytest 仍由宿主统一执行。
def create_isolated_executor(workspace_directory: Path) -> WorkspaceToolExecutor:
    policy = WorkspaceExecutionPolicy(workspace_directory, read_only_file_names=frozenset({"AGENTS.md"}))
    return WorkspaceToolExecutor(
        policy,
        create_workspace_file_tools(),
        InMemoryToolAuditLog(),
        WorkspaceBashTool(workspace_directory),
    )


# 将生成的核心函数源码发布到固定输出目录。
def publish_submission(isolated_run: IsolatedRun, output_directory: Path = OUTPUT_DIRECTORY) -> Path | None:
    if not isolated_run.solution_path.is_file():
        return None
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / isolated_run.solution_path.name
    shutil.copyfile(isolated_run.solution_path, output_path)
    return output_path


# 为隔离 Agent 构造最小授权上下文；仅代码 Agent 可额外取得编译所需的 Bash 能力。
def create_file_only_context(agent_id: str, task_id: str, allow_bash: bool = False) -> ToolExecutionContext:
    capabilities = {ToolCapability.FILE_READ, ToolCapability.FILE_WRITE, ToolCapability.FILE_EDIT}
    approvals = {ToolApproval.OVERWRITE_FILE}
    if allow_bash:
        capabilities.add(ToolCapability.BASH_EXECUTE)
        approvals.add(ToolApproval.RUN_BASH)
    return ToolExecutionContext(
        agent_id=agent_id,
        task_id=task_id,
        capabilities=frozenset(capabilities),
        approvals=frozenset(approvals),
    )


# 由宿主执行 pytest 并保留输出，以便失败时将可操作的诊断交给代码 Agent。
def run_pytest(root_directory: Path, test_file: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(test_file)],
        cwd=root_directory,
        check=False,
        capture_output=True,
        text=True,
    )
    result_event = "pytest 通过" if completed.returncode == 0 else "pytest 失败"
    result_color = ANSI_GREEN if completed.returncode == 0 else ANSI_RED
    if completed.stdout:
        print_log_block("宿主", result_event, completed.stdout.rstrip(), result_color)
    if completed.stderr:
        print_log_block("宿主", result_event, completed.stderr.rstrip(), result_color, sys.stderr)
    return completed


# 构造只允许修复生产代码的失败诊断，防止 Agent 通过修改测试规避问题。
def create_code_retry_prompt(
    isolated_run: IsolatedRun,
    source_file: str,
    test_file: str,
    core_function: str,
    requirements: str,
    pytest_output: str,
) -> str:
    return (
        f"宿主 pytest 未通过。项目目录是 `{isolated_run.root_directory}`。只读取 AGENTS.md 和 "
        f"`{isolated_run.source_file}`，并只修改 "
        f"`{isolated_run.source_file}`。\n源码文件：{source_file}\n测试文件：{test_file}\n"
        f"核心函数：{core_function}\n行为要求：{requirements}\n"
        f"pytest 输出：\n{bounded_log(pytest_output)}\n"
        "这是代码编写节点的下一轮，不是独立的修复角色。根据失败信息修改实现；"
        "不要读取或修改测试文件，不要削弱测试断言。"
        f"必须使用 bash 运行 `python -m py_compile {source_file}`，不要使用 `cd` 或 `&&`；"
        "只有编译通过后才能结束本轮；"
        "最终 pytest 验证由宿主执行。"
    )


# 按顺序编写核心函数和 pytest 测试，再由宿主运行 pytest。
async def run_workflow(problem: str, skill_catalog: SkillCatalog | None = None) -> bool:
    model = create_model()
    print_log_block("工作流", "1/4 任务拆分", "正在确定核心函数和文件……", ANSI_CYAN)
    coordinator = create_code_test_task_coordinator(model)
    coordinator_logger = AgentRunLogger("协调 Agent", show_model_text=False)
    coordinator_result, _ = await run_observed(
        coordinator,
        problem,
        on_response=coordinator_logger.on_response,
        on_event=coordinator_logger.on_event,
    )
    allocation = normalize_task_plan(problem, coordinator_result.output)
    coordinator_logger.log_structured_result(json.dumps(allocation.model_dump(), ensure_ascii=False, indent=2))
    if allocation.human_checkpoints:
        print_log_block(
            "人工检查点",
            "任务计划",
            "\n".join(f"- {checkpoint}" for checkpoint in allocation.human_checkpoints),
            ANSI_YELLOW,
        )
    selected_skills = ()
    if skill_catalog is not None:
        selected_skills = SkillSelector(skill_catalog).select(
            SkillTaskContext(
                task=f"{problem}\n{allocation.requirements}",
                role="python_code",
                target_paths=(allocation.source_file,),
                symbols=(allocation.core_function,),
            )
        )
        selection_summary = "\n".join(
            f"- {skill.name}: {'；'.join(skill.reasons)}" for skill in selected_skills
        ) or "未找到相关 Skill，继续使用固定角色说明。"
        print_log_block("Skill 路由", "选择结果", selection_summary, ANSI_MAGENTA)
    skill_instructions = render_skill_instructions(selected_skills)
    print_log_block("工作流", "1/4 任务拆分", "核心函数和文件已确定。", ANSI_GREEN)
    pause_interactive_logs()
    try:
        confirmation = input("允许 Agent 在隔离目录中修改文件，并让代码和测试 Agent 编译各自文件吗？[y/N] ").strip().lower()
    finally:
        resume_interactive_logs()
    if confirmation != "y":
        print_log_block("工作流", "已取消", "未取得确认，流程结束。", ANSI_YELLOW)
        return False

    RUNS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="run-", dir=RUNS_DIRECTORY) as temporary_directory:
        isolated_run = create_isolated_run(
            Path(temporary_directory),
            allocation.source_file,
            allocation.test_file,
            allocation.visualization_file if allocation.needs_visualization else None,
        )
        print_log_block("工作流", "2/4 隔离项目", "已创建隔离的 pytest 项目。", ANSI_CYAN)
        try:
            executor = create_isolated_executor(isolated_run.root_directory)
            test_agent = create_python_test_agent(
                model,
                executor,
                create_file_only_context("python-test", "write-pytest-tests", allow_bash=True),
            )
            code_agent = create_python_code_agent(
                model,
                executor,
                create_file_only_context("python-code", "implement-core-function", allow_bash=True),
                skill_instructions=skill_instructions,
            )
            test_logger = AgentRunLogger("测试 Agent")
            code_logger = AgentRunLogger("代码 Agent")
            test_prompt = (
                f"项目目录是 `{isolated_run.root_directory}`。只读取 AGENTS.md，并只修改 `{isolated_run.test_file}`。\n"
                f"源码文件：{allocation.source_file}\n核心函数：{allocation.core_function}\n"
                f"行为要求：{allocation.requirements}\n验证策略：{allocation.validation_strategy}\n"
                "使用 pytest 编写正常、边界和错误场景测试；不要读取或修改源码文件。"
                f"必须使用 bash 运行 `python -m py_compile {allocation.test_file}`，不要使用 `cd` 或 `&&`；"
                "只有编译通过后才能结束本轮。"
            )
            code_prompt = (
                f"项目目录是 `{isolated_run.root_directory}`。只读取 AGENTS.md，并只修改 `{isolated_run.source_file}`。\n"
                f"源码文件：{allocation.source_file}\n测试文件：{allocation.test_file}\n"
                f"核心函数：{allocation.core_function}\n行为要求：{allocation.requirements}\n"
                f"验证策略：{allocation.validation_strategy}\n"
                "实现核心函数；不要读取或修改测试文件。"
                f"必须使用 bash 运行 `python -m py_compile {allocation.source_file}`，不要使用 `cd` 或 `&&`；"
                "只有编译通过后才能结束本轮。"
                "最终 pytest 测试由宿主执行。"
            )

            async def generate_tests() -> None:
                print_log_block("工作流", "3/4 测试 Agent", "已启动。", ANSI_CYAN)
                await run_observed(
                    test_agent,
                    test_prompt,
                    on_response=test_logger.on_response,
                    on_event=test_logger.on_event,
                )
                print_log_block("工作流", "3/4 测试 Agent", "测试文件已编译通过。", ANSI_GREEN)

            async def generate_code(code_authoring_prompt: str, event: str) -> None:
                print_log_block("工作流", event, "已启动。", ANSI_CYAN)
                await run_observed(
                    code_agent,
                    code_authoring_prompt,
                    on_response=code_logger.on_response,
                    on_event=code_logger.on_event,
                )
                print_log_block("工作流", event, "源码已编译通过。", ANSI_GREEN)

            # 代码先完成，避免两个 Agent 同时写日志；测试 Agent 随后仅依据需求验证公开行为。
            await generate_code(code_prompt, "3/4 代码 Agent")
            await generate_tests()
            if allocation.needs_visualization and isolated_run.visualization_file is not None:
                pause_interactive_logs()
                try:
                    visualization_approval = input(
                        "是否生成数学可视化程序 "
                        f"{allocation.visualization_file}？[y/N] "
                    ).strip().lower() == "y"
                finally:
                    resume_interactive_logs()
                if visualization_approval:
                    visualization_prompt = (
                        f"项目目录是 `{isolated_run.root_directory}`。只读取 AGENTS.md 和 "
                        f"`{isolated_run.source_file}`，只修改 `{isolated_run.visualization_file}`。\n"
                        f"核心函数：{allocation.core_function}\n行为要求：{allocation.requirements}\n"
                        f"验证策略：{allocation.validation_strategy}\n"
                        f"必须将 PNG 图片保存为 `{isolated_run.root_directory / 'artifacts' / 'visualization.png'}`。"
                        "使用 matplotlib 或任务适合的可视化库，生成可重复运行的可视化脚本；"
                        "脚本必须从源码导入核心函数，不要修改源码或测试文件。"
                    )
                    print_log_block("工作流", "3/4 可视化 Agent", "已启动。", ANSI_CYAN)
                    await run_observed(
                        code_agent,
                        visualization_prompt,
                        on_response=code_logger.on_response,
                        on_event=code_logger.on_event,
                    )
                    print_log_block("工作流", "3/4 可视化 Agent", "可视化脚本已生成。", ANSI_GREEN)
                    visualization_result = run_visualization(isolated_run)
                    preview_visualization(visualization_result)
            print_log_block("工作流", "4/4 pytest", "正在运行 pytest。", ANSI_CYAN)
            completed = run_pytest(isolated_run.root_directory, isolated_run.test_file)
            for retry_attempt in range(1, MAX_VALIDATION_RETRIES + 1):
                if completed.returncode == 0:
                    return True
                print_log_block(
                    "工作流",
                    "回到代码编写",
                    f"pytest 未通过，代码 Agent 正在进行第 {retry_attempt} 次重新编写；测试 Agent 不会再次运行。",
                    ANSI_YELLOW,
                )
                retry_prompt = create_code_retry_prompt(
                    isolated_run,
                    allocation.source_file,
                    allocation.test_file,
                    allocation.core_function,
                    allocation.requirements,
                    completed.stdout + completed.stderr,
                )
                await generate_code(retry_prompt, f"3/4 代码 Agent（第 {retry_attempt} 次回退）")
                print_log_block("工作流", "4/4 pytest", "正在重新运行 pytest。", ANSI_CYAN)
                completed = run_pytest(isolated_run.root_directory, isolated_run.test_file)
            return completed.returncode == 0
        finally:
            output_path = publish_submission(isolated_run)
            if output_path is not None:
                print_log_block("工作流", "生成结果", f"生成代码已保存到：{output_path}", ANSI_GREEN)


if __name__ == "__main__":
    try:
        completed = asyncio.run(run_workflow(input("请输入 Python 开发需求：").strip()))
    except UnexpectedModelBehavior as error:
        print_log_block("工作流", "模型错误", f"模型未能完成本轮生成：{error}", ANSI_RED, sys.stderr)
        completed = False
    raise SystemExit(0 if completed else 1)
