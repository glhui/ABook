"""使用 LangGraph 编排代码、测试、验证和修复 Agent 的示例。"""

import asyncio
from collections.abc import Callable
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Final, Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel

ROOT_DIRECTORY: Final[Path] = Path(__file__).resolve().parents[1]
if str(ROOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(ROOT_DIRECTORY))

from agent_profiles import (
    CodeTestTaskAllocation,
    create_code_test_task_coordinator,
    create_python_code_agent,
    create_python_test_agent,
)
from agent_runtime import run_observed
from examples.python_code_test_agents import (
    ANSI_CYAN,
    ANSI_GREEN,
    ANSI_MAGENTA,
    ANSI_RED,
    ANSI_YELLOW,
    MAX_REPAIR_ATTEMPTS,
    RUNS_DIRECTORY,
    AgentRunLogger,
    IsolatedRun,
    create_file_only_context,
    create_isolated_executor,
    create_isolated_run,
    create_model,
    create_repair_prompt,
    print_log_block,
    publish_submission,
    run_pytest,
)


# 保存图节点之间传递的最小业务状态；模型和 Agent 实例保留在节点闭包中，不写入状态。
class CodeTestWorkflowState(TypedDict, total=False):
    problem: str
    allocation: CodeTestTaskAllocation
    approved: bool
    isolated_run: IsolatedRun
    validation_result: subprocess.CompletedProcess[str]
    repair_attempt: int
    workflow_succeeded: bool


WorkflowRoute = Literal["prepare_workspace", "cancel", "repair", "finish"]


# 构建只负责流程转换的 LangGraph，具体 Agent、工具和宿主验证逻辑沿用现有示例。
def create_code_test_workflow(
    model: Model,
    run_directory: Path,
    confirm: Callable[[], bool],
) -> CompiledStateGraph[CodeTestWorkflowState, None, CodeTestWorkflowState, CodeTestWorkflowState]:
    graph = StateGraph(CodeTestWorkflowState)
    coordinator_logger = AgentRunLogger("协调 Agent", show_model_text=False)
    code_logger = AgentRunLogger("代码 Agent")
    test_logger = AgentRunLogger("测试 Agent")

    # 将自然语言需求收敛为源文件、测试文件和公开行为契约。
    async def plan(state: CodeTestWorkflowState) -> CodeTestWorkflowState:
        print_log_block("工作流", "1/4 任务拆分", "正在确定核心函数和文件……", ANSI_CYAN)
        result, _ = await run_observed(
            create_code_test_task_coordinator(model),
            state["problem"],
            on_response=coordinator_logger.on_response,
            on_event=coordinator_logger.on_event,
        )
        allocation = result.output
        print_log_block(
            "协调 Agent",
            "结构化结果",
            allocation.model_dump_json(indent=2),
            ANSI_MAGENTA,
        )
        return {"allocation": allocation, "approved": confirm(), "repair_attempt": 0}

    # 在确认后建立隔离项目，确保后续所有节点共享同一份源码与测试文件。
    def prepare_workspace(state: CodeTestWorkflowState) -> CodeTestWorkflowState:
        allocation = state["allocation"]
        isolated_run = create_isolated_run(run_directory, allocation.source_file, allocation.test_file)
        print_log_block("工作流", "2/4 隔离项目", "已创建隔离的 pytest 项目。", ANSI_CYAN)
        return {"isolated_run": isolated_run}

    # 代码节点只负责实现生产代码；验证节点会把失败诊断送往单独的修复节点。
    async def implement_code(state: CodeTestWorkflowState) -> CodeTestWorkflowState:
        allocation = state["allocation"]
        isolated_run = state["isolated_run"]
        code_agent = create_python_code_agent(
            model,
            create_isolated_executor(isolated_run.root_directory),
            create_file_only_context("python-code", "implement-core-function", allow_bash=True),
        )
        prompt = (
            f"项目目录是 `{isolated_run.root_directory}`。只读取 AGENTS.md，并只修改 `{isolated_run.source_file}`。\n"
            f"源码文件：{allocation.source_file}\n测试文件：{allocation.test_file}\n"
            f"核心函数：{allocation.core_function}\n行为要求：{allocation.requirements}\n"
            "实现核心函数；不要读取或修改测试文件。修改后可使用 bash 对源码运行 `python -m py_compile`。"
            "最终 pytest 测试由宿主执行。"
        )
        print_log_block("工作流", "3/4 代码 Agent", "已启动。", ANSI_CYAN)
        await run_observed(code_agent, prompt, on_response=code_logger.on_response, on_event=code_logger.on_event)
        print_log_block("工作流", "3/4 代码 Agent", "已完成核心函数。", ANSI_GREEN)
        return {}

    # 测试节点根据同一份公开契约编写测试，不能以失败为由修改生产代码。
    async def write_tests(state: CodeTestWorkflowState) -> CodeTestWorkflowState:
        allocation = state["allocation"]
        isolated_run = state["isolated_run"]
        test_agent = create_python_test_agent(
            model,
            create_isolated_executor(isolated_run.root_directory),
            create_file_only_context("python-test", "write-pytest-tests"),
        )
        prompt = (
            f"项目目录是 `{isolated_run.root_directory}`。只读取 AGENTS.md，并只修改 `{isolated_run.test_file}`。\n"
            f"源码文件：{allocation.source_file}\n核心函数：{allocation.core_function}\n"
            f"行为要求：{allocation.requirements}\n"
            "使用 pytest 编写正常、边界和错误场景测试；不要读取或修改源码文件。"
        )
        print_log_block("工作流", "3/4 测试 Agent", "已启动。", ANSI_CYAN)
        await run_observed(test_agent, prompt, on_response=test_logger.on_response, on_event=test_logger.on_event)
        print_log_block("工作流", "3/4 测试 Agent", "已完成 pytest 测试。", ANSI_GREEN)
        return {}

    # 宿主独立运行 pytest，并把真实结果写回图状态供条件边选择下一节点。
    def validate(state: CodeTestWorkflowState) -> CodeTestWorkflowState:
        isolated_run = state["isolated_run"]
        print_log_block("工作流", "4/4 pytest", "正在运行 pytest。", ANSI_CYAN)
        return {"validation_result": run_pytest(isolated_run.root_directory, isolated_run.test_file)}

    # 修复节点只把实际 pytest 诊断提供给代码 Agent，随后回到验证节点形成有界循环。
    async def repair(state: CodeTestWorkflowState) -> CodeTestWorkflowState:
        allocation = state["allocation"]
        isolated_run = state["isolated_run"]
        attempt = state["repair_attempt"] + 1
        validation_result = state["validation_result"]
        print_log_block("工作流", "4/4 修复", f"pytest 未通过，正在进行第 {attempt} 次修复。", ANSI_YELLOW)
        code_agent = create_python_code_agent(
            model,
            create_isolated_executor(isolated_run.root_directory),
            create_file_only_context("python-code", "repair-core-function", allow_bash=True),
        )
        prompt = create_repair_prompt(
            isolated_run,
            allocation.source_file,
            allocation.test_file,
            allocation.core_function,
            allocation.requirements,
            validation_result.stdout + validation_result.stderr,
        )
        await run_observed(code_agent, prompt, on_response=code_logger.on_response, on_event=code_logger.on_event)
        return {"repair_attempt": attempt}

    # 将终止节点与正常失败区别开，调用方只需要读取一个明确的布尔结果。
    def finish(state: CodeTestWorkflowState) -> CodeTestWorkflowState:
        passed = state["validation_result"].returncode == 0
        return {"workflow_succeeded": passed}

    # 用户拒绝写入时不创建隔离文件，也不调用后续 Agent。
    def cancel(_: CodeTestWorkflowState) -> CodeTestWorkflowState:
        print_log_block("工作流", "已取消", "未取得确认，流程结束。", ANSI_YELLOW)
        return {"workflow_succeeded": False}

    # 只依据真实 pytest 退出码和有界重试次数决定循环，避免由模型文本控制流程。
    def route_after_validation(state: CodeTestWorkflowState) -> WorkflowRoute:
        if state["validation_result"].returncode == 0 or state["repair_attempt"] >= MAX_REPAIR_ATTEMPTS:
            return "finish"
        return "repair"

    # 确认分支在协调结果生成后发生，保持原示例的交互顺序。
    def route_after_plan(state: CodeTestWorkflowState) -> WorkflowRoute:
        return "prepare_workspace" if state["approved"] else "cancel"

    graph.add_node("plan", plan)
    graph.add_node("prepare_workspace", prepare_workspace)
    graph.add_node("implement_code", implement_code)
    graph.add_node("write_tests", write_tests)
    graph.add_node("validate", validate)
    graph.add_node("repair", repair)
    graph.add_node("finish", finish)
    graph.add_node("cancel", cancel)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", route_after_plan, {"prepare_workspace": "prepare_workspace", "cancel": "cancel"})
    graph.add_edge("prepare_workspace", "implement_code")
    graph.add_edge("implement_code", "write_tests")
    graph.add_edge("write_tests", "validate")
    graph.add_conditional_edges("validate", route_after_validation, {"repair": "repair", "finish": "finish"})
    graph.add_edge("repair", "validate")
    graph.add_edge("finish", END)
    graph.add_edge("cancel", END)
    return graph.compile()


# 保持原示例的命令行交互和临时目录清理行为，仅将步骤推进委托给 LangGraph。
async def run_workflow(problem: str) -> bool:
    model: OpenAIChatModel = create_model()
    RUNS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # 将交互式确认封装为图节点可调用的无参回调，便于测试时替换。
    def confirm() -> bool:
        answer = input("允许 Agent 在隔离目录中修改文件，并让代码 Agent 编译源码吗？[y/N] ")
        return answer.strip().lower() == "y"

    with TemporaryDirectory(prefix="langgraph-run-", dir=RUNS_DIRECTORY) as temporary_directory:
        workflow = create_code_test_workflow(model, Path(temporary_directory), confirm)
        result = cast(CodeTestWorkflowState, await workflow.ainvoke({"problem": problem}))
        isolated_run = result.get("isolated_run")
        if isolated_run is not None:
            output_path = publish_submission(isolated_run)
            if output_path is not None:
                print_log_block("工作流", "生成结果", f"生成代码已保存到：{output_path}", ANSI_GREEN)
        return result["workflow_succeeded"]


if __name__ == "__main__":
    try:
        completed = asyncio.run(run_workflow(input("请输入 Python 开发需求：").strip()))
    except UnexpectedModelBehavior as error:
        print_log_block("工作流", "运行错误", str(error), ANSI_RED, sys.stderr)
        completed = False
    raise SystemExit(0 if completed else 1)
