"""通过 DeepSeek 与 Pydantic AI 进行带工具调用的非流式多轮对话。"""

import asyncio
import ast
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Final

from dotenv import load_dotenv
from pydantic_ai import Agent, FunctionToolCallEvent, FunctionToolResultEvent
from pydantic_ai.messages import HandleResponseEvent, ModelRequest, ModelResponse, TextPart, ToolReturnPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

ROOT_DIRECTORY: Final[Path] = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY: Final[Path] = ROOT_DIRECTORY / "src"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from agent_runtime import run_turn
from terminal import CYAN, GREEN, MAGENTA, RED, RESET, YELLOW
from tool_execution import (
    AuthorizedWorkspaceTools,
    InMemoryToolAuditLog,
    ToolApproval,
    ToolCapability,
    ToolExecutionContext,
    WorkspaceExecutionPolicy,
    WorkspaceToolExecutor,
)
from workspace_tools import WorkspaceBashTool, create_workspace_file_tools

DIVIDER: Final[str] = "=" * 72


# 安全计算器仅允许基础算术表达式，避免执行任意 Python 代码。
def evaluate_expression(expression: str) -> int | float:
    expression_tree = ast.parse(expression, mode="eval")
    return evaluate_node(expression_tree.body)


# 递归求值经白名单验证过的抽象语法树节点。
def evaluate_node(node: ast.expr) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = evaluate_node(node.operand)
        return operand if isinstance(node.op, ast.UAdd) else -operand

    if isinstance(node, ast.BinOp):
        left = evaluate_node(node.left)
        right = evaluate_node(node.right)
        operators: dict[type[ast.operator], object] = {
            ast.Add: lambda: left + right,
            ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right,
            ast.Div: lambda: left / right,
            ast.FloorDiv: lambda: left // right,
            ast.Mod: lambda: left % right,
            ast.Pow: lambda: left**right,
        }
        operation = operators.get(type(node.op))
        if operation is not None:
            return operation()  # type: ignore[operator]

    raise ValueError("表达式只支持数字与 +、-、*、/、//、%、**、括号。")


# 将输出限制为单行，便于在终端中检查工具轨迹。
def compact(value: object) -> str:
    return " ".join(str(value).split())


# 打印带颜色的区块标题和内容。
def print_block(color: str, title: str, content: str) -> None:
    print(f"{color}{DIVIDER}\n{title}{RESET}")
    print(content)


# 返回本机当前时间，供模型回答时间相关问题。
def get_local_time() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


# 计算受限的算术表达式，例如 (18 + 6) * 3。
def calculate(expression: str) -> str:
    return str(evaluate_expression(expression))


# 从 .env 创建指向 OpenAI 兼容 DeepSeek API 的模型。
def create_agent() -> Agent[None, str]:
    load_dotenv(ROOT_DIRECTORY / ".env")
    model_name = os.getenv("ABOOK_MODEL", "deepseek-chat")
    api_key = os.getenv("ABOOK_API_KEY")
    base_url = os.getenv("ABOOK_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        raise ValueError("请先在项目根目录 .env 中设置 ABOOK_API_KEY。")

    model = OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(base_url=base_url, api_key=api_key),
    )
    workspace_tools = create_workspace_file_tools()
    bash_tool = WorkspaceBashTool(ROOT_DIRECTORY)
    executor = WorkspaceToolExecutor(
        WorkspaceExecutionPolicy(ROOT_DIRECTORY),
        workspace_tools,
        InMemoryToolAuditLog(),
        bash_tool,
    )
    # 此示例预先授予只读和 Bash 调用确认，用于演示执行层接入；生产环境应在用户确认后再写入 approvals。
    execution_context = ToolExecutionContext(
        agent_id="example-chat",
        task_id="interactive-chat",
        capabilities=frozenset({ToolCapability.FILE_READ, ToolCapability.BASH_EXECUTE}),
        approvals=frozenset({ToolApproval.RUN_BASH}),
    )
    authorized_tools = AuthorizedWorkspaceTools(executor, execution_context)
    agent = Agent(
        model,
        instructions=(
            "你是一个中文助手。需要当前本地时间时调用 get_local_time；"
            "需要精确算术计算时调用 calculate。需要查看工作区文件内容时调用 read_file；"
            "需要列出或搜索工作区文件、查看只读 Git 状态时调用 bash。"
            "当前在 Windows 时，bash 工具实际执行 PowerShell，应使用 Get-ChildItem、rg 或 git status，"
            "不要使用 ls -la 等 Bash 专用参数。"
            "不得调用 bash 修改文件、安装依赖、访问网络或提交 Git。工具调用完成后，用中文简洁说明结果。"
        ),
        tools=[get_local_time, calculate, *authorized_tools.as_pydantic_tools()],
    )
    return agent


# 打印一条完整模型响应中的文本部分。
def print_model_response(response: ModelResponse) -> None:
    text_parts = [part.content for part in response.parts if isinstance(part, TextPart)]
    if text_parts:
        print_block(GREEN, "模型回复", "".join(text_parts))


# 打印当前示例关心的函数工具事件，其他事件可在此处继续扩展。
def print_tool_event(event: HandleResponseEvent) -> None:
    if isinstance(event, FunctionToolCallEvent):
        arguments = event.part.args
        content = json.dumps(arguments, ensure_ascii=False) if isinstance(arguments, dict) else compact(arguments)
        print_block(MAGENTA, f"工具调用  {event.part.tool_name}", f"参数: {content}")
    elif isinstance(event, FunctionToolResultEvent):
        result_content = event.part.content if isinstance(event.part, ToolReturnPart) else event.content
        print_block(YELLOW, f"工具结果  {event.part.tool_name}", compact(result_content))


# 循环读取终端输入，并保留历史消息以支持多轮对话。
async def run_chat() -> None:
    agent = create_agent()
    history: list[ModelRequest | ModelResponse] = []
    print_block(CYAN, "DeepSeek + Pydantic AI 工具对话", "输入问题开始对话；输入 /quit、/exit 或按 Ctrl+C 结束。")

    while True:
        try:
            user_prompt = input(f"{CYAN}\n你 > {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{CYAN}对话结束。{RESET}")
            return

        if user_prompt.lower() in {"/quit", "/exit"}:
            print(f"{CYAN}对话结束。{RESET}")
            return
        if not user_prompt:
            continue

        print(f"{RED}{DIVIDER}\n请求模型（非流式）…{RESET}")
        history = await run_turn(
            agent,
            user_prompt,
            history,
            on_response=print_model_response,
            on_event=print_tool_event,
        )


if __name__ == "__main__":
    asyncio.run(run_chat())
