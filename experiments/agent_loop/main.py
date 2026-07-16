"""运行最小上下文 Runtime 与工具型 Agent 的命令行入口。"""

import os
from pathlib import Path
import sys
from typing import Callable

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .agent_runtime import create_agent, run_agent
from .context import (
    AgentContext,
    AgentDependencies,
    ContextRuntime,
    TaskState,
    WorkspaceContextBuilder,
)


EXIT_COMMANDS = frozenset({"/quit", "/exit", "quit", "exit"})


def create_model() -> OpenAIChatModel:
    """使用项目 .env 中的 OpenAI 兼容配置创建模型。"""
    load_dotenv()
    provider = OpenAIProvider(
        base_url=os.environ["ABOOK_BASE_URL"],
        api_key=os.environ["ABOOK_API_KEY"],
    )
    return OpenAIChatModel(os.environ["ABOOK_MODEL"], provider=provider)


def run_conversation(
    agent: Agent[AgentDependencies, str],
    runtime: ContextRuntime,
    root_context: AgentContext,
    initial_request: str,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    """在一个 root AgentContext 中持续处理终端输入。

    每轮通过同一个 ``root_context`` 调用 Agent，因此消息历史、任务状态、证据和
    子 Agent 交接会持续进入后续上下文。退出命令不会发送给模型，避免把会话控制
    文本误当作用户任务。
    """
    request = initial_request.strip()
    while True:
        if request:
            result = run_agent(agent, runtime, root_context, request)
            output_fn(result.output)

        try:
            request = input_fn("You> ").strip()
        except EOFError:
            return
        if request.casefold() in EXIT_COMMANDS:
            return


def main() -> None:
    """创建一个 root 会话，并在终端中持续处理用户请求。"""
    initial_request = " ".join(sys.argv[1:]).strip()
    if not initial_request:
        try:
            initial_request = input("You> ").strip()
        except EOFError:
            return
    if not initial_request or initial_request.casefold() in EXIT_COMMANDS:
        return

    experiment_root = Path(__file__).parent
    workspace_context = WorkspaceContextBuilder(
        workspace_root=Path.cwd(),
        skills_root=experiment_root / "skills",
    ).build()
    runtime = ContextRuntime(
        workspace_context, TaskState(goal=initial_request)
    )
    root_context = runtime.create_agent_context(
        "root", initial_request, "general"
    )
    run_conversation(
        create_agent(create_model()),
        runtime,
        root_context,
        initial_request,
    )


if __name__ == "__main__":
    main()
