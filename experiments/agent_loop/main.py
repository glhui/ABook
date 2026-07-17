"""运行最小上下文 Runtime 与工具型 Agent 的命令行入口。"""

import os
from pathlib import Path
import sys
from typing import Callable

from dotenv import load_dotenv
from openai.types.chat import (
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from pydantic_ai import Agent
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openai import OpenAIChatModelSettings
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


class DeepSeekThinkingChatModel(OpenAIChatModel):
    """为 DeepSeek 思考模式省略不兼容的 ``tool_choice`` 请求字段。

    PydanticAI 在提供函数工具时默认发送 ``tool_choice='auto'``。DeepSeek
    的思考模式虽然支持工具定义，却拒绝这个显式字段并返回 HTTP 400。省略后
    Chat Completions API 的默认行为仍是自动选择工具，因此不会削弱 Agent 的
    工具调用能力。
    """

    def _get_tool_choice(
        self,
        model_settings: OpenAIChatModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[
        list[ChatCompletionToolParam], ChatCompletionToolChoiceOptionParam | None
    ]:
        """保留工具定义，但不把默认的自动选择显式发送给 DeepSeek。"""
        tools, _tool_choice = super()._get_tool_choice(
            model_settings, model_request_parameters
        )
        return tools, None


def create_model() -> OpenAIChatModel:
    """使用项目 .env 中的 OpenAI 兼容配置创建模型。

    DeepSeek 模型 ID 使用专用适配类，以兼容思考模式与函数工具同时启用的
    Chat Completions 请求；其他 OpenAI 兼容模型保持 PydanticAI 默认行为。
    """
    load_dotenv()
    provider = OpenAIProvider(
        base_url=os.environ["ABOOK_BASE_URL"],
        api_key=os.environ["ABOOK_API_KEY"],
    )
    model_name = os.environ["ABOOK_MODEL"]
    model_type = (
        DeepSeekThinkingChatModel
        if model_name.casefold().startswith("deepseek")
        else OpenAIChatModel
    )
    return model_type(model_name, provider=provider)


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
            # ``input_fn`` 的提示词已经标识用户输入；为避免终端中的模型回答
            # 与用户文本混在一起，这里为每轮最终回答加上对应角色标记。
            output_fn(f"Assistant> {result.output}")

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
