"""运行最小上下文 Runtime 与工具型 Agent 的命令行入口。"""

import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from openai.types.chat import (
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openai import OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

from .agents.agent_runtime import create_coordinator_agent
from .coordination.conversation import EXIT_COMMANDS, run_conversation
from .context import (
    ContextRuntime,
    TaskState,
    WorkspaceContextBuilder,
)
from .persistence import RuntimeStateStore


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


def main() -> None:
    """创建一个任务协调会话，并在终端中持续处理用户请求。"""
    initial_request = " ".join(sys.argv[1:]).strip()
    if not initial_request:
        try:
            initial_request = input("You> ").strip()
        except EOFError:
            return
    if not initial_request or initial_request.casefold() in EXIT_COMMANDS:
        return

    experiment_root = Path(__file__).parent.parent
    workspace_context = WorkspaceContextBuilder(
        workspace_root=Path.cwd(),
        skills_root=experiment_root / "skills",
    ).build()
    state_store = RuntimeStateStore(Path.cwd() / ".abook" / "runtime-state.json")
    runtime = state_store.load(workspace_context)
    if runtime is None:
        runtime = ContextRuntime(
            workspace_context, TaskState(goal=initial_request)
        )
        coordinator_context = runtime.create_agent_context(
            "coordinator", initial_request, "general", role="coordinator"
        )
    else:
        coordinator_context = next(
            context
            for context in runtime.agent_contexts.values()
            if context.role == "coordinator"
        )
    runtime.persistence_handler = state_store.save
    runtime.persist()
    run_conversation(
        create_coordinator_agent(create_model()),
        runtime,
        coordinator_context,
        initial_request,
    )


if __name__ == "__main__":
    main()
