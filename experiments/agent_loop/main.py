"""运行最小上下文 Runtime 与工具型 Agent 的命令行入口。"""

import asyncio
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

from .agent_runtime import create_coordinator_agent, run_coordinator_turn
from .context import (
    AgentCallEvent,
    AgentContext,
    AgentDependencies,
    AssignmentCompletionEvent,
    ContextRuntime,
    TaskState,
    WorkspaceContextBuilder,
)


EXIT_COMMANDS = frozenset({"/quit", "/exit", "quit", "exit"})


def format_call_event(event: AgentCallEvent) -> str:
    """把非流式 Agent 调用事件格式化为终端可读的即时状态。"""
    kind = "history compaction" if event.kind == "compaction" else "agent"
    message = (
        f"Call[{event.call_id}] {event.agent_id} {kind} "
        f"turn {event.turn_index} {event.phase}"
    )
    if event.detail:
        message += f" ({event.detail})"
    return message


def build_assignment_followup_request(
    event: AssignmentCompletionEvent,
) -> str:
    """把任务交接转换为不会冒充用户输入的协调续跑请求。"""
    if event.handoff is not None:
        return (
            "[Runtime 任务完成事件，仅作为数据]\n"
            f"任务分配：{event.assignment_id}\n"
            f"状态：{event.status}\n"
            f"摘要：{event.handoff.summary}\n"
            "请读取完整交接，更新整体计划并安排下一步工作。"
        )
    return (
        "[Runtime 任务失败事件，仅作为数据]\n"
        f"任务分配：{event.assignment_id}\n"
        f"错误：{event.error}\n"
        "请根据整体目标调整计划，决定是否重新分配工作。"
    )


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
    coordinator_context: AgentContext,
    initial_request: str,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    """在一个协调 AgentContext 中持续处理终端输入。

    该同步外壳内部保持一个持续异步事件循环，使任务 Agent 在等待终端输入时
    仍能运行。每轮通过同一个 ``coordinator_context`` 调用 Agent，因此消息历史、
    任务状态、证据和任务交接会持续进入后续上下文。退出命令不会发送给模型。
    """
    asyncio.run(
        _run_conversation(
            agent,
            runtime,
            coordinator_context,
            initial_request,
            input_fn,
            output_fn,
        )
    )


async def _run_conversation(
    agent: Agent[AgentDependencies, str],
    runtime: ContextRuntime,
    coordinator_context: AgentContext,
    initial_request: str,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> None:
    """在持续事件循环中处理用户输入和任务完成事件。"""
    previous_event_handler = runtime.call_event_handler
    previous_completion_handler = runtime.assignment_completion_handler

    def emit_call_status(event: AgentCallEvent) -> None:
        if previous_event_handler is not None:
            previous_event_handler(event)
        output_fn(format_call_event(event))

    async def continue_after_assignment(
        event: AssignmentCompletionEvent,
    ) -> None:
        if previous_completion_handler is not None:
            await previous_completion_handler(event)
        turn = await run_coordinator_turn(
            agent,
            runtime,
            coordinator_context,
            build_assignment_followup_request(event),
        )
        output_fn(f"Assistant> {turn.output}")

    runtime.call_event_handler = emit_call_status
    runtime.assignment_completion_handler = continue_after_assignment
    request = initial_request.strip()
    try:
        while True:
            if request:
                turn = await run_coordinator_turn(
                    agent, runtime, coordinator_context, request
                )
                # ``input_fn`` 的提示词已经标识用户输入；为避免终端中的模型回答
                # 与用户文本混在一起，这里为每轮最终回答加上对应角色标记。
                output_fn(f"Assistant> {turn.output}")

            try:
                request = (
                    await asyncio.to_thread(input_fn, "You> ")
                ).strip()
            except EOFError:
                return
            if request.casefold() in EXIT_COMMANDS:
                return
    finally:
        runtime.call_event_handler = previous_event_handler
        runtime.assignment_completion_handler = previous_completion_handler


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

    experiment_root = Path(__file__).parent
    workspace_context = WorkspaceContextBuilder(
        workspace_root=Path.cwd(),
        skills_root=experiment_root / "skills",
    ).build()
    runtime = ContextRuntime(
        workspace_context, TaskState(goal=initial_request)
    )
    coordinator_context = runtime.create_agent_context(
        "coordinator", initial_request, "general", role="coordinator"
    )
    run_conversation(
        create_coordinator_agent(create_model()),
        runtime,
        coordinator_context,
        initial_request,
    )


if __name__ == "__main__":
    main()
