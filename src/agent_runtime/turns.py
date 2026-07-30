"""按步骤执行 Pydantic AI 对话轮次。"""

from collections.abc import Callable, Sequence
from typing import TypeVar

from pydantic_ai import Agent, CallToolsNode
from pydantic_ai.messages import HandleResponseEvent, ModelMessage, ModelResponse


AgentDepsT = TypeVar("AgentDepsT")
AgentOutputT = TypeVar("AgentOutputT")
ModelResponseHandler = Callable[[ModelResponse], None]
ToolEventHandler = Callable[[HandleResponseEvent], None]


# 执行一轮非流式模型请求，并将响应和工具事件交给调用方处理。
async def run_turn(
    agent: Agent[AgentDepsT, AgentOutputT],
    user_prompt: str,
    history: Sequence[ModelMessage] | None = None,
    on_response: ModelResponseHandler | None = None,
    on_event: ToolEventHandler | None = None,
) -> list[ModelMessage]:
    async with agent.iter(user_prompt, message_history=history) as agent_run:
        async for node in agent_run:
            if not isinstance(node, CallToolsNode):
                continue

            if on_response is not None:
                on_response(node.model_response)

            async with node.stream(agent_run.ctx) as events:
                async for event in events:
                    if on_event is not None:
                        on_event(event)

    return agent_run.all_messages()
