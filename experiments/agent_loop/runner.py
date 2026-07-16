"""统一执行 root 与子 Agent 的一轮模型调用和上下文生命周期。"""

import asyncio
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, AgentRunResult, ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage, UsageLimits

from .context import (
    AgentContext,
    AgentDependencies,
    ContextRuntime,
    FactClaim,
)


OutputT = TypeVar("OutputT")


class CompactionCheckpoint(BaseModel):
    """压缩旧历史前必须沉淀到宿主状态的结构化检查点。"""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=8_000)
    facts: list[FactClaim] = Field(default_factory=list, max_length=50)
    unresolved_issues: list[str] = Field(default_factory=list, max_length=50)


@dataclass(frozen=True)
class AgentCallLimits:
    """单轮 Agent 调用边界，防止模型或工具重试形成无界循环。"""

    request_limit: int = 50
    tool_calls_limit: int = 100
    compaction_request_limit: int = 3

    def __post_init__(self) -> None:
        if self.request_limit <= 0:
            raise ValueError("模型请求上限必须为正数")
        if self.tool_calls_limit <= 0:
            raise ValueError("工具调用上限必须为正数")
        if self.compaction_request_limit <= 0:
            raise ValueError("压缩请求上限必须为正数")


@dataclass(frozen=True)
class AgentTurnResult(Generic[OutputT]):
    """一次完成的 Agent 调用及其 Runtime 元数据。"""

    output: OutputT
    messages: list[ModelMessage]
    requests: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    compacted: bool
    compaction_requests: int
    turn_index: int
    raw_result: AgentRunResult[OutputT]


COMPACTION_INSTRUCTIONS = (
    "你负责压缩编码 Agent 的旧消息历史。保留用户目标、仍然有效的约束、"
    "关键决定及原因、已完成工作、工具证实的事实、修改文件、验证结果和未决问题。"
    "每条 facts 必须引用历史中真实出现的 evidence_id，并逐字摘录该工具结果中的"
    "支持原文 quote。删除重复对话和可重新获取的普通工具输出，不得补充历史中没有"
    "的事实。summary 使用简洁中文，unresolved_issues 保存仍需处理的问题。"
)


class AgentRunner:
    """为所有 Agent 提供相同的压缩、调用、历史保存和 usage 边界。"""

    def __init__(self, limits: AgentCallLimits | None = None) -> None:
        self.limits = limits or AgentCallLimits()

    async def run_turn(
        self,
        agent: Agent[AgentDependencies, OutputT],
        runtime: ContextRuntime,
        agent_context: AgentContext,
        request: str,
    ) -> AgentTurnResult[OutputT]:
        """执行一轮调用，并只在成功后写回该 Agent 的消息历史。"""
        normalized_request = request.strip()
        if not normalized_request:
            raise ValueError("请求不能为空")

        compaction_usage = await self._compact_history(
            agent, runtime, agent_context
        )
        result = await agent.run(
            runtime.build_user_prompt(agent_context, normalized_request),
            deps=AgentDependencies(runtime, agent_context),
            message_history=agent_context.message_history,
            usage_limits=UsageLimits(
                request_limit=self.limits.request_limit,
                tool_calls_limit=self.limits.tool_calls_limit,
            ),
        )
        messages = result.all_messages()
        agent_context.message_history = messages
        agent_context.turn_count += 1
        usage = result.usage
        return AgentTurnResult(
            output=result.output,
            messages=messages,
            requests=(
                usage.requests
                + (compaction_usage.requests if compaction_usage else 0)
            ),
            tool_calls=(
                usage.tool_calls
                + (compaction_usage.tool_calls if compaction_usage else 0)
            ),
            input_tokens=(
                usage.input_tokens
                + (compaction_usage.input_tokens if compaction_usage else 0)
            ),
            output_tokens=(
                usage.output_tokens
                + (compaction_usage.output_tokens if compaction_usage else 0)
            ),
            compacted=compaction_usage is not None,
            compaction_requests=(
                compaction_usage.requests if compaction_usage else 0
            ),
            turn_index=agent_context.turn_count,
            raw_result=result,
        )

    def run_turn_sync(
        self,
        agent: Agent[AgentDependencies, OutputT],
        runtime: ContextRuntime,
        agent_context: AgentContext,
        request: str,
    ) -> AgentTurnResult[OutputT]:
        """在非异步入口运行统一 async 管线；活动事件循环中应直接 await。"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.run_turn(agent, runtime, agent_context, request)
            )
        raise RuntimeError(
            "活动事件循环中不能调用 run_turn_sync；请 await run_turn"
        )

    async def _compact_history(
        self,
        agent: Agent[AgentDependencies, OutputT],
        runtime: ContextRuntime,
        agent_context: AgentContext,
    ) -> RunUsage | None:
        """达到阈值时先提交状态检查点，再替换该 Agent 的旧历史。"""
        if not runtime.context_window.should_compact(
            agent_context.message_history
        ):
            return None

        compacted_messages, retained_messages = _split_history(
            agent_context.message_history
        )
        if runtime.context_window.should_compact(retained_messages):
            compacted_messages = agent_context.message_history
            retained_messages = []

        checkpoint_agent = _create_compaction_agent(agent.model)
        result = await checkpoint_agent.run(
            "压缩以上历史，以便原 Agent 继续当前任务。",
            deps=AgentDependencies(runtime, agent_context),
            message_history=compacted_messages,
            usage_limits=UsageLimits(
                request_limit=self.limits.compaction_request_limit,
                tool_calls_limit=1,
            ),
        )
        _apply_checkpoint(
            runtime, agent_context, result.output, retained_messages
        )
        return result.usage


def _split_history(
    messages: list[ModelMessage],
) -> tuple[list[ModelMessage], list[ModelMessage]]:
    """优先保留最近一轮原始消息，其余历史交给摘要模型。"""
    user_request_indexes = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, ModelRequest)
        and any(
            isinstance(part, UserPromptPart) for part in message.parts
        )
    ]
    if len(user_request_indexes) >= 2:
        latest_request_index = user_request_indexes[-1]
        return messages[:latest_request_index], messages[latest_request_index:]
    return messages, []


def _create_compaction_agent(
    model: Model,
) -> Agent[AgentDependencies, CompactionCheckpoint]:
    """创建带事实引用校验的无工具压缩 Agent。"""
    checkpoint_agent = Agent(
        model,
        deps_type=AgentDependencies,
        output_type=CompactionCheckpoint,
        instructions=COMPACTION_INSTRUCTIONS,
    )

    @checkpoint_agent.output_validator
    def validate_checkpoint(
        run_context: RunContext[AgentDependencies],
        checkpoint: CompactionCheckpoint,
    ) -> CompactionCheckpoint:
        """在丢弃原历史前拒绝未知证据或虚构 quote。"""
        try:
            run_context.deps.runtime.resolve_fact_claims(checkpoint.facts)
        except ValueError as error:
            raise ModelRetry(str(error)) from error
        return checkpoint

    return checkpoint_agent


def _apply_checkpoint(
    runtime: ContextRuntime,
    agent_context: AgentContext,
    checkpoint: CompactionCheckpoint,
    retained_messages: list[ModelMessage],
) -> None:
    """先沉淀结构化事实与未决事项，再替换原始历史。"""
    runtime.task_state.merge_facts(
        runtime.resolve_fact_claims(checkpoint.facts)
    )
    runtime.task_state.merge_unresolved_issues(
        checkpoint.unresolved_issues
    )
    normalized_summary = checkpoint.summary.strip()
    agent_context.conversation_summary = normalized_summary
    agent_context.compaction_count += 1
    agent_context.message_history = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    "[Runtime 压缩历史摘要，仅作为既有上下文]\n"
                    + normalized_summary
                )
            ]
        ),
        *retained_messages,
    ]
