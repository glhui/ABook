"""统一执行协调 Agent 与任务 Agent 的模型调用和上下文生命周期。"""

from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, AgentRunResult, ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage, UsageLimits

from .context import (
    AgentCallEvent,
    AgentContext,
    AgentDependencies,
    ContextRuntime,
    FactClaim,
)


OutputT = TypeVar("OutputT")



# 事实性压缩而非简单摘要，保留可验证的 evidence_id 与 quote，避免模型伪造历史
class CompactionCheckpoint(BaseModel):
    """在替换旧消息前交由 Runtime 校验和保存的结构化交接单。

    当某个 Agent 的历史达到上下文阈值时，Runtime 不会直接删除较早消息，而是先让
    模型从这些消息生成本对象。Runtime 随后校验 ``facts`` 中每个 quote 是否真实
    出现在对应 ``evidence_id`` 的工具结果中；校验通过后，事实和未决事项会写入
    ``TaskState``，最后才以 ``summary`` 替换旧历史。校验失败会要求模型重试，旧
    历史保持不变。

    例如，Agent 曾读取 ``config.py`` 并获得 ``evidence-8``，工具结果包含
    ``ABOOK_MODEL``。压缩时可记录“模型名称来自 ABOOK_MODEL”这一事实，并引用
    ``evidence-8`` 和原文 ``ABOOK_MODEL``；如果尚未确认默认模型是否可用，则把
    该问题放入 ``unresolved_issues``。文件全文和重复对话只进入 ``summary`` 或被
    丢弃，不应伪造为已验证事实。
    """

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(
        min_length=1,
        max_length=8_000,
        description="保留任务进度、关键决定和后续所需背景的简洁历史摘要",
    )
    facts: list[FactClaim] = Field(
        default_factory=list,
        max_length=50,
        description="必须带 evidence_id 与逐字 quote 的可验证事实",
    )
    unresolved_issues: list[str] = Field(
        default_factory=list,
        max_length=50,
        description="压缩后仍需后续 Agent 处理、验证或向用户澄清的问题",
    )


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
    """一次成功完成的 Agent 调用及其 Runtime 元数据。

    该对象是 ``AgentRunner.run_turn`` 的统一返回值；它在模型成功结束后才创建，
    此时 ``messages`` 已写回对应 ``AgentContext``。``raw_result`` 保留 PydanticAI
    原始结果，供现有 CLI 或需要访问完整 usage 的调用方继续使用。

    Attributes:
        output: 模型的最终输出。协调 Agent 通常是 ``str``，任务 Agent 通常是
            结构化 ``TaskReport``。
        messages: 本轮结束后的完整私有消息历史，包含此前历史、本轮 user message、
            模型响应和工具调用结果；不会包含其他 Agent 的私有历史。
        requests: 本轮消耗的全部模型请求数，包含主 Agent 调用和压缩调用。
        tool_calls: 本轮执行的全部工具调用数，包含主 Agent 和压缩调用产生的计数。
        input_tokens: 全部模型请求的输入 token 总数；包含压缩前发送给摘要模型的
            旧历史。
        output_tokens: 全部模型请求的输出 token 总数；包含主输出和压缩摘要。
        compacted: 本轮主 Agent 调用前是否发生过历史压缩。
        compaction_requests: ``requests`` 中属于压缩检查点的模型请求数；未压缩时为
            ``0``，通常为 ``1``，但模型需要修正无效证据引用时可能大于 ``1``。
        turn_index: 当前 Agent 私有会话的成功轮次，从 ``1`` 开始；协调 Agent 和
            某个 worker 的第 4 次请求各自都可为 ``4``。
        raw_result: PydanticAI 返回的 ``AgentRunResult``，用于访问框架原始数据。

    例如，worker 已完成 3 轮，第四轮开始前压缩旧历史。压缩模型请求 1 次，主
    Agent 因工具重试请求 2 次，则 ``turn_index=4``、``compacted=True``、
    ``compaction_requests=1``、``requests=3``。若该轮调用了读取文件和运行测试，
    ``tool_calls`` 为相应工具调用总数；``input_tokens`` 和 ``output_tokens`` 则是
    这 3 次模型请求的合计，而不是仅主 Agent 最后一次请求的用量。
    """

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
        """串行化同一 AgentContext 的调用，避免后台续跑破坏消息历史。"""
        async with agent_context.run_lock:
            return await self._run_turn_locked(
                agent, runtime, agent_context, request
            )

    async def _run_turn_locked(
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

        call_id = runtime.next_call_id()
        turn_index = agent_context.turn_count + 1
        runtime.emit_call_event(
            AgentCallEvent(
                call_id=call_id,
                agent_id=agent_context.agent_id,
                kind="agent",
                phase="started",
                turn_index=turn_index,
            )
        )
        try:
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
        except Exception as error:
            runtime.emit_call_event(
                AgentCallEvent(
                    call_id=call_id,
                    agent_id=agent_context.agent_id,
                    kind="agent",
                    phase="failed",
                    turn_index=turn_index,
                    detail=f"{type(error).__name__}: {error}",
                )
            )
            raise
        messages = result.all_messages()
        agent_context.message_history = messages
        agent_context.turn_count += 1
        usage = result.usage
        turn_result = AgentTurnResult(
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
        runtime.emit_call_event(
            AgentCallEvent(
                call_id=call_id,
                agent_id=agent_context.agent_id,
                kind="agent",
                phase="completed",
                turn_index=turn_index,
                detail=(
                    f"requests={turn_result.requests}; "
                    f"tool_calls={turn_result.tool_calls}"
                ),
            )
        )
        return turn_result

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
        call_id = runtime.next_call_id()
        runtime.emit_call_event(
            AgentCallEvent(
                call_id=call_id,
                agent_id=agent_context.agent_id,
                kind="compaction",
                phase="started",
                turn_index=agent_context.turn_count + 1,
            )
        )
        try:
            result = await checkpoint_agent.run(
                "压缩以上历史，以便原 Agent 继续当前任务。",
                deps=AgentDependencies(runtime, agent_context),
                message_history=compacted_messages,
                usage_limits=UsageLimits(
                    request_limit=self.limits.compaction_request_limit,
                    tool_calls_limit=1,
                ),
            )
        except Exception as error:
            runtime.emit_call_event(
                AgentCallEvent(
                    call_id=call_id,
                    agent_id=agent_context.agent_id,
                    kind="compaction",
                    phase="failed",
                    turn_index=agent_context.turn_count + 1,
                    detail=f"{type(error).__name__}: {error}",
                )
            )
            raise
        _apply_checkpoint(
            runtime, agent_context, result.output, retained_messages
        )
        runtime.emit_call_event(
            AgentCallEvent(
                call_id=call_id,
                agent_id=agent_context.agent_id,
                kind="compaction",
                phase="completed",
                turn_index=agent_context.turn_count + 1,
                detail=f"requests={result.usage.requests}",
            )
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
