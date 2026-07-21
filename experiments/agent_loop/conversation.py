"""管理协调会话的事件订阅、自动续跑和终端交互生命周期。"""

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from typing import Callable

from pydantic_ai import Agent

from .agent_runtime import initialize_assignment_scheduler, run_coordinator_turn
from .context import (
    AgentCallEvent,
    AgentContext,
    AgentDependencies,
    AssignmentCompletionEvent,
    ContextRuntime,
)
from .runner import AgentTurnResult


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
    events: AssignmentCompletionEvent | list[AssignmentCompletionEvent],
) -> str:
    """把一批任务事件转换为不会冒充用户输入的协调续跑请求。"""
    event_batch = (
        [events] if isinstance(events, AssignmentCompletionEvent) else events
    )
    summaries = []
    for event in event_batch:
        detail = (
            f"摘要：{event.handoff.summary}"
            if event.handoff is not None
            else f"错误：{event.error}"
        )
        summaries.append(
            f"- 任务分配：{event.assignment_id}\n"
            f"  状态：{event.status}\n  {detail}"
        )
    return (
        "[Runtime 任务完成事件批次，仅作为数据]\n"
        + "\n".join(summaries)
        + "\n请读取相关完整交接，统一更新整体计划并安排下一步工作。"
    )


@dataclass(frozen=True)
class CoordinatorResumePolicy:
    """限制完成事件触发协调 Agent 自动续跑时的重试次数和退避间隔。"""

    max_attempts: int = 3
    retry_delay_seconds: float = 0.05

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("协调续跑次数必须为正数")
        if self.retry_delay_seconds < 0:
            raise ValueError("协调续跑退避不能为负数")


class ConversationSession:
    """在持续事件循环中处理用户输入和任务完成事件。

    会话通过事件通道订阅 Runtime，而不是覆盖单一回调。关闭时只取消自己的
    订阅，因此日志、监控或同一 Runtime 的其他宿主处理器不会被覆盖或误删。
    """

    def __init__(
        self,
        agent: Agent[AgentDependencies, str],
        runtime: ContextRuntime,
        coordinator_context: AgentContext,
        input_fn: Callable[[str], str],
        output_fn: Callable[[str], None],
        resume_policy: CoordinatorResumePolicy | None = None,
    ) -> None:
        self.agent = agent
        self.runtime = runtime
        self.coordinator_context = coordinator_context
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.resume_policy = resume_policy or CoordinatorResumePolicy()
        self.completion_events: asyncio.Queue[AssignmentCompletionEvent] = (
            asyncio.Queue()
        )

    def emit_call_status(self, event: AgentCallEvent) -> None:
        """调用状态转发器，用于在终端中显示 Agent 调用的即时状态。

        该函数会在每次 Agent 调用事件发生时被触发。
        """
        self.output_fn(format_call_event(event))

    async def continue_after_assignment(
        self, event: AssignmentCompletionEvent
    ) -> None:
        """把属于当前协调 Agent 的任务完成事件转入会话私有队列。"""
        # 任务完成事件转发器，该函数会在每次任务完成事件发生时被触发。
        if event.coordinator_id == self.coordinator_context.agent_id:
            await self.completion_events.put(event)

    async def process_completion_events(self) -> None:
        """合并同一调度波次的完成事件，避免重复触发协调模型。

        这是长期运行的异步协程，作为后台任务启动：
        1. 持续监听事件；
        2. 如果在 10 毫秒内有多个完成事件，则合并为一个批次；
        3. 批量取出剩余事件；
        4. 触发协调模型，处理批次事件。
        """
        while True:
            first_event = await self.completion_events.get()
            await asyncio.sleep(0.01)
            batch = [first_event]
            while not self.completion_events.empty():
                batch.append(self.completion_events.get_nowait())

            turn = await self._resume_coordinator(batch)
            if turn is None:
                continue
            self.output_fn(f"Assistant> {turn.output}")
            # 处理完批次后，从待处理列表中移除已完成的事件，避免重复处理。
            for event in batch:
                if event in self.runtime.pending_completion_events:
                    self.runtime.pending_completion_events.remove(event)
            self.runtime.persist()

    async def _resume_coordinator(
        self, batch: list[AssignmentCompletionEvent]
    ) -> AgentTurnResult[str] | None:
        """按有界策略重试协调续跑；耗尽后保留事件供重启恢复。"""
        last_error: Exception | None = None
        for attempt in range(1, self.resume_policy.max_attempts + 1):
            try:
                return await run_coordinator_turn(
                    self.agent,
                    self.runtime,
                    self.coordinator_context,
                    build_assignment_followup_request(batch),
                )
            except Exception as error:
                last_error = error
                if attempt < self.resume_policy.max_attempts:
                    await asyncio.sleep(
                        self.resume_policy.retry_delay_seconds * attempt
                    )

        assert last_error is not None
        for event in batch:
            assignment = self.runtime.assignments.get(event.assignment_id)
            if assignment is not None:
                assignment.error = (
                    "协调 Agent 自动续跑失败："
                    f"{type(last_error).__name__}: {last_error}；"
                    f"已重试 {self.resume_policy.max_attempts} 次"
                )
        self.runtime.persist()
        return None

    async def run(self, initial_request: str) -> None:
        """启动订阅和调度器恢复，然后持续处理用户请求直到退出。"""
        unsubscribe_calls = self.runtime.call_events.subscribe(
            self.emit_call_status
        )
        unsubscribe_assignments = self.runtime.assignment_events.subscribe(
            self.continue_after_assignment
        )
        completion_processor = asyncio.create_task(
            self.process_completion_events()
        )
        initialize_assignment_scheduler(self.runtime, self.agent.model)
        for pending_event in list(self.runtime.pending_completion_events):
            await self.continue_after_assignment(pending_event)
        request = initial_request.strip()
        try:
            while True:
                if request:
                    turn = await run_coordinator_turn(
                        self.agent,
                        self.runtime,
                        self.coordinator_context,
                        request,
                    )
                    # ``input_fn`` 的提示词已经标识用户输入；为避免终端中的模型回答
                    # 与用户文本混在一起，这里为每轮最终回答加上对应角色标记。
                    self.output_fn(f"Assistant> {turn.output}")

                try:
                    request = (
                        await asyncio.to_thread(self.input_fn, "You> ")
                    ).strip()
                except EOFError:
                    return
                if request.casefold() in EXIT_COMMANDS:
                    return
        finally:
            completion_processor.cancel()
            with suppress(asyncio.CancelledError):
                await completion_processor
            unsubscribe_assignments()
            unsubscribe_calls()


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
    session = ConversationSession(
        agent, runtime, coordinator_context, input_fn, output_fn
    )
    asyncio.run(session.run(initial_request))
