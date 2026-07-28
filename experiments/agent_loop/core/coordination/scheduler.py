"""为任务分配提供非阻塞队列、依赖、重试和取消语义。"""

import asyncio
import heapq
from typing import Protocol

from ..context import (
    AgentCompletionEvent,
    AssignmentSession,
    ContextRuntime,
)


class AssignmentExecutor(Protocol):
    """执行一种任务会话并返回统一完成事件的可替换协议。

    调度器只依赖该协议，不要求任务必须由 PydanticAI Agent 完成；后续可以注入
    本地脚本、静态分析或人工审批执行器，而不修改优先级和依赖调度算法。
    """

    async def __call__(
        self, assignment: AssignmentSession, request: str
    ) -> AgentCompletionEvent:
        """执行一轮任务并返回其最终事件。"""
        ...


class AssignmentScheduler:
    """集中启动满足依赖的工作包，并限制 Runtime 级并发数。

    工具调用只负责入队。调度器在下一个事件循环周期按优先级选择任务，避免同一
    批次中先登记的低优先级任务抢占全部并发位。``background_task`` 是等待本轮
    最终结果的兼容句柄；实际模型调用句柄由本类私有持有。
    """

    def __init__(
        self,
        runtime: ContextRuntime,
        executor: AssignmentExecutor,
    ) -> None:
        self.runtime = runtime
        self.executor = executor
        self._pending: list[tuple[int, int, str]] = []
        self._sequence = 0
        self._active: dict[str, asyncio.Task[None]] = {}
        self._waiters: dict[str, asyncio.Future[None]] = {}
        self._dispatch_scheduled = False

    def enqueue(self, assignment: AssignmentSession, request: str) -> None:
        """登记一轮请求并立即返回，不等待依赖或并发位。"""
        normalized_request = request.strip()
        if not normalized_request:
            raise ValueError("任务请求不能为空")
        assignment.pending_request = normalized_request
        assignment.status = "queued"
        assignment.error = None
        loop = asyncio.get_running_loop()
        waiter = loop.create_future()
        self._waiters[assignment.assignment_id] = waiter
        assignment.background_task = asyncio.create_task(self._wait(waiter))
        self._push(assignment)
        self.runtime.persist()
        self._schedule_dispatch()

    def restore_queued(self) -> None:
        """把快照中尚未结束的任务重新放回当前事件循环。"""
        for assignment in self.runtime.assignments.values():
            if assignment.status == "queued" and assignment.pending_request:
                self.enqueue(assignment, assignment.pending_request)

    async def cancel(self, assignment_id: str, reason: str) -> bool:
        """取消排队或运行中的任务；已经终止的任务返回 ``False``。"""
        assignment = self.runtime.assignments.get(assignment_id)
        if assignment is None or assignment.status not in {"queued", "running"}:
            return False
        assignment.status = "cancelled"
        assignment.error = reason.strip() or "任务已由协调 Agent 取消"
        running = self._active.get(assignment_id)
        if running is not None:
            running.cancel()
        else:
            await self._finish(
                assignment,
                AgentCompletionEvent(
                    assignment_id=assignment.assignment_id,
                    coordinator_id=assignment.coordinator_id,
                    status="cancelled",
                    error=assignment.error,
                ),
            )
        self.runtime.persist()
        self._schedule_dispatch()
        return True

    @staticmethod
    async def _wait(waiter: asyncio.Future[None]) -> None:
        await waiter

    def _push(self, assignment: AssignmentSession) -> None:
        self._sequence += 1
        heapq.heappush(
            self._pending,
            (-assignment.priority, self._sequence, assignment.assignment_id),
        )

    def _schedule_dispatch(self) -> None:
        if self._dispatch_scheduled:
            return
        self._dispatch_scheduled = True
        asyncio.get_running_loop().call_soon(self._dispatch)

    def _dispatch(self) -> None:
        """按优先级启动当前可运行任务，并保留尚未满足的依赖任务。"""
        self._dispatch_scheduled = False
        deferred: list[tuple[int, int, str]] = []
        while (
            self._pending
            and len(self._active) < self.runtime.max_concurrent_assignments
        ):
            entry = heapq.heappop(self._pending)
            assignment = self.runtime.assignments[entry[2]]
            if assignment.status != "queued":
                continue
            dependency_sessions = {
                dependency: self.runtime.assignments.get(dependency)
                for dependency in assignment.depends_on
            }
            missing_dependencies = [
                dependency
                for dependency, session in dependency_sessions.items()
                if session is None
            ]
            failed_dependencies = [
                dependency
                for dependency, session in dependency_sessions.items()
                if session is not None
                and session.status in {"failed", "blocked", "cancelled"}
            ]
            if missing_dependencies or failed_dependencies:
                assignment.status = "blocked"
                reasons: list[str] = []
                if missing_dependencies:
                    reasons.append(
                        "依赖任务不存在：" + ", ".join(missing_dependencies)
                    )
                if failed_dependencies:
                    reasons.append(
                        "依赖任务未成功完成："
                        + ", ".join(failed_dependencies)
                    )
                assignment.error = "；".join(reasons)
                asyncio.create_task(
                    self._finish(
                        assignment,
                        AgentCompletionEvent(
                            assignment_id=assignment.assignment_id,
                            coordinator_id=assignment.coordinator_id,
                            status="blocked",
                            error=assignment.error,
                        ),
                    )
                )
                continue
            if any(
                session is not None and session.status != "completed"
                for session in dependency_sessions.values()
            ):
                deferred.append(entry)
                continue
            assignment.status = "running"
            assignment.attempts += 1
            self.runtime.persist()
            task = asyncio.create_task(self._run(assignment))
            self._active[assignment.assignment_id] = task
        for entry in deferred:
            heapq.heappush(self._pending, entry)

    async def _run(self, assignment: AssignmentSession) -> None:
        event: AgentCompletionEvent
        try:
            event = await self.executor(assignment, assignment.pending_request)
        except asyncio.CancelledError:
            event = AgentCompletionEvent(
                assignment_id=assignment.assignment_id,
                coordinator_id=assignment.coordinator_id,
                status="cancelled",
                error=assignment.error or "任务已取消",
            )
        finally:
            self._active.pop(assignment.assignment_id, None)

        if (
            assignment.status == "failed"
            and assignment.attempts < assignment.max_attempts
        ):
            assignment.status = "queued"
            self._push(assignment)
            self.runtime.persist()
        else:
            await self._finish(assignment, event)
        self._schedule_dispatch()

    async def _finish(
        self,
        assignment: AssignmentSession,
        event: AgentCompletionEvent,
    ) -> None:
        """完成等待句柄并把最终事件交给宿主；处理器失败不改写任务结果。"""
        if event not in self.runtime.pending_completion_events:
            self.runtime.pending_completion_events.append(event)
        self.runtime.persist()
        waiter = self._waiters.pop(assignment.assignment_id, None)
        if waiter is not None and not waiter.done():
            waiter.set_result(None)
        try:
            await self.runtime.emit_assignment_event(event)
        except Exception as error:
            assignment.error = (
                "协调 Agent 自动续跑失败："
                f"{type(error).__name__}: {error}"
            )
            self.runtime.persist()
