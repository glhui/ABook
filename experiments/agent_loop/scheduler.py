"""为任务分配提供非阻塞队列、依赖、重试和取消语义。"""

import asyncio
import heapq
from collections.abc import Awaitable, Callable

from .context import (
    AssignmentCompletionEvent,
    AssignmentSession,
    ContextRuntime,
)


AssignmentExecutor = Callable[
    [AssignmentSession, str], Awaitable[AssignmentCompletionEvent]
]


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
                AssignmentCompletionEvent(
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
            dependency_states = [
                self.runtime.assignments[dependency].status
                for dependency in assignment.depends_on
            ]
            failed_dependencies = [
                dependency
                for dependency in assignment.depends_on
                if self.runtime.assignments[dependency].status
                in {"failed", "blocked", "cancelled"}
            ]
            if failed_dependencies:
                assignment.status = "blocked"
                assignment.error = (
                    "依赖任务未成功完成：" + ", ".join(failed_dependencies)
                )
                asyncio.create_task(
                    self._finish(
                        assignment,
                        AssignmentCompletionEvent(
                            assignment_id=assignment.assignment_id,
                            coordinator_id=assignment.coordinator_id,
                            status="blocked",
                            error=assignment.error,
                        ),
                    )
                )
                continue
            if any(state != "completed" for state in dependency_states):
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
        event: AssignmentCompletionEvent
        try:
            event = await self.executor(assignment, assignment.pending_request)
        except asyncio.CancelledError:
            event = AssignmentCompletionEvent(
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
        event: AssignmentCompletionEvent,
    ) -> None:
        """完成等待句柄并把最终事件交给宿主；处理器失败不改写任务结果。"""
        if event not in self.runtime.pending_completion_events:
            self.runtime.pending_completion_events.append(event)
        self.runtime.persist()
        waiter = self._waiters.pop(assignment.assignment_id, None)
        if waiter is not None and not waiter.done():
            waiter.set_result(None)
        handler = self.runtime.assignment_completion_handler
        if handler is not None:
            try:
                await handler(event)
            except Exception as error:
                assignment.error = (
                    "协调 Agent 自动续跑失败："
                    f"{type(error).__name__}: {error}"
                )
                self.runtime.persist()
