"""提供不绑定 CLI 或调度器的 Runtime 进程内事件通道。"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar


EventT = TypeVar("EventT")
Unsubscribe = Callable[[], None]


@dataclass
class SyncEventChannel(Generic[EventT]):
    """按注册顺序同步广播事件，并允许多个宿主独立订阅。

    订阅返回的取消函数只移除本次注册，因而 CLI、测试和监控代码不需要通过
    保存并覆盖 Runtime 单一回调来共享事件。处理器异常会继续向发布者传播，
    由最了解调用边界的上层决定是否降级或记录错误。
    """

    _handlers: list[Callable[[EventT], None]] = field(
        default_factory=list, repr=False
    )

    def subscribe(self, handler: Callable[[EventT], None]) -> Unsubscribe:
        """注册同步处理器并返回幂等的取消订阅函数。"""
        self._handlers.append(handler)

        def unsubscribe() -> None:
            if handler in self._handlers:
                self._handlers.remove(handler)

        return unsubscribe

    def publish(self, event: EventT) -> None:
        """向当前订阅者快照广播事件，允许处理器在回调中取消订阅。

        单个订阅者失败不会阻止其他订阅者收到事件；广播结束后重新抛出首个异常，
        让发布边界仍能记录失败。
        """
        first_error: Exception | None = None
        for handler in tuple(self._handlers):
            try:
                handler(event)
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


@dataclass
class AsyncEventChannel(Generic[EventT]):
    """按注册顺序异步广播事件，并支持多个完成事件消费者。"""

    _handlers: list[Callable[[EventT], Awaitable[None]]] = field(
        default_factory=list, repr=False
    )

    def subscribe(
        self, handler: Callable[[EventT], Awaitable[None]]
    ) -> Unsubscribe:
        """注册异步处理器并返回幂等的取消订阅函数。"""
        self._handlers.append(handler)

        def unsubscribe() -> None:
            if handler in self._handlers:
                self._handlers.remove(handler)

        return unsubscribe

    async def publish(self, event: EventT) -> None:
        """依次等待所有订阅者，保证同一事件的处理顺序可预测。

        与同步通道一致，一个失败订阅者不会饿死后续订阅者；首个异常在广播完成后
        交回发布者处理。
        """
        first_error: Exception | None = None
        for handler in tuple(self._handlers):
            try:
                await handler(event)
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
