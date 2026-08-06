"""提供可复用的普通与交互式终端日志渲染。"""

from dataclasses import dataclass
import os
import shutil
import sys
from threading import Event, Lock, Thread
from typing import Protocol, TextIO


@dataclass(frozen=True)
class LogEntry:
    """保存一条可折叠的终端日志。"""

    scope: str
    event: str
    content: str
    color: str


class LogRenderer(Protocol):
    """定义日志输出后端的最小接口。"""

    def emit(self: "LogRenderer", entry: LogEntry) -> None:
        ...


class PlainLogRenderer:
    """以兼容重定向和 CI 的方式逐条输出日志。"""

    def __init__(self: "PlainLogRenderer", stream: TextIO | None = None) -> None:
        self._stream = sys.stdout if stream is None else stream

    def emit(self: "PlainLogRenderer", entry: LogEntry) -> None:
        print(f"== [{entry.scope}][{entry.event}] ==\n{entry.content}", file=self._stream, flush=True)


class InteractiveLogRenderer:
    """使用 Rich 重绘终端，并允许用方向键和回车展开日志正文。"""

    def __init__(self: "InteractiveLogRenderer", max_entries: int = 200) -> None:
        self._entries: list[LogEntry] = []
        self._expanded: set[int] = set()
        self._selected = 0
        self._scroll_offset = 0
        self._max_entries = max_entries
        self._lock = Lock()
        self._stop = Event()
        self._thread: Thread | None = None
        self._live = None
        self._input = None

    def emit(self: "InteractiveLogRenderer", entry: LogEntry) -> None:
        self._ensure_started()
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max_entries:
                self._entries.pop(0)
                self._expanded = {index - 1 for index in self._expanded if index > 0}
            self._selected = min(self._selected, max(len(self._entries) - 1, 0))
            self._ensure_selected_visible()
            self._refresh()

    def close(self: "InteractiveLogRenderer") -> None:
        self._pause()

    def pause(self: "InteractiveLogRenderer") -> None:
        """暂时释放键盘和终端控制权，供普通 input() 使用。"""
        self._pause()

    def resume(self: "InteractiveLogRenderer") -> None:
        """恢复日志重绘和键盘导航。"""
        if self._live is None:
            self._stop.clear()
            self._start()

    def _pause(self: "InteractiveLogRenderer") -> None:
        self._stop.set()
        if self._input is not None:
            self._input.close()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self._live is not None:
            self._live.stop()
        self._input = None
        self._live = None
        self._thread = None

    def _ensure_started(self: "InteractiveLogRenderer") -> None:
        if self._live is None:
            self._start()

    def _refresh(self: "InteractiveLogRenderer") -> None:
        if self._live is None:
            return
        self._live.update(self._renderable(), refresh=True)

    def _renderable(self: "InteractiveLogRenderer"):
        from rich.console import Group
        from rich.markup import escape
        from rich.text import Text

        lines = []
        for index, entry in enumerate(self._entries):
            # ASCII markers keep legacy Windows code pages usable.
            marker = "v" if index in self._expanded else ">"
            prefix = "* " if index == self._selected else "  "
            lines.append(Text(f"{prefix}{marker} [{entry.scope}][{entry.event}]", style=f"bold {self._style(entry.color)}"))
            if index in self._expanded:
                lines.extend(
                    Text(escape(line), style=self._style(entry.color))
                    for line in entry.content.splitlines()
                )
        height = max(shutil.get_terminal_size((120, 30)).lines - 1, 5)
        visible_lines = lines[self._scroll_offset : self._scroll_offset + height]
        return Group(*visible_lines)

    # 让当前选中标题始终位于可见日志视口中。
    def _ensure_selected_visible(self: "InteractiveLogRenderer") -> None:
        lines_before = 0
        for index, entry in enumerate(self._entries):
            if index == self._selected:
                break
            lines_before += 1 + (len(entry.content.splitlines()) if index in self._expanded else 0)
        height = max(shutil.get_terminal_size((120, 30)).lines - 1, 5)
        if lines_before < self._scroll_offset:
            self._scroll_offset = lines_before
        elif lines_before >= self._scroll_offset + height:
            self._scroll_offset = lines_before - height + 1

    def _style(self: "InteractiveLogRenderer", color: str) -> str:
        return {
            "\033[34m": "blue",
            "\033[36m": "cyan",
            "\033[32m": "green",
            "\033[35m": "magenta",
            "\033[31m": "red",
            "\033[33m": "yellow",
        }.get(color, "white")

    def _start(self: "InteractiveLogRenderer") -> None:
        from prompt_toolkit.input import create_input
        from rich.live import Live

        self._input = create_input()
        self._live = Live(self._renderable(), refresh_per_second=10, screen=False)
        self._live.start(refresh=True)
        self._thread = Thread(target=self._read_keys, daemon=True)
        self._thread.start()

    def _read_keys(self: "InteractiveLogRenderer") -> None:
        assert self._input is not None
        with self._input.raw_mode():
            while not self._stop.is_set():
                for key in self._input.read_keys():
                    with self._lock:
                        self._handle_key(key.key)
                        self._refresh()

    # 统一处理不同终端和 Windows 控制台产生的按键名称。
    def _handle_key(self: "InteractiveLogRenderer", key: str) -> None:
        if key in {"up", "k"}:
            self._selected = max(0, self._selected - 1)
        elif key in {"down", "j"}:
            self._selected = min(len(self._entries) - 1, self._selected + 1)
        elif key in {"pageup", "page-up", "scroll-up"}:
            self._scroll_offset = max(0, self._scroll_offset - 5)
        elif key in {"pagedown", "page-down", "scroll-down"}:
            self._scroll_offset = min(self._max_scroll_offset(), self._scroll_offset + 5)
        elif key in {"enter", "c-m", "c-j", "space", " "} and self._entries:
            if self._selected in self._expanded:
                self._expanded.remove(self._selected)
            else:
                self._expanded.add(self._selected)
            self._ensure_selected_visible()

    # 根据当前展开内容计算日志视口允许的最大偏移。
    def _max_scroll_offset(self: "InteractiveLogRenderer") -> int:
        total_lines = sum(
            1 + (len(entry.content.splitlines()) if index in self._expanded else 0)
            for index, entry in enumerate(self._entries)
        )
        height = max(shutil.get_terminal_size((120, 30)).lines - 1, 5)
        return max(0, total_lines - height)


def create_log_renderer(interactive: bool | None = None) -> LogRenderer:
    """按终端能力选择交互式渲染，非 TTY 自动回退为普通输出。"""

    enabled = interactive if interactive is not None else os.getenv("ABOOK_INTERACTIVE_LOGS", "1") != "0"
    if enabled and sys.stdout.isatty():
        try:
            import prompt_toolkit  # noqa: F401
            import rich  # noqa: F401

            # 延迟启动键盘监听，避免模块导入阶段抢占应用的 input()。
            return InteractiveLogRenderer()
        except (ImportError, OSError):
            pass
    return PlainLogRenderer()
