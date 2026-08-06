"""终端日志渲染器测试。"""

from io import StringIO
import unittest

from terminal import InteractiveLogRenderer, LogEntry, PlainLogRenderer, create_log_renderer


class TerminalLoggingTests(unittest.TestCase):
    def test_plain_renderer_keeps_header_and_content(self: "TerminalLoggingTests") -> None:
        output = StringIO()
        PlainLogRenderer(output).emit(LogEntry("代码 Agent", "模型轮次 1", "详情", ""))

        self.assertIn("[代码 Agent][模型轮次 1]", output.getvalue())
        self.assertIn("详情", output.getvalue())

    def test_renderer_falls_back_when_interactive_dependencies_are_unavailable(
        self: "TerminalLoggingTests",
    ) -> None:
        renderer = create_log_renderer(interactive=False)

        self.assertIsInstance(renderer, PlainLogRenderer)

    def test_windows_enter_alias_expands_selected_entry(self: "TerminalLoggingTests") -> None:
        renderer = InteractiveLogRenderer()
        renderer._entries.append(LogEntry("scope", "event", "details", ""))

        renderer._handle_key("c-m")

        self.assertEqual(renderer._expanded, {0})
