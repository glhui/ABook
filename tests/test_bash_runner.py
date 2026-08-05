import os
from unittest.mock import patch
import unittest

from workspace_tools.bash import LocalBashRunner


# 验证 Windows PowerShell 后端兼容模型常见命令并按本地代码页解码诊断。
class LocalBashRunnerTests(unittest.TestCase):
    # Windows PowerShell 5.1 不支持 Bash 的 &&，执行前必须转换为可解析的命令分隔符。
    def test_windows_command_replaces_bash_and_operator(self: "LocalBashRunnerTests") -> None:
        runner = LocalBashRunner()

        with patch("workspace_tools.bash.os.name", "nt"):
            arguments = runner._command_arguments('cd "C:\\workspace" && python -m py_compile src\\module.py')

        self.assertEqual(arguments[:4], ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command"])
        self.assertEqual(arguments[4], 'cd "C:\\workspace" ; python -m py_compile src\\module.py')

    # 超时路径的字节输出也应使用 PowerShell 所在系统的代码页，避免中文错误乱码。
    @unittest.skipUnless(os.name == "nt", "仅 Windows PowerShell 使用本地代码页")
    def test_windows_timeout_output_uses_system_code_page(self: "LocalBashRunnerTests") -> None:
        runner = LocalBashRunner()
        source = "错误信息"

        with patch("workspace_tools.bash.os.name", "nt"):
            self.assertEqual(runner._as_text(source.encode("cp936")), source)
