"""Agent Profile 配置示例的离线测试。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from examples.agent_profile_configuration import create_configured_agent


# 示例应在不访问网络或真实模型凭据的情况下展示配置覆盖结果。
class AgentProfileConfigurationExampleTests(unittest.TestCase):
    def test_create_configured_agent_reports_role_and_overrides(self: "AgentProfileConfigurationExampleTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            instruction_file, max_tokens, temperature = create_configured_agent(Path(temporary_directory))

        self.assertEqual(instruction_file, "python-code-agent.md")
        self.assertEqual(max_tokens, 512)
        self.assertEqual(temperature, 0.2)
