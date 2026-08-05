"""预定义 Agent 角色的测试。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pydantic_ai.models.test import TestModel

from agent_profiles import (
    create_code_test_task_coordinator,
    create_python_code_agent,
    create_python_code_context,
    create_python_test_agent,
    create_python_test_context,
    create_python_validator_agent,
    create_python_validator_context,
)
from agent_profiles.role_instructions import load_role_instructions
from tool_execution import (
    InMemoryToolAuditLog,
    ToolApproval,
    ToolCapability,
    ToolExecutionContext,
    WorkspaceExecutionPolicy,
    WorkspaceToolExecutor,
)
from workspace_tools import create_workspace_file_tools


# 验证 Python 代码角色固定工具范围，同时允许调用方为每项任务单独创建上下文。
class PythonCodeAgentProfileTests(unittest.TestCase):
    def test_create_context_uses_fixed_identity_capabilities_and_explicit_approval(
        self: "PythonCodeAgentProfileTests",
    ) -> None:
        context = create_python_code_context(
            "add-email-validation",
            frozenset({ToolApproval.OVERWRITE_FILE}),
        )

        self.assertEqual(context.agent_id, "python-code")
        self.assertEqual(context.task_id, "add-email-validation")
        self.assertEqual(
            context.capabilities,
            frozenset(
                {
                    ToolCapability.FILE_READ,
                    ToolCapability.FILE_WRITE,
                    ToolCapability.FILE_EDIT,
                    ToolCapability.BASH_EXECUTE,
                }
            ),
        )
        self.assertEqual(context.approvals, frozenset({ToolApproval.OVERWRITE_FILE}))

    def test_create_agent_registers_only_python_code_tools(self: "PythonCodeAgentProfileTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            executor = self._create_executor(Path(temporary_directory))

            agent = create_python_code_agent(TestModel(), executor, create_python_code_context("task-1"))

            self.assertEqual(set(agent._function_toolset.tools), {"read_file", "write_file", "replace_text"})
            self.assertEqual(agent.model_settings.get("max_tokens"), 8_192)
            self.assertTrue(any("Python 代码 Agent 工作说明" in instruction for instruction in agent._instructions))

    # 代码 Agent 应把路由器选中的 Skill 放在固定角色说明之后。
    def test_create_code_agent_appends_selected_skill_instructions(self: "PythonCodeAgentProfileTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            executor = self._create_executor(Path(temporary_directory))

            agent = create_python_code_agent(
                TestModel(),
                executor,
                create_python_code_context("task-with-skill"),
                skill_instructions="## Skill: example\n\n只用于当前任务。",
            )

            self.assertTrue(any("Skill: example" in instruction for instruction in agent._instructions))

    def test_create_agent_rejects_identity_outside_fixed_profile(self: "PythonCodeAgentProfileTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            executor = self._create_executor(Path(temporary_directory))
            context = ToolExecutionContext(
                agent_id="another-agent",
                task_id="task-1",
                capabilities=frozenset({ToolCapability.FILE_READ}),
            )

            with self.assertRaisesRegex(ValueError, "agent_id"):
                create_python_code_agent(TestModel(), executor, context)

    def test_python_test_agent_registers_workspace_tools(self: "PythonCodeAgentProfileTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            executor = self._create_executor(Path(temporary_directory))

            agent = create_python_test_agent(TestModel(), executor, create_python_test_context("task-1"))

            self.assertEqual(set(agent._function_toolset.tools), {"read_file", "write_file", "replace_text"})
            self.assertTrue(any("Python 测试 Agent 工作说明" in instruction for instruction in agent._instructions))

    def test_validator_agent_cannot_read_implementation_or_tests(self: "PythonCodeAgentProfileTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            executor = self._create_executor(Path(temporary_directory))

            context = create_python_validator_context("black-box-validation")
            agent = create_python_validator_agent(TestModel(), executor, context)

            self.assertEqual(context.agent_id, "python-validator")
            self.assertEqual(
                context.capabilities,
                frozenset({ToolCapability.FILE_WRITE, ToolCapability.BASH_EXECUTE}),
            )
            self.assertEqual(set(agent._function_toolset.tools), {"write_file"})

    def test_task_coordinator_does_not_register_workspace_tools(self: "PythonCodeAgentProfileTests") -> None:
        agent = create_code_test_task_coordinator(TestModel())

        self.assertEqual(set(agent._function_toolset.tools), set())

    def test_each_implementation_role_has_distinct_versioned_instructions(self: "PythonCodeAgentProfileTests") -> None:
        code_instructions = load_role_instructions("python_code")
        test_instructions = load_role_instructions("python_test")

        self.assertIn("生产代码", code_instructions)
        self.assertIn("自动化测试", test_instructions)
        self.assertNotEqual(code_instructions, test_instructions)

    # 创建角色工厂测试所需的受控工作区执行器。
    def _create_executor(self: "PythonCodeAgentProfileTests", workspace_root: Path) -> WorkspaceToolExecutor:
        return WorkspaceToolExecutor(
            WorkspaceExecutionPolicy(workspace_root),
            create_workspace_file_tools(),
            InMemoryToolAuditLog(),
        )
