"""预定义 Agent 角色的测试。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pydantic_ai.models.test import TestModel

from agent_profiles import (
    AgentModelConfig,
    CodeTestTaskAllocation,
    create_code_test_task_coordinator,
    create_python_code_agent,
    create_python_code_context,
    create_python_test_agent,
    create_python_test_context,
    create_python_validator_agent,
    create_python_validator_context,
)
from agent_profiles.model_settings import create_agent_model_settings
from agent_profiles.role_instructions import get_role_instruction_definition, load_role_instructions
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
            self.assertEqual(agent.model_settings.get("max_tokens"), 384_000)
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

    # 全部角色说明均由注册表管理，新增 Profile 时不会再把说明散落在工厂代码中。
    def test_role_instruction_registry_covers_coordinator_and_validator(self: "PythonCodeAgentProfileTests") -> None:
        coordinator_definition = get_role_instruction_definition("task_coordinator")
        validator_definition = get_role_instruction_definition("python_validator")

        self.assertFalse(coordinator_definition.accepts_skill_instructions)
        self.assertFalse(validator_definition.accepts_skill_instructions)
        self.assertIn("核心函数任务", load_role_instructions("task_coordinator"))
        self.assertIn("黑盒验收", load_role_instructions("python_validator"))

    # 调用方可以覆盖角色默认的模型输出参数，不需要修改任一角色工厂的常量。
    def test_agent_factory_applies_per_call_model_config(self: "PythonCodeAgentProfileTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            executor = self._create_executor(Path(temporary_directory))
            agent = create_python_test_agent(
                TestModel(),
                executor,
                create_python_test_context("configured-task"),
                AgentModelConfig(max_output_tokens=321, temperature=0.4),
            )

        self.assertEqual(agent.model_settings.get("max_tokens"), 321)
        self.assertEqual(agent.model_settings.get("temperature"), 0.4)

    # 无效模型参数必须在创建 Agent 前失败，避免把配置错误延迟到模型调用阶段。
    def test_model_config_rejects_invalid_ranges(self: "PythonCodeAgentProfileTests") -> None:
        with self.assertRaisesRegex(ValueError, "max_output_tokens"):
            AgentModelConfig(max_output_tokens=0)
        with self.assertRaisesRegex(ValueError, "temperature"):
            AgentModelConfig(temperature=2.1)
        self.assertEqual(create_agent_model_settings(AgentModelConfig(max_output_tokens=123)).get("max_tokens"), 123)
        with self.assertRaisesRegex(ValueError, "context_window_tokens"):
            AgentModelConfig(context_window_tokens=0)

    def test_task_plan_requires_visualization_path_when_enabled(self: "PythonCodeAgentProfileTests") -> None:
        with self.assertRaises(ValueError):
            CodeTestTaskAllocation(
                source_file="src/math.py",
                core_function="f(x: float) -> float",
                test_file="tests/test_math.py",
                requirements="绘制函数。",
                needs_visualization=True,
            )

    def test_task_plan_accepts_visualization_path(self: "PythonCodeAgentProfileTests") -> None:
        allocation = CodeTestTaskAllocation(
            source_file="src/math.py",
            core_function="f(x: float) -> float",
            test_file="tests/test_math.py",
            requirements="绘制函数。",
            needs_visualization=True,
            visualization_file="visualizations/plot.py",
            validation_strategy="使用 pytest 和性质测试。",
            human_checkpoints=("确认单位",),
        )

        self.assertEqual(allocation.visualization_file, "visualizations/plot.py")

    # 创建角色工厂测试所需的受控工作区执行器。
    def _create_executor(self: "PythonCodeAgentProfileTests", workspace_root: Path) -> WorkspaceToolExecutor:
        return WorkspaceToolExecutor(
            WorkspaceExecutionPolicy(workspace_root),
            create_workspace_file_tools(),
            InMemoryToolAuditLog(),
        )
