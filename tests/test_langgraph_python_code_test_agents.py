"""LangGraph 工作流结构的离线测试。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pydantic_ai.models.test import TestModel

from agent_profiles import CodeTestTaskAllocation
from examples.langgraph_python_code_test_agents import (
    SKILL_CATALOG_DIRECTORY,
    create_code_test_workflow,
    select_code_skills,
)
from skill_loading import SkillCatalog


# 验证迁移后的图保留代码、测试、验证和回到代码节点的有界循环。
class LangGraphCodeTestWorkflowTests(unittest.TestCase):
    def test_workflow_returns_failed_validation_to_code_node_without_rewriting_tests(
        self: "LangGraphCodeTestWorkflowTests",
    ) -> None:
        # 用具名回调保持测试中的确认行为与交互式入口一致。
        def approve() -> bool:
            return True

        with TemporaryDirectory() as temporary_directory:
            workflow = create_code_test_workflow(TestModel(), Path(temporary_directory), approve)
            graph = workflow.get_graph()
            node_names = set(graph.nodes)
            edges = {(edge.source, edge.target) for edge in graph.edges}

        self.assertTrue(
            {"plan", "prepare_workspace", "implement_code", "write_tests", "validate", "finish", "cancel"}
            <= node_names
        )
        self.assertIn(("validate", "implement_code"), edges)
        self.assertNotIn("repair", node_names)

    # LangGraph 在协调结果确定后按需求选择 Skill，并只为代码 Agent 准备注入内容。
    def test_select_code_skills_loads_scipy_and_its_numpy_dependency(self: "LangGraphCodeTestWorkflowTests") -> None:
        allocation = CodeTestTaskAllocation(
            source_file="src/optimizer.py",
            core_function="minimize_loss(values: list[float]) -> float",
            test_file="tests/test_optimizer.py",
            requirements="使用 SciPy 优化求解最小值，并验证收敛失败路径。",
        )

        selected = select_code_skills(
            SkillCatalog.discover(SKILL_CATALOG_DIRECTORY),
            allocation,
            "实现一个 scipy 最小化函数。",
        )

        self.assertEqual([skill.name for skill in selected], ["python-core-function", "numpy", "scipy"])
        self.assertTrue(all(skill.content for skill in selected))
