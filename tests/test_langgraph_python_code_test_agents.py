"""LangGraph 工作流结构的离线测试。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pydantic_ai.models.test import TestModel

from examples.langgraph_python_code_test_agents import create_code_test_workflow


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
