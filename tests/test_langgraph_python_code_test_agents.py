"""LangGraph 工作流结构的离线测试。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pydantic_ai.models.test import TestModel

from examples.langgraph_python_code_test_agents import create_code_test_workflow


# 验证迁移后的图保留代码、测试、验证和有界修复循环等关键节点。
class LangGraphCodeTestWorkflowTests(unittest.TestCase):
    def test_workflow_contains_expected_nodes_and_repair_loop(self: "LangGraphCodeTestWorkflowTests") -> None:
        # 用具名回调保持测试中的确认行为与交互式入口一致。
        def approve() -> bool:
            return True

        with TemporaryDirectory() as temporary_directory:
            workflow = create_code_test_workflow(TestModel(), Path(temporary_directory), approve)
            graph = workflow.get_graph()
            node_names = set(graph.nodes)
            edges = {(edge.source, edge.target) for edge in graph.edges}

        self.assertTrue(
            {"plan", "prepare_workspace", "implement_code", "write_tests", "validate", "repair", "finish", "cancel"}
            <= node_names
        )
        self.assertIn(("repair", "validate"), edges)
