import tempfile
import unittest
from pathlib import Path

from pydantic_ai.models.test import TestModel

from abook_agent.agent import AgentDependencies, abook_agent
from abook_agent.knowledge import KnowledgeStore


class AgentTests(unittest.TestCase):
    def test_store_and_search(self) -> None:
        """写入后应能按正文中的多个关键词检索到同一文档。"""
        with tempfile.TemporaryDirectory() as directory:
            store = KnowledgeStore(Path(directory) / "knowledge.json")
            store.add("PydanticAI", "PydanticAI builds typed agent workflows.")

            matches = store.search("typed workflows")

            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0].title, "PydanticAI")

    def test_agent_runs_without_external_api(self) -> None:
        """使用 TestModel 验证工具注册和依赖注入，不访问真实模型接口。"""
        with tempfile.TemporaryDirectory() as directory:
            deps = AgentDependencies(
                store=KnowledgeStore(Path(directory) / "knowledge.json")
            )
            model = TestModel(call_tools=["list_documents"])

            result = abook_agent.run_sync("List the notes", deps=deps, model=model)

            self.assertIn("The knowledge base is empty.", result.output)


if __name__ == "__main__":
    unittest.main()
