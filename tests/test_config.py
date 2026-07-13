import os
import unittest
from unittest.mock import patch

from pydantic_ai.models.openai import OpenAIChatModel

from abook_agent.config import Settings, create_model


class ModelConfigurationTests(unittest.TestCase):
    def test_complete_environment_builds_compatible_model(self) -> None:
        """三个配置项应足以创建模型，且构造过程不发起网络请求。"""
        environment = {
            "ABOOK_MODEL": "custom-chat-model",
            "ABOOK_API_KEY": "test-key",
            "ABOOK_BASE_URL": "https://gateway.example.com/v1",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = Settings.from_env()
            model = create_model(settings)

        self.assertIsInstance(model, OpenAIChatModel)
        self.assertEqual(model.model_name, "custom-chat-model")
        self.assertEqual(settings.api_key, "test-key")
        self.assertEqual(settings.base_url, "https://gateway.example.com/v1")

    def test_missing_configuration_is_reported_together(self) -> None:
        """启动失败时应一次列出全部缺失项，避免逐项试错。"""
        with patch.dict(os.environ, {"ABOOK_MODEL": "test-model"}, clear=True):
            with self.assertRaisesRegex(
                ValueError, "ABOOK_API_KEY, ABOOK_BASE_URL"
            ):
                Settings.from_env()


if __name__ == "__main__":
    unittest.main()
