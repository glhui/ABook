"""集中定义开发工作流 Agent 的模型输出设置。"""

from typing import Final

from pydantic_ai.settings import ModelSettings


AGENT_MAX_OUTPUT_TOKENS: Final[int] = 8_192


# 为每个 Agent 返回独立设置，避免共享可变映射在运行期间被意外修改。
def create_agent_model_settings() -> ModelSettings:
    return ModelSettings(max_tokens=AGENT_MAX_OUTPUT_TOKENS)
