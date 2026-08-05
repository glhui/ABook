"""定义 Agent 可复用且可覆盖的模型输出配置。"""

from dataclasses import dataclass
from typing import Final

from pydantic_ai.settings import ModelSettings


AGENT_MAX_OUTPUT_TOKENS: Final[int] = 8_192


# 表示角色默认值或调用方覆盖值，避免模型参数分散在各个 Agent 工厂中。
@dataclass(frozen=True)
class AgentModelConfig:
    max_output_tokens: int = AGENT_MAX_OUTPUT_TOKENS
    temperature: float | None = None

    # 在构建底层可变映射前校验配置，尽早报告无效角色或调用方参数。
    def __post_init__(self: "AgentModelConfig") -> None:
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens 必须大于 0。")
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise ValueError("temperature 必须位于 0 到 2 之间。")


DEFAULT_AGENT_MODEL_CONFIG: Final[AgentModelConfig] = AgentModelConfig()


# 为每个 Agent 返回独立设置，避免共享可变映射在运行期间被意外修改。
def create_agent_model_settings(config: AgentModelConfig | None = None) -> ModelSettings:
    selected_config = config or DEFAULT_AGENT_MODEL_CONFIG
    settings = ModelSettings(max_tokens=selected_config.max_output_tokens)
    if selected_config.temperature is not None:
        settings["temperature"] = selected_config.temperature
    return settings
