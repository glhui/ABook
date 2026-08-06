"""定义 Agent 可复用且可覆盖的模型输出配置。"""

from dataclasses import dataclass
from math import ceil
from typing import Final

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai.settings import ModelSettings


DEEPSEEK_CONTEXT_WINDOW_TOKENS: Final[int] = 1_000_000
DEEPSEEK_MAX_OUTPUT_TOKENS: Final[int] = 384_000
CONTEXT_COMPACTION_THRESHOLD: Final[float] = 0.70


# 表示角色默认值或调用方覆盖值，避免模型参数分散在各个 Agent 工厂中。
@dataclass(frozen=True)
class AgentModelConfig:
    context_window_tokens: int = DEEPSEEK_CONTEXT_WINDOW_TOKENS
    max_output_tokens: int = DEEPSEEK_MAX_OUTPUT_TOKENS
    temperature: float | None = None # 温度参数，控制输出的随机性，范围为0到2，默认值为None表示使用模型默认值。
    compaction_threshold: float = CONTEXT_COMPACTION_THRESHOLD

    # 在构建底层可变映射前校验配置，尽早报告无效角色或调用方参数。
    def __post_init__(self: "AgentModelConfig") -> None:
        if self.context_window_tokens <= 0:
            raise ValueError("context_window_tokens 必须大于 0。")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens 必须大于 0。")
        if self.max_output_tokens > self.context_window_tokens:
            raise ValueError("max_output_tokens 不能超过 context_window_tokens。")
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise ValueError("temperature 必须位于 0 到 2 之间。")
        if not 0 < self.compaction_threshold < 1:
            raise ValueError("compaction_threshold 必须位于 0 到 1 之间。")


DEFAULT_AGENT_MODEL_CONFIG: Final[AgentModelConfig] = AgentModelConfig()


# 为每个 Agent 返回独立设置，避免共享可变映射在运行期间被意外修改。
def create_agent_model_settings(config: AgentModelConfig | None = None) -> ModelSettings:
    selected_config = config or DEFAULT_AGENT_MODEL_CONFIG
    settings = ModelSettings(max_tokens=selected_config.max_output_tokens)
    if selected_config.temperature is not None:
        settings["temperature"] = selected_config.temperature
    return settings


# 使用 PydanticAI 的稳定消息序列化格式估算历史 token；模型未提供 usage 时只能使用保守近似。
def estimate_message_tokens(history: list[ModelMessage] | tuple[ModelMessage, ...]) -> int:
    serialized = ModelMessagesTypeAdapter.dump_json(list(history))
    return ceil(len(serialized) / 4)


# 判断历史是否达到配置的压缩阈值。
def should_compact_history(
    history: list[ModelMessage] | tuple[ModelMessage, ...],
    config: AgentModelConfig | None = None,
) -> bool:
    selected_config = config or DEFAULT_AGENT_MODEL_CONFIG
    return estimate_message_tokens(history) >= int(
        selected_config.context_window_tokens * selected_config.compaction_threshold
    )
