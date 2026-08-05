"""展示如何以角色 Profile 创建并覆盖单次 Agent 模型配置。"""

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator

from pydantic_ai.models.test import TestModel

from agent_profiles import AgentModelConfig, create_python_code_agent, create_python_code_context
from agent_profiles.role_instructions import get_role_instruction_definition
from tool_execution import InMemoryToolAuditLog, WorkspaceExecutionPolicy, WorkspaceToolExecutor
from workspace_tools import create_workspace_file_tools


# 保留示例的具名输出，同时支持调用方按元组方式解包三个配置值。
@dataclass(frozen=True)
class ConfiguredAgentDetails:
    instruction_file: str
    max_tokens: int
    temperature: float

    # 让最小示例可直接解包，而不把无语义的裸元组暴露为公共返回类型。
    def __iter__(self: "ConfiguredAgentDetails") -> Iterator[str | int | float]:
        yield self.instruction_file
        yield self.max_tokens
        yield self.temperature


# 创建离线 TestModel Agent，验证角色说明注册和本次调用的模型参数覆盖。
def create_configured_agent(workspace_root: Path) -> ConfiguredAgentDetails:
    executor = WorkspaceToolExecutor(
        WorkspaceExecutionPolicy(workspace_root=workspace_root),
        create_workspace_file_tools(),
        InMemoryToolAuditLog(),
    )
    model_config = AgentModelConfig(max_output_tokens=512, temperature=0.2)
    agent = create_python_code_agent(
        TestModel(),
        executor,
        create_python_code_context("profile-configuration"),
        model_config=model_config,
    )
    role_definition = get_role_instruction_definition("python_code")
    temperature = agent.model_settings.get("temperature")
    if not isinstance(temperature, float):
        raise RuntimeError("代码 Agent 未应用 temperature 配置。")
    return ConfiguredAgentDetails(
        instruction_file=role_definition.instruction_file,
        max_tokens=model_config.max_output_tokens,
        temperature=temperature,
    )


# 直接运行时输出离线示例的已解析配置。
def main() -> None:
    details = create_configured_agent(Path.cwd())
    print(f"instruction_file={details.instruction_file}")
    print(f"max_tokens={details.max_tokens}")
    print(f"temperature={details.temperature}")


if __name__ == "__main__":
    main()
