"""创建将开发请求收敛为单个核心函数任务的无工具 Agent。"""

from pydantic import BaseModel, Field
from pydantic_ai import Agent, PromptedOutput
from pydantic_ai.models import Model

from agent_profiles.model_settings import AgentModelConfig, create_agent_model_settings
from agent_profiles.role_instructions import load_role_instructions

# 表示测试 Agent 与代码 Agent 共享的最小任务边界。
class CodeTestTaskAllocation(BaseModel):
    source_file: str = Field(description="需要编写的 Python 源码文件相对路径。")
    core_function: str = Field(description="需要实现和测试的核心函数签名。")
    test_file: str = Field(description="需要编写的 pytest 测试文件相对路径。")
    requirements: str = Field(description="核心函数的简短行为要求和关键边界条件。")


TASK_COORDINATOR_INSTRUCTIONS = (
    "你是开发任务协调者。把用户的 Python 开发请求收敛为一个可由 pytest 验证的核心函数。"
    "只返回源码文件、核心函数签名、pytest 测试文件和简短行为要求；"
    "源码文件必须位于 src/，测试文件必须位于 tests/，路径均为相对路径；"
    "核心函数签名必须包含完整参数和返回类型，行为要求只保留实现与测试所需的信息。"
    "不要设计私有测试包、标准输入输出协议、验证器或修复方案，也不要编写代码或调用工具。"
)


# 创建不具备工作区工具、通过提示词 JSON 输出拆分任务的协调 Agent。
def create_code_test_task_coordinator(
    model: Model,
    model_config: AgentModelConfig | None = None,
) -> Agent[None, CodeTestTaskAllocation]:
    return Agent(
        model,
        instructions=f"{TASK_COORDINATOR_INSTRUCTIONS}\n\n{load_role_instructions('task_coordinator')}",
        model_settings=create_agent_model_settings(model_config),
        output_type=PromptedOutput(CodeTestTaskAllocation),
    )
