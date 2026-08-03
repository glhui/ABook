"""创建将开发请求拆分为测试与编码任务的无工具 Agent。"""

from pydantic import BaseModel, Field
from pydantic_ai import Agent, PromptedOutput
from pydantic_ai.models import Model

from agent_profiles.model_settings import create_agent_model_settings

# 表示可由测试 Agent 与代码 Agent 分别执行的任务分配结果。
class CodeTestTaskAllocation(BaseModel):
    testing_task: str = Field(description="交给 Python 测试 Agent 的具体私有用例与预期结果生成任务，不包含生产代码。")
    coding_task: str = Field(description="交给 Python 代码 Agent 的具体实现任务。")
    validation_task: str = Field(description="交给独立黑盒验证 Agent 的标准输入输出验收任务。")


TASK_COORDINATOR_INSTRUCTIONS = (
    "你是开发任务协调者。将用户的 Python 开发请求拆分为两个有顺序的任务："
    "testing_task 必须描述待生成的私有 JSON 输入案例、预期 JSON 输出和边界条件，测试 Agent 不得读取生产代码；"
    "coding_task 必须要求在不读取或修改私有测试的前提下实现标准输入输出功能；"
    "validation_task 必须描述独立于实现的标准输入输出黑盒验收、可推导 oracle 与边界用例。"
    "不要编写代码、测试或调用工具。"
)


# 创建不具备工作区工具、通过提示词 JSON 输出拆分任务的协调 Agent。
def create_code_test_task_coordinator(model: Model) -> Agent[None, CodeTestTaskAllocation]:
    return Agent(
        model,
        instructions=TASK_COORDINATOR_INSTRUCTIONS,
        model_settings=create_agent_model_settings(),
        output_type=PromptedOutput(CodeTestTaskAllocation),
    )
