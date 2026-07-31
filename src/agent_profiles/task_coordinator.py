"""创建将开发请求拆分为测试与编码任务的无工具 Agent。"""

from pydantic import BaseModel, Field
from pydantic_ai import Agent, PromptedOutput
from pydantic_ai.models import Model


# 表示可由测试 Agent 与代码 Agent 分别执行的任务分配结果。
class CodeTestTaskAllocation(BaseModel):
    testing_task: str = Field(description="交给 Python 测试 Agent 的具体测试编写任务。")
    coding_task: str = Field(description="交给 Python 代码 Agent 的具体实现任务。")


TASK_COORDINATOR_INSTRUCTIONS = (
    "你是开发任务协调者。将用户的 Python 开发请求拆分为两个有顺序的任务："
    "testing_task 必须先描述待编写的行为测试和边界条件；"
    "coding_task 必须要求在不修改这些测试的前提下实现功能并通过测试。"
    "不要编写代码、测试或调用工具。"
)


# 创建不具备工作区工具、通过提示词 JSON 输出拆分任务的协调 Agent。
def create_code_test_task_coordinator(model: Model) -> Agent[None, CodeTestTaskAllocation]:
    return Agent(
        model,
        instructions=TASK_COORDINATOR_INSTRUCTIONS,
        output_type=PromptedOutput(CodeTestTaskAllocation),
    )
