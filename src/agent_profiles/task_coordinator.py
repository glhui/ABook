"""创建将开发请求收敛为单个核心函数任务的无工具 Agent。"""

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent, PromptedOutput
from pydantic_ai.models import Model

from agent_profiles.model_settings import AgentModelConfig, create_agent_model_settings
from agent_profiles.role_instructions import load_role_instructions


VISUALIZATION_TRIGGERS: tuple[str, ...] = (
    "绘图",
    "绘制",
    "可视化",
    "示意图",
    "图表",
    "plot",
    "visualiz",
    "chart",
    "diagram",
)

# 表示测试 Agent 与代码 Agent 共享的最小任务边界。
class CodeTestTaskAllocation(BaseModel):
    source_file: str = Field(description="需要编写的 Python 源码文件相对路径。")
    core_function: str = Field(description="需要实现和测试的核心函数签名。")
    test_file: str = Field(description="需要编写的 pytest 测试文件相对路径。")
    requirements: str = Field(description="核心函数的简短行为要求和关键边界条件。")
    needs_visualization: bool = Field(default=False, description="任务是否需要生成可视化程序或图表。")
    visualization_file: str | None = Field(default=None, description="可视化源码相对路径；需要可视化时必须位于 visualizations/。")
    validation_strategy: str = Field(default="使用 pytest 覆盖正常、边界和错误场景。", description="宿主和 Agent 应采用的验证策略。")
    human_checkpoints: tuple[str, ...] = Field(default=(), description="需要人工确认的关键决策点。")

    # 可视化任务必须提供隔离目录内的目标脚本，避免工作流进入无法落地的分支。
    @model_validator(mode="after")
    def validate_visualization_plan(self: "CodeTestTaskAllocation") -> "CodeTestTaskAllocation":
        if self.needs_visualization and not self.visualization_file:
            raise ValueError("needs_visualization 为 true 时必须提供 visualization_file。")
        if self.visualization_file is not None and not self.visualization_file.startswith("visualizations/"):
            raise ValueError("visualization_file 必须位于 visualizations/ 目录。")
        return self


TASK_COORDINATOR_INSTRUCTIONS = (
    "你是开发任务协调者。把用户的 Python 开发请求收敛为一个可由 pytest 验证的核心函数。"
    "返回源码文件、核心函数签名、pytest 测试文件、行为要求、验证策略和人工检查点；"
    "源码文件必须位于 src/，测试文件必须位于 tests/，路径均为相对路径；"
    "核心函数签名必须包含完整参数和返回类型；数学或数据分析任务需要可视化时，needs_visualization 设为 true，"
    "并将 visualization_file 设为 visualizations/ 下的 Python 文件；不需要时设为 false 和 null。"
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


# 用用户原始需求补强模型计划，避免模型漏判明确的可视化意图。
def normalize_task_plan(problem: str, allocation: CodeTestTaskAllocation) -> CodeTestTaskAllocation:
    normalized_problem = problem.casefold()
    needs_visualization = allocation.needs_visualization or any(
        trigger.casefold() in normalized_problem for trigger in VISUALIZATION_TRIGGERS
    )
    if not needs_visualization:
        return allocation
    visualization_file = allocation.visualization_file or "visualizations/visualization.py"
    checkpoints = allocation.human_checkpoints or ("确认可视化方案和输出图片。",)
    return allocation.model_copy(
        update={
            "needs_visualization": True,
            "visualization_file": visualization_file,
            "human_checkpoints": checkpoints,
        }
    )
