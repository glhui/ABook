"""加载按角色维护的 Agent 工作说明。"""

from pathlib import Path
from dataclasses import dataclass
from typing import Final, Literal


RoleInstructionName = Literal["task_coordinator", "python_code", "python_test", "python_validator"]
ROLE_INSTRUCTIONS_DIRECTORY: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "agent-profiles"


# 描述一个可用角色的版本化说明来源和可选 Skill 注入能力。
@dataclass(frozen=True)
class RoleInstructionDefinition:
    name: RoleInstructionName
    instruction_file: str
    accepts_skill_instructions: bool


ROLE_INSTRUCTION_DEFINITIONS: Final[dict[RoleInstructionName, RoleInstructionDefinition]] = {
    "task_coordinator": RoleInstructionDefinition("task_coordinator", "task-coordinator-agent.md", False),
    "python_code": RoleInstructionDefinition("python_code", "python-code-agent.md", True),
    "python_test": RoleInstructionDefinition("python_test", "python-test-agent.md", False),
    "python_validator": RoleInstructionDefinition("python_validator", "python-validator-agent.md", False),
}


# 读取随仓库版本控制的角色说明，使职责演进不必修改 Agent 工厂代码。
def load_role_instructions(role: RoleInstructionName) -> str:
    instruction_path = ROLE_INSTRUCTIONS_DIRECTORY / ROLE_INSTRUCTION_DEFINITIONS[role].instruction_file
    content = instruction_path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"角色说明不能为空：{instruction_path}")
    return content


# 返回角色注册信息，供 Profile 工厂统一决定说明与 Skill 注入规则。
def get_role_instruction_definition(role: RoleInstructionName) -> RoleInstructionDefinition:
    return ROLE_INSTRUCTION_DEFINITIONS[role]
