"""加载按角色维护的 Agent 工作说明。"""

from pathlib import Path
from typing import Final, Literal


RoleInstructionName = Literal["python_code", "python_test"]
ROLE_INSTRUCTIONS_DIRECTORY: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "agent-profiles"
ROLE_INSTRUCTION_FILES: Final[dict[RoleInstructionName, str]] = {
    "python_code": "python-code-agent.md",
    "python_test": "python-test-agent.md",
}


# 读取随仓库版本控制的角色说明，使职责演进不必修改 Agent 工厂代码。
def load_role_instructions(role: RoleInstructionName) -> str:
    instruction_path = ROLE_INSTRUCTIONS_DIRECTORY / ROLE_INSTRUCTION_FILES[role]
    content = instruction_path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"角色说明不能为空：{instruction_path}")
    return content
