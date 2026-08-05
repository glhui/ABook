import asyncio
from pathlib import Path
import sys
from typing import Final

from pydantic_ai.exceptions import UnexpectedModelBehavior


EXAMPLES_DIRECTORY: Final[Path] = Path(__file__).resolve().parent
ROOT_DIRECTORY: Final[Path] = EXAMPLES_DIRECTORY.parent
SOURCE_DIRECTORY: Final[Path] = ROOT_DIRECTORY / "src"
SKILL_DIRECTORY: Final[Path] = EXAMPLES_DIRECTORY / "skill_catalog"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from python_code_test_agents import ANSI_RED, print_log_block, run_workflow
from skill_loading import SkillCatalog


# 加载示例 catalog；此时只读取 manifest，正文由工作流完成路由后按需读取。
def create_example_catalog() -> SkillCatalog:
    return SkillCatalog.discover(SKILL_DIRECTORY)


# 复用代码—测试—pytest 流程，并为代码 Agent 注入本轮选中的少量 Skill。
def main() -> int:
    problem = input("请输入 Python 开发需求：").strip()
    try:
        completed = asyncio.run(run_workflow(problem, create_example_catalog()))
    except UnexpectedModelBehavior as error:
        print_log_block("工作流", "模型错误", f"模型未能完成本轮生成：{error}", ANSI_RED, sys.stderr)
        completed = False
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
