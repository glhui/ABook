"""展示 Skill 加载和思考、生成、检查 Agent 循环的最小实现。"""

import os
from pathlib import Path
import re
import sys

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


SKILL_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


class SkillMetadata(BaseModel):
    """保存在 ``skill.json`` 中、用于发现 Skill 的元数据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=SKILL_ID_PATTERN.pattern)
    description: str = Field(min_length=1, max_length=500)


class Skill(BaseModel):
    """运行时加载的 Skill，完整指令只来自 Markdown 文件。"""

    metadata: SkillMetadata
    content: str


class Review(BaseModel):
    """从检查 Agent 的简单文本协议转换出的结论。"""

    approved: bool
    feedback: str


class TaskState(BaseModel):
    """多次 Agent 调用之间显式传递的最小任务状态。"""

    request: str
    plan: str = ""
    draft: str = ""
    feedback: str = "首次生成"
    attempts: int = 0
    approved: bool = False


class SkillRuntime:
    """按 ID 从固定文件名加载 JSON 元数据和 Markdown 正文。"""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=True)

    def load(self, skill_id: str) -> Skill:
        """加载 ``<skill_id>/skill.json`` 和 ``<skill_id>/SKILL.md``。"""
        if not SKILL_ID_PATTERN.fullmatch(skill_id):
            raise ValueError(f"Invalid Skill ID: {skill_id!r}")

        skill_directory = (self.root / skill_id).resolve(strict=True)
        if not skill_directory.is_relative_to(self.root):
            raise ValueError(f"Skill escapes configured root: {skill_id!r}")

        metadata = SkillMetadata.model_validate_json(
            (skill_directory / "skill.json").read_text(encoding="utf-8")
        )
        if metadata.name != skill_id:
            raise ValueError("Skill name must match its directory name")

        content = (skill_directory / "SKILL.md").read_text(
            encoding="utf-8"
        ).strip()
        if not content:
            raise ValueError("SKILL.md must not be empty")
        return Skill(metadata=metadata, content=content)


class AgentLoop:
    """运行一次思考，并在检查失败时有限次地重新生成。"""

    def __init__(self, model: OpenAIChatModel, skill: Skill) -> None:
        self.skill = skill
        self.thinker = Agent(
            model,
            instructions="输出简洁的执行计划，不要直接回答用户。",
        )
        self.generator = Agent(
            model,
            instructions="根据 Skill、计划和反馈生成候选答案。",
        )
        self.checker = Agent(
            model,
            instructions=(
                "检查草稿是否满足用户请求和 Skill。第一行只能输出 PASS 或 "
                "FAIL，后续内容给出简短、具体的检查结论。"
            ),
        )

    def run(self, request: str, max_attempts: int = 2) -> TaskState:
        """执行有界循环并返回可观察的任务进度和最终草稿。"""
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        state = TaskState(request=request)
        context = f"用户请求：\n{request}\n\nSkill：\n{self.skill.content}"
        state.plan = self.thinker.run_sync(context).output

        while state.attempts < max_attempts and not state.approved:
            state.attempts += 1
            state.draft = self.generator.run_sync(
                f"{context}\n\n计划：\n{state.plan}\n\n检查反馈：\n{state.feedback}"
            ).output
            review_text = self.checker.run_sync(
                f"{context}\n\n计划：\n{state.plan}\n\n草稿：\n{state.draft}"
            ).output
            review = self._parse_review(review_text)
            state.approved = review.approved
            state.feedback = review.feedback

        return state

    @staticmethod
    def _parse_review(review_text: str) -> Review:
        """验证检查 Agent 的首行状态；未知状态按未通过处理。"""
        decision, _, feedback = review_text.strip().partition("\n")
        normalized_decision = decision.strip().upper()
        if normalized_decision not in {"PASS", "FAIL"}:
            return Review(approved=False, feedback=review_text.strip())
        return Review(
            approved=normalized_decision == "PASS",
            feedback=feedback.strip() or normalized_decision,
        )


def create_model() -> OpenAIChatModel:
    """使用项目 ``.env`` 中的 OpenAI 兼容配置创建模型。"""
    load_dotenv()
    provider = OpenAIProvider(
        base_url=os.environ["ABOOK_BASE_URL"],
        api_key=os.environ["ABOOK_API_KEY"],
    )
    return OpenAIChatModel(os.environ["ABOOK_MODEL"], provider=provider)


def main() -> None:
    """从命令行读取请求并打印完整的任务状态。"""
    request = " ".join(sys.argv[1:]).strip()
    if not request:
        request = input("请输入请求：").strip()
    if not request:
        raise SystemExit("请求不能为空")

    skills_root = Path(__file__).parent / "skills"
    skill = SkillRuntime(skills_root).load("general")
    state = AgentLoop(create_model(), skill).run(request)
    print(state.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
