"""展示由 PydanticAI 驱动的 reason、act、observe、continue Agent Loop。"""

from typing import Annotated
import os
from pathlib import Path
import re
import sys

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, AgentRunResult, UsageLimits
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


SKILL_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
MAX_MODEL_REQUESTS = 6
MAX_TOOL_CALLS = 4
MAX_TEXT_LENGTH = 20_000


class SkillMetadata(BaseModel):
    """保存在 skill.json 中、用于发现 Skill 的元数据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=SKILL_ID_PATTERN.pattern)
    description: str = Field(min_length=1, max_length=500)


class Skill(BaseModel):
    """运行时加载的 Skill，完整指令只来自 Markdown 文件。"""

    metadata: SkillMetadata
    content: str


class TextInspection(BaseModel):
    """inspect_text 返回的确定性文本指标。

    total_characters 使用 Python Unicode 字符数，包含空白和标点；
    non_whitespace_characters 排除所有 Unicode 空白；paragraph_count 将一个
    或多个空白行视为段落分隔符。模型可根据用户对“字数”的具体定义选择指标。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_characters: int = Field(ge=0)
    non_whitespace_characters: int = Field(ge=0)
    line_count: int = Field(ge=0)
    paragraph_count: int = Field(ge=0)


class SkillRuntime:
    """按 ID 从固定文件名加载 JSON 元数据和 Markdown 正文。"""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=True)

    def load(self, skill_id: str) -> Skill:
        """加载指定 Skill 的元数据和 Markdown 指令。"""
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


def inspect_text(
    text: Annotated[
        str,
        Field(
            min_length=1,
            max_length=MAX_TEXT_LENGTH,
            description="需要检查的完整候选文本",
        ),
    ],
) -> TextInspection:
    """检查候选文本并返回确定性指标；该只读工具不修改内容。"""
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    stripped_text = normalized_text.strip()
    paragraphs = (
        re.split(r"\n(?:[ \t]*\n)+", stripped_text)
        if stripped_text
        else []
    )
    return TextInspection(
        total_characters=len(text),
        non_whitespace_characters=sum(
            not character.isspace() for character in text
        ),
        line_count=normalized_text.count("\n") + 1,
        paragraph_count=len(paragraphs),
    )


def create_agent(model: Model, skill: Skill) -> Agent:
    """创建单一执行 Agent，并注册构成行动与观察闭环的只读工具。

    PydanticAI 会把模型的工具调用交给 inspect_text，再把返回值作为工具结果
    加入同一次运行的消息历史。模型随后可以继续调用工具或提交最终答案。
    """
    return Agent(
        model,
        instructions=(
            "你是一个执行型 Agent。根据用户目标在内部执行 "
            "reason -> act -> observe -> continue 循环，但不要输出隐藏推理过程。"
            "需要客观文本指标时调用 inspect_text，并把工具结果视为事实。"
            "用户提出字数、字符数、行数或段落数限制时，必须先检查完整候选文本；"
            "不满足时修订并再次检查，满足后才给出最终答案。"
            "只向用户输出最终可用答案，不输出计划、工具协议或检查日志。\n\n"
            f"Skill：\n{skill.content}"
        ),
        tools=[inspect_text],
    )


def run_agent_loop(
    agent: Agent,
    request: str,
    usage_limits: UsageLimits | None = None,
) -> AgentRunResult[str]:
    """运行一个有界 Agent Loop 并返回包含完整消息历史的结果。

    Args:
        agent: 已注册所需工具的执行 Agent。
        request: 用户目标；空白请求会被拒绝。
        usage_limits: 可选的调用预算，主要用于测试或受控运行。

    Returns:
        最终答案以及 PydanticAI 记录的模型请求、工具调用和工具结果。

    Raises:
        ValueError: request 为空白。
        UsageLimitExceeded: 模型请求或工具调用超过预算。
    """
    normalized_request = request.strip()
    if not normalized_request:
        raise ValueError("请求不能为空")

    effective_limits = usage_limits or UsageLimits(
        request_limit=MAX_MODEL_REQUESTS,
        tool_calls_limit=MAX_TOOL_CALLS,
    )
    return agent.run_sync(
        normalized_request,
        usage_limits=effective_limits,
    )


def create_model() -> OpenAIChatModel:
    """使用项目 .env 中的 OpenAI 兼容配置创建模型。"""
    load_dotenv()
    provider = OpenAIProvider(
        base_url=os.environ["ABOOK_BASE_URL"],
        api_key=os.environ["ABOOK_API_KEY"],
    )
    return OpenAIChatModel(os.environ["ABOOK_MODEL"], provider=provider)


def main() -> None:
    """读取用户请求，运行工具型 Agent Loop，并只打印最终答案。"""
    request = " ".join(sys.argv[1:]).strip()
    if not request:
        request = input("请输入请求：").strip()
    if not request:
        raise SystemExit("请求不能为空")

    skills_root = Path(__file__).parent / "skills"
    skill = SkillRuntime(skills_root).load("general")
    result = run_agent_loop(create_agent(create_model(), skill), request)
    print(result.output)


if __name__ == "__main__":
    main()
