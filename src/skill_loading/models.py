from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


SkillKind = Literal["package", "repository", "workflow"]


# 描述 Skill 的轻量路由信息；正文始终保存在独立的 SKILL.md 中。
class SkillManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    version: str = Field(min_length=1)
    kind: SkillKind
    description: str = Field(min_length=1)
    roles: tuple[str, ...] = ()
    triggers: tuple[str, ...] = ()
    path_patterns: tuple[str, ...] = ()
    import_names: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    conflicts_with: tuple[str, ...] = ()
    priority: int = Field(default=0, ge=0, le=100)
    estimated_tokens: int = Field(default=1_000, gt=0)

    # 去除路由字段两端空白，避免看似相同的线索产生不同匹配结果。
    @field_validator(
        "roles",
        "triggers",
        "path_patterns",
        "import_names",
        "symbols",
        "depends_on",
        "conflicts_with",
    )
    @classmethod
    def validate_non_empty_items(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("Skill manifest 的列表字段不能包含空字符串。")
        return normalized


# 保存 manifest 的来源，但不预先读取 Skill 正文。
class SkillDescriptor(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    manifest: SkillManifest
    directory: Path
    content_path: Path


# 汇总一次路由所需的稳定信号，调用方可来自任务解析、目标文件和轻量代码检索。
class SkillTaskContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    task: str
    role: str
    target_paths: tuple[str, ...] = ()
    import_names: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    explicit_skills: tuple[str, ...] = ()


# 限制最终注入的 Skill 数量和声明 token 总量。
class SkillSelectionPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_skills: int = Field(default=3, gt=0)
    max_tokens: int = Field(default=6_000, gt=0)
    max_dependency_depth: int = Field(default=1, ge=0, le=3)


# 返回可解释的选择结果，便于日志、测试与后续路由评测。
class SelectedSkill(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    kind: SkillKind
    score: int
    reasons: tuple[str, ...]
    estimated_tokens: int
    content: str
