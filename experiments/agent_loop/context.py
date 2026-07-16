"""构建共享工作区上下文，并保存每个 Agent 的私有运行状态。"""

from dataclasses import dataclass, field
from pathlib import Path
import re
import subprocess
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage


SKILL_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
MAX_REPOSITORY_STATUS_LINES = 50
GIT_CONTEXT_TIMEOUT_SECONDS = 5


class SkillMetadata(BaseModel):
    """保存在 skill.json 中、用于发现 Skill 的最小元数据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=SKILL_ID_PATTERN.pattern)
    description: str = Field(min_length=1, max_length=500)


class Skill(BaseModel):
    """当前运行明确选择的 Skill。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metadata: SkillMetadata
    content: str


class ProjectInstruction(BaseModel):
    """一个适用于当前工作目录的 AGENTS.md 指令源。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    content: str


class RepositoryContext(BaseModel):
    """进入初始上下文的有界 Git 仓库快照。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    is_repository: bool
    branch: str | None
    status_lines: tuple[str, ...]
    status_truncated: bool


class WorkspaceContext(BaseModel):
    """所有 Agent 共享的不可变工作区上下文。

    这里只保存运行位置、项目约束和可发现的 Skill。用户任务、当前 Skill 与
    消息历史属于各自的 AgentContext，避免把全局事实和单个 Agent 状态混在一起。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    workspace_root: str
    working_directory: str
    skills_root: str
    project_instructions: tuple[ProjectInstruction, ...]
    repository: RepositoryContext
    available_skills: tuple[SkillMetadata, ...]

    def render_instructions(self) -> str:
        """按作用域优先级渲染传给模型的系统指令。"""
        rendered_project_instructions = "\n\n".join(
            f"### {instruction.path}\n{instruction.content}"
            for instruction in self.project_instructions
        )
        if not rendered_project_instructions:
            rendered_project_instructions = "未发现适用的 AGENTS.md。"

        if self.repository.is_repository:
            branch = self.repository.branch or "detached HEAD"
            repository_status = "\n".join(self.repository.status_lines)
            if not repository_status:
                repository_status = "工作区干净。"
            if self.repository.status_truncated:
                repository_status += "\n... Git 状态已截断。"
            rendered_repository = (
                f"当前分支：{branch}\n工作区状态：\n{repository_status}"
            )
        else:
            rendered_repository = "当前工作区不是 Git 仓库。"

        rendered_skill_catalog = "\n".join(
            f"- {metadata.name}: {metadata.description}"
            for metadata in self.available_skills
        )
        if not rendered_skill_catalog:
            rendered_skill_catalog = "未发现可用 Skill。"

        return (
            "## 运行环境\n"
            f"工作区根目录：{self.workspace_root}\n"
            f"当前工作目录：{self.working_directory}\n\n"
            "## 项目指令\n"
            "以下指令按作用域从宽到窄排列；更接近当前工作目录的指令优先。\n\n"
            f"{rendered_project_instructions}\n\n"
            f"## Git 快照\n{rendered_repository}\n\n"
            f"## 可用 Skill\n{rendered_skill_catalog}"
        )


class ValidationResult(BaseModel):
    """由宿主命令工具记录的一次验证结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    command: str
    exit_code: int | None
    timed_out: bool


@dataclass
class TaskState:
    """当前任务的结构化工作状态。

    计划、事实和完成条件由 root Agent 显式更新；修改文件和验证结果由实际工具
    调用确定性记录，避免模型把未发生的操作写成已经完成的事实。
    """

    goal: str
    plan: list[str] = field(default_factory=list)
    completed_steps: list[str] = field(default_factory=list)
    important_facts: list[str] = field(default_factory=list)
    completion_criteria: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    validation_results: list[ValidationResult] = field(default_factory=list)
    status: Literal["in_progress", "complete", "blocked"] = "in_progress"

    def __post_init__(self) -> None:
        """规范化目标，并拒绝没有实际任务的状态。"""
        self.goal = self.goal.strip()
        if not self.goal:
            raise ValueError("任务目标不能为空")

    def record_modified_file(self, path: str) -> None:
        """记录一次实际成功的文件修改，并保持路径列表去重。"""
        if path not in self.modified_files:
            self.modified_files.append(path)

    def record_validation(
        self,
        command: str,
        exit_code: int | None,
        timed_out: bool,
    ) -> None:
        """记录实际执行过的测试、编译或依赖检查结果。"""
        self.validation_results.append(
            ValidationResult(
                command=command,
                exit_code=exit_code,
                timed_out=timed_out,
            )
        )

    def render(self) -> str:
        """以紧凑、确定性的格式渲染当前任务状态。"""
        def render_items(items: list[str]) -> str:
            return "\n".join(f"- {item}" for item in items) or "- 无"

        validations = [
            (
                f"{result.command}: "
                f"{'timed out' if result.timed_out else f'exit {result.exit_code}'}"
            )
            for result in self.validation_results
        ]
        return (
            f"目标：{self.goal}\n"
            f"状态：{self.status}\n\n"
            f"计划：\n{render_items(self.plan)}\n\n"
            f"已完成：\n{render_items(self.completed_steps)}\n\n"
            f"重要事实：\n{render_items(self.important_facts)}\n\n"
            f"完成条件：\n{render_items(self.completion_criteria)}\n\n"
            f"已修改文件：\n{render_items(self.modified_files)}\n\n"
            f"验证结果：\n{render_items(validations)}"
        )


@dataclass
class AgentContext:
    """一个 Agent 私有的任务上下文；消息历史只属于该 Agent。"""

    agent_id: str
    task: str
    skill: Skill
    message_history: list[ModelMessage] = field(default_factory=list)


@dataclass
class AgentSession:
    """保存可继续调用的子 Agent 执行实例。"""

    template: str
    agent: Agent


class SkillRuntime:
    """从固定根目录安全加载 Skill 元数据和完整指令。"""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=True)

    def load(self, skill_id: str) -> Skill:
        """加载指定 Skill，并拒绝越过配置根目录的路径。"""
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

    def discover(self) -> tuple[SkillMetadata, ...]:
        """只读取 manifest，返回可供上下文展示的 Skill 目录。"""
        discovered_skills: list[SkillMetadata] = []
        for skill_directory in sorted(self.root.iterdir()):
            manifest_path = skill_directory / "skill.json"
            if not skill_directory.is_dir() or not manifest_path.is_file():
                continue
            metadata = SkillMetadata.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            if metadata.name != skill_directory.name:
                raise ValueError("Skill name must match its directory name")
            discovered_skills.append(metadata)
        return tuple(discovered_skills)


@dataclass
class ContextRuntime:
    """持有共享工作区上下文，并为每个 Agent 保存独立上下文。

    Runtime 不让模型维护这份结构。它只负责确定性地创建 AgentContext、组合
    全局指令与当前 Skill，并保存可继续调用的子 Agent 会话。
    """

    workspace: WorkspaceContext
    task_state: TaskState
    agent_contexts: dict[str, AgentContext] = field(default_factory=dict)
    subagents: dict[str, AgentSession] = field(default_factory=dict)

    def create_agent_context(
        self, agent_id: str, task: str, skill_id: str
    ) -> AgentContext:
        """创建并登记一个 Agent 私有上下文。"""
        normalized_task = task.strip()
        if not normalized_task:
            raise ValueError("任务不能为空")
        if agent_id in self.agent_contexts:
            raise ValueError(f"Agent 上下文已存在：{agent_id}")
        skill = SkillRuntime(Path(self.workspace.skills_root)).load(skill_id)
        agent_context = AgentContext(
            agent_id=agent_id,
            task=normalized_task,
            skill=skill,
        )
        self.agent_contexts[agent_id] = agent_context
        return agent_context

    def render_agent_instructions(
        self, agent_context: AgentContext
    ) -> str:
        """把共享上下文与指定 Agent 的当前 Skill 组合为模型指令。"""
        return (
            f"{self.workspace.render_instructions()}\n\n"
            f"## 当前 Skill\n{agent_context.skill.content}\n\n"
            f"## 当前任务状态\n{self.task_state.render()}"
        )

    def next_subagent_id(self, template: str) -> str:
        """生成当前父任务内可读的子 Agent ID。"""
        return f"{template}-{len(self.subagents) + 1}"


@dataclass(frozen=True)
class AgentDependencies:
    """把同一 Runtime 和当前 AgentContext 注入工具。"""

    runtime: ContextRuntime
    agent_context: AgentContext

    @property
    def workspace_root(self) -> Path:
        """返回所有 Agent 共享的工作区根目录。"""
        return Path(self.runtime.workspace.workspace_root)

    @property
    def working_directory(self) -> Path:
        """返回所有 Agent 共享的当前工作目录。"""
        return Path(self.runtime.workspace.working_directory)

    @property
    def skills_root(self) -> Path:
        """返回 Runtime 配置的 Skill 根目录。"""
        return Path(self.runtime.workspace.skills_root)


class WorkspaceContextBuilder:
    """确定性组织所有 Agent 共享的最小工作区上下文。

    该构建器不调用模型，也不执行工具。它只校验工作区作用域，加载从工作区根
    目录到当前工作目录依次生效的 AGENTS.md，并发现可供 Agent 选择的 Skill。
    """

    def __init__(self, workspace_root: Path, skills_root: Path) -> None:
        self.workspace_root = workspace_root.resolve(strict=True)
        if not self.workspace_root.is_dir():
            raise ValueError("工作区根目录必须是目录")
        self.skill_runtime = SkillRuntime(skills_root)

    def build(
        self,
        working_directory: Path | None = None,
    ) -> WorkspaceContext:
        """构建当前父任务内所有 Agent 共享的工作区上下文。

        Args:
            working_directory: 当前任务目录，默认使用工作区根目录。

        Returns:
            包含运行位置、层级项目指令和 Skill 目录的共享上下文。

        Raises:
            ValueError: 工作目录不在工作区内。
        """
        resolved_working_directory = (
            working_directory or self.workspace_root
        ).resolve(strict=True)
        if not resolved_working_directory.is_dir():
            raise ValueError("工作目录必须是目录")
        if not resolved_working_directory.is_relative_to(self.workspace_root):
            raise ValueError("工作目录必须位于工作区根目录内")

        return WorkspaceContext(
            workspace_root=self.workspace_root.as_posix(),
            working_directory=resolved_working_directory.as_posix(),
            skills_root=self.skill_runtime.root.as_posix(),
            project_instructions=tuple(
                self._load_project_instructions(resolved_working_directory)
            ),
            repository=self._inspect_repository(),
            available_skills=self.skill_runtime.discover(),
        )

    def _load_project_instructions(
        self, working_directory: Path
    ) -> list[ProjectInstruction]:
        """从宽到窄加载当前工作目录路径上的 AGENTS.md。"""
        candidate_directories = [self.workspace_root]
        current_directory = self.workspace_root
        relative_directory = working_directory.relative_to(self.workspace_root)
        for path_part in relative_directory.parts:
            current_directory /= path_part
            candidate_directories.append(current_directory)

        instructions: list[ProjectInstruction] = []
        for directory in candidate_directories:
            instruction_path = directory / "AGENTS.md"
            if not instruction_path.is_file():
                continue
            instructions.append(
                ProjectInstruction(
                    path=instruction_path.relative_to(
                        self.workspace_root
                    ).as_posix(),
                    content=instruction_path.read_text(
                        encoding="utf-8"
                    ).strip(),
                )
            )
        return instructions

    def _inspect_repository(self) -> RepositoryContext:
        """读取有界 Git 状态；非仓库或未安装 Git 时返回明确空快照。"""
        repository_check = self._run_git(
            "rev-parse", "--is-inside-work-tree"
        )
        if repository_check is None or repository_check.returncode != 0:
            return RepositoryContext(
                is_repository=False,
                branch=None,
                status_lines=(),
                status_truncated=False,
            )

        branch_result = self._run_git("branch", "--show-current")
        branch = (
            branch_result.stdout.strip()
            if branch_result is not None and branch_result.returncode == 0
            else None
        )
        status_result = self._run_git(
            "status", "--short", "--untracked-files=normal"
        )
        status_lines = (
            status_result.stdout.splitlines()
            if status_result is not None and status_result.returncode == 0
            else []
        )
        return RepositoryContext(
            is_repository=True,
            branch=branch or None,
            status_lines=tuple(status_lines[:MAX_REPOSITORY_STATUS_LINES]),
            status_truncated=len(status_lines) > MAX_REPOSITORY_STATUS_LINES,
        )

    def _run_git(
        self, *arguments: str
    ) -> subprocess.CompletedProcess[str] | None:
        """运行固定只读 Git 子命令，不接受模型生成的命令文本。"""
        try:
            return subprocess.run(
                ["git", *arguments],
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=GIT_CONTEXT_TIMEOUT_SECONDS,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
