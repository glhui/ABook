"""展示类似 Codex 的上下文 Runtime、Skill 选择与模板子 Agent。"""

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Annotated, Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, AgentRunResult, ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.toolsets import FunctionToolset, WrapperToolset
from pydantic_ai.toolsets.abstract import ToolsetTool


SKILL_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
IGNORED_WORKSPACE_DIRECTORIES = frozenset(
    {".git", ".venv", "__pycache__"}
)
SENSITIVE_WORKSPACE_FILE_NAMES = frozenset({".env"})
MAX_LISTED_FILES = 100
MAX_READ_CHARACTERS = 20_000
MAX_SEARCH_MATCHES = 50
MAX_SEARCHED_FILES = 500
MAX_MATCH_LINE_LENGTH = 300
MAX_REPLACEMENT_TEXT_LENGTH = 20_000
MAX_COMMAND_LENGTH = 2_000
MAX_COMMAND_OUTPUT_CHARACTERS = 20_000
MAX_COMMAND_TIMEOUT_SECONDS = 120
MAX_REPOSITORY_STATUS_LINES = 50
GIT_CONTEXT_TIMEOUT_SECONDS = 5
POWERSHELL_EXECUTABLE = (
    Path(os.environ.get("SystemRoot", "C:/Windows"))
    / "System32"
    / "WindowsPowerShell"
    / "v1.0"
    / "powershell.exe"
)
ALLOWED_POWERSHELL_COMMANDS = frozenset(
    {
        "get-childitem",
        "get-command",
        "get-item",
        "get-location",
        "resolve-path",
        "test-path",
        "git",
        "rg",
        "python",
        "python.exe",
    }
)
ALLOWED_GIT_SUBCOMMANDS = frozenset(
    {"diff", "log", "ls-files", "rev-parse", "status"}
)
FORBIDDEN_POWERSHELL_SYNTAX = (
    "\n",
    "\r",
    ";",
    "|",
    ">",
    "<",
    "`",
    "$(",
    "@(",
    "&&",
    "||",
)


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


class CommandResult(BaseModel):
    """受限 PowerShell 命令的稳定执行结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    command: str
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool


class AgentTemplate(BaseModel):
    """父 Agent 可用于创建子 Agent 会话的固定模板。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    instructions: str
    can_write: bool


class DelegationResult(BaseModel):
    """子 Agent 完成一轮任务后返回给父 Agent 的摘要。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    template: str
    skill: str
    output: str
    model_requests: int
    tool_calls: int


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


AGENT_TEMPLATES = {
    template.name: template
    for template in (
        AgentTemplate(
            name="explorer",
            description="只读探索代码、定位文件并汇总证据",
            instructions=(
                "只读取和搜索工作区，不修改文件。先收集证据，再给出简洁结论。"
            ),
            can_write=False,
        ),
        AgentTemplate(
            name="worker",
            description="实现用户已经明确授权的代码修改并运行验证",
            instructions=(
                "完成一个范围明确的实现任务。只有任务明确要求修改时才写文件，"
                "修改后运行相关验证并报告结果。"
            ),
            can_write=True,
        ),
        AgentTemplate(
            name="reviewer",
            description="只读审查实现、测试和潜在风险",
            instructions=(
                "审查现有代码和修改，不写文件。结论必须引用工具获得的证据。"
            ),
            can_write=False,
        ),
    )
}


class RecoverableToolError(Exception):
    """表示模型可通过修改参数或改用其他工具修正的失败。"""


class RetryToolset(WrapperToolset["AgentDependencies"]):
    """在单一边界把业务可恢复错误转换为 PydanticAI 重试信号。"""

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext["AgentDependencies"],
        tool: ToolsetTool["AgentDependencies"],
    ) -> Any:
        """执行被包装工具，并让模型观察可修正错误后重新决策。"""
        try:
            return await self.wrapped.call_tool(
                name, tool_args, ctx, tool
            )
        except RecoverableToolError as error:
            raise ModelRetry(str(error)) from error


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
            f"## 当前 Skill\n{agent_context.skill.content}"
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


def _resolve_workspace_path(
    workspace_root: Path,
    requested_path: str,
) -> Path:
    """解析工作区相对路径，并拒绝目录穿越和外部符号链接。"""
    try:
        resolved_path = (workspace_root / requested_path).resolve(strict=True)
    except FileNotFoundError as error:
        raise RecoverableToolError(
            f"工作区路径不存在：{requested_path}。请重新选择路径。"
        ) from error
    if not resolved_path.is_relative_to(workspace_root):
        raise RecoverableToolError("路径必须位于工作区根目录内")
    if resolved_path.name in SENSITIVE_WORKSPACE_FILE_NAMES:
        raise RecoverableToolError("工具不能访问包含凭据的敏感文件")
    return resolved_path


def _iter_workspace_files(directory: Path) -> list[Path]:
    """按稳定顺序枚举目录中的文件，并跳过运行数据目录。"""
    files: list[Path] = []
    for directory_path, directory_names, file_names in os.walk(directory):
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name not in IGNORED_WORKSPACE_DIRECTORIES
        )
        current_directory = Path(directory_path)
        files.extend(
            current_directory / name
            for name in sorted(file_names)
            if name not in SENSITIVE_WORKSPACE_FILE_NAMES
        )
    return files


def list_workspace_files(
    ctx: RunContext[AgentDependencies],
    directory: Annotated[
        str,
        Field(
            min_length=1,
            max_length=500,
            description="工作区根目录下要枚举的相对目录",
        ),
    ] = ".",
    limit: Annotated[
        int,
        Field(ge=1, le=MAX_LISTED_FILES, description="最多返回的文件数量"),
    ] = 50,
) -> str:
    """递归列出工作区文件；无效目录会反馈给模型重新选择。"""
    resolved_directory = _resolve_workspace_path(
        ctx.deps.workspace_root, directory
    )
    if not resolved_directory.is_dir():
        raise RecoverableToolError(
            "directory 必须指向工作区内存在的目录"
        )

    files = _iter_workspace_files(resolved_directory)
    if not files:
        return "No files found."

    displayed_files = files[:limit]
    lines = [
        path.relative_to(ctx.deps.workspace_root).as_posix()
        for path in displayed_files
    ]
    if len(files) > limit:
        lines.append(f"... truncated; {len(files)} files found.")
    return "\n".join(lines)


def read_workspace_file(
    ctx: RunContext[AgentDependencies],
    path: Annotated[
        str,
        Field(
            min_length=1,
            max_length=500,
            description="工作区根目录下要读取的 UTF-8 文件相对路径",
        ),
    ],
    max_characters: Annotated[
        int,
        Field(
            ge=1,
            le=MAX_READ_CHARACTERS,
            description="最多返回的字符数",
        ),
    ] = 10_000,
) -> str:
    """读取 UTF-8 工作区文件；无效路径会反馈给模型重新选择。"""
    resolved_path = _resolve_workspace_path(ctx.deps.workspace_root, path)
    if not resolved_path.is_file():
        raise RecoverableToolError("path 必须指向工作区文件")
    try:
        content = resolved_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise RecoverableToolError(
            "文件不是 UTF-8 文本，请改用其他文件或工具。"
        ) from error
    if len(content) <= max_characters:
        return content
    return (
        content[:max_characters]
        + f"\n... truncated after {max_characters} characters."
    )


def search_workspace_text(
    ctx: RunContext[AgentDependencies],
    query: Annotated[
        str,
        Field(
            min_length=1,
            max_length=200,
            description="要查找的大小写不敏感字面文本",
        ),
    ],
    directory: Annotated[
        str,
        Field(
            min_length=1,
            max_length=500,
            description="工作区根目录下要搜索的相对目录",
        ),
    ] = ".",
    limit: Annotated[
        int,
        Field(ge=1, le=MAX_SEARCH_MATCHES, description="最多返回的匹配行数量"),
    ] = 20,
) -> str:
    """搜索 UTF-8 工作区文件；无效目录会反馈给模型重新选择。"""
    resolved_directory = _resolve_workspace_path(
        ctx.deps.workspace_root, directory
    )
    if not resolved_directory.is_dir():
        raise RecoverableToolError(
            "directory 必须指向工作区内存在的目录"
        )

    normalized_query = query.casefold()
    matches: list[str] = []
    skipped_non_utf8_files = 0
    files = _iter_workspace_files(resolved_directory)
    search_truncated = len(files) > MAX_SEARCHED_FILES
    for file_path in files[:MAX_SEARCHED_FILES]:
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            skipped_non_utf8_files += 1
            continue
        for line_number, line in enumerate(content.splitlines(), start=1):
            if normalized_query not in line.casefold():
                continue
            relative_path = file_path.relative_to(
                ctx.deps.workspace_root
            ).as_posix()
            matches.append(
                f"{relative_path}:{line_number}:"
                f"{line[:MAX_MATCH_LINE_LENGTH]}"
            )
            if len(matches) == limit:
                return "\n".join(matches + ["... match results truncated."])

    if not matches:
        matches.append("No matching text found.")
    if search_truncated:
        matches.append(
            f"... search stopped after {MAX_SEARCHED_FILES} files."
        )
    if skipped_non_utf8_files:
        matches.append(
            f"Skipped {skipped_non_utf8_files} non-UTF-8 files."
        )
    return "\n".join(matches)


def replace_workspace_text(
    ctx: RunContext[AgentDependencies],
    path: Annotated[
        str,
        Field(
            min_length=1,
            max_length=500,
            description="工作区根目录下要修改的 UTF-8 文件相对路径",
        ),
    ],
    old_text: Annotated[
        str,
        Field(
            min_length=1,
            max_length=MAX_REPLACEMENT_TEXT_LENGTH,
            description="必须在文件中精确出现的原文本",
        ),
    ],
    new_text: Annotated[
        str,
        Field(
            max_length=MAX_REPLACEMENT_TEXT_LENGTH,
            description="替换后的新文本；允许为空字符串",
        ),
    ],
    expected_replacements: Annotated[
        int,
        Field(
            ge=1,
            le=100,
            description="预期替换次数；实际次数不一致时拒绝写入",
        ),
    ] = 1,
) -> str:
    """仅在用户明确要求修改代码或文件时执行精确文本替换。

    该工具会修改文件。替换前必须验证原文本出现次数，避免模型使用模糊匹配
    意外修改额外位置；校验失败时文件保持不变。
    """
    resolved_path = _resolve_workspace_path(ctx.deps.workspace_root, path)
    if not resolved_path.is_file():
        raise RecoverableToolError("path 必须指向工作区文件")
    try:
        content = resolved_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise RecoverableToolError(
            "文件不是 UTF-8 文本，请改用其他文件或工具。"
        ) from error
    actual_replacements = content.count(old_text)
    if actual_replacements != expected_replacements:
        raise RecoverableToolError(
            "原文本出现次数与 expected_replacements 不一致："
            f"expected={expected_replacements}, actual={actual_replacements}。"
            "请重新读取文件并提供精确原文本。"
        )

    updated_content = content.replace(
        old_text, new_text, expected_replacements
    )
    resolved_path.write_text(updated_content, encoding="utf-8")
    return (
        f"Replaced {actual_replacements} occurrence(s) in "
        f"{resolved_path.relative_to(ctx.deps.workspace_root).as_posix()}."
    )


def _validate_powershell_command(command: str) -> list[str]:
    """验证受限 PowerShell 单命令，并返回用于策略检查的参数列表。

    该检查不是操作系统沙箱。它只允许一条无管道、重定向、子表达式或环境变量
    展开的命令，并进一步限制 Git 和 Python 子命令。
    """
    if any(token in command for token in FORBIDDEN_POWERSHELL_SYNTAX):
        raise RecoverableToolError(
            "PowerShell 命令不能包含管道、重定向或复合语法"
        )
    if "$" in command:
        raise RecoverableToolError("PowerShell 命令不能展开变量")
    if ".env" in command.casefold():
        raise RecoverableToolError(
            "PowerShell 命令不能访问敏感配置文件"
        )

    try:
        arguments = [
            argument.strip("\"'")
            for argument in shlex.split(command, posix=False)
        ]
    except ValueError as error:
        raise RecoverableToolError(
            "PowerShell 命令引号不完整，请修正命令语法"
        ) from error
    if not arguments:
        raise RecoverableToolError("PowerShell 命令不能为空")

    executable_name = Path(arguments[0]).name.casefold()
    if executable_name not in ALLOWED_POWERSHELL_COMMANDS:
        raise RecoverableToolError(f"不允许执行命令：{arguments[0]}")

    for argument in arguments[1:]:
        if ".." in re.split(r"[\\/]", argument):
            raise RecoverableToolError("命令参数不能访问工作区父目录")
        if re.match(r"^[a-zA-Z]:[\\/]", argument):
            raise RecoverableToolError("命令参数不能使用绝对路径")
        if argument.startswith(("\\", "/")):
            raise RecoverableToolError("命令参数不能使用根路径")
        if re.match(r"^[a-zA-Z]+:", argument):
            raise RecoverableToolError(
                "命令参数不能访问 PowerShell Provider"
            )

    if executable_name == "git":
        if len(arguments) < 2 or arguments[1].casefold() not in (
            ALLOWED_GIT_SUBCOMMANDS
        ):
            raise RecoverableToolError("只允许只读 Git 子命令")
    if executable_name in {"python", "python.exe"}:
        normalized_arguments = [argument.casefold() for argument in arguments]
        allowed_python_command = (
            normalized_arguments[1:] == ["--version"]
            or normalized_arguments[1:3] in (
                ["-m", "unittest"],
                ["-m", "compileall"],
            )
            or normalized_arguments[1:4] == ["-m", "pip", "check"]
        )
        if not allowed_python_command:
            raise RecoverableToolError(
                "Python 只允许版本、测试、编译和 pip check 命令"
            )
    if executable_name == "rg" and any(
        argument.casefold().startswith("--pre") for argument in arguments[1:]
    ):
        raise RecoverableToolError("rg 不能执行预处理命令")
    return arguments


def _truncate_command_output(output: str | bytes | None) -> str:
    """将命令输出转换为文本并限制进入模型上下文的长度。"""
    if output is None:
        return ""
    if isinstance(output, bytes):
        normalized_output = output.decode("utf-8", errors="replace")
    else:
        normalized_output = output
    if len(normalized_output) <= MAX_COMMAND_OUTPUT_CHARACTERS:
        return normalized_output
    return (
        normalized_output[:MAX_COMMAND_OUTPUT_CHARACTERS]
        + f"\n... truncated after {MAX_COMMAND_OUTPUT_CHARACTERS} characters."
    )


def run_powershell_command(
    ctx: RunContext[AgentDependencies],
    command: Annotated[
        str,
        Field(
            min_length=1,
            max_length=MAX_COMMAND_LENGTH,
            description=(
                "单条受限 PowerShell 命令；不允许管道、重定向或复合语法"
            ),
        ),
    ],
    working_directory: Annotated[
        str | None,
        Field(
            max_length=500,
            description=(
                "可选的工作区相对目录；默认使用当前 Runtime 工作目录"
            ),
        ),
    ] = None,
    timeout_seconds: Annotated[
        int,
        Field(
            ge=1,
            le=MAX_COMMAND_TIMEOUT_SECONDS,
            description="命令超时秒数",
        ),
    ] = 30,
) -> CommandResult:
    """运行受限的查询、Git 检查或本地验证命令。

    该工具不允许安装依赖、网络访问、Git 写操作或任意 PowerShell 复合语法。
    Python 测试和编译可能在工作区产生常规缓存文件。
    """
    _validate_powershell_command(command)
    command_working_directory = (
        ctx.deps.working_directory
        if working_directory is None
        else _resolve_workspace_path(
            ctx.deps.workspace_root, working_directory
        )
    )
    if not command_working_directory.is_dir():
        raise RecoverableToolError(
            "working_directory 必须指向工作区内存在的目录"
        )

    try:
        completed_process = subprocess.run(
            [
                str(POWERSHELL_EXECUTABLE),
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command,
            ],
            cwd=command_working_directory,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return CommandResult(
            command=command,
            exit_code=None,
            stdout=_truncate_command_output(error.stdout),
            stderr=_truncate_command_output(error.stderr),
            timed_out=True,
        )

    return CommandResult(
        command=command,
        exit_code=completed_process.returncode,
        stdout=_truncate_command_output(completed_process.stdout),
        stderr=_truncate_command_output(completed_process.stderr),
        timed_out=False,
    )


def select_skill(
    ctx: RunContext[AgentDependencies],
    skill_id: Annotated[
        str,
        Field(
            pattern=SKILL_ID_PATTERN.pattern,
            description="从上下文 Skill 目录中选择的 Skill ID",
        ),
    ],
) -> str:
    """按需加载 Skill，并更新当前 Agent 私有上下文。"""
    try:
        skill = SkillRuntime(ctx.deps.skills_root).load(skill_id)
    except (FileNotFoundError, ValueError) as error:
        raise RecoverableToolError(
            f"无法选择 Skill {skill_id!r}：{error}"
        ) from error
    ctx.deps.agent_context.skill = skill
    return f"# Skill: {skill.metadata.name}\n\n{skill.content}"


def _create_workspace_toolset(can_write: bool) -> RetryToolset:
    """为父子 Agent 创建统一重试边界下的工作区工具集。"""
    tools = [
        list_workspace_files,
        read_workspace_file,
        search_workspace_text,
        run_powershell_command,
    ]
    if can_write:
        tools.append(replace_workspace_text)
    return RetryToolset(
        FunctionToolset[AgentDependencies](tools=tools, max_retries=2)
    )


async def delegate_task(
    ctx: RunContext[AgentDependencies],
    template: Annotated[
        str,
        Field(
            min_length=1,
            max_length=50,
            description="子 Agent 模板：explorer、worker 或 reviewer",
        ),
    ],
    task: Annotated[
        str,
        Field(
            min_length=1,
            max_length=4_000,
            description="交给子 Agent 的单一、范围明确的任务",
        ),
    ],
    skill_id: Annotated[
        str,
        Field(
            pattern=SKILL_ID_PATTERN.pattern,
            description="子 Agent 使用的 Skill ID",
        ),
    ] = "general",
) -> DelegationResult:
    """创建不可递归委派的子 Agent，并保存可继续交互的内存会话。"""
    agent_template = AGENT_TEMPLATES.get(template)
    if agent_template is None:
        raise RecoverableToolError(
            f"未知 Agent 模板 {template!r}；可用模板为 "
            f"{', '.join(AGENT_TEMPLATES)}"
        )
    runtime = ctx.deps.runtime
    try:
        session_id = runtime.next_subagent_id(agent_template.name)
        child_context = runtime.create_agent_context(
            session_id, task, skill_id
        )
    except (FileNotFoundError, ValueError) as error:
        raise RecoverableToolError(
            f"无法为子 Agent 加载 Skill {skill_id!r}：{error}"
        ) from error

    child_agent = Agent(
        ctx.model,
        deps_type=AgentDependencies,
        instructions=(
            f"你是由父 Agent 创建的 {agent_template.name} 子 Agent。"
            "只完成委派任务，不扩展任务范围，也不能创建其他 Agent。\n\n"
            f"## 模板指令\n{agent_template.instructions}"
        ),
        toolsets=[_create_workspace_toolset(agent_template.can_write)],
    )

    @child_agent.instructions
    def child_runtime_context(
        run_context: RunContext[AgentDependencies],
    ) -> str:
        """在子 Agent 每次运行前组合共享上下文与其私有上下文。"""
        return run_context.deps.runtime.render_agent_instructions(
            run_context.deps.agent_context
        )

    child_result = await child_agent.run(
        child_context.task,
        deps=AgentDependencies(runtime, child_context),
    )
    child_context.message_history = child_result.all_messages()
    runtime.subagents[session_id] = AgentSession(
        template=agent_template.name,
        agent=child_agent,
    )
    return DelegationResult(
        session_id=session_id,
        template=agent_template.name,
        skill=child_context.skill.metadata.name,
        output=child_result.output,
        model_requests=child_result.usage.requests,
        tool_calls=child_result.usage.tool_calls,
    )


async def continue_subagent(
    ctx: RunContext[AgentDependencies],
    session_id: Annotated[
        str,
        Field(
            min_length=1,
            max_length=100,
            description="delegate_task 返回的子 Agent 会话 ID",
        ),
    ],
    task: Annotated[
        str,
        Field(
            min_length=1,
            max_length=4_000,
            description="给同一子 Agent 的验证反馈或后续任务",
        ),
    ],
) -> DelegationResult:
    """把反馈交回同一个子 Agent，并显式传入它之前的消息历史。"""
    runtime = ctx.deps.runtime
    session = runtime.subagents.get(session_id)
    if session is None:
        raise RecoverableToolError(
            f"未知子 Agent 会话 {session_id!r}；请先调用 delegate_task"
        )

    child_context = runtime.agent_contexts[session_id]
    child_result = await session.agent.run(
        task,
        deps=AgentDependencies(runtime, child_context),
        message_history=child_context.message_history,
    )
    child_context.message_history = child_result.all_messages()
    return DelegationResult(
        session_id=session_id,
        template=session.template,
        skill=child_context.skill.metadata.name,
        output=child_result.output,
        model_requests=child_result.usage.requests,
        tool_calls=child_result.usage.tool_calls,
    )


def create_agent(model: Model) -> Agent:
    """创建使用专用文件工具和受限 PowerShell 的执行 Agent。

    ContextRuntime 组合共享工作区上下文与当前 Agent 私有上下文；模型必须以
    工具返回值为工作区事实来源。
    """
    orchestration_tools = RetryToolset(
        FunctionToolset[AgentDependencies](
            tools=[select_skill, delegate_task, continue_subagent],
            max_retries=2,
        )
    )
    template_catalog = "\n".join(
        f"- {template.name}: {template.description}"
        for template in AGENT_TEMPLATES.values()
    )
    agent = Agent(
        model,
        deps_type=AgentDependencies,
        instructions=(
            "你是一个在本地工作区中协助用户完成任务的执行 Agent。"
            "只把已提供的项目指令和 Skill 当作持久上下文。"
            "需要工作区事实时，使用工具列出文件、读取文件或搜索文本。"
            "需要 Git 查询、PowerShell 查询或运行本地测试时，使用受限命令工具。"
            "需要其他 Skill 时先调用 select_skill。"
            "只有任务能被拆成范围明确的子任务时才调用 delegate_task；"
            "该工具会返回 session_id。验证失败或需要补充修改时，"
            "调用 continue_subagent 把反馈交回同一个子 Agent。"
            "简单任务由你直接完成。"
            "只有用户明确要求修改代码或文件时，才允许调用精确文本替换工具。"
            "不得把模型记忆或猜测描述为工作区内容。"
            "当前不能安装依赖、访问网络、执行破坏性命令或提交 Git 变更。"
            "超出能力时应明确说明。\n\n"
            f"## 子 Agent 模板\n{template_catalog}"
        ),
        toolsets=[
            _create_workspace_toolset(can_write=True),
            orchestration_tools,
        ],
    )

    @agent.instructions
    def runtime_context(
        run_context: RunContext[AgentDependencies],
    ) -> str:
        """在当前 Agent 每次运行前组合共享上下文与其私有上下文。"""
        return run_context.deps.runtime.render_agent_instructions(
            run_context.deps.agent_context
        )

    return agent


def run_agent(
    agent: Agent,
    runtime: ContextRuntime,
    agent_context: AgentContext,
    request: str | None = None,
) -> AgentRunResult[str]:
    """使用 Agent 自己的历史运行一轮，并把新历史写回其上下文。"""
    current_request = (request or agent_context.task).strip()
    if not current_request:
        raise ValueError("请求不能为空")
    result = agent.run_sync(
        current_request,
        deps=AgentDependencies(runtime, agent_context),
        message_history=agent_context.message_history,
    )
    agent_context.message_history = result.all_messages()
    return result


def create_model() -> OpenAIChatModel:
    """使用项目 .env 中的 OpenAI 兼容配置创建模型。"""
    load_dotenv()
    provider = OpenAIProvider(
        base_url=os.environ["ABOOK_BASE_URL"],
        api_key=os.environ["ABOOK_API_KEY"],
    )
    return OpenAIChatModel(os.environ["ABOOK_MODEL"], provider=provider)


def main() -> None:
    """组织上下文，运行带工作区工具的执行 Agent，并打印回答。"""
    request = " ".join(sys.argv[1:]).strip()
    if not request:
        request = input("请输入请求：").strip()
    if not request:
        raise SystemExit("请求不能为空")

    experiment_root = Path(__file__).parent
    workspace_context = WorkspaceContextBuilder(
        workspace_root=Path.cwd(),
        skills_root=experiment_root / "skills",
    ).build()
    runtime = ContextRuntime(workspace_context)
    root_context = runtime.create_agent_context("root", request, "general")
    agent = create_agent(create_model())
    result = run_agent(agent, runtime, root_context)
    print(result.output)


if __name__ == "__main__":
    main()
