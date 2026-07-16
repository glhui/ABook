"""提供有界工作区文件操作、PowerShell 执行和统一工具重试边界。"""

import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.toolsets import FunctionToolset, WrapperToolset
from pydantic_ai.toolsets.abstract import ToolsetTool

from .context import AgentDependencies


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


class CommandResult(BaseModel):
    """受限 PowerShell 命令的稳定执行结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    command: str
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool


class RecoverableToolError(Exception):
    """表示模型可通过修改参数或改用其他工具修正的失败。"""


class RetryToolset(WrapperToolset[AgentDependencies]):
    """在单一边界把业务可恢复错误转换为 PydanticAI 重试信号。"""

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDependencies],
        tool: ToolsetTool[AgentDependencies],
    ) -> Any:
        """执行被包装工具，并让模型观察可修正错误后重新决策。"""
        try:
            return await self.wrapped.call_tool(name, tool_args, ctx, tool)
        except RecoverableToolError as error:
            raise ModelRetry(str(error)) from error


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
    relative_path = resolved_path.relative_to(
        ctx.deps.workspace_root
    ).as_posix()
    ctx.deps.runtime.task_state.record_modified_file(relative_path)
    return (
        f"Replaced {actual_replacements} occurrence(s) in "
        f"{relative_path}."
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


def _is_validation_command(arguments: list[str]) -> bool:
    """判断允许的 Python 命令是否属于测试、编译或依赖检查。"""
    executable_name = Path(arguments[0]).name.casefold()
    normalized_arguments = [argument.casefold() for argument in arguments]
    return executable_name in {"python", "python.exe"} and (
        normalized_arguments[1:3] in (
            ["-m", "unittest"],
            ["-m", "compileall"],
        )
        or normalized_arguments[1:4] == ["-m", "pip", "check"]
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
    arguments = _validate_powershell_command(command)
    is_validation = _is_validation_command(arguments)
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
        if is_validation:
            ctx.deps.runtime.task_state.record_validation(
                command=command,
                exit_code=None,
                timed_out=True,
            )
        return CommandResult(
            command=command,
            exit_code=None,
            stdout=_truncate_command_output(error.stdout),
            stderr=_truncate_command_output(error.stderr),
            timed_out=True,
        )

    if is_validation:
        ctx.deps.runtime.task_state.record_validation(
            command=command,
            exit_code=completed_process.returncode,
            timed_out=False,
        )
    return CommandResult(
        command=command,
        exit_code=completed_process.returncode,
        stdout=_truncate_command_output(completed_process.stdout),
        stderr=_truncate_command_output(completed_process.stderr),
        timed_out=False,
    )


def create_workspace_toolset(can_write: bool) -> RetryToolset:
    """为 Agent 创建统一重试边界下的工作区工具集。"""
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
