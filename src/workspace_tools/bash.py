"""提供不含授权策略的 Bash 命令执行能力。"""

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Protocol


# 表示一条 Bash 命令的完整可序列化执行结果。
@dataclass(frozen=True)
class BashResult:
    command: str
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool


# 抽象 Bash 进程启动方式，使执行器可在测试或 Sandbox 后端中替换实现。
class BashRunner(Protocol):
    def run(self: "BashRunner", command: str, working_directory: Path, timeout_seconds: float | None) -> BashResult:
        ...


# 使用本机平台 Shell 执行命令；该实现不承担命令授权或隔离责任。
class LocalBashRunner:
    def __init__(self: "LocalBashRunner", executable: str | None = None) -> None:
        self._executable = executable

    # 在指定目录运行 Bash，并将进程失败与超时转换为结构化结果。
    def run(self: "LocalBashRunner", command: str, working_directory: Path, timeout_seconds: float | None) -> BashResult:
        try:
            completed_process = subprocess.run(
                self._command_arguments(command),
                cwd=working_directory,
                capture_output=True,
                check=False,
                encoding="utf-8",
                errors="replace",
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            return BashResult(
                command=command,
                stdout=self._as_text(error.stdout),
                stderr=self._as_text(error.stderr),
                exit_code=None,
                timed_out=True,
            )
        except OSError as error:
            return BashResult(command=command, stdout="", stderr=str(error), exit_code=None, timed_out=False)
        return BashResult(
            command=command,
            stdout=completed_process.stdout,
            stderr=completed_process.stderr,
            exit_code=completed_process.returncode,
            timed_out=False,
        )

    # 在 Unix 使用 Bash，在 Windows 使用 PowerShell；显式 executable 始终优先。
    def _command_arguments(self: "LocalBashRunner", command: str) -> list[str]:
        if self._executable is not None:
            return [self._executable, "-lc", command]
        if os.name == "nt":
            return ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command]
        return ["bash", "-lc", command]

    # 将 TimeoutExpired 在不同 Python 配置下返回的字节或文本统一为文本。
    def _as_text(self: "LocalBashRunner", value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value


# 将命令执行绑定到默认工作区目录，具体授权由外层执行器处理。
class WorkspaceBashTool:
    def __init__(self: "WorkspaceBashTool", workspace_root: Path, runner: BashRunner | None = None) -> None:
        self._workspace_root = workspace_root.resolve()
        self._runner = runner if runner is not None else LocalBashRunner()

    # 在工作区根目录运行 Bash 并返回标准输出、标准错误和退出状态。
    def run(self: "WorkspaceBashTool", command: str, timeout_seconds: float | None = None) -> BashResult:
        return self._runner.run(command, self._workspace_root, timeout_seconds)
