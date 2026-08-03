"""提供接受绝对路径的基础文本文件工具。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic import Field
from pydantic_ai.tools import Tool


AbsoluteFilePath = Annotated[str, Field(description="UTF-8 文本文件的绝对路径。")]
StartLine = Annotated[int, Field(description="从 1 开始的首行行号。")]
EndLine = Annotated[int | None, Field(description="从 1 开始的末行行号，省略时读取至文件末尾。")]


# 表示一次有界文本读取的结果，保留原始文本便于后续精确编辑。
@dataclass(frozen=True)
class ReadFileResult:
    path: str
    exists: bool
    error: str | None
    content: str
    start_line: int
    end_line: int
    total_lines: int
    truncated: bool


# 表示一次整文件写入的成功结果。
@dataclass(frozen=True)
class WriteFileResult:
    path: str
    bytes_written: int


# 表示一次精确文本替换的成功结果。
@dataclass(frozen=True)
class EditFileResult:
    path: str
    replacements: int


# 提供无授权策略的绝对路径文件操作；访问范围由调用方的执行策略负责。
class WorkspaceFileTools:
    # 按行读取 UTF-8 文本，默认返回全部内容并提供总行数以支持后续分段读取。
    def read_file(
        self: "WorkspaceFileTools",
        path: AbsoluteFilePath,
        start_line: StartLine = 1,
        end_line: EndLine = None,
    ) -> ReadFileResult:
        target_path = self._resolve_path(path)
        content = self._read_text(target_path)
        lines = content.splitlines(keepends=True)
        total_lines = len(lines)
        first_line = max(start_line, 1)
        last_line = max(end_line, first_line) if end_line is not None else total_lines
        selected_lines = lines[first_line - 1 : last_line]
        selected_end_line = min(last_line, total_lines)
        return ReadFileResult(
            path=str(target_path),
            exists=True,
            error=None,
            content="".join(selected_lines),
            start_line=first_line,
            end_line=selected_end_line,
            total_lines=total_lines,
            truncated=selected_end_line < total_lines,
        )

    # 写入 UTF-8 文本文件；调用方负责决定是否允许覆盖和创建父目录。
    def write_file(self: "WorkspaceFileTools", path: AbsoluteFilePath, content: str) -> WriteFileResult:
        target_path = self._resolve_path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_text(target_path, content)
        return WriteFileResult(path=str(target_path), bytes_written=len(content.encode("utf-8")))

    # 将所有匹配的旧文本替换为新文本，并返回实际替换次数供调用方判断结果。
    def replace_text(
        self: "WorkspaceFileTools",
        path: AbsoluteFilePath,
        old_text: str,
        new_text: str,
    ) -> EditFileResult:
        target_path = self._resolve_path(path)
        content = self._read_text(target_path)
        if not old_text:
            return EditFileResult(path=str(target_path), replacements=0)
        replacements = content.count(old_text)
        if replacements:
            self._write_text(target_path, content.replace(old_text, new_text))
        return EditFileResult(path=str(target_path), replacements=replacements)

    # 返回带模型可见描述和参数 schema 的 Pydantic AI 工具，供示例直接注册。
    def as_pydantic_tools(self: "WorkspaceFileTools") -> list[Tool[None]]:
        return [
            Tool(
                self.read_file,
                description="读取已获授权的 UTF-8 文本文件绝对路径，可按行范围读取。",
            ),
            Tool(
                self.write_file,
                description="向已获授权的 UTF-8 文本文件绝对路径写入完整内容。",
            ),
            Tool(
                self.replace_text,
                description="将已获授权的 UTF-8 文本文件绝对路径中的所有旧文本替换为新文本。",
            ),
        ]

    # 拒绝相对路径，避免进程当前目录意外成为隐式的访问边界。
    def _resolve_path(self: "WorkspaceFileTools", path: str) -> Path:
        target_path = Path(path)
        if not target_path.is_absolute():
            raise ValueError("文件路径必须是绝对路径。")
        return target_path.resolve()

    # 使用 UTF-8 和保留换行符的方式读取文本，底层文件错误直接交由调用方处理。
    def _read_text(self: "WorkspaceFileTools", path: Path) -> str:
        with path.open("r", encoding="utf-8", newline="") as source_file:
            return source_file.read()

    # 直接写入目标文件；原子性和并发写入策略由更高层协调。
    def _write_text(self: "WorkspaceFileTools", path: Path, content: str) -> None:
        with path.open("w", encoding="utf-8", newline="") as target_file:
            target_file.write(content)


# 创建供多个示例复用的绝对路径文件工具集合。
def create_workspace_file_tools() -> WorkspaceFileTools:
    return WorkspaceFileTools()
