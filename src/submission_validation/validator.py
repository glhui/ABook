"""使用私有测试包验证标准输入输出提交。"""

import json
from pathlib import Path
import subprocess
import sys

from submission_validation.models import JsonValue, PrivateTestBundle, ValidationFailure, ValidationResult, ValidationStage


DEFAULT_TIMEOUT_SECONDS = 5.0
MAX_DIAGNOSTIC_CHARACTERS = 1_000


# 宿主验证器通过子进程编译并调用提交程序，测试包不会写入提交工作区。
class SubmissionValidator:
    def __init__(
        self: "SubmissionValidator",
        submission_path: Path,
        test_bundle: PrivateTestBundle,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._submission_path = submission_path.resolve()
        self._test_bundle = test_bundle
        self._timeout_seconds = timeout_seconds

    # 先验证 Python 语法，再按私有用例执行 stdin/stdout 协议。
    def validate(self: "SubmissionValidator") -> ValidationResult:
        compilation_failure = self._compile_submission()
        if compilation_failure is not None:
            return ValidationResult(passed=False, failure=compilation_failure)

        for case in self._test_bundle.cases:
            failure = self._validate_case(case.case_id, case.input_value, case.expected_value)
            if failure is not None:
                return ValidationResult(passed=False, failure=failure)
        return ValidationResult(passed=True, failure=None)

    # 调用 Python 编译器以获得与实际执行一致的语法错误信息。
    def _compile_submission(self: "SubmissionValidator") -> ValidationFailure | None:
        compile_script = (
            "from pathlib import Path; import sys; "
            "compile(Path(sys.argv[1]).read_text(encoding='utf-8'), sys.argv[1], 'exec')"
        )
        result = self._run_process(
            [sys.executable, "-c", compile_script, str(self._submission_path)],
            input_text=None,
        )
        if result is None:
            return ValidationFailure(
                stage=ValidationStage.COMPILE,
                case_id=None,
                summary="编译命令超时。",
                detail=f"超过 {self._timeout_seconds} 秒仍未完成。",
            )
        if result.returncode == 0:
            return None
        return ValidationFailure(
            stage=ValidationStage.COMPILE,
            case_id=None,
            summary="提交程序无法编译。",
            detail=self._diagnostic(result.stderr or result.stdout),
        )

    # 对单个私有用例隔离启动提交程序，并检查 JSON 输出与私有期望值完全一致。
    def _validate_case(
        self: "SubmissionValidator",
        case_id: str,
        input_value: JsonValue,
        expected_value: JsonValue,
    ) -> ValidationFailure | None:
        input_text = json.dumps(input_value, ensure_ascii=False)
        result = self._run_process([sys.executable, str(self._submission_path)], input_text=input_text)
        if result is None:
            return ValidationFailure(
                stage=ValidationStage.EXECUTION,
                case_id=case_id,
                summary="提交程序执行超时。",
                detail=f"超过 {self._timeout_seconds} 秒仍未完成。",
            )
        if result.returncode != 0:
            return ValidationFailure(
                stage=ValidationStage.EXECUTION,
                case_id=case_id,
                summary=f"提交程序以退出码 {result.returncode} 结束。",
                detail=self._diagnostic(result.stderr),
            )
        try:
            actual_value = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            return ValidationFailure(
                stage=ValidationStage.OUTPUT,
                case_id=case_id,
                summary="stdout 不是单个合法 JSON 值。",
                detail=self._diagnostic(str(error)),
            )
        if actual_value != expected_value:
            return ValidationFailure(
                stage=ValidationStage.ASSERTION,
                case_id=case_id,
                summary="输出值不符合私有验收条件。",
                detail="请检查该边界条件的输入解析、计算逻辑和 JSON 输出格式。",
            )
        return None

    # 所有被测进程均从提交文件所在目录运行，不产生宿主日志文件。
    def _run_process(
        self: "SubmissionValidator",
        arguments: list[str],
        input_text: str | None,
    ) -> subprocess.CompletedProcess[str] | None:
        try:
            return subprocess.run(
                arguments,
                cwd=self._submission_path.parent,
                input=input_text,
                capture_output=True,
                check=False,
                encoding="utf-8",
                errors="replace",
                text=True,
                timeout=self._timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return None

    # 截断编译器和被测程序诊断，避免异常输出污染交互上下文。
    def _diagnostic(self: "SubmissionValidator", value: str) -> str:
        compact_value = value.strip()
        if not compact_value:
            return "未输出错误详情。"
        return compact_value[:MAX_DIAGNOSTIC_CHARACTERS]
