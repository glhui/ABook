"""定义私有测试包和宿主验证的结构化结果。"""

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


# 表示测试 Agent 编写、但实现 Agent 不可读取的一条 JSON 标准输入输出用例。
@dataclass(frozen=True)
class PrivateTestCase:
    case_id: str
    input_value: JsonValue
    expected_value: JsonValue


# 表示由宿主保存的私有测试包，验证器是唯一允许读取该包和提交代码的组件。
@dataclass(frozen=True)
class PrivateTestBundle:
    cases: tuple[PrivateTestCase, ...]


# 区分编译、进程和行为错误，使修复反馈不需要暴露私有 oracle。
class ValidationStage(StrEnum):
    COMPILE = "compile"
    EXECUTION = "execution"
    OUTPUT = "output"
    ASSERTION = "assertion"


# 记录一次失败的有限诊断；详情经过截断，避免日志无限增长或泄露测试答案。
@dataclass(frozen=True)
class ValidationFailure:
    stage: ValidationStage
    case_id: str | None
    summary: str
    detail: str


# 表示一轮私有验证的结果，供宿主决定成功、反馈修复或停止重试。
@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    failure: ValidationFailure | None

    # 只生成足够修复的摘要，绝不回传 expected_value 或私有测试源码。
    def feedback(self: "ValidationResult") -> str:
        if self.passed:
            return "私有验证通过。"
        if self.failure is None:
            return "私有验证失败，但未获得可用诊断。"
        case_part = f"；用例：{self.failure.case_id}" if self.failure.case_id is not None else ""
        return f"验证失败（阶段：{self.failure.stage.value}{case_part}）：{self.failure.summary}\n{self.failure.detail}"
