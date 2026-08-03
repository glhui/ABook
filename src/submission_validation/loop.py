"""实现有限次数的宿主验证与修复闭环。"""

from collections.abc import Awaitable, Callable

from submission_validation.models import ValidationResult
from submission_validation.validator import SubmissionValidator


RepairCallback = Callable[[ValidationResult, int], Awaitable[None]]


# 在每次失败后把脱敏诊断交给修复回调；达到上限时保留最后一次真实结果。
async def validate_with_repairs(
    validator: SubmissionValidator,
    repair: RepairCallback,
    max_attempts: int,
) -> ValidationResult:
    if max_attempts < 1:
        raise ValueError("max_attempts 必须至少为 1。")

    for attempt in range(1, max_attempts + 1):
        result = validator.validate()
        if result.passed or attempt == max_attempts:
            return result
        await repair(result, attempt)
    raise AssertionError("有限重试循环必须在最大尝试次数内返回。")
