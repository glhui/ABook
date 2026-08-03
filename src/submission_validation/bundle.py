"""加载测试 Agent 生成的私有 JSON 用例包。"""

import json
from pathlib import Path

from submission_validation.models import JsonValue, PrivateTestBundle, PrivateTestCase


# 从私有目录加载严格的测试包格式，拒绝含糊或不可 JSON 序列化的测试数据。
def load_private_test_bundle(path: Path) -> PrivateTestBundle:
    source_value = _read_json(path)
    if not isinstance(source_value, dict):
        raise ValueError("测试包根节点必须是对象。")
    cases_value = source_value.get("cases")
    if not isinstance(cases_value, list) or not cases_value:
        raise ValueError("测试包必须包含非空 cases 数组。")

    cases: list[PrivateTestCase] = []
    case_ids: set[str] = set()
    for item in cases_value:
        if not isinstance(item, dict):
            raise ValueError("每个测试用例必须是对象。")
        case_id = item.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("每个测试用例必须包含非空字符串 id。")
        if case_id in case_ids:
            raise ValueError(f"测试用例 id 重复：{case_id}")
        if "input" not in item or "expected" not in item:
            raise ValueError(f"测试用例 {case_id} 必须包含 input 和 expected。")
        cases.append(
            PrivateTestCase(
                case_id=case_id,
                input_value=_as_json_value(item["input"]),
                expected_value=_as_json_value(item["expected"]),
            )
        )
        case_ids.add(case_id)
    return PrivateTestBundle(cases=tuple(cases))


# 将 json 模块的动态边界值递归缩窄为项目使用的 JSON 类型。
def _read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as source_file:
        return json.load(source_file)


# 验证每一层对象都满足 JSON 值约束，避免运行时测试包混入不可序列化对象。
def _as_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_as_json_value(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON 对象键必须是字符串。")
        return {key: _as_json_value(item) for key, item in value.items()}
    raise ValueError("测试包只能包含 JSON 值。")
