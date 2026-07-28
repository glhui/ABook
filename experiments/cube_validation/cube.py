"""二阶魔方的角块模型和确定性复原验证。

状态以八个角块的置换与朝向表示，不依赖贴纸颜色或外部魔方库。该表示只接受
标准面转动 ``U R F D L B``、逆转和二次转动，适合作为 Agent 结论的可重复裁判。
"""

from dataclasses import dataclass


_MOVE_TABLES: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "U": ((3, 0, 1, 2, 4, 5, 6, 7), (0, 0, 0, 0, 0, 0, 0, 0)),
    "R": ((4, 1, 2, 0, 7, 5, 6, 3), (2, 0, 0, 1, 1, 0, 0, 2)),
    "F": ((1, 5, 2, 3, 0, 4, 6, 7), (1, 2, 0, 0, 2, 1, 0, 0)),
    "D": ((0, 1, 2, 3, 5, 6, 7, 4), (0, 0, 0, 0, 0, 0, 0, 0)),
    "L": ((0, 2, 6, 3, 4, 1, 5, 7), (0, 1, 2, 0, 0, 2, 1, 0)),
    "B": ((0, 1, 3, 7, 4, 5, 2, 6), (0, 0, 1, 2, 0, 0, 2, 1)),
}
_FACE_MOVES = tuple(_MOVE_TABLES)


@dataclass(frozen=True)
class CubeState:
    """一个合法二阶魔方的角块状态。"""

    permutation: tuple[int, ...] = tuple(range(8))
    orientation: tuple[int, ...] = (0,) * 8

    def __post_init__(self) -> None:
        """拒绝不是八个独立角块或不满足朝向守恒的状态。"""
        if sorted(self.permutation) != list(range(8)):
            raise ValueError("角块置换必须恰好包含 0 到 7")
        if len(self.orientation) != 8 or any(value not in range(3) for value in self.orientation):
            raise ValueError("角块朝向必须由八个 0、1 或 2 组成")
        if sum(self.orientation) % 3:
            raise ValueError("角块朝向之和必须是 3 的倍数")

    @property
    def is_solved(self) -> bool:
        """判断所有角块是否回到位置且朝向正确。"""
        return self == CubeState()

    def apply(self, notation: str) -> "CubeState":
        """按空白分隔的标准记号执行转动，返回新状态。"""
        state = self
        for token in notation.split():
            if not token or token[0] not in _MOVE_TABLES or token[1:] not in ("", "'", "2"):
                raise ValueError(f"不支持的二阶魔方记号：{token!r}")
            turns = 3 if token.endswith("'") else 2 if token.endswith("2") else 1
            for _ in range(turns):
                permutation, orientation = _MOVE_TABLES[token[0]]
                state = CubeState(
                    tuple(state.permutation[index] for index in permutation),
                    tuple((state.orientation[index] + orientation[position]) % 3 for position, index in enumerate(permutation)),
                )
        return state


def verify_solution(scramble: str, solution: str) -> bool:
    """验证候选动作是否能把给定打乱复原；非法记号会抛出 ``ValueError``。"""
    return CubeState().apply(scramble).apply(solution).is_solved


def find_solution(scramble: str, max_depth: int = 7) -> str | None:
    """用有界深度优先搜索求复原动作，适合短公式验证而非通用竞速求解器。"""
    if max_depth < 0:
        raise ValueError("max_depth 不能为负数")
    start = CubeState().apply(scramble)
    if start.is_solved:
        return ""

    def search(state: CubeState, remaining: int, previous_face: str | None, path: list[str]) -> list[str] | None:
        if state.is_solved:
            return path
        if remaining == 0:
            return None
        for face in _FACE_MOVES:
            if face == previous_face:
                continue
            for suffix in ("", "'", "2"):
                result = search(state.apply(face + suffix), remaining - 1, face, [*path, face + suffix])
                if result is not None:
                    return result
        return None

    for depth in range(1, max_depth + 1):
        result = search(start, depth, None, [])
        if result is not None:
            return " ".join(result)
    return None
