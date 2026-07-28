"""二阶魔方验证实验的离线领域测试。"""

import unittest

from experiments.cube_validation import CubeState, find_solution, verify_solution
from experiments.cube_validation.flow import CubeToolsets


class CubeValidationTests(unittest.TestCase):
    """保证 Agent 使用的确定性裁判不依赖模型输出。"""

    def test_inverse_formula_restores_cube(self) -> None:
        """每个标准面转动与其逆转组合后必须回到已复原状态。"""
        for face in "URFDLB":
            self.assertTrue(CubeState().apply(f"{face} {face}'").is_solved)
            self.assertTrue(CubeState().apply(f"{face}2 {face}2").is_solved)

    def test_short_search_result_is_verified_by_independent_simulation(self) -> None:
        """搜索结果必须由独立验证入口确认，而非只检查非空文本。"""
        scramble = "R U R' U'"
        solution = find_solution(scramble, max_depth=4)

        self.assertIsNotNone(solution)
        self.assertTrue(verify_solution(scramble, solution or ""))

    def test_invalid_notation_is_rejected(self) -> None:
        """非标准记号不能被静默当作无操作。"""
        with self.assertRaisesRegex(ValueError, "不支持"):
            CubeState().apply("Rw")

    def test_toolsets_are_declared_by_capability(self) -> None:
        """求解与验证角色从同一工厂选择能力，不复制工具清单。"""
        bundle = CubeToolsets("R U")
        solver_names = {tool.name for tool in bundle.create(can_search=True, can_verify=True).tools.values()}
        verifier_names = {tool.name for tool in bundle.create(can_search=False, can_verify=True).tools.values()}

        self.assertEqual({"inspect_state", "search_solution", "check_solution"}, solver_names)
        self.assertEqual({"inspect_state", "check_solution"}, verifier_names)


if __name__ == "__main__":
    unittest.main()
