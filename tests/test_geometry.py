"""测试三角形面积计算函数，覆盖正常路径、边界、无效输入和异常类型。"""

import math
import unittest

from src.abook_agent.geometry import triangle_area, _validate_triangle


class TestTriangleArea(unittest.TestCase):
    """验证 triangle_area 在不同输入下的正确性和稳定性。"""

    # ── 正常路径 ──────────────────────────────────────────────

    def test_known_right_triangle(self) -> None:
        """3-4-5 直角三角形的面积应为 6.0。"""
        self.assertAlmostEqual(triangle_area(3, 4, 5), 6.0)

    def test_known_isosceles_triangle(self) -> None:
        """已知底 6 腰 5 的等腰三角形，面积应为 12.0。

        使用勾股定理推算高：sqrt(5² - (6/2)²) = 4，底 6 * 高 4 / 2 = 12。
        """
        self.assertAlmostEqual(triangle_area(5, 5, 6), 12.0)

    def test_equilateral_triangle(self) -> None:
        """边长为 1 的等边三角形，面积 = sqrt(3) / 4。"""
        expected = math.sqrt(3) / 4.0
        self.assertAlmostEqual(triangle_area(1, 1, 1), expected)

    def test_large_triangle(self) -> None:
        """大数值三角形，确保浮点计算不会溢出或异常。"""
        # 边长 1000000 的等边三角形
        expected = math.sqrt(3) / 4.0 * 1_000_000 ** 2
        self.assertAlmostEqual(triangle_area(1_000_000, 1_000_000, 1_000_000),
                               expected, places=0)

    def test_small_triangle(self) -> None:
        """极小边长，验证浮点精度未导致信息丢失。"""
        # 边长为 0.001 的等边三角形
        expected = math.sqrt(3) / 4.0 * 0.001 ** 2
        self.assertAlmostEqual(triangle_area(0.001, 0.001, 0.001), expected)

    def test_float_inputs(self) -> None:
        """浮点边长直接传入，应正常计算。"""
        area = triangle_area(3.0, 4.0, 5.0)
        self.assertAlmostEqual(area, 6.0)

    def test_mixed_int_float(self) -> None:
        """混合 int 和 float 类型，验证 float() 转换兼容性。"""
        area = triangle_area(3, 4.0, 5)
        self.assertAlmostEqual(area, 6.0)

    # ── 边界 ──────────────────────────────────────────────────

    def test_extremely_small_positive(self) -> None:
        """极小的正边长（接近浮点数下限），应能被识别为合法三角形。"""
        # 边长为 1e-10 的等边三角形
        tiny = 1e-10
        # 只要三角不等式成立即可，面积会非常接近 0
        area = triangle_area(tiny, tiny, tiny)
        self.assertGreater(area, 0.0)

    def test_degenerate_triangle_inequality_equal(self) -> None:
        """两边之和等于第三边 → 退化三角形，应拒绝。"""
        with self.assertRaises(ValueError) as context:
            triangle_area(1, 2, 3)
        self.assertIn("[三角不等式]", str(context.exception))

    def test_just_over_triangle_inequality(self) -> None:
        """略微超过三角不等式边界，应可构成三角形。"""
        # 2 + 3 = 5 刚好退化，2 + 3 > 4.9999999 则构成极扁平三角形
        area = triangle_area(2, 3, 4.9999999)
        self.assertGreater(area, 0.0)

    def test_very_flat_triangle(self) -> None:
        """极扁平三角形（两边之和略大于第三边），面积应接近 0 但为正。"""
        area = triangle_area(1, 1, 1.9999999)
        self.assertGreater(area, 0.0)
        self.assertLess(area, 1e-3)

    # ── 非正数 ────────────────────────────────────────────────

    def test_zero_side(self) -> None:
        """某条边为 0，应抛出 ValueError 并包含 [非正数]。"""
        with self.assertRaises(ValueError) as context:
            triangle_area(0, 4, 5)
        self.assertIn("[非正数]", str(context.exception))

    def test_negative_side(self) -> None:
        """某条边为负数，应抛出 ValueError 并包含 [非正数]。"""
        with self.assertRaises(ValueError) as context:
            triangle_area(-3, 4, 5)
        self.assertIn("[非正数]", str(context.exception))

    def test_multiple_non_positive(self) -> None:
        """多条边为 0 或负数，应检测并报告。"""
        with self.assertRaises(ValueError) as context:
            triangle_area(0, -1, 5)
        self.assertIn("[非正数]", str(context.exception))

    def test_all_non_positive(self) -> None:
        """所有边均非法，应抛出。"""
        with self.assertRaises(ValueError) as context:
            triangle_area(-1, -2, -3)
        self.assertIn("[非正数]", str(context.exception))

    # ── 三角不等式失效 ────────────────────────────────────────

    def test_sum_two_sides_less_than_third(self) -> None:
        """两边之和小于第三边，应拒绝并报告 [三角不等式]。"""
        with self.assertRaises(ValueError) as context:
            triangle_area(1, 2, 4)
        self.assertIn("[三角不等式]", str(context.exception))

    def test_large_side_dominates(self) -> None:
        """一条边远大于另外两条之和，应拒绝。"""
        with self.assertRaises(ValueError) as context:
            triangle_area(1, 2, 100)
        self.assertIn("[三角不等式]", str(context.exception))

    def test_inequality_violation_second_pair(self) -> None:
        """不同的两边组合违反三角不等式（a + c <= b）。"""
        with self.assertRaises(ValueError) as context:
            triangle_area(2, 5, 3)
        self.assertIn("[三角不等式]", str(context.exception))

    def test_inequality_violation_third_pair(self) -> None:
        """第三种违反组合（b + c <= a）。"""
        with self.assertRaises(ValueError) as context:
            triangle_area(5, 2, 3)
        self.assertIn("[三角不等式]", str(context.exception))

    # ── 非法类型 ──────────────────────────────────────────────

    def test_string_input(self) -> None:
        """字符串作为边长，应抛出 TypeError。"""
        with self.assertRaises(TypeError):
            triangle_area("3", 4, 5)

    def test_none_input(self) -> None:
        """None 作为边长，应抛出 TypeError。"""
        with self.assertRaises(TypeError):
            triangle_area(None, 4, 5)

    def test_list_input(self) -> None:
        """列表作为边长，应抛出 TypeError。"""
        with self.assertRaises(TypeError):
            triangle_area([3], 4, 5)

    # ── 浮点与边界精度 ────────────────────────────────────────

    def test_large_values_no_overflow(self) -> None:
        """极大数值不应导致 OverflowError 或 nan。"""
        # 边长为 1e50，等边三角形（乘积约 10^199 < float 上限 1.8e308）
        area = triangle_area(1e50, 1e50, 1e50)
        self.assertTrue(math.isfinite(area))
        self.assertGreater(area, 0.0)

    def test_almost_degenerate_precision(self) -> None:
        """非常接近退化三角形的输入，验证浮点健壮性。"""
        # 2 + 3 = 5 是退化边界，2 + 3 + tiny_eps 应能构成极小面积
        eps = 1e-12
        area = triangle_area(2, 3, 5 - eps)
        self.assertGreater(area, 0.0)

    def test_reordered_sides(self) -> None:
        """相同三边不同顺序应得到相同面积。"""
        area1 = triangle_area(6, 8, 10)
        area2 = triangle_area(8, 10, 6)
        area3 = triangle_area(10, 6, 8)
        self.assertAlmostEqual(area1, area2)
        self.assertAlmostEqual(area2, area3)


class TestValidateTriangle(unittest.TestCase):
    """直接测试 _validate_triangle 的内部验证逻辑。"""

    def test_valid_passes(self) -> None:
        """合法三角形不应抛出异常。"""
        _validate_triangle(3, 4, 5)  # 不抛出即为通过

    def test_non_positive(self) -> None:
        """非正数应触发 ValueError 并包含标签。"""
        with self.assertRaises(ValueError) as context:
            _validate_triangle(0, 4, 5)
        self.assertIn("[非正数]", str(context.exception))

    def test_inequality_violation(self) -> None:
        """三角不等式被违反应触发 ValueError 并包含标签。"""
        with self.assertRaises(ValueError) as context:
            _validate_triangle(1, 2, 4)
        self.assertIn("[三角不等式]", str(context.exception))

    def test_type_error(self) -> None:
        """非法类型应触发 TypeError。"""
        with self.assertRaises(TypeError):
            _validate_triangle("abc", 4, 5)


if __name__ == "__main__":
    unittest.main()
