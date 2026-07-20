"""三角形几何计算工具。

当前提供基于海伦公式的三角形面积计算，附带严格的输入验证。
所有函数为纯计算函数，不依赖外部存储或模型接口。
"""

import math


def triangle_area(a: float, b: float, c: float) -> float:
    """计算并返回三角形的面积（海伦公式）。

    三条边必须满足以下条件才会被视为有效三角形：
    1. 所有边长均为正数。
    2. 任意两边之和大于第三边（三角不等式）。

    Args:
        a: 第一条边长，必须为正数。
        b: 第二条边长，必须为正数。
        c: 第三条边长，必须为正数。

    Returns:
        三角形的面积，使用 ``math.sqrt`` 计算。

    Raises:
        ValueError: 当任意边不是正数时，附带 ``"[非正数]"`` 标签说明。
        ValueError: 当三边不满足三角不等式时，附带 ``"[三角不等式]"`` 标签说明。

    示例:

    >>> triangle_area(3, 4, 5)
    6.0
    >>> triangle_area(5, 5, 6)
    12.0
    """
    _validate_triangle(a, b, c)

    semiperimeter = (a + b + c) / 2.0
    area = math.sqrt(
        semiperimeter
        * (semiperimeter - a)
        * (semiperimeter - b)
        * (semiperimeter - c)
    )
    return area


def _validate_triangle(a: float, b: float, c: float) -> None:
    """验证三边能否构成有效三角形（纯验证，无副作用）。

    检查顺序：先检查非正数，再检查三角不等式，确保调用方能通过
    异常消息准确判断失败原因。

    Args:
        a: 第一条边长。
        b: 第二条边长。
        c: 第三条边长。

    Raises:
        ValueError: 任意边非正数时抛出，消息包含 ``"[非正数]"``。
        ValueError: 三角不等式不成立时抛出，消息包含 ``"[三角不等式]"``。
        TypeError: 输入类型无法转换为 ``float`` 时抛出；调用方可借此在
            边界测试中确认类型校验行为。
    """
    # 尝试转换为 float，确保数字类型可接受；int 和 float 可正常转换，
    # 字符串或 None 等非法类型会触发 TypeError。
    try:
        a_f = float(a)
        b_f = float(b)
        c_f = float(c)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"边长必须为数值类型（int 或 float），收到 {type(a).__name__}/"
            f"{type(b).__name__}/{type(c).__name__}"
        ) from exc

    # 检查是否所有边长都为正数
    if a_f <= 0 or b_f <= 0 or c_f <= 0:
        non_positive = []
        if a_f <= 0:
            non_positive.append(f"a={a_f}")
        if b_f <= 0:
            non_positive.append(f"b={b_f}")
        if c_f <= 0:
            non_positive.append(f"c={c_f}")
        raise ValueError(
            f"三角形边长必须为正数，收到非正数：{', '.join(non_positive)} "
            f"[非正数]"
        )

    # 检查三角不等式：任意两边之和大于第三边
    if a_f + b_f <= c_f:
        raise ValueError(
            f"a + b <= c（{a_f} + {b_f} <= {c_f}），无法构成三角形 [三角不等式]"
        )
    if a_f + c_f <= b_f:
        raise ValueError(
            f"a + c <= b（{a_f} + {c_f} <= {b_f}），无法构成三角形 [三角不等式]"
        )
    if b_f + c_f <= a_f:
        raise ValueError(
            f"b + c <= a（{b_f} + {c_f} <= {a_f}），无法构成三角形 [三角不等式]"
        )
