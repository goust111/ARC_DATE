# Combo variant of 45a5af55 (index 14)
# combo_spec: D9/0+D13/0
# Name: 分隔行编码间隙两阶段嵌套

from __future__ import annotations

import numpy as np

"""
Design analysis for the combo variant
=====================================

a) 具体化与行为追踪
   - 原题输入是若干个全宽纯色横条，自底向上读取；最底条变成中心实心方块，其余各条按
     自身高度作为厚度向外包裹同色方框。
   - 融合后，输入仍是全宽横条，但任意相邻彩条之间都插入 1 或 2 行背景色 `0` 分隔。
     这些 `0` 行不是新图层，而是“层间距离编码”。
   - 第一阶段仍先忽略分隔，只根据彩条高度构造基础嵌套关系。第二阶段再读取每段分隔
     的真实高度 `g`，在对应两层之间插入宽度恰为 `g` 的背景色方环。
   - 小例子（底->顶）：
       条纹: `(2,h=2)`, `(5,h=1)`, `(7,h=2)`
       分隔高度: `2` 与 `5` 之间是 2，`5` 与 `7` 之间是 1
     原题:
       1. `2` -> 2x2 中心块
       2. `5` 贴着包 1 层 -> 4x4
       3. `7` 再包 2 层 -> 8x8
     融合后:
       1. 先得到同样的基础层级
       2. 在 `2` 与 `5` 之间先留 2 像素背景环，再画厚度 1 的 `5` 框
       3. 在 `5` 与 `7` 之间再留 1 像素背景环，再画厚度 2 的 `7` 框
     差异:
       输出边长从 `8` 变成 `14`，且不同层对之间出现不同宽度的背景间隙。

b) 复杂度校准
   - 只使用两种分隔高度 `1` 和 `2`，避免自由度过多，同时让“间隙宽度受输入控制”足够
     明显。
   - 生成器强制每个样例里同时出现 1 行分隔和 2 行分隔，这样人类能在单张图里直接对比
     “窄间隙”和“宽间隙”，不会误读成固定装饰边。
   - 条纹层数固定在 3-6 层，既能展示多次编码，又能把输出尺寸稳定控制在 30 以内。

c) 隐式假设审查
   - 原 `main()` 假设每一行都属于有效彩条；融合后 `0` 行只是分隔，不能当作颜色层。
   - 原逻辑默认新外框总是紧贴旧结构：`new_n = cur.shape[0] + 2 * h`。现在这个假设失效，
     因为层与层之间可能还要再加 `2 * gap_height` 的边长增量。
   - 原逻辑从底部向上读 run 时，只需要 `(color, height)`；融合后还必须额外保留每对彩条
     之间的 `(gap_height)`，否则第二阶段无法恢复真实间距。
   - 因此新实现分成三步：
       1. 解析底->顶彩条与对应分隔高度
       2. 计算不带间隙的基础内外边长
       3. 将累计分隔宽度平移到各外层，最后统一渲染

d) 放置合法性
   - 生成器只构造“彩条 run / 背景 run / 彩条 run ...”的垂直交替结构，且每行都是整行
     纯色，因此不会出现局部重叠或歧义。
   - 输出始终是同心正方形。总边长为
     `h0 + 2*sum(外层厚度) + 2*sum(分隔高度)`，生成器显式检查其不超过 30。
   - 输入高度 `sum(条纹高度) + sum(分隔高度)` 也限制在 30 以内，避免越界。

e) 视觉可观察性
   - 观察输入时，人类能直接看到整行 `0` 作为层间分隔；观察输出时，能看到对应层间出现
     等宽的黑色空环。
   - 因为样例中同时包含 1 行和 2 行分隔，解题者可以从 2-3 个样例中归纳“分隔越厚，输出
     间隙越宽”，而不是只看见“有无间隙”。
   - 规则完全由可见的空间厚度决定，不依赖颜色数值大小等不可视因素。

f) 可视化整洁度
   - 输入仍是单一方向的整齐水平条纹，新增元素只有全宽背景分隔行。
   - 输出仍是单中心的同心正方形，新增元素只有背景空环，不会引入交叉、遮挡或不规则边界。

g) 变化的可推断性
   - 没有颜色映射；彩条在输出中保持原色，`0` 固定代表背景与间隙。
   - 颜色只负责区分层身份，真正的变化由“条纹高度”和“分隔高度”这两个可见几何量共同
     决定，因此从少量样例中可推断。

h) 新 solver 假设与 generator 保证
   - solver 假设每一行都是全宽单色，且背景色只用 `0`。
   - solver 允许没有分隔的边界，但 generator 实际上总会在每对相邻彩条之间放置 1 或 2
     行背景，从而稳定触发融合行为。
   - generator 还保证至少出现一个 1 行分隔和一个 2 行分隔，使“高度编码间隙宽度”不是
     退化成固定常量的伪变化。
"""


BG = 0


def _parse_structure(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[int]]:
    """
    Parse full-width monochrome runs into bottom->top stripes and gap heights.

    Returns:
        stripes: [(color, height), ...] ordered inner->outer
        gap_heights: [g0, g1, ...] where gi is the separator height between
            stripes[i] and stripes[i+1]. Missing separators are encoded as 0.
    """
    g = np.array(grid, dtype=int)
    height, _width = g.shape

    row_colors: list[int] = []
    for r in range(height):
        col = int(g[r, 0])
        assert bool((g[r] == col).all())
        row_colors.append(col)

    runs: list[tuple[int, int]] = []
    r = 0
    while r < height:
        col = row_colors[r]
        start = r
        r += 1
        while r < height and row_colors[r] == col:
            r += 1
        runs.append((int(col), int(r - start)))

    while runs and runs[0][0] == BG:
        runs.pop(0)
    while runs and runs[-1][0] == BG:
        runs.pop()
    assert runs

    bottom_to_top = list(reversed(runs))
    assert bottom_to_top[0][0] != BG

    stripes: list[tuple[int, int]] = []
    gap_heights: list[int] = []
    idx = 0
    while idx < len(bottom_to_top):
        col, stripe_h = bottom_to_top[idx]
        assert col != BG
        stripes.append((int(col), int(stripe_h)))
        idx += 1
        if idx >= len(bottom_to_top):
            break
        if bottom_to_top[idx][0] == BG:
            gap_heights.append(int(bottom_to_top[idx][1]))
            idx += 1
            assert idx < len(bottom_to_top)
        else:
            gap_heights.append(0)

    assert len(stripes) >= 2
    assert len(gap_heights) == len(stripes) - 1
    return stripes, gap_heights


def _base_layer_specs(stripes: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """
    Stage 1: ignore separators and compute nested-square geometry.

    Each layer spec is `(color, inner_size, outer_size)` from inner->outer.
    """
    center_color, center_size = stripes[0]
    specs: list[tuple[int, int, int]] = [(int(center_color), 0, int(center_size))]
    cur_size = int(center_size)
    for col, stripe_h in stripes[1:]:
        inner = cur_size
        cur_size = int(cur_size + 2 * int(stripe_h))
        specs.append((int(col), int(inner), int(cur_size)))
    return specs


def _insert_gap_widths(
    base_specs: list[tuple[int, int, int]], gap_heights: list[int]
) -> list[tuple[int, int, int]]:
    """
    Stage 2: convert input separator heights into output background ring widths.

    A gap of height g between layers i and i+1 contributes `2*g` to the side
    length of layer i+1 and every outer layer.
    """
    decorated: list[tuple[int, int, int]] = []
    extra = 0
    for idx, (col, inner, outer) in enumerate(base_specs):
        if idx > 0:
            extra += 2 * int(gap_heights[idx - 1])
        decorated.append((int(col), int(inner + extra), int(outer + extra)))
    return decorated


def _render_layers(specs: list[tuple[int, int, int]]) -> np.ndarray:
    """Render centered square annuli onto a background canvas."""
    final_side = int(specs[-1][2])
    out = np.full((final_side, final_side), BG, dtype=int)

    for col, inner, outer in reversed(specs):
        outer = int(outer)
        inner = int(inner)
        outer_margin = (final_side - outer) // 2
        out[outer_margin : outer_margin + outer, outer_margin : outer_margin + outer] = int(col)
        if inner > 0:
            inner_margin = (final_side - inner) // 2
            out[inner_margin : inner_margin + inner, inner_margin : inner_margin + inner] = BG

    return out


def _build_output(
    stripes: list[tuple[int, int]], gap_heights: list[int]
) -> np.ndarray:
    base_specs = _base_layer_specs(stripes)
    decorated_specs = _insert_gap_widths(base_specs, gap_heights)
    return _render_layers(decorated_specs)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a nested square arrangement from full-width horizontal stripes.

    Combo rule:
    1. Read non-background stripes bottom->top and build the original nesting.
    2. Read the separator height between each adjacent stripe pair and insert a
       background gap of the same width between the corresponding output layers.
    """
    stripes, gap_heights = _parse_structure(np.array(grid, dtype=int))
    return _build_output(stripes, gap_heights)


def _gen_attempt() -> np.ndarray | None:
    n_stripes = int(np.random.randint(3, 7))
    width = int(np.random.randint(7, 16))

    stripe_heights = [int(np.random.randint(1, 5)) for _ in range(n_stripes)]
    gap_heights = [int(np.random.randint(1, 3)) for _ in range(n_stripes - 1)]

    if 1 not in gap_heights or 2 not in gap_heights:
        return None

    output_side = int(
        stripe_heights[0] + 2 * sum(stripe_heights[1:]) + 2 * sum(gap_heights)
    )
    input_height = int(sum(stripe_heights) + sum(gap_heights))
    if output_side > 30 or input_height > 30:
        return None

    colors = list(np.random.choice(np.arange(1, 10), size=n_stripes, replace=False))
    stripes = [(int(colors[i]), int(stripe_heights[i])) for i in range(n_stripes)]  # bottom -> top

    rows: list[np.ndarray] = []
    for idx in range(n_stripes - 1, -1, -1):
        col, stripe_h = stripes[idx]
        rows.append(np.full((stripe_h, width), col, dtype=int))
        if idx > 0:
            rows.append(np.full((gap_heights[idx - 1], width), BG, dtype=int))
    grid = np.vstack(rows)

    parsed_stripes, parsed_gaps = _parse_structure(grid)
    assert parsed_stripes == stripes
    assert parsed_gaps == gap_heights

    expected = _build_output(stripes, gap_heights)
    if not np.array_equal(main(grid), expected):
        return None

    original_like = _render_layers(_base_layer_specs(stripes))
    if np.array_equal(expected, original_like):
        return None

    bool_gap_like = _build_output(
        stripes, [1 if int(gap_h) > 0 else 0 for gap_h in gap_heights]
    )
    if np.array_equal(expected, bool_gap_like):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
