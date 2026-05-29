# Variant of 45a5af55
# Dimension: D9 — 多步处理与迭代结构 (multi_step)
# Change: 单阶段迭代：逐层读取色条并包裹框架，无额外处理步骤
#   → 两阶段处理：先构建嵌套框架，再根据色条间的分隔行进行额外装饰

from __future__ import annotations

import numpy as np

"""
Design analysis for the D9 variant
==================================

a) 具体化与行为追踪
   - 输入仍是“全宽水平条纹”的纵向堆叠，但现在允许相邻彩色条纹之间插入若干行 `0`
     作为背景分隔。彩色条纹颜色都非 0，且每一行都是整行纯色。
   - 输出的第一阶段与原题一致：自底向上读取彩色条纹，最底层条纹变成中心实心方块，
     之后每层以自身高度为厚度向外包裹同色方框。
   - 第二阶段新增：如果输入中第 i 层与第 i+1 层条纹之间存在任意高度的 `0` 分隔行，
     则在输出中这两层之间额外插入一圈“1 像素宽”的 `0` 间隙。注意这里读的是
     “是否存在分隔”，不是“分隔有几行”，因此输入分隔高度和输出间隙宽度不相等。
   - 小例子（底→顶）：
       条纹: `(2,h=2)`, `(5,h=1)`, `(7,h=2)`
       分隔: `2` 与 `5` 之间有 `0` 行，`5` 与 `7` 之间没有。
     原始逻辑:
       1. `2` 生成 2x2 中心块
       2. `5` 直接包一圈厚度 1，得到 4x4
       3. `7` 再包一圈厚度 2，得到 8x8
     变体逻辑:
       1. 先得到同样的基础层级关系
       2. 因为 `2` 和 `5` 之间有分隔，所以在中心块外围先留出 1 像素 `0` 空隙，
          再画 `5` 的框；`5` 和 `7` 之间无分隔，所以 `7` 仍直接贴着 `5`。
       差异: 输出边长从 8 变为 10，且中心块与第二层框之间出现可见黑色空环。

b) 复杂度校准
   - 分隔信号只取二值：有 / 无。这样规则清晰，但不会退化成原题，因为同一张图里同时
     保留“有间隙的层对”和“无间隙的层对”，解题者必须同时理解两个阶段。
   - 条纹数固定在 3-6 层，既能重复展示“是否分隔”的对比，又不会让输出过大或杂乱。
   - 分隔行高度可为 1-3，但输出永远只插 1 像素间隙；这让第二阶段是“读取结构信号再
     做统一装饰”，而不是简单复制输入厚度。

c) 隐式假设审查
   - 原 `main()` 逐行向上合并同色 run，默认所有 run 都是有效条纹；在变体中这会把 `0`
     分隔误当作要画的颜色层，因此必须先显式区分“彩色条纹”和“背景分隔”。
   - 原 `while rr >= 0 and (g[rr] == col).all()` 依赖“所有 run 都参与输出”。现在仍可
     用来统计 run 高度，但后续处理必须按 `col == 0` 分支。
   - 原 `nxt[...] = cur` 代表新框总是紧贴旧形状；该前提在变体中不成立，因为有些层
     之间需要先插一个统一宽度的背景环。
   - 新实现因此分成两步：
       1. 先计算基础层级的内外边长（不带间隙）
       2. 再根据分隔布尔值对外层边长整体平移，并绘制带空洞的方框环

d) 放置合法性
   - 生成器保证输入只由“彩色条纹 run”和“0 分隔 run”交替组成，不出现前后连续两个
     `0` 块之外的歧义，也不在最顶/最底放额外背景边框。
   - 输出是同心正方形，没有对象间碰撞问题；边长通过
     `h0 + 2*sum(外层厚度) + 2*sum(有分隔的层对)` 控制在 30 以内。

e) 视觉可观察性
   - 每个样例至少包含一个有分隔的层对和一个无分隔的层对，因此观察 2-3 组样例时，
     人类能直接对比“有些层之间留黑环，有些没有”，而不是把所有样例误读成固定有边框。
   - 条纹颜色全部互异，避免同色层在输出中贴合后难以辨认。

f) 可视化整洁度
   - 仍然只使用同心方框这一种结构，新增元素只是 1 像素背景环，不会引入局部重叠、
     交叉或不规则形状，因此图面整洁。

g) 颜色变化可推断性
   - 没有颜色映射。`0` 固定表示背景/间隙；每个彩色条纹在输出中仍使用自身颜色。
   - 颜色只承担“身份标记”作用，规则核心由位置和分隔关系决定，便于从少量样例推断。
"""


BG = 0


def _parse_runs(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[bool]]:
    """
    Parse the input into bottom->top color stripes plus separator flags.

    Returns:
        stripes: [(color, height), ...] from inner-most to outer-most
        separators: bool list where separators[i] describes whether there is at
            least one background run between stripes[i] and stripes[i+1].
    """
    g = np.array(grid, dtype=int)
    H, _W = g.shape

    row_colors: list[int] = []
    for r in range(H):
        col = int(g[r, 0])
        assert bool((g[r] == col).all())
        row_colors.append(col)

    runs: list[tuple[int, int]] = []
    r = 0
    while r < H:
        col = row_colors[r]
        start = r
        r += 1
        while r < H and row_colors[r] == col:
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
    separators: list[bool] = []
    i = 0
    while i < len(bottom_to_top):
        col, h = bottom_to_top[i]
        assert col != BG
        stripes.append((int(col), int(h)))
        i += 1
        if i >= len(bottom_to_top):
            break
        if bottom_to_top[i][0] == BG:
            separators.append(True)
            i += 1
            assert i < len(bottom_to_top)
        else:
            separators.append(False)

    assert len(stripes) >= 2
    assert len(separators) == len(stripes) - 1
    return stripes, separators


def _base_layer_specs(stripes: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """
    Stage 1: compute the nested-square geometry ignoring separator gaps.

    Each spec is (color, inner_size, outer_size), ordered inner->outer.
    The center square has inner_size == 0.
    """
    c0, h0 = stripes[0]
    specs: list[tuple[int, int, int]] = [(int(c0), 0, int(h0))]
    cur_size = int(h0)
    for col, h in stripes[1:]:
        inner = cur_size
        cur_size = int(cur_size + 2 * int(h))
        specs.append((int(col), int(inner), int(cur_size)))
    return specs


def _decorate_specs(
    base_specs: list[tuple[int, int, int]], separators: list[bool]
) -> list[tuple[int, int, int]]:
    """
    Stage 2: expand outer layers when the corresponding stripe pair had a
    separator run in the input. Each such separator contributes a 1-pixel gap,
    i.e. +2 to the side length of that layer and every layer outside it.
    """
    decorated: list[tuple[int, int, int]] = []
    extra = 0
    for idx, (col, inner, outer) in enumerate(base_specs):
        if idx > 0 and separators[idx - 1]:
            extra += 2
        decorated.append((int(col), int(inner + extra), int(outer + extra)))
    return decorated


def _render_layers(specs: list[tuple[int, int, int]]) -> np.ndarray:
    """Render centered square annuli from outer to inner on a background canvas."""
    final_n = int(specs[-1][2])
    out = np.full((final_n, final_n), BG, dtype=int)

    for col, inner, outer in reversed(specs):
        outer = int(outer)
        inner = int(inner)
        outer_margin = (final_n - outer) // 2
        out[outer_margin : outer_margin + outer, outer_margin : outer_margin + outer] = int(col)
        if inner > 0:
            inner_margin = (final_n - inner) // 2
            out[inner_margin : inner_margin + inner, inner_margin : inner_margin + inner] = BG

    return out


def main(grid: np.ndarray) -> np.ndarray:
    stripes, separators = _parse_runs(np.array(grid, dtype=int))
    base_specs = _base_layer_specs(stripes)
    decorated_specs = _decorate_specs(base_specs, separators)
    return _render_layers(decorated_specs)


def _gen_attempt() -> np.ndarray | None:
    n_stripes = int(np.random.randint(3, 7))
    widths = int(np.random.randint(7, 16))

    heights = [int(np.random.randint(1, 5)) for _ in range(n_stripes)]
    separators = [bool(np.random.randint(0, 2)) for _ in range(n_stripes - 1)]
    if not any(separators):
        return None
    if all(separators):
        return None

    separator_heights = [int(np.random.randint(1, 4)) if flag else 0 for flag in separators]
    final_side = int(heights[0] + 2 * sum(heights[1:]) + 2 * sum(int(flag) for flag in separators))
    input_height = int(sum(heights) + sum(separator_heights))
    if final_side > 30 or input_height > 30:
        return None

    colors = list(np.random.choice(np.arange(1, 10), size=n_stripes, replace=False))
    stripes = [(int(colors[i]), int(heights[i])) for i in range(n_stripes)]  # bottom -> top

    rows: list[np.ndarray] = []
    for idx in range(n_stripes - 1, -1, -1):
        col, h = stripes[idx]
        rows.append(np.full((h, widths), col, dtype=int))
        if idx > 0 and separators[idx - 1]:
            rows.append(np.full((separator_heights[idx - 1], widths), BG, dtype=int))

    grid = np.vstack(rows)

    parsed_stripes, parsed_separators = _parse_runs(grid)
    assert parsed_stripes == stripes
    assert parsed_separators == separators

    out = main(grid)
    if out.shape[0] != final_side or out.shape[1] != final_side:
        return None
    if np.array_equal(out, _render_layers(_base_layer_specs(stripes))):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
