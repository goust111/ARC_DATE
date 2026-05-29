# Variant of dfadab01
# Dimension: D5 — 字形的固定几何形状 (meta)
# Change: 四种固定4x4字形图案：空心方块（边框填充）、菱形（对角连接环）、对角块（2x2块对角排列）、X形（角和中心交叉）... → 字形为旋转变体
#
# design_analysis:
#   a_concretization_and_trace:
#     base_glyph:
#       - 使用同一个 4x4 非对称基础字形，恰好 8 个像素，四次 90 度旋转后得到 4 个互不相同的字形。
#       - 标记色仍然是 {2,3,5,8}，输出色仍然是 {4,1,6,7}，所以题目的“控制像素展开成 4x4 图案”骨架不变。
#       - 与原题不同的是，四个字形不再是四个彼此无关的模板，而是 base / rot90 / rot180 / rot270。
#     pipeline_trace:
#       - 原逻辑示例：若输入在 (6, 10) 放置颜色 5，则输出在同一左上角画固定的“对角 2x2 块”。
#       - 变体逻辑：若输入在 (6, 10) 放置颜色 5，则输出在同一左上角画“基础字形旋转 180 度”的版本。
#       - 差异点发生在字形选择环节：原题按颜色查 4 个彼此独立的模板；变体按颜色选择“同一模板的旋转角度”。
#   b_complexity_calibration:
#     - 如果基础字形本身有 180 度或镜像对称，旋转关系会退化为少数几种外观，人类无法稳定看出“同源旋转”。
#     - 因此基础字形必须明显不对称，且四个旋转全部不同；这里选用弯折阶梯状 8 像素图案。
#     - 生成器偏向 3-4 种标记类型并常带 1-2 个图例，避免样例长期只出现单一旋转而看不出变化。
#   c_implicit_assumption_audit:
#     - `assert unique colors subset ...` 依赖输入只含背景、标记色和图例字形的输出色；生成器保持这一点。
#     - `_has_source_glyph(...)` 依赖图例字形完整落在网格内，且标签像素正好位于字形 bbox 的右下外侧 (+4,+4)；生成器只放置完整图例。
#     - 主循环按 marker_color 固定取 `(glyph_color, mask)`；变体中这个前提仍成立，只是 `mask` 由 `BASE_GLYPH` 旋转得到。
#     - `_paint_clipped(...)` 默认多个真实字形不会互相覆盖；生成器继续保证行列间距 >= 4。
#     - 原题里 4 个 mask 彼此独立，顺序无关；变体里虽然同源，但只要颜色映射固定、旋转互异，就不会引入覆盖或顺序依赖。
#   d_placement_legality:
#     - 真实标记仍放在规则网格上，行列间距取 4-6，保证 4x4 字形之间最多只发生边界裁剪，不发生彼此重叠。
#     - 图例只放在完全空白的 5x5 角区：4x4 字形区域必须全空，标签点也必须空，避免覆盖真实标记或其他图例。
#     - 生成器在返回前显式验证 `main(input) == expected_output`。
#   e_visual_observability:
#     - 看到几个样例后，人类能观察到四种输出图案轮廓相同，只是朝向不同，而不是四个完全不同的符号。
#     - 为了让这一点稳定可见，生成器强制至少 3 种真实标记，且多数样本使用 20x20 网格留出重复实例。
#   f_visual_cleanliness:
#     - 图例放在空区域，不让输出色和标记色叠在一起。
#     - 不使用额外辅助标记、排序线或背景噪声，保持与原题近似的“黑底 + 稀疏控制点/图例”版式。
#   g_color_inference:
#     - 颜色映射没有新增自由度，仍是固定的 marker→output 色对；可观察变化只来自几何旋转。
#     - 不依赖颜色 id 的数值大小进行排序或运算，只把颜色当作离散类别。
#   post_change_main_assumptions:
#     - 新 solver 额外依赖“基础字形四个旋转彼此可区分”；代码中直接用非对称 `BASE_MASK` 保证这一点。
#     - 生成器通过至少 3 种真实标记与 1-2 个图例，保证这个新假设在训练样例式观看下是可见的。

from __future__ import annotations

import random

import numpy as np


BASE_MASK = np.array(
    [
        [1, 1, 1, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 1, 0],
    ],
    dtype=bool,
)

MARKER_TO_COLOR: dict[int, int] = {2: 4, 3: 1, 5: 6, 8: 7}
MARKER_ROTATIONS: dict[int, int] = {2: 0, 3: 1, 5: 2, 8: 3}
GLYPHS: dict[int, tuple[int, np.ndarray]] = {
    marker_color: (
        glyph_color,
        np.rot90(BASE_MASK, k=rotation).copy(),
    )
    for marker_color, glyph_color in MARKER_TO_COLOR.items()
    for rotation in [MARKER_ROTATIONS[marker_color]]
}
ALLOWED_COLORS = {0, *MARKER_TO_COLOR.keys(), *MARKER_TO_COLOR.values()}


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> bool:
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + 4 > h or c0 + 4 > w:
        return False
    patch = g[r0 : r0 + 4, c0 : c0 + 4]
    return bool(np.all(patch[mask] == color) and np.all(patch[~mask] != color))


def _paint_clipped(out: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    h, w = out.shape
    for dr in range(4):
        rr = r0 + dr
        if rr < 0 or rr >= h:
            continue
        for dc in range(4):
            cc = c0 + dc
            if 0 <= cc < w and bool(mask[dr, dc]):
                out[rr, cc] = color


def _render_expected(shape: tuple[int, int], markers: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for marker_color, r, c in markers:
        glyph_color, mask = GLYPHS[marker_color]
        _paint_clipped(out, r, c, glyph_color, mask)
    return out


def _enumerate_grid_positions(
    side: int,
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    feasible_row_spacings = [s for s in range(4, 7) if 1 + (side - 1) // s >= 2]
    row_spacing = rng.choice(feasible_row_spacings)
    col_spacing = rng.choice(feasible_row_spacings)

    max_rows = min(5, 1 + (side - 1) // row_spacing)
    max_cols = min(5, 1 + (side - 1) // col_spacing)

    min_rows = 3 if max_rows >= 3 else 2
    min_cols = 3 if max_cols >= 3 else 2
    n_rows = rng.randint(min_rows, max_rows)
    n_cols = rng.randint(min_cols, max_cols)

    slack_r = side - 1 - (n_rows - 1) * row_spacing
    slack_c = side - 1 - (n_cols - 1) * col_spacing
    if rng.random() < 0.45:
        start_r = slack_r
    else:
        start_r = rng.randint(0, slack_r)
    if rng.random() < 0.45:
        start_c = slack_c
    else:
        start_c = rng.randint(0, slack_c)

    rows = [start_r + i * row_spacing for i in range(n_rows)]
    cols = [start_c + i * col_spacing for i in range(n_cols)]
    return rows, cols


def _place_legend(
    inp: np.ndarray,
    marker_color: int,
    rng: random.Random,
) -> bool:
    h, w = inp.shape
    glyph_color, mask = GLYPHS[marker_color]
    candidates: list[tuple[int, int]] = []
    for r0 in range(h - 4):
        for c0 in range(w - 4):
            if np.any(inp[r0 : r0 + 4, c0 : c0 + 4] != 0):
                continue
            if inp[r0 + 4, c0 + 4] != 0:
                continue
            candidates.append((r0, c0))
    if not candidates:
        return False

    r0, c0 = rng.choice(candidates)
    for dr in range(4):
        for dc in range(4):
            if bool(mask[dr, dc]):
                inp[r0 + dr, c0 + dc] = glyph_color
    inp[r0 + 4, c0 + 4] = marker_color
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each colored control pixel into one quarter-turn of a shared 4x4 glyph.

    Input structure:
    - Black background with single-pixel markers in colors 2, 3, 5, and/or 8.
    - Some inputs also include 1-2 legend entries: a full 4x4 glyph drawn in the output color plus a matching marker one cell below-right of that glyph's bbox.

    Rule:
    - Replace each real marker by the glyph in its mapped output color, anchored at the marker as the glyph's top-left corner.
    - The four marker types no longer map to unrelated templates; they map to the 0/90/180/270 degree rotations of one asymmetric base glyph.
    - Ignore legend samples themselves and suppress the marker pixel that labels each legend; clip any painted glyph at the grid boundary.

    Key insight:
    - The task still treats colored pixels as control points, but the glyph alphabet is now generated by quarter-turning one prototype rather than memorizing four unrelated masks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset(ALLOWED_COLORS)

    out = np.zeros_like(g)
    seen_marker = False
    for marker_color, (glyph_color, mask) in GLYPHS.items():
        coords = np.argwhere(g == marker_color)
        if coords.size:
            seen_marker = True
        for r, c in coords:
            if _has_source_glyph(g, int(r) - 4, int(c) - 4, glyph_color, mask):
                continue
            _paint_clipped(out, int(r), int(c), glyph_color, mask)

    assert seen_marker
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(400):
        side = 20 if rng.random() < 0.8 else 10
        rows, cols = _enumerate_grid_positions(side, rng)
        cells = [(r, c) for r in rows for c in cols]
        rng.shuffle(cells)

        n_types = 4 if len(cells) >= 6 and rng.random() < 0.6 else 3
        used_types = rng.sample(list(MARKER_TO_COLOR), k=n_types)

        min_fill = max(n_types + 1, int(np.ceil(0.6 * len(cells))))
        filled_count = rng.randint(min_fill, len(cells))
        filled_cells = cells[:filled_count]

        markers: list[tuple[int, int, int]] = []
        for marker_color, (r, c) in zip(used_types, filled_cells):
            markers.append((marker_color, r, c))
        for r, c in filled_cells[len(used_types) :]:
            markers.append((rng.choice(used_types), r, c))

        inp = np.zeros((side, side), dtype=int)
        for marker_color, r, c in markers:
            inp[r, c] = marker_color

        expected = _render_expected(inp.shape, markers)

        legend_target = 2 if side == 20 and n_types == 4 and rng.random() < 0.5 else 1
        legend_types = rng.sample(used_types, k=min(legend_target, len(used_types)))
        ok = True
        for marker_color in legend_types:
            if not _place_legend(inp, marker_color, rng):
                ok = False
                break
        if not ok:
            continue

        solved = main(inp)
        if not np.array_equal(solved, expected):
            continue

        real_marker_types = {
            marker_color
            for marker_color, r, c in markers
            if not _has_source_glyph(inp, r - 4, c - 4, GLYPHS[marker_color][0], GLYPHS[marker_color][1])
        }
        if len(real_marker_types) < 3:
            continue

        return inp

    raise RuntimeError("failed to generate a valid rotated-glyph instance")
