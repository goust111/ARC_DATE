# Variant of dfadab01
# Dimension: D5 — 字形的固定几何形状 (meta)
# Change: 四种固定4x4字形图案：空心方块（边框填充）、菱形（对角连接环）、对角块（2x2块对角排列）、X形（角和中心交叉）... → 字形大小可变

from __future__ import annotations

import random

import numpy as np


"""
design_analysis:
  a_concretization_and_trace:
    original_scene:
      - 输入里的真实对象只有单像素标记；输出把每个标记扩成固定 4x4 字形。
      - 若输入额外出现一个样本字形，则它只是图例；其右下角的同色标记是标签，不参与输出。
    variant_scene:
      - 保留四种标记色 -> 四种输出色/图案家族的对应关系，但把字形边界框尺寸改成依颜色而定：
        2 -> 3x3 空心方块, 3 -> 4x4 菱形环, 5 -> 5x5 对角块, 8 -> 6x6 X。
      - 图例标签不再固定在 (+4,+4)，而是位于该颜色样本字形的右下外侧一格，即 (+size,+size)。
      - 输出仍以标记像素为字形左上角，仍允许超出网格时裁剪。
    concrete_trace:
      - 原逻辑中，若输入在 (7, 8) 放一个颜色 5 标记，就总是在输出的 [7:11, 8:12] 里画固定 4x4 的两块对角 2x2。
      - 变体中，同样位置的颜色 5 标记要在 [7:12, 8:13] 里画 5x5 字形；颜色 2 则只画 3x3。
      - 因此差异不只是 mask 内容，还包括 label 偏移、边界裁剪范围、以及“不会重叠”的最小间距。
  b_complexity_calibration:
    - 只让尺寸在四个离散档位 {3,4,5,6} 中变化，避免自由度过多。
    - 同一输入至少使用两种标记色，并尽量让每种已使用颜色出现至少两次，保证人类能看到重复支撑。
    - 保留原题图例机制，用可见样本把“颜色 -> 图案/尺寸”联立提示出来，而不是引入不可见编码。
  c_implicit_assumption_review_of_main:
    - 原 _has_source_glyph() 假设 bbox 恒为 4x4；变体里必须改成使用 mask.shape。
    - 原 _paint_clipped() 固定循环 4 次；变体里必须按实际 size 循环，否则会漏画或越界。
    - 原 main() 用 (r-4, c-4) 检查图例标签；变体里必须改成 (r-size, c-size)。
    - 原“不重叠”只需要 spacing >= 4；变体里若仍用 4，会导致 5x5/6x6 字形互相覆盖，所以 generator 必须升级 spacing 下界。
    - 原 assert 的允许颜色集合不变，因为仍然只有背景、四种标记色、四种输出色。
  d_placement_legality:
    - 真实标记放在规则网格上，行列间距只从 {6,7} 选取，确保最大 6x6 字形也不会与相邻字形重叠。
    - 图例样本要求完整落在网格内，且其整块 bbox 与标签像素都落在当前空白区域，避免与真实标记或其他图例纠缠。
    - 真实标记允许靠近下边/右边，以保留“裁剪”这一原题行为。
  e_visual_observability:
    - 尺寸变化是直接可见的几何差异，不依赖颜色 id 的排序或数值运算。
    - 采用固定四档尺寸并复用原图案家族，解题者能从少量样例观察到：相同标记色总是扩成同一种颜色、同一种图案、同一种大小。
  f_visual_tidiness:
    - 图例不与真实标记混叠，真实输出字形之间也不重叠，因此每个对象的来源都清楚。
    - 小尺寸只分配给视觉上仍然清晰的图案：3x3 给边框方块，4x4 给菱形环，5x5 给对角块，6x6 给 X。
  g_color_change_inferability:
    - 颜色映射仍保持原题的一一对应，没有增加额外颜色机制。
    - 变体的新增信息完全由可见几何尺寸承载，所以不会出现“颜色变了但无结构线索”的问题。
  generator_guarantees_for_new_main:
    - 每个输入至少有一个真实标记，因此 main() 的 seen_marker 断言可满足。
    - 每个输入至少包含两种真实标记色，因此新维度 D5 在绝大多数样本里都可观察。
    - 所有图例标签都严格位于对应样本字形的 (+size,+size)，因此 main() 的 skip 条件有构造性保证。
"""


def _frame_mask(size: int) -> np.ndarray:
    rr, cc = np.indices((size, size))
    return (rr == 0) | (rr == size - 1) | (cc == 0) | (cc == size - 1)


def _diamond_ring_mask(size: int) -> np.ndarray:
    center = (size - 1) / 2.0
    radius = (size - 1) / 2.0
    rr, cc = np.indices((size, size))
    dist = np.abs(rr - center) + np.abs(cc - center)
    return (dist >= radius - 0.5) & (dist <= radius + 0.5)


def _diag_blocks_mask(size: int) -> np.ndarray:
    block = max(1, size // 2)
    mask = np.zeros((size, size), dtype=bool)
    mask[:block, :block] = True
    mask[size - block :, size - block :] = True
    return mask


def _x_mask(size: int) -> np.ndarray:
    rr, cc = np.indices((size, size))
    return (rr == cc) | (rr + cc == size - 1)


GLYPHS: dict[int, tuple[int, int, np.ndarray]] = {
    2: (4, 3, _frame_mask(3)),
    3: (1, 4, _diamond_ring_mask(4)),
    5: (6, 5, _diag_blocks_mask(5)),
    8: (7, 6, _x_mask(6)),
}


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> bool:
    h, w = g.shape
    gh, gw = mask.shape
    if r0 < 0 or c0 < 0 or r0 + gh > h or c0 + gw > w:
        return False
    patch = g[r0 : r0 + gh, c0 : c0 + gw]
    return bool(np.all(patch[mask] == color) and np.all(patch[~mask] != color))


def _paint_clipped(out: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    h, w = out.shape
    gh, gw = mask.shape
    for dr in range(gh):
        rr = r0 + dr
        if rr < 0 or rr >= h:
            continue
        for dc in range(gw):
            cc = c0 + dc
            if 0 <= cc < w and bool(mask[dr, dc]):
                out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each colored control pixel into its corresponding size-coded glyph.

    Input structure:
    - Black background with single-pixel markers in colors 2, 3, 5, and/or 8.
    - Some inputs also show legend samples: a glyph already drawn in its output color, plus a same-type marker one cell below-right of that sample's own bounding box.

    Rule:
    - Replace each real marker by the glyph tied to its color, using both that color's output hue and that color's dedicated size: 2->3x3 frame, 3->4x4 diamond ring, 5->5x5 diagonal blocks, 8->6x6 X.
    - Paint each glyph with its top-left corner at the marker pixel, clipping anything that extends past the grid edge.
    - Remove legend samples and skip the label markers that sit at (+size, +size) from their matching sample glyphs.

    Key insight:
    - Marker color now encodes two visible parameters at once: which glyph family to draw and how large its bounding box should be; the legend offset changes with that size instead of staying fixed at 4.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({0, *GLYPHS.keys(), 1, 4, 6, 7})

    out = np.zeros_like(g)
    seen_marker = False
    for marker_color, (glyph_color, size, mask) in GLYPHS.items():
        coords = np.argwhere(g == marker_color)
        if coords.size:
            seen_marker = True
        for r, c in coords:
            if _has_source_glyph(g, int(r) - size, int(c) - size, glyph_color, mask):
                continue
            _paint_clipped(out, int(r), int(c), glyph_color, mask)

    assert seen_marker
    return out


GRID_SIZE_OPTIONS = (20, 24)
ROWCOL_COUNT_RANGE = (2, 4)
SPACING_OPTIONS = (6, 7)
FILL_RATE_RANGE = (0.65, 1.0)
GEN_RETRIES = 400


def _all_top_left_positions(height: int, width: int, size: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(height - size) for c in range(width - size)]


def _choose_real_markers(
    anchors: list[tuple[int, int]],
    used_colors: list[int],
) -> list[tuple[int, int, int]]:
    cell_count = len(anchors)
    min_fill = max(len(used_colors) * 2, int(np.ceil(cell_count * random.uniform(*FILL_RATE_RANGE))))
    min_fill = min(cell_count, max(len(used_colors), min_fill))
    fill_count = random.randint(min_fill, cell_count)

    chosen_positions = random.sample(anchors, fill_count)
    assignments = used_colors * 2
    while len(assignments) < fill_count:
        assignments.append(random.choice(used_colors))
    assignments = assignments[:fill_count]
    random.shuffle(assignments)
    return [(r, c, color) for (r, c), color in zip(chosen_positions, assignments, strict=True)]


def _try_place_legend(
    grid: np.ndarray,
    marker_color: int,
) -> bool:
    glyph_color, size, mask = GLYPHS[marker_color]
    candidates = _all_top_left_positions(grid.shape[0], grid.shape[1], size)
    random.shuffle(candidates)
    for r0, c0 in candidates:
        label = (r0 + size, c0 + size)
        if label[0] >= grid.shape[0] or label[1] >= grid.shape[1]:
            continue
        patch = grid[r0 : r0 + size, c0 : c0 + size]
        if np.any(patch != 0) or grid[label] != 0:
            continue
        patch[mask] = glyph_color
        grid[label] = marker_color
        return True
    return False


def _build_expected(height: int, width: int, markers: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.zeros((height, width), dtype=int)
    for r, c, marker_color in markers:
        glyph_color, _size, mask = GLYPHS[marker_color]
        _paint_clipped(out, r, c, glyph_color, mask)
    return out


def generate_input() -> np.ndarray:
    """
    Construct a regular marker lattice plus 1-2 legend entries for the variable-size glyph rule.
    """
    for _ in range(GEN_RETRIES):
        height = width = random.choice(GRID_SIZE_OPTIONS)
        row_count = random.randint(*ROWCOL_COUNT_RANGE)
        col_count = random.randint(*ROWCOL_COUNT_RANGE)
        row_spacing = random.choice(SPACING_OPTIONS)
        col_spacing = random.choice(SPACING_OPTIONS)
        max_row_start = height - 1 - (row_count - 1) * row_spacing
        max_col_start = width - 1 - (col_count - 1) * col_spacing
        if max_row_start < 0 or max_col_start < 0:
            continue

        row0 = random.randint(0, max_row_start)
        col0 = random.randint(0, max_col_start)
        anchors = [(row0 + rr * row_spacing, col0 + cc * col_spacing) for rr in range(row_count) for cc in range(col_count)]

        used_count = random.randint(2, 4)
        used_colors = random.sample(list(GLYPHS), used_count)
        real_markers = _choose_real_markers(anchors, used_colors)

        grid = np.zeros((height, width), dtype=int)
        for r, c, marker_color in real_markers:
            grid[r, c] = marker_color

        legend_colors = random.sample(used_colors, 1 + int(len(used_colors) > 1 and random.random() < 0.35))
        placed_legends = 0
        for marker_color in legend_colors:
            placed_legends += int(_try_place_legend(grid, marker_color))
        if placed_legends == 0:
            continue

        expected = _build_expected(height, width, real_markers)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid variable-size dfadab01 instance")
