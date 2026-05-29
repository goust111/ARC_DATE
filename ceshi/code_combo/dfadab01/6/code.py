# Combo variant of dfadab01 (index 6)
# combo_spec: D5/2+D6/0
# Name: 可变尺寸字形+网格自动扩展

from __future__ import annotations

import random

import numpy as np


"""
design_analysis:
  a_concretization_and_trace:
    original_scene:
      - 输入中的真实对象是规则栅格上的单像素标记；输出把每个标记扩成固定 4x4 字形。
      - 若输入里额外出现一个样本字形，则它只是图例；其右下角固定偏移 (+4,+4) 的同类标记是标签，不参与输出。
      - 即便真实标记靠近下边或右边，原题也只是把越界部分裁掉，输出尺寸不变。
    fused_scene:
      - 保留四种标记色 -> 四种输出色/图案家族的关系，但每种颜色的 bbox 尺寸不同：
        2 -> 3x3 空心方框, 3 -> 4x4 菱形环, 5 -> 5x5 对角块, 8 -> 6x6 X。
      - 图例标签的偏移也随尺寸变化，必须位于对应样本字形的 (+size,+size)。
      - 真实标记仍以自己的像素作为字形左上角，但不再裁剪；若任一字形越过原输入边界，输出网格向下和向右扩展到能完整容纳所有真实字形。
    concrete_trace:
      - 例: 输入高宽都是 18，颜色 8 的真实标记在 (13, 14)。原题会尝试画 4x4，且只保留落在 18x18 内的部分。
      - 融合后，同一标记要画 6x6 的 X，覆盖行 13..18、列 14..19，因此输出至少扩成 19x20。
      - 若同一输入里还有颜色 2 的标记在 (2, 3)，它只会展开成 3x3 方框；因此解题者必须同时推断“颜色决定图案”和“颜色决定尺寸”，而输出尺寸还要看最靠边的真实标记能延伸多远。
  b_complexity_calibration:
    - 尺寸只使用四个离散档位 {3,4,5,6}，避免自由度过大，同时让差异直接可见。
    - 生成器强制每题至少出现两种标记色，并保证至少一种非 4x4 颜色真实出现多次，这样“颜色 -> 尺寸”有重复支撑。
    - 生成器还强制至少一个真实标记发生越界扩展，使 D6/0 的扩网格行为不是偶发，而是样本的核心现象。
  c_implicit_assumption_review_of_main:
    - 原 GLYPHS 只存 output_color 和 4x4 mask；融合后必须把 size 一并显式编码，或从 mask.shape 读取。
    - 原 _has_source_glyph() 默认检查固定 4x4 样本；融合后必须按各颜色自己的 mask 尺寸检查，否则图例标签会误判。
    - 原 _paint_clipped() 和 out = np.zeros_like(g) 依赖“输出尺寸固定且越界裁剪”；融合后这两个前提都失效，必须先收集真实标记，再按最远 extents 分配更大的输出。
    - 原 main() 用 (r-4, c-4) 判断“这个 marker 是否只是图例标签”；融合后必须改为 (r-size, c-size)。
    - 原题只要 spacing >= 4 就不会重叠；融合后最大字形是 6x6，因此 generator 至少要保证相邻锚点相隔 6。
  d_placement_legality:
    - 真实标记仍放在规则锚点栅格上，行列间距从 {6,7} 取值，保证任意两枚真实字形即使都是 6x6 也不会重叠。
    - 其中一个锚点被强制分配给会越界的真实标记，并且该颜色选自 {2,5,8}，确保同时触发“非 4x4”与“扩网格”。
    - 图例样本要求整块 glyph bbox 与标签像素都完整落在输入网格内部，且只写入当前空白区域，因此不会与真实标记或其他图例混叠。
  e_visual_observability:
    - 人类直接能看到同色单像素总是变成同一颜色、同一图案、同一大小；尺寸不是通过数值计算隐含编码，而是画面本身可见。
    - 输出有时比输入更大，且增长量与最靠边标记的尺寸相关，所以“扩网格”不是隐藏机制。
    - 图例仍提供局部示范，帮助把 marker 色和输出字形/尺寸绑定起来。
  f_visual_tidiness:
    - 真实标记排布在规则栅格上，输出字形之间不重叠；图例单独占据空白区域，输入结构仍然清楚。
    - 扩网格只在右边和下边增加背景边界，不引入额外对象层叠，因此大尺寸字形仍易于辨认。
  g_color_change_inferability:
    - 颜色映射保持原题的一一对应，没有引入新的颜色语义。
    - 新增难点完全来自几何尺寸和输出尺寸变化，而不是不可见的颜色数值运算。
  generator_guarantees_for_new_main:
    - main() 假设只有图例会在输入中出现输出色字形；generator 保证真实对象始终是单像素 marker。
    - main() 假设标签 marker 精确位于对应图例的 (+size,+size)；generator 按该偏移构造图例。
    - main() 按右下 extents 扩张输出，不处理负坐标外扩；generator 只在输入网格内部放 marker，因此所有扩展都发生在下边和右边。
    - main() 不显式检测真实字形重叠；generator 用 spacing >= 6 的规则锚点避免此类冲突。
    - main() 允许 legend 数量为 0，但 generator 至少放置 1 个 legend，并且至少 1 个真实 marker 发生越界扩展，从而避免退化回原题。
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


INPUT_SIZE_OPTIONS = (18, 20, 22)
ROW_COUNT_RANGE = (2, 3)
COL_COUNT_RANGE = (3, 4)
SPACING_OPTIONS = (6, 7)
FILL_RATE_RANGE = (0.75, 1.0)
GEN_RETRIES = 500


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> bool:
    h, w = g.shape
    gh, gw = mask.shape
    if r0 < 0 or c0 < 0 or r0 + gh > h or c0 + gw > w:
        return False
    patch = g[r0 : r0 + gh, c0 : c0 + gw]
    return bool(np.all(patch[mask] == color) and np.all(patch[~mask] != color))


def _paint_mask(out: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    gh, gw = mask.shape
    patch = out[r0 : r0 + gh, c0 : c0 + gw]
    patch[mask] = color


def _render_markers(base_shape: tuple[int, int], markers: list[tuple[int, int, int]]) -> np.ndarray:
    height, width = map(int, base_shape)
    out_h, out_w = int(height), int(width)
    render_items: list[tuple[int, int, int, np.ndarray]] = []
    for r, c, marker_color in markers:
        glyph_color, size, mask = GLYPHS[int(marker_color)]
        out_h = max(out_h, int(r) + int(size))
        out_w = max(out_w, int(c) + int(size))
        render_items.append((int(r), int(c), int(glyph_color), mask))

    out = np.zeros((out_h, out_w), dtype=int)
    for r, c, glyph_color, mask in render_items:
        _paint_mask(out, r, c, glyph_color, mask)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each colored control pixel into its corresponding size-coded glyph and enlarge the canvas if needed.

    Input structure:
    - Black background with single-pixel markers in colors 2, 3, 5, and/or 8.
    - Some inputs also include legend samples: a completed glyph already drawn in its output color, plus a same-type marker placed one cell diagonally beyond that sample's own lower-right corner, i.e. at (+size, +size).

    Rule:
    - Replace each real marker by the glyph family tied to its color, using that color's dedicated output hue and bounding-box size: 2->3x3 frame, 3->4x4 diamond ring, 5->5x5 diagonal blocks, 8->6x6 X.
    - Treat the marker pixel as the glyph's top-left anchor. Do not clip: if any real glyph would cross the original lower or right border, expand the output grid just enough to contain the full set of real glyphs.
    - Remove legend samples entirely and skip the label markers that sit at (+size, +size) from their matching sample glyphs.

    Key insight:
    - Marker color now determines both which pattern to draw and how large its footprint is, and that larger footprint can force the output shape itself to grow; the legend offset changes with glyph size, so sample detection must also be size-aware.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({0, *GLYPHS.keys(), 1, 4, 6, 7})

    real_markers: list[tuple[int, int, int]] = []
    seen_marker = False
    for marker_color, (glyph_color, size, mask) in GLYPHS.items():
        coords = np.argwhere(g == marker_color)
        if coords.size:
            seen_marker = True
        for r, c in coords:
            rr, cc = int(r), int(c)
            if _has_source_glyph(g, rr - size, cc - size, glyph_color, mask):
                continue
            real_markers.append((rr, cc, int(marker_color)))

    assert seen_marker
    return _render_markers(g.shape, real_markers)


def _grid_anchors(height: int, width: int, row_count: int, col_count: int, spacing: int, overflow_size: int) -> list[tuple[int, int]]:
    row_lb = max(0, height - overflow_size + 1 - (row_count - 1) * spacing)
    row_ub = height - 1 - (row_count - 1) * spacing
    col_lb = max(0, width - overflow_size + 1 - (col_count - 1) * spacing)
    col_ub = width - 1 - (col_count - 1) * spacing
    if row_lb > row_ub or col_lb > col_ub:
        raise ValueError("cannot place overflow anchor with chosen dimensions")

    row0 = random.randint(int(row_lb), int(row_ub))
    col0 = random.randint(int(col_lb), int(col_ub))
    return [
        (row0 + rr * spacing, col0 + cc * spacing)
        for rr in range(row_count)
        for cc in range(col_count)
    ]


def _choose_real_markers(
    anchors: list[tuple[int, int]],
    used_colors: list[int],
    overflow_color: int,
) -> list[tuple[int, int, int]]:
    cell_count = len(anchors)
    fill_floor = max(len(used_colors) * 2, int(np.ceil(cell_count * random.uniform(*FILL_RATE_RANGE))))
    fill_floor = min(cell_count, fill_floor)
    fill_count = random.randint(fill_floor, cell_count)

    overflow_anchor = max(anchors)
    remaining = [anchor for anchor in anchors if anchor != overflow_anchor]
    chosen = random.sample(remaining, fill_count - 1)
    chosen.append(overflow_anchor)

    assignments = list(used_colors)
    assignments.append(int(overflow_color))
    pool = used_colors + [overflow_color]
    while len(assignments) < fill_count:
        assignments.append(random.choice(pool))
    assignments = assignments[:fill_count]
    random.shuffle(assignments)

    overflow_index = chosen.index(overflow_anchor)
    if assignments[overflow_index] != overflow_color:
        swap_index = assignments.index(overflow_color)
        assignments[overflow_index], assignments[swap_index] = assignments[swap_index], assignments[overflow_index]

    return [(int(r), int(c), int(color)) for (r, c), color in zip(chosen, assignments, strict=True)]


def _all_legend_positions(height: int, width: int, size: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(height - size) for c in range(width - size)]


def _try_place_legend(grid: np.ndarray, marker_color: int) -> bool:
    glyph_color, size, mask = GLYPHS[int(marker_color)]
    candidates = _all_legend_positions(grid.shape[0], grid.shape[1], int(size))
    random.shuffle(candidates)
    for r0, c0 in candidates:
        label = (r0 + size, c0 + size)
        patch = grid[r0 : r0 + size, c0 : c0 + size]
        if np.any(patch != 0) or int(grid[label]) != 0:
            continue
        patch[mask] = int(glyph_color)
        grid[label] = int(marker_color)
        return True
    return False


def _old_clipped_fixed4_baseline(grid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(grid)
    fixed = {
        2: (4, np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]], dtype=bool)),
        3: (1, np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]], dtype=bool)),
        5: (6, np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=bool)),
        8: (7, np.array([[1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1]], dtype=bool)),
    }
    for marker_color, (glyph_color, mask) in fixed.items():
        coords = np.argwhere(grid == marker_color)
        for r, c in coords:
            rr, cc = int(r), int(c)
            if _has_source_glyph(grid, rr - 4, cc - 4, glyph_color, mask):
                continue
            for dr in range(4):
                for dc in range(4):
                    if not bool(mask[dr, dc]):
                        continue
                    tr, tc = rr + dr, cc + dc
                    if 0 <= tr < out.shape[0] and 0 <= tc < out.shape[1]:
                        out[tr, tc] = int(glyph_color)
    return out


def generate_input() -> np.ndarray:
    """
    Construct a marker lattice whose real glyphs use color-specific sizes and force output expansion.
    """
    for _ in range(GEN_RETRIES):
        height = width = random.choice(INPUT_SIZE_OPTIONS)
        row_count = random.randint(*ROW_COUNT_RANGE)
        col_count = random.randint(*COL_COUNT_RANGE)
        spacing = random.choice(SPACING_OPTIONS)
        cell_count = row_count * col_count

        max_used_count = min(len(GLYPHS), cell_count // 2)
        if max_used_count < 2:
            continue
        used_count = random.randint(2, max_used_count)
        used_colors = random.sample(list(GLYPHS), used_count)
        overflow_candidates = [color for color in used_colors if GLYPHS[int(color)][1] != 4]
        overflow_color = random.choice(overflow_candidates)
        overflow_size = GLYPHS[int(overflow_color)][1]

        try:
            anchors = _grid_anchors(height, width, row_count, col_count, spacing, overflow_size)
        except ValueError:
            continue

        real_markers = _choose_real_markers(anchors, used_colors, overflow_color)
        grid = np.zeros((height, width), dtype=int)
        for r, c, marker_color in real_markers:
            grid[int(r), int(c)] = int(marker_color)

        legend_target_count = 1 + int(len(used_colors) > 1 and random.random() < 0.5)
        legend_colors = random.sample(used_colors, legend_target_count)
        placed_legends = sum(int(_try_place_legend(grid, color)) for color in legend_colors)
        if placed_legends == 0:
            continue

        expected = _render_markers(grid.shape, real_markers)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue

        baseline = _old_clipped_fixed4_baseline(grid)
        if actual.shape == baseline.shape and np.array_equal(actual, baseline):
            continue

        if expected.shape == grid.shape:
            continue

        real_colors = {color for _r, _c, color in real_markers}
        if len(real_colors) < 2:
            continue

        non_four_count = sum(int(GLYPHS[int(color)][1] != 4) for _r, _c, color in real_markers)
        if non_four_count < 2:
            continue

        return grid

    raise RuntimeError("failed to generate a valid expanded variable-size dfadab01 instance")
