# Combo variant of dfadab01 (index 11)
# combo_spec: D1/0+D5/2
# Name: 字形放大+可变尺寸

from __future__ import annotations

import random

import numpy as np


"""
design_analysis:
  a_concretization_and_trace:
    original_scene:
      - 真实对象是若干单像素 marker；输出把每个 marker 直接替换成固定 4x4 字形。
      - 若输入中额外出现一个样本字形，则它只是图例；右下外侧的同类 marker 是标签，不会在输出里展开。
    fused_scene:
      - 保留四种 marker 色到四种输出色/图案家族的映射，但先给每种颜色分配不同基础尺寸：
        2 -> 3x3 空心框, 3 -> 4x4 菱形环, 5 -> 5x5 对角块, 8 -> 6x6 X。
      - 然后对所有基础字形统一做 2 倍像素放大，所以实际绘制 bbox 变为：
        2 -> 6x6, 3 -> 8x8, 5 -> 10x10, 8 -> 12x12。
      - 图例标签必须跟着“实际样本 bbox”移动，因此偏移是 (+6,+6) / (+8,+8) / (+10,+10) / (+12,+12)，而不是原题固定 (+4,+4)。
    concrete_trace:
      - 原题里，若在 (7, 9) 放一个颜色 5 marker，则输出在 [7:11, 9:13] 内画固定 4x4 对角块。
      - 融合后，同样位置的颜色 5 marker 先对应 5x5 基础图案，再统一放大 2 倍，所以输出在 [7:17, 9:19] 内画 10x10 对角块。
      - 颜色 2 的 marker 则只对应 6x6 放大框；因此 solver 必须同时处理“不同颜色的基础 size 不同”和“所有颜色共享同一 scale”。
  b_complexity_calibration:
    - 基础尺寸只取四个离散档位 {3,4,5,6}，统一倍率固定为 2，避免自由参数过多。
    - 每个输入至少使用两种 marker 色，且每种已使用颜色至少出现两次，让人能从重复实例看出颜色和最终大小的稳定对应。
    - 仍保留 1-2 个图例样本，帮助解题者把 marker 色、输出色、图案族、最终 bbox 尺寸联系起来。
  c_implicit_assumption_review_of_main:
    - 原 `GLYPHS` 假设所有 mask 都是 4x4；现在必须存放大后的 mask，shape 随颜色变化。
    - 原 `_has_source_glyph()` 用常数 4 裁 patch；现在必须读取 `mask.shape`，否则无法识别大图例。
    - 原 `_paint_clipped()` 固定循环 4 次；现在也必须改为遍历实际放大后的 `gh, gw`。
    - 原 `main()` 用 `(r-4, c-4)` 回溯图例标签来源；现在必须按该颜色的最终绘制尺寸 `(r-gh, c-gw)` 检查。
    - 原 generator 只需 spacing >= 4；融合后若 spacing 小于 12，最大 12x12 的 X 会和相邻字形重叠，所以 spacing 下界必须提升。
  d_placement_legality:
    - 真实 marker 放在规则锚点格上，行列间距从 {12,13,14} 中选，保证最大 12x12 字形也不互相覆盖。
    - 图例样本必须整块落在空白区域，且标签像素也必须为空；否则会误把真实 marker 当图例标签或反过来。
    - 真实 marker 允许靠近右边或下边，从而保留“放大后更容易被裁剪”的融合现象。
  e_visual_observability:
    - 在输出里，6x6 / 8x8 / 10x10 / 12x12 的 bbox 差异直接可见，不依赖颜色数值。
    - 同一输入至少两种颜色重复出现，因此解题者能看到“相同 marker 色 -> 相同输出色 + 相同图案族 + 相同放大后尺寸”。
    - 图例给出一个可直接比对的样本；其标签偏移与样本实际 bbox 一致，也帮助观察者意识到尺寸已改变。
  f_visual_tidiness:
    - 真实字形之间不重叠；图例与真实 marker 分开摆放，避免多层覆盖造成阅读混乱。
    - 小基础 size 分配给仍清晰的图案，放大后边界更明显，避免 3x3/4x4 在原尺度下过于细碎。
  g_change_inferability:
    - 颜色映射仍是一一对应，没有加入额外的颜色重编码。
    - 新信息完全通过可见的最终几何尺寸和统一 2 倍像素块体现；若按原题 4x4 规则去解释，同一输入上的输出面积、图例偏移、边缘裁剪范围都会系统性不匹配。
  generator_guarantees_for_new_main:
    - 每个输入至少含一个真实 marker，因此 `seen_marker` 总能成立。
    - 每个输入至少含两种真实 marker 色，并且每种已使用颜色至少出现两次，保证多尺寸现象可观察。
    - 每个图例标签都严格位于其放大样本的 `(r + final_h, c + final_w)`，因此 `main()` 的 skip 条件有构造性保证。
    - generator 会用 `main(grid) == expected` 自检，同时额外约束样本包含至少两种最终尺寸，避免退化成“只有一种大小时看不出 D5/D1 融合”。
"""


SCALE = 2


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


def _scale_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(mask.astype(np.uint8), np.ones((scale, scale), dtype=np.uint8)).astype(bool)


BASE_GLYPHS: dict[int, tuple[int, int, np.ndarray]] = {
    2: (4, 3, _frame_mask(3)),
    3: (1, 4, _diamond_ring_mask(4)),
    5: (6, 5, _diag_blocks_mask(5)),
    8: (7, 6, _x_mask(6)),
}

GLYPHS: dict[int, tuple[int, int, int, np.ndarray]] = {
    marker_color: (
        glyph_color,
        base_size,
        base_size * SCALE,
        _scale_mask(base_mask, SCALE),
    )
    for marker_color, (glyph_color, base_size, base_mask) in BASE_GLYPHS.items()
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
    Expand each colored control pixel into a size-coded glyph that is uniformly magnified.

    Input structure:
    - Black background with single-pixel markers in colors 2, 3, 5, and/or 8.
    - Some inputs also include legend samples: an already-drawn enlarged glyph in its output color, plus a same-type marker one cell beyond that sample's lower-right corner.

    Rule:
    - Replace each real marker with the glyph family tied to its color, using that color's output hue and base size, then enlarge the whole glyph by a shared factor of 2 before painting.
    - The resulting painted sizes are 2->6x6 frame, 3->8x8 diamond ring, 5->10x10 diagonal blocks, and 8->12x12 X; draw each with its top-left corner at the marker location and clip at grid edges.
    - Remove legend samples and skip the label markers that sit at (+final_height, +final_width) from their matching enlarged sample glyphs.

    Key insight:
    - Marker color controls two linked visible parameters at once: which glyph family to use and what its pre-scale base size is; every glyph is then expanded by the same hidden global scale, so both the painted bbox and the legend offset grow together.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({0, *GLYPHS.keys(), 1, 4, 6, 7})

    out = np.zeros_like(g)
    seen_marker = False
    for marker_color, (glyph_color, _base_size, _final_size, mask) in GLYPHS.items():
        coords = np.argwhere(g == marker_color)
        if coords.size:
            seen_marker = True
        gh, gw = mask.shape
        for r, c in coords:
            if _has_source_glyph(g, int(r) - gh, int(c) - gw, glyph_color, mask):
                continue
            _paint_clipped(out, int(r), int(c), glyph_color, mask)

    assert seen_marker
    return out


GRID_SIZE_OPTIONS = (24, 28, 30)
ROWCOL_COUNT_RANGE = (2, 3)
SPACING_OPTIONS = (12, 13, 14)
FILL_RATE_RANGE = (0.65, 1.0)
GEN_RETRIES = 500


def _all_top_left_positions(height: int, width: int, gh: int, gw: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(height - gh + 1) for c in range(width - gw + 1)]


def _choose_real_markers(
    anchors: list[tuple[int, int]],
    used_colors: list[int],
) -> list[tuple[int, int, int]]:
    cell_count = len(anchors)
    min_required = len(used_colors) * 2
    if cell_count < min_required:
        raise ValueError("not enough anchors to repeat each used color twice")

    min_fill = max(min_required, int(np.ceil(cell_count * random.uniform(*FILL_RATE_RANGE))))
    fill_count = random.randint(min_fill, cell_count)
    chosen_positions = random.sample(anchors, fill_count)

    assignments = used_colors * 2
    while len(assignments) < fill_count:
        assignments.append(random.choice(used_colors))
    assignments = assignments[:fill_count]
    random.shuffle(assignments)
    return [(r, c, color) for (r, c), color in zip(chosen_positions, assignments, strict=True)]


def _try_place_legend(grid: np.ndarray, marker_color: int) -> bool:
    glyph_color, _base_size, _final_size, mask = GLYPHS[marker_color]
    gh, gw = mask.shape
    candidates = _all_top_left_positions(grid.shape[0], grid.shape[1], gh, gw)
    random.shuffle(candidates)
    for r0, c0 in candidates:
        label = (r0 + gh, c0 + gw)
        if label[0] >= grid.shape[0] or label[1] >= grid.shape[1]:
            continue
        patch = grid[r0 : r0 + gh, c0 : c0 + gw]
        if np.any(patch != 0) or grid[label] != 0:
            continue
        patch[mask] = glyph_color
        grid[label] = marker_color
        return True
    return False


def _build_expected(height: int, width: int, markers: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.zeros((height, width), dtype=int)
    for r, c, marker_color in markers:
        glyph_color, _base_size, _final_size, mask = GLYPHS[marker_color]
        _paint_clipped(out, r, c, glyph_color, mask)
    return out


def _has_multiple_final_sizes(markers: list[tuple[int, int, int]]) -> bool:
    sizes = {GLYPHS[color][3].shape[0] for _, _, color in markers}
    return len(sizes) >= 2


def _has_clipped_marker(height: int, width: int, markers: list[tuple[int, int, int]]) -> bool:
    for r, c, color in markers:
        gh, gw = GLYPHS[color][3].shape
        if r + gh > height or c + gw > width:
            return True
    return False


def generate_input() -> np.ndarray:
    """
    Construct a regular marker lattice plus enlarged legend samples for the fused size+scale rule.
    """
    for _ in range(GEN_RETRIES):
        height = width = random.choice(GRID_SIZE_OPTIONS)
        row_count = random.randint(*ROWCOL_COUNT_RANGE)
        col_count = random.randint(*ROWCOL_COUNT_RANGE)
        cell_count = row_count * col_count

        max_used = min(4, cell_count // 2)
        if max_used < 2:
            continue
        used_count = random.randint(2, max_used)

        row_spacing = random.choice(SPACING_OPTIONS)
        col_spacing = random.choice(SPACING_OPTIONS)
        max_row_start = height - 1 - (row_count - 1) * row_spacing
        max_col_start = width - 1 - (col_count - 1) * col_spacing
        if max_row_start < 0 or max_col_start < 0:
            continue

        row0 = random.randint(0, max_row_start)
        col0 = random.randint(0, max_col_start)
        anchors = [
            (row0 + rr * row_spacing, col0 + cc * col_spacing)
            for rr in range(row_count)
            for cc in range(col_count)
        ]

        used_colors = random.sample(list(GLYPHS), used_count)
        real_markers = _choose_real_markers(anchors, used_colors)
        if not _has_multiple_final_sizes(real_markers):
            continue

        grid = np.zeros((height, width), dtype=int)
        for r, c, marker_color in real_markers:
            grid[r, c] = marker_color

        legend_target_count = 1 + int(len(used_colors) > 1 and random.random() < 0.45)
        legend_colors = random.sample(used_colors, legend_target_count)
        placed_legends = 0
        for marker_color in legend_colors:
            placed_legends += int(_try_place_legend(grid, marker_color))
        if placed_legends == 0:
            continue

        expected = _build_expected(height, width, real_markers)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue
        if not (_has_multiple_final_sizes(real_markers) and (placed_legends >= 1)):
            continue
        if not (_has_clipped_marker(height, width, real_markers) or random.random() < 0.55):
            continue
        return grid

    raise RuntimeError("failed to generate a valid enlarged variable-size dfadab01 instance")
