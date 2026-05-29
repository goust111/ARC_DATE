# Combo variant of dfadab01 (index 24)
# combo_spec: D3/0+D5/2
# Name: 2x2色块标记+可变尺寸字形

from __future__ import annotations

import random

import numpy as np


"""
design_analysis:
  a_concretization_and_trace:
    original_scene:
      - 原题把颜色 2/3/5/8 的单像素当作控制标记；输出时每个像素都扩成固定 4x4 字形。
      - 若输入里还出现一个已画好的样本字形，则它只是图例；其右下角的同色单像素标签不参与输出。
    fused_scene:
      - 真实标记与图例标签都改成 2x2 同色色块，字形锚点取该色块左上角。
      - 颜色仍决定图案家族与输出颜色，但现在还决定 bbox 尺寸：2->3x3，3->4x4，5->5x5，8->6x6。
      - 图例标签也随尺寸变化，位于样本字形左上角的 (+size, +size)，且自身占 2x2。
    concrete_trace:
      - 例如输入在左上角 (8, 11) 放一个颜色 5 的 2x2 色块，原题若把四个像素都当单标记，会错误地产生四个互相平移的 4x4 字形。
      - 融合后 solver 必须先把这四个像素识别为“一个 2x2 标记块”，再只用其左上角 (8, 11) 画一个 5x5 的对角双块字形。
      - 若另有颜色 3 的图例样本放在 (2, 14)，则其 4x4 蓝色菱形环只是提示；位于 (6, 18) 的绿色 2x2 标签块要被识别并跳过，而不是扩写。
      - 因此和原题相比，变化同时发生在“标记粒度”“图例标签几何”“字形尺寸”“裁剪范围”四个环节上。
  b_complexity_calibration:
    - 尺寸只用四个离散档位 {3,4,5,6}，与四种颜色一一对应，避免自由度过高。
    - 每个样例至少使用两种真实标记色，并给每种已用颜色至少两个真实标记，保证人类能看到重复支撑。
    - 图例仍保留 1-2 个，提供“颜色 -> 输出颜色/图案/尺寸”的直接可见线索，而不是依赖颜色数值。
  c_implicit_assumption_review_of_main:
    - 原 main() 直接对 `np.argwhere(g == marker_color)` 的每个像素扩写；融合后这会把一个 2x2 标记误当成四个标记，因此必须改成先检测孤立 2x2 色块。
    - 原 `_has_source_glyph()` 与 `_paint_clipped()` 默认 4x4；融合后都必须改成读取 `mask.shape`。
    - 原图例跳过条件使用固定偏移 `(r-4, c-4)`；融合后必须按当前颜色对应的 `size` 回推 `(r-size, c-size)`。
    - 原 seen_marker 只需见到任一控制像素；融合后应统计检测到的 2x2 标记块，否则单个噪声像素会错误满足断言。
    - 新增的 `_is_marker_block()` 依赖“标记块是孤立精确 2x2，同色邻接不会继续延展”的前提，因此 generator 需要显式保证该结构。
  d_placement_legality:
    - 真实 2x2 标记块放在规则锚点格上，行列间距从 {6,7} 取值，确保最大 6x6 输出字形也不会与邻居重叠。
    - 锚点只要求 2x2 标记本身完全落在网格内，因此靠近下边或右边时仍可触发输出裁剪。
    - 图例样本要求整个 glyph bbox、右下 2x2 标签块、以及其外侧一圈缓冲区都落在空白区域内，避免与真实标记或其他图例缠在一起。
  e_visual_observability:
    - 输入里 2x2 标记块与输出里的 3x3~6x6 字形形成明显的尺度层级，人眼可直接区分“控制标记”和“被展开的图形”。
    - 不同颜色反复出现时，解题者能同时观察到三条稳定关系：同色 2x2 块总是映射到同一种输出颜色、同一种图案、同一种尺寸。
    - 图例标签本身也是 2x2 块，进一步提示“真正需要读取的是一个小色块，而不是色块中的四个独立像素”。
  f_visual_tidiness:
    - 真实输出字形之间不重叠，图例区域与真实标记网格分离，因此网格不会出现难以拆分来源的覆盖。
    - 2x2 标记块比单像素更显眼，但仍远小于 3x3~6x6 字形，不会喧宾夺主。
    - 图例标签与样本字形对角相邻，关系清晰且不与样本本体融合成更大块。
  g_color_change_inferability:
    - 颜色映射仍沿用原题的四组固定对应，没有额外引入新的颜色机制。
    - 新增难点完全来自几何层面：解题者需要从颜色稳定地推断尺寸，而不是从不可见的颜色 id 数值做排序或计算。
  generator_guarantees_for_new_main:
    - 每个输入至少有一个真实 2x2 标记块，且至少两种真实颜色，因此 D3 与 D5 都会实际参与。
    - 所有真实标记块和图例标签块都满足“孤立精确 2x2”的检测条件，因此 solver 的块识别不会退化为逐像素处理。
    - 所有图例标签都严格放在对应样本的 (+size, +size) 位置，保证 `main()` 的 skip 分支有构造性支撑。
    - 在这些样本上，原始单像素/固定 4x4 solver 会把每个 2x2 块扩成多个 4x4 图形，故其输出与融合 solver 明显不同，可用于反退化检查。
"""


MARKER_BLOCK_SIZE = 2


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


def _is_marker_block(g: np.ndarray, r0: int, c0: int, color: int) -> bool:
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + MARKER_BLOCK_SIZE > h or c0 + MARKER_BLOCK_SIZE > w:
        return False
    if not np.all(g[r0 : r0 + MARKER_BLOCK_SIZE, c0 : c0 + MARKER_BLOCK_SIZE] == color):
        return False

    rs = max(0, r0 - 1)
    re = min(h, r0 + MARKER_BLOCK_SIZE + 1)
    cs = max(0, c0 - 1)
    ce = min(w, c0 + MARKER_BLOCK_SIZE + 1)
    neighborhood = g[rs:re, cs:ce]
    same_color = neighborhood == color
    allowed = np.zeros_like(same_color, dtype=bool)
    allowed[r0 - rs : r0 - rs + MARKER_BLOCK_SIZE, c0 - cs : c0 - cs + MARKER_BLOCK_SIZE] = True
    return not np.any(same_color & ~allowed)


def _find_marker_blocks(g: np.ndarray, color: int) -> list[tuple[int, int]]:
    h, w = g.shape
    return [
        (r, c)
        for r in range(max(0, h - MARKER_BLOCK_SIZE + 1))
        for c in range(max(0, w - MARKER_BLOCK_SIZE + 1))
        if _is_marker_block(g, r, c, color)
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each 2x2 control block into the variable-size glyph assigned to its color.

    Input structure:
    - Black background with isolated 2x2 marker blocks in colors 2, 3, 5, and/or 8; each block's top-left cell is the placement anchor.
    - Some inputs also include 1-2 legend samples: a glyph already drawn in its output color, plus a matching 2x2 label block placed one glyph-size down and right from that sample.

    Rule:
    - Replace every real 2x2 marker block by its color's output glyph, using both the mapped output color and the mapped size: 2->3x3 frame, 3->4x4 diamond ring, 5->5x5 diagonal blocks, 8->6x6 X.
    - Paint each glyph with its top-left corner aligned to the marker block's top-left cell, clipping any part that extends beyond the grid.
    - Remove legend samples and skip the 2x2 label blocks that sit at (+size, +size) from a matching sample glyph.

    Key insight:
    - The control object is no longer a single pixel but an entire 2x2 same-color block, and that block's color now determines both which glyph family to draw and how large the glyph should be.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({0, *GLYPHS.keys(), 1, 4, 6, 7})

    out = np.zeros_like(g)
    marker_block_count = 0
    for marker_color, (glyph_color, size, mask) in GLYPHS.items():
        blocks = _find_marker_blocks(g, marker_color)
        marker_block_count += len(blocks)
        for r, c in blocks:
            if _has_source_glyph(g, r - size, c - size, glyph_color, mask):
                continue
            _paint_clipped(out, r, c, glyph_color, mask)

    assert marker_block_count > 0
    return out


GRID_SIZE_OPTIONS = (20, 24)
ROWCOL_COUNT_RANGE = (2, 4)
SPACING_OPTIONS = (6, 7)
FILL_RATE_RANGE = (0.65, 1.0)
GEN_RETRIES = 500


def _choose_real_markers(
    anchors: list[tuple[int, int]],
    used_colors: list[int],
) -> list[tuple[int, int, int]]:
    cell_count = len(anchors)
    min_fill = max(
        len(used_colors) * 2,
        int(np.ceil(cell_count * random.uniform(*FILL_RATE_RANGE))),
    )
    min_fill = min(cell_count, max(len(used_colors), min_fill))
    fill_count = random.randint(min_fill, cell_count)

    chosen_positions = random.sample(anchors, fill_count)
    assignments = used_colors * 2
    while len(assignments) < fill_count:
        assignments.append(random.choice(used_colors))
    assignments = assignments[:fill_count]
    random.shuffle(assignments)
    return [(r, c, color) for (r, c), color in zip(chosen_positions, assignments, strict=True)]


def _can_place_legend(grid: np.ndarray, r0: int, c0: int, size: int) -> bool:
    h, w = grid.shape
    if r0 < 0 or c0 < 0 or r0 + size + MARKER_BLOCK_SIZE > h or c0 + size + MARKER_BLOCK_SIZE > w:
        return False
    rs = max(0, r0 - 1)
    re = min(h, r0 + size + MARKER_BLOCK_SIZE + 1)
    cs = max(0, c0 - 1)
    ce = min(w, c0 + size + MARKER_BLOCK_SIZE + 1)
    return not np.any(grid[rs:re, cs:ce] != 0)


def _try_place_legend(grid: np.ndarray, marker_color: int) -> bool:
    glyph_color, size, mask = GLYPHS[marker_color]
    candidates = [
        (r, c)
        for r in range(max(0, grid.shape[0] - size - MARKER_BLOCK_SIZE + 1))
        for c in range(max(0, grid.shape[1] - size - MARKER_BLOCK_SIZE + 1))
    ]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        if not _can_place_legend(grid, r0, c0, size):
            continue
        grid[r0 : r0 + size, c0 : c0 + size][mask] = glyph_color
        grid[
            r0 + size : r0 + size + MARKER_BLOCK_SIZE,
            c0 + size : c0 + size + MARKER_BLOCK_SIZE,
        ] = marker_color
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
    Construct a sparse lattice of 2x2 marker blocks plus 1-2 legends for the fused rule.
    """
    for _ in range(GEN_RETRIES):
        height = width = random.choice(GRID_SIZE_OPTIONS)
        row_count = random.randint(*ROWCOL_COUNT_RANGE)
        col_count = random.randint(*ROWCOL_COUNT_RANGE)
        row_spacing = random.choice(SPACING_OPTIONS)
        col_spacing = random.choice(SPACING_OPTIONS)
        max_row_start = height - MARKER_BLOCK_SIZE - (row_count - 1) * row_spacing
        max_col_start = width - MARKER_BLOCK_SIZE - (col_count - 1) * col_spacing
        if max_row_start < 0 or max_col_start < 0:
            continue

        row0 = random.randint(0, max_row_start)
        col0 = random.randint(0, max_col_start)
        anchors = [
            (row0 + rr * row_spacing, col0 + cc * col_spacing)
            for rr in range(row_count)
            for cc in range(col_count)
        ]

        max_used = min(4, len(anchors) // 2)
        if max_used < 2:
            continue
        used_count = random.randint(2, max_used)
        used_colors = random.sample(list(GLYPHS), used_count)
        real_markers = _choose_real_markers(anchors, used_colors)

        grid = np.zeros((height, width), dtype=int)
        for r, c, marker_color in real_markers:
            grid[r : r + MARKER_BLOCK_SIZE, c : c + MARKER_BLOCK_SIZE] = marker_color

        legend_target_count = 1 + int(len(used_colors) > 2 and random.random() < 0.4)
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
        return grid

    raise RuntimeError("failed to generate a valid fused dfadab01 instance")
