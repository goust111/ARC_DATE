# Variant of dfadab01
# Dimension: D1 — 字形替换变换方式 (transform)
# Change: 每个标记像素被替换为固定4x4字形图案，标记位置为字形左上角，字形使用与标记色不同的固定输出色... → 字形镜像替换

from __future__ import annotations

import random

import numpy as np


# ---------------------------------------------------------------------------
# Design analysis for the D1 variant
# ---------------------------------------------------------------------------
# design:
#   core_change:
#     - 原题把每个单像素 marker 直接替换成固定朝向的 4x4 字形。
#     - 本变体保留“marker 颜色 -> 输出颜色/字形”的字典、左上角锚定、边界裁剪、legend 删除，
#       但字形朝向不再固定，而是由 marker 所在的规则网格行列索引决定：
#       行索引奇偶控制左右镜像，列索引奇偶控制上下镜像。
#   a_concretization_and_trace:
#     input:
#       - 黑底，真实控制像素仍然只有 {2,3,5,8} 四种颜色。
#       - 真实 marker 排列在 3x3 到 5x5 的规则网格上；每一条网格行和网格列至少有一个 marker，
#         因此“第 0/1/2... 行、第 0/1/2... 列”的奇偶性是可见的。
#       - 可选 legend 仍是一个 4x4 样例字形 + 其右下方的同色标签 marker；legend 使用字典的
#         canonical 朝向，不参与镜像规则，只用于提示映射且最终被删除。
#     output:
#       - 对每个真实 marker，先查颜色对应的 canonical 4x4 mask，再按其网格行列索引变换：
#         odd row -> 左右翻转；odd col -> 上下翻转；两者都 odd -> 两次翻转。
#       - 仍以 marker 为字形左上角绘制，超界部分裁剪；legend 和其标签都不出现在输出里。
#     concrete_trace:
#       - 例：某个灰色 5-marker 位于真实 marker 网格的第 1 行第 2 列。
#       - 原逻辑：直接画出 5 对应的固定 4x4 图案。
#       - 新逻辑：第 1 行是奇数，所以先对 5 的 canonical 图案做左右镜像；第 2 列是偶数，
#         不做上下镜像；最后再按原题方式落到输出网格。
#       - 差异在于：颜色映射不变，但同一种 marker 在不同网格行/列里会呈现系统性的朝向变化。
#   b_complexity_calibration:
#     - 原题四个字形里大多是水平/垂直对称的，直接镜像几乎不可见，所以本变体把四个 4x4 字形
#       都改成了非对称 mask；其尺寸、颜色字典、稀疏程度仍保持接近原题。
#     - 规则只用两位可见自由度：行 parity 和列 parity，共四种朝向分支；复杂度明显高于原题，
#       但仍能在少量样例中归纳。
#     - 生成器强制至少出现 (even,even)、(odd,even)、(even,odd)、(odd,odd) 四类位置中的若干类，
#       并保证至少有一处镜像后输出确实不同于固定朝向输出，避免退化。
#   c_implicit_assumptions_review:
#     original_main_dependencies:
#       - 原 `main()` 默认 `mask` 与位置无关；本变体中该假设失效，必须在绘制前根据 marker 的
#         网格行列索引派生 oriented mask。
#       - 原代码逐颜色独立扫描 marker，但没有收集全局行列；本变体需要先剔除 legend 标签，
#         再从全部真实 marker 提取有序唯一行/列坐标，才能稳定计算 parity。
#       - 原 `_has_source_glyph()` 默认 legend 显示 canonical 字形；本变体沿用这一点，因此生成器
#         也必须始终画 canonical legend，不能把 legend 自身也做镜像，否则标签剔除会失效。
#     new_solver_assumptions:
#       - 每条真实 marker 网格行/列至少各有一个 marker，否则“缺失整行/整列”会让后续 parity
#         基于压缩后的索引，变得不适合作为训练样例。
#       - legend 标签只会出现在 canonical legend 的右下角 (+4,+4)。
#     generator_guarantees:
#       - 占位矩阵先覆盖所有行列，再补随机空位，所以不会缺整行/整列。
#       - legend 只在空白区域放置，且始终使用 canonical mask。
#   d_placement_legality:
#     - 行/列间距仍在 4..6，保证 4x4 输出字形之间最多接触、不重叠。
#     - 网格尺寸按行列跨度反推，确保所有 marker 像素在边界内；最后一行/列的字形可被裁剪，
#       这与原题一致且不会影响 solver。
#     - legend 搜索独立空区域，避免覆盖 marker 或制造伪 marker 行列。
#   e_visual_observability:
#     - 同一颜色的 marker 会在不同奇偶行列中呈现同一家族的不同朝向；因为字形非对称，
#       人类可直接看到“第二行左右反了、第二列上下反了”的规律。
#     - 每个样例都至少包含奇数行和奇数列，因此不会长期退化成“全部正常朝向”。
#   f_cleanliness:
#     - 仍然只有黑底、单像素 marker、可选单个 legend，视觉元素层级没有变复杂。
#     - 非对称字形依然限制在 4x4、6-7 个前景像素，输出密度接近原题，不会变成杂乱实心块。
#   g_color_rule_inference:
#     - 颜色映射本身不变，且不依赖颜色 id 排序或任何不可见数值。
#     - 可见规律完全来自位置：同色 marker 处在不同网格行/列，输出朝向按 parity 系统变化。


GLYPHS: dict[int, tuple[int, np.ndarray]] = {
    2: (
        4,
        np.array(
            [
                [1, 1, 1, 0],
                [1, 0, 0, 0],
                [1, 0, 1, 1],
                [0, 0, 0, 1],
            ],
            dtype=bool,
        ),
    ),
    3: (
        1,
        np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    5: (
        6,
        np.array(
            [
                [1, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    8: (
        7,
        np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 1, 1],
            ],
            dtype=bool,
        ),
    ),
}


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


def _orient_mask(mask: np.ndarray, row_idx: int, col_idx: int) -> np.ndarray:
    out = mask
    if row_idx % 2 == 1:
        out = np.fliplr(out)
    if col_idx % 2 == 1:
        out = np.flipud(out)
    return out


def _extract_real_markers(g: np.ndarray) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    for marker_color, (glyph_color, mask) in GLYPHS.items():
        for r, c in np.argwhere(g == marker_color):
            rr = int(r)
            cc = int(c)
            if _has_source_glyph(g, rr - 4, cc - 4, glyph_color, mask):
                continue
            markers.append((rr, cc, int(marker_color)))
    return markers


def _render_markers(shape: tuple[int, int], markers: list[tuple[int, int, int]], *, mirrored: bool) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    if not markers:
        return out
    row_ids = {int(r): idx for idx, r in enumerate(sorted({int(r) for r, _c, _mc in markers}))}
    col_ids = {int(c): idx for idx, c in enumerate(sorted({int(c) for _r, c, _mc in markers}))}
    for r, c, marker_color in markers:
        glyph_color, base_mask = GLYPHS[int(marker_color)]
        mask = base_mask if not mirrored else _orient_mask(base_mask, row_ids[int(r)], col_ids[int(c)])
        _paint_clipped(out, int(r), int(c), int(glyph_color), mask)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each colored control pixel into a 4x4 glyph whose orientation depends on marker-lattice row/column parity.

    Input structure:
    - Black background with single-pixel markers in colors 2, 3, 5, and/or 8 arranged on a sparse regular lattice.
    - Some inputs also include one legend entry: a canonical 4x4 glyph in its output color and a matching label marker
      placed one cell below-right of that glyph's bounding box.

    Rule:
    - Ignore legend glyphs and skip their label markers exactly as in the original task.
    - For each real marker, look up its fixed output color and canonical 4x4 glyph, then mirror that glyph according to
      the marker's lattice position: odd marker rows flip left-right, odd marker columns flip up-down.
    - Paint the resulting glyph with its top-left corner at the marker pixel, clipping at the grid boundary.

    Key insight:
    - Marker color still chooses the glyph family, but the visible orientation is controlled by the marker's row/column
      index within the marker lattice rather than by any extra symbol drawn in the input.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({0, *GLYPHS.keys(), 1, 4, 6, 7})

    real_markers = _extract_real_markers(g)
    assert real_markers
    return _render_markers(g.shape, real_markers, mirrored=True)


# --------------- generator (output-first mirrored glyph substitution) ---------------


def _build_occupancy(rng: random.Random, rows: int, cols: int) -> np.ndarray:
    occ = np.zeros((rows, cols), dtype=bool)

    # Force visible parity cases near the top-left of the lattice.
    occ[0, 0] = True
    occ[0, 1] = True
    occ[1, 0] = True
    occ[1, 1] = True

    for r in range(rows):
        occ[r, rng.randrange(cols)] = True
    for c in range(cols):
        occ[rng.randrange(rows), c] = True

    min_fill = max(4, int(np.ceil(0.7 * rows * cols)))
    target_fill = rng.randint(min_fill, rows * cols)
    vacant = [(r, c) for r in range(rows) for c in range(cols) if not occ[r, c]]
    rng.shuffle(vacant)
    for r, c in vacant[: max(0, target_fill - int(occ.sum()))]:
        occ[r, c] = True
    return occ


def _place_legend(
    rng: random.Random,
    grid: np.ndarray,
    legend_marker_color: int,
) -> bool:
    glyph_color, mask = GLYPHS[int(legend_marker_color)]
    h, w = grid.shape
    if h < 5 or w < 5:
        return False

    for _ in range(200):
        r0 = rng.randint(0, h - 5)
        c0 = rng.randint(0, w - 5)
        needed = [(r0 + 4, c0 + 4)]
        for dr in range(4):
            for dc in range(4):
                if bool(mask[dr, dc]):
                    needed.append((r0 + dr, c0 + dc))
        if all(int(grid[r, c]) == 0 for r, c in needed):
            for dr in range(4):
                for dc in range(4):
                    if bool(mask[dr, dc]):
                        grid[r0 + dr, c0 + dc] = int(glyph_color)
            grid[r0 + 4, c0 + 4] = int(legend_marker_color)
            return True
    return False


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    rows = rng.randint(3, 5)
    cols = rng.randint(3, 5)

    row_step = rng.randint(4, 6)
    col_steps = [rng.randint(4, 6) for _ in range(cols - 1)]
    row_span = (rows - 1) * row_step
    col_span = sum(col_steps)

    h = rng.randint(max(10, row_span + 1), row_span + 5)
    w = rng.randint(max(10, col_span + 1), col_span + 5)
    top = rng.randint(0, h - 1 - row_span)
    left = rng.randint(0, w - 1 - col_span)

    row_positions = [top + i * row_step for i in range(rows)]
    col_positions = [left]
    for step in col_steps:
        col_positions.append(col_positions[-1] + step)

    occ = _build_occupancy(rng, rows, cols)
    occupied_cells = [(r, c) for r in range(rows) for c in range(cols) if bool(occ[r, c])]
    rng.shuffle(occupied_cells)

    marker_types = rng.sample(sorted(GLYPHS), k=rng.randint(2, 4))
    assigned_colors: list[int] = marker_types[:]
    while len(assigned_colors) < len(occupied_cells):
        assigned_colors.append(rng.choice(marker_types))
    rng.shuffle(assigned_colors)

    inp = np.zeros((h, w), dtype=int)
    real_markers: list[tuple[int, int, int]] = []
    for (rr, cc), marker_color in zip(occupied_cells, assigned_colors):
        r = row_positions[rr]
        c = col_positions[cc]
        inp[r, c] = int(marker_color)
        real_markers.append((int(r), int(c), int(marker_color)))

    if rng.random() < 0.65:
        _place_legend(rng, inp, rng.choice(sorted({mc for _r, _c, mc in real_markers})))

    expected = _render_markers(inp.shape, real_markers, mirrored=True)
    baseline = _render_markers(inp.shape, real_markers, mirrored=False)

    # Reject degenerate samples where the new transform is visually identical to the old one.
    assert any(
        (ri % 2 == 1) or (ci % 2 == 1)
        for ri in range(rows)
        for ci in range(cols)
        if bool(occ[ri, ci])
    )
    assert not np.array_equal(expected, baseline)
    return inp, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        inp, expected = _generate_case(rng)
        got = main(inp)
        if np.array_equal(got, expected):
            return inp
    raise RuntimeError("failed to generate a valid mirrored-glyph sample")
