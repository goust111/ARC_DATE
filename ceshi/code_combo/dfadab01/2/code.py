# Combo variant of dfadab01 (index 2)
# combo_spec: D1/2+D5/1
# Name: 奇偶翻转+非对称字形

from __future__ import annotations

import random

import numpy as np


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
# design:
#   core_change:
#     - 原题把 marker 颜色直接映射到固定朝向的 4x4 对称字形，所以同色 marker 在任何位置都长一样。
#     - 本融合变体同时修改两件事：
#       1) 四个 canonical 字形都换成明显非对称的 4x4 图案。
#       2) 放置时只看 marker 所在“可见 marker 行”的奇偶性：偶数行保持 canonical 朝向，
#          奇数行做左右翻转。列位置不再参与变换。
#     - 这样同一种颜色在不同 marker 行会出现两种可见朝向，且因为字形非对称，水平翻转可直接观察到。
#   a_concretization_and_trace:
#     input:
#       - 黑底；真实对象仍然只是颜色 {2,3,5,8} 的单像素 marker，排成 3x2 到 5x5 的稀疏规则网格。
#       - 可选 legend 仍是 canonical 4x4 字形，使用输出颜色绘制，并在其右下角 (+4,+4) 放一个同类 marker。
#       - legend 本身不参与奇偶翻转，它只负责提示“这种 marker 会变成哪一种非对称字形/颜色”。
#     output:
#       - 对每个真实 marker，先按 marker 颜色查 canonical mask 和输出颜色。
#       - 若该 marker 属于第 0、2、4... 条可见 marker 行，就直接画 canonical mask；
#         若属于第 1、3... 条可见 marker 行，就对 mask 做左右翻转后再画。
#       - 仍以 marker 为字形左上角，边界外部分裁剪；legend 和其标签 marker 从输出中消失。
#     concrete_trace:
#       - 例：某个灰色 5-marker 位于真实 marker 网格的第二条可见行（行索引 1）。
#       - 原题：无论它在哪一行，都画同一个固定 4x4 图案。
#       - 本变体：先取 5 对应的非对称 canonical 图案，再左右翻转，因为它在奇数行；
#         若另一个 5-marker 在第 0 行，则保留 canonical 朝向。
#       - 差异是：颜色映射仍固定，但朝向由“所在行 parity”控制，而不是完全固定。
#   b_complexity_calibration:
#     - 只引入一个新的结构自由度：marker 行 parity，取值仅有 even / odd 两档，易于从少量样例总结。
#     - 为了避免“翻了也看不出来”，四个字形都选成水平非对称图案，但仍限制在 4x4、6-7 个像素内，
#       视觉密度接近原题，不会膨胀成大块实心区域。
#     - 生成器强制至少两条可见 marker 行，并保证至少一种颜色在偶数行和奇数行都重复出现，
#       让解题者能比较出“同色同字形家族，但第二行是水平镜像”。
#   c_implicit_assumptions_review:
#     original_main_dependencies:
#       - 原 `main()` 假设 mask 与位置无关；这里失效，必须先抽取真实 marker，再根据其行索引派生朝向。
#       - 原代码逐颜色扫描即可；这里不够，因为 parity 依赖全部真实 marker 的有序行集合。
#       - 原 `_has_source_glyph()` 假设 legend 使用固定 canonical 字形；本变体仍保留这个假设，
#         所以 generator 不能把 legend 也做翻转。
#     new_solver_assumptions:
#       - 每一条可见 marker 行都至少有一个真实 marker，否则压缩后的行索引不稳定，训练样例会模糊。
#       - legend 标签只会出现在 canonical legend 的右下角 (+4,+4)，且周围 legend 区域不会混入别的颜色。
#       - 所有用于翻转的字形都与其左右镜像不同，否则奇偶规则会局部不可见。
#     generator_guarantees:
#       - 占位矩阵先覆盖每一行和每一列，再追加随机填充，因此不会缺整行/整列。
#       - 至少一对同色 marker 被固定放在第 0 行和第 1 行，用来提供跨 parity 对照。
#       - legend 只放在空白区，且始终使用 canonical mask。
#   d_placement_legality:
#     - 行间距 4..6、列间距 4..6，保证 4x4 输出字形最多接触、不重叠。
#     - marker 像素始终在边界内；最后一行/列的字形允许被裁剪，这与原题一致。
#     - legend 搜索 5x5 空白区域，避免覆盖真实 marker 或伪造新的 marker 行。
#   e_visual_observability:
#     - 因为同一颜色会跨奇偶行重复，解题者能直接看到“上面这一行是 L 形，下面同色变成反向 L”。
#     - 规则只与可见行顺序相关，不依赖颜色数值、扫描顺序之外的隐式信息。
#     - 输入仍保留 optional legend，帮助先学到颜色→字形映射，再从多处重复中观察行 parity 翻转。
#   f_cleanliness:
#     - 输入仍是黑底 + 单像素 marker + 最多一个 legend，没有引入新的装饰层。
#     - 输出里只出现四类非对称 4x4 字形，且间距足够，局部结构清楚，不会互相糊在一起。
#   g_change_inferability:
#     - 颜色规则仍是固定字典，没有额外颜色条件；变化只发生在字形朝向上。
#     - 生成器拒绝与“无翻转原题”输出完全相同的样本，并统计像素差异作为非退化指标，
#       保证融合后的核心行为在样本中实际生效。
#   post_main_assumption_audit:
#     - `_extract_real_markers()` 依赖 legend 是 canonical 朝向且标签位置固定，generator 已满足。
#     - `_render_markers()` 依赖同一可见行的 marker 共享同一 y 坐标；规则网格构造满足这一点。
#     - `_orient_mask_for_row()` 只根据排序后的唯一行集合取 parity，因此 generator 必须保证没有整行缺失。


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


def _orient_mask_for_row(mask: np.ndarray, row_idx: int) -> np.ndarray:
    if row_idx % 2 == 1:
        return np.fliplr(mask)
    return mask


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


def _render_markers(shape: tuple[int, int], markers: list[tuple[int, int, int]], *, flip_by_row: bool) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    if not markers:
        return out
    row_ids = {int(r): idx for idx, r in enumerate(sorted({int(r) for r, _c, _mc in markers}))}
    for r, c, marker_color in markers:
        glyph_color, base_mask = GLYPHS[int(marker_color)]
        mask = base_mask if not flip_by_row else _orient_mask_for_row(base_mask, row_ids[int(r)])
        _paint_clipped(out, int(r), int(c), int(glyph_color), mask)
    return out


def _row_ids_from_markers(markers: list[tuple[int, int, int]]) -> dict[int, int]:
    return {int(r): idx for idx, r in enumerate(sorted({int(r) for r, _c, _mc in markers}))}


def _cross_parity_same_color(markers: list[tuple[int, int, int]]) -> bool:
    row_ids = _row_ids_from_markers(markers)
    color_parities: dict[int, set[int]] = {}
    for r, _c, marker_color in markers:
        color_parities.setdefault(int(marker_color), set()).add(int(row_ids[int(r)] % 2))
    return any(len(parities) >= 2 for parities in color_parities.values())


def _flip_delta_pixels(markers: list[tuple[int, int, int]], shape: tuple[int, int]) -> int:
    flipped = _render_markers(shape, markers, flip_by_row=True)
    baseline = _render_markers(shape, markers, flip_by_row=False)
    return int(np.count_nonzero(flipped != baseline))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each control pixel into a non-symmetric 4x4 glyph, flipping it horizontally on odd marker rows.

    Input structure:
    - Black background with single-pixel markers in colors 2, 3, 5, and/or 8 placed on a sparse regular lattice.
    - Some inputs also include one legend entry: a canonical 4x4 glyph in its output color plus a matching label marker
      one cell below-right of the glyph's bounding box.

    Rule:
    - Remove legend glyphs and skip their label markers just as in the base task.
    - For each real marker, look up its fixed output color and canonical non-symmetric 4x4 glyph.
    - If the marker lies on an even-indexed visible marker row, draw the glyph as-is; if it lies on an odd-indexed
      visible marker row, draw the left-right mirror of that glyph. Anchor every glyph at the marker pixel and clip at
      the grid boundary.

    Key insight:
    - Marker color still determines which glyph family appears, but the glyph's facing direction is controlled by row
      parity within the marker lattice, so the same color can produce two mirrored shapes on alternating rows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({0, *GLYPHS.keys(), 1, 4, 6, 7})

    real_markers = _extract_real_markers(g)
    assert real_markers
    return _render_markers(g.shape, real_markers, flip_by_row=True)


# --------------- generator (output-first row-parity mirrored substitution) ---------------


def _build_occupancy(rng: random.Random, rows: int, cols: int) -> np.ndarray:
    occ = np.zeros((rows, cols), dtype=bool)

    # Force visible support on both parities near the top-left.
    occ[0, 0] = True
    occ[1, 0] = True
    if cols >= 2:
        occ[0, 1] = True
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


def _place_legend(rng: random.Random, grid: np.ndarray, legend_marker_color: int) -> bool:
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


def _choose_anchor_colors(rng: random.Random, marker_types: list[int], cols: int) -> dict[tuple[int, int], int]:
    anchors: dict[tuple[int, int], int] = {}
    first_color = rng.choice(marker_types)
    anchors[(0, 0)] = first_color
    anchors[(1, 0)] = first_color
    if cols >= 2:
        second_color = rng.choice(marker_types)
        anchors[(0, 1)] = second_color
        anchors[(1, 1)] = second_color
    return anchors


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray, int]:
    rows = rng.randint(3, 5)
    cols = rng.randint(2, 5)

    row_step = rng.randint(4, 6)
    col_steps = [rng.randint(4, 6) for _ in range(cols - 1)]
    row_span = (rows - 1) * row_step
    col_span = sum(col_steps)

    min_h = max(10, row_span + 1)
    max_h = max(min_h, row_span + 5)
    min_w = max(10, col_span + 1)
    max_w = max(min_w, col_span + 5)
    h = rng.randint(min_h, max_h)
    w = rng.randint(min_w, max_w)
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
    anchors = _choose_anchor_colors(rng, marker_types, cols)

    inp = np.zeros((h, w), dtype=int)
    real_markers: list[tuple[int, int, int]] = []
    used_colors = set(anchors.values())

    remaining_types = marker_types[:]
    rng.shuffle(remaining_types)
    extras: list[int] = []
    while len(used_colors | set(extras)) < len(marker_types):
        extras.append(remaining_types.pop())

    free_cells = [cell for cell in occupied_cells if cell not in anchors]
    rng.shuffle(free_cells)
    forced_assignments = list(zip(free_cells, extras))
    forced_cells = {cell for cell, _marker_color in forced_assignments}

    for rr, cc in occupied_cells:
        if (rr, cc) in anchors:
            marker_color = anchors[(rr, cc)]
        elif (rr, cc) in forced_cells:
            marker_color = next(color for cell, color in forced_assignments if cell == (rr, cc))
        else:
            marker_color = rng.choice(marker_types)
        r = row_positions[rr]
        c = col_positions[cc]
        inp[r, c] = int(marker_color)
        real_markers.append((int(r), int(c), int(marker_color)))

    if rng.random() < 0.7:
        _place_legend(rng, inp, rng.choice(sorted({mc for _r, _c, mc in real_markers})))

    expected = _render_markers(inp.shape, real_markers, flip_by_row=True)
    delta = _flip_delta_pixels(real_markers, inp.shape)

    assert _cross_parity_same_color(real_markers)
    assert any(_row_ids_from_markers(real_markers)[int(r)] % 2 == 1 for r, _c, _mc in real_markers)
    assert delta > 0
    return inp, expected, delta


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        inp, expected, _delta = _generate_case(rng)
        got = main(inp)
        if np.array_equal(got, expected):
            return inp
    raise RuntimeError("failed to generate a valid row-parity mirrored glyph sample")
