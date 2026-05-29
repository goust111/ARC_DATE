# Combo variant of dfadab01 (index 28)
# combo_spec: D2/1+D11/0+D12/1
# Name: 动态映射+全样本展示+更多标记色

from __future__ import annotations

import random

import numpy as np


# Design analysis
# ===============
# a) 具体化与行为追踪
#    - 保留“单像素 marker 展开成 4x4 glyph”的骨架，但把 marker 扩到 6 种：
#      {1,2,3,5,6,8}。输入顶部固定是一块 2x3 的完整图例区，共 6 个 legend。
#    - 每个 legend 仍由“完整 4x4 glyph 样本 + 其右下角 (+4,+4) 的 label marker”
#      组成；只是现在 6 种 marker 全都会出现，且 glyph 不再由全局颜色表硬编码，
#      而是本题局部置换决定。
#    - 具体追踪：若本题 legend 中 `(label=2)` 的左上 4x4 是紫色对角块，而
#      `(label=6)` 的左上 4x4 是黄色 X，那么原题会把 `2` 固定画成黄色边框、
#      `6` 不存在；融合版则必须把所有真实 `2` 画成紫色对角块，把所有真实 `6`
#      画成黄色 X。差异点在于 solver 先读完整图例，再批量复用映射。
#    - 输出只保留真实 marker 的展开结果；6 个 legend glyph 和 6 个 legend label
#      都消失，贴边仍然裁剪。
#
# b) 复杂度校准
#    - 只扩到 6 种 marker / 6 个 glyph，不继续放大自由度；这已经明显高于原题的
#      4 种，同时仍能在单张图中完整展示。
#    - 6 个 glyph 的轮廓固定、彼此可区分；glyph 颜色只使用 {4,7,9} 三色循环复用，
#      避免把“新增种类数”完全压在颜色记忆上，仍以形状为主线索。
#    - 真实区域强制 6 种 marker 都至少重复 2 次，保证解题者真的需要读完整字典，
#      而不是只用一两个 legend。
#
# c) 隐式假设审查
#    - 原 `GLYPHS` 把 marker_color -> glyph 直接写死；D2/1 后该前提失效，必须先
#      调 `_decode_legend_mapping()`。
#    - 原 `main()` 可以一边扫 marker 一边立即展开；现在不行，因为在不知道局部映射
#      前无法判断某颜色该画什么，也无法判断某个 marker 是否是 legend label。
#    - 原 `_has_source_glyph(... fixed color/mask ...)` 依赖“已知该 marker 对应哪个
#      glyph”；现在要对全部 glyph 候选逐一尝试匹配。
#    - D12/1 引入 6 个 marker 后，输入颜色集合不再是原题那 4 个 marker 颜色；solver
#      的合法颜色断言与生成器都必须同步扩展。
#    - D11/0 要求完整 legend，因此 `main()` 新增了“6 个 marker 颜色都能在 legend
#      中被解码”的假设；生成器通过固定 6 个 legend 全展示来满足。
#
# d) 放置合法性
#    - 画布固定为 24x24。legend 用 2x3 槽位，槽位步长 6；每个 legend 占 5x5，
#      因而互不重叠，且与下方真实区域完全分带。
#    - 真实 marker 从第 11 行开始走规则点阵，行列间距取 4 或 5，因此展开后的 4x4
#      glyph 不会互相覆盖；最后一行/列允许贴边，从而保留裁剪现象。
#    - 真实 marker 永不进入 legend 区，也不进入任何 label 单元，所以“左上刚好有
#      一个完整 glyph 样本”的 marker 唯一地就是 legend label。
#
# e) 视觉可观察性
#    - 解题者看到输入时会先看到 6 个完整样本，再看到下半部分大量单像素 marker。
#      从 2-3 个样例中很容易归纳出“先按图例解码，再替换所有同色 marker”。
#    - 与原题相比，差异是肉眼可见的：marker 颜色更多，legend 总是完整出现，而且
#      同一种 marker 在不同题里不再固定对应同一个 glyph。
#
# f) 可视化整洁度
#    - legend 区和真实区分带摆放，没有对象叠压。
#    - 真实 marker 只占规则网格点，输出仍由互不重叠的 4x4 模块构成，阅读负担可控。
#
# g) 变化的可推断性
#    - 颜色变化有明确结构性线索：每个 marker 颜色都在 legend 里配有一个完整样本。
#    - glyph 颜色允许重复，但形状不重复，因此解题者不会被“6 种新颜色”淹没，而是
#      通过“label marker + 样本轮廓”稳定解码。
#
# 9) 新 main() 假设与生成器保证
#    - 假设 1：6 个 marker 颜色都在图例中恰好给出一条映射。生成器固定绘制 6 个 legend。
#    - 假设 2：legend label 是唯一一种“其左上 4x4 恰好是某个 glyph 样本”的 marker。
#      生成器通过顶部独立图例区保证这一点。
#    - 假设 3：真实 marker 至少各出现两次，且全部 6 色都被实际使用。生成器显式构造
#      `2 * 6` 个基础 marker，再随机补充。
#    - 假设 4：局部映射不是静态默认表。生成器只接受 6 元全错位置换，并计算与默认表
#      的像素级差异 `static_disagreement > 0`，用来拒绝退化样本。


MARKER_COLORS: tuple[int, ...] = (1, 2, 3, 5, 6, 8)
LEGEND_SLOTS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (0, 6),
    (0, 12),
    (6, 0),
    (6, 6),
    (6, 12),
)
GRID_SHAPE = (24, 24)
REAL_MARKER_MIN_ROW = 11
BASE_GLYPH_BY_MARKER: dict[int, int] = {marker_color: idx for idx, marker_color in enumerate(MARKER_COLORS)}

GLYPH_LIBRARY: tuple[tuple[int, np.ndarray], ...] = (
    (
        4,
        np.array(
            [
                [1, 1, 1, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 1, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    (
        7,
        np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [0, 1, 1, 0],
            ],
            dtype=bool,
        ),
    ),
    (
        9,
        np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    (
        4,
        np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [1, 0, 0, 1],
            ],
            dtype=bool,
        ),
    ),
    (
        7,
        np.array(
            [
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    (
        9,
        np.array(
            [
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
            ],
            dtype=bool,
        ),
    ),
)
GLYPH_COLORS = {glyph_color for glyph_color, _ in GLYPH_LIBRARY}
ALLOWED_COLORS = {0, *MARKER_COLORS, *GLYPH_COLORS}


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


def _decode_legend_mapping(
    g: np.ndarray,
) -> tuple[dict[int, tuple[int, np.ndarray]], set[tuple[int, int]]]:
    mapping: dict[int, tuple[int, np.ndarray]] = {}
    legend_labels: set[tuple[int, int]] = set()

    for marker_color in MARKER_COLORS:
        for r, c in np.argwhere(g == marker_color):
            r_i = int(r)
            c_i = int(c)
            matches: list[int] = []
            for glyph_idx, (glyph_color, mask) in enumerate(GLYPH_LIBRARY):
                if _has_source_glyph(g, r_i - 4, c_i - 4, glyph_color, mask):
                    matches.append(glyph_idx)
            if not matches:
                continue
            assert len(matches) == 1, f"ambiguous legend for marker {marker_color} at {(r_i, c_i)}"
            glyph_color, mask = GLYPH_LIBRARY[matches[0]]
            if marker_color in mapping:
                prev_color, prev_mask = mapping[marker_color]
                assert prev_color == glyph_color and np.array_equal(prev_mask, mask)
            mapping[marker_color] = (glyph_color, mask)
            legend_labels.add((r_i, c_i))

    return mapping, legend_labels


def _render_with_mapping(
    shape: tuple[int, int],
    markers: list[tuple[int, int, int]],
    mapping: dict[int, int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c, marker_color in markers:
        glyph_color, mask = GLYPH_LIBRARY[mapping[marker_color]]
        _paint_clipped(out, r, c, glyph_color, mask)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a six-symbol marker field by first decoding the puzzle-local 4x4 legend dictionary.

    Input structure:
    - Black background with single-pixel markers in six possible colors: 1, 2, 3, 5, 6, and 8.
    - The top band shows all six glyph types as full legend entries: each entry is a complete 4x4 sample glyph plus a label marker one cell below-right of that sample.
    - The lower band contains the real marker lattice whose colors reuse the same six marker colors.

    Rule:
    - For each legend entry, the label marker color names the glyph drawn immediately up-left of it; together, the six entries define the per-puzzle marker-to-glyph mapping.
    - Replace every non-legend marker by its decoded glyph, placing the glyph's top-left corner at the marker position and clipping any part that extends past the grid edge.
    - Remove the legend glyphs and their label markers entirely from the output.

    Key insight:
    - Neither a fixed global lookup nor glyph color alone is enough: the full visible legend must be read first, because the mapping changes from puzzle to puzzle and several glyph colors are intentionally reused across different shapes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))).issubset(ALLOWED_COLORS)

    mapping, legend_labels = _decode_legend_mapping(g)
    assert set(mapping) == set(MARKER_COLORS), "expected a complete 6-entry legend"

    present_marker_colors = {int(color) for color in np.unique(g) if int(color) in MARKER_COLORS}
    assert present_marker_colors == set(MARKER_COLORS), "expected all 6 marker colors to appear"

    out = np.zeros_like(g)
    for marker_color in MARKER_COLORS:
        glyph_color, mask = mapping[marker_color]
        for r, c in np.argwhere(g == marker_color):
            cell = (int(r), int(c))
            if cell in legend_labels:
                continue
            _paint_clipped(out, cell[0], cell[1], glyph_color, mask)

    return out


def _sample_mapping() -> dict[int, int]:
    while True:
        perm = random.sample(range(len(GLYPH_LIBRARY)), len(GLYPH_LIBRARY))
        if all(perm[idx] != idx for idx in range(len(perm))):
            return {marker_color: perm[idx] for idx, marker_color in enumerate(MARKER_COLORS)}


def _sample_real_lattice(h: int, w: int) -> tuple[list[int], list[int]]:
    while True:
        rows = random.randint(3, 4)
        cols = random.randint(4, 5)
        if rows * cols < 2 * len(MARKER_COLORS):
            continue
        row_spacing = random.randint(4, 5)
        col_spacing = random.randint(4, 5)
        max_row_start = h - 1 - row_spacing * (rows - 1)
        max_col_start = w - 1 - col_spacing * (cols - 1)
        if max_row_start < REAL_MARKER_MIN_ROW or max_col_start < 0:
            continue
        start_r = random.randint(REAL_MARKER_MIN_ROW, max_row_start)
        start_c = random.randint(0, max_col_start)
        row_positions = [start_r + i * row_spacing for i in range(rows)]
        col_positions = [start_c + i * col_spacing for i in range(cols)]
        return row_positions, col_positions


def _sample_real_markers(
    row_positions: list[int],
    col_positions: list[int],
) -> list[tuple[int, int, int]]:
    cells = [(r, c) for r in row_positions for c in col_positions]
    total_cells = len(cells)
    min_fill = max(2 * len(MARKER_COLORS), (7 * total_cells + 9) // 10)
    fill_count = random.randint(min_fill, total_cells)
    chosen_cells = random.sample(cells, fill_count)

    colors = [marker_color for marker_color in MARKER_COLORS for _ in range(2)]
    extra = fill_count - len(colors)
    if extra > 0:
        colors.extend(random.choices(MARKER_COLORS, k=extra))
    random.shuffle(colors)

    return [(r, c, colors[idx]) for idx, (r, c) in enumerate(chosen_cells)]


def _draw_legends(inp: np.ndarray, mapping: dict[int, int]) -> None:
    slot_order = list(LEGEND_SLOTS)
    random.shuffle(slot_order)
    marker_order = list(MARKER_COLORS)
    random.shuffle(marker_order)

    for marker_color, (r0, c0) in zip(marker_order, slot_order):
        glyph_color, mask = GLYPH_LIBRARY[mapping[marker_color]]
        _paint_clipped(inp, r0, c0, glyph_color, mask)
        inp[r0 + 4, c0 + 4] = marker_color


def _build_case() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    h, w = GRID_SHAPE

    mapping = _sample_mapping()
    row_positions, col_positions = _sample_real_lattice(h, w)
    real_markers = _sample_real_markers(row_positions, col_positions)

    inp = np.zeros((h, w), dtype=int)
    for r, c, marker_color in real_markers:
        inp[r, c] = marker_color
    _draw_legends(inp, mapping)

    out = _render_with_mapping((h, w), real_markers, mapping)
    solved = main(inp)
    assert np.array_equal(solved, out)

    default_out = _render_with_mapping((h, w), real_markers, BASE_GLYPH_BY_MARKER)
    real_counts = {marker_color: 0 for marker_color in MARKER_COLORS}
    for _r, _c, marker_color in real_markers:
        real_counts[marker_color] += 1

    used_colors = {marker_color for marker_color, count in real_counts.items() if count > 0}
    moved_count = sum(int(mapping[color] != BASE_GLYPH_BY_MARKER[color]) for color in MARKER_COLORS)
    clipped_markers = sum(1 for r, c, _ in real_markers if r > h - 4 or c > w - 4)
    static_disagreement = int(np.count_nonzero(out != default_out))
    legend_label_count = sum(
        1
        for marker_color in MARKER_COLORS
        for r, c in np.argwhere(inp == marker_color)
        if any(_has_source_glyph(inp, int(r) - 4, int(c) - 4, glyph_color, mask) for glyph_color, mask in GLYPH_LIBRARY)
    )

    assert used_colors == set(MARKER_COLORS)
    assert min(real_counts.values()) >= 2
    assert moved_count == len(MARKER_COLORS)
    assert legend_label_count == len(MARKER_COLORS)
    assert static_disagreement > 0

    meta = {
        "mapping": {color: mapping[color] for color in MARKER_COLORS},
        "moved_count": moved_count,
        "legend_count": len(MARKER_COLORS),
        "used_color_count": len(used_colors),
        "real_counts": real_counts,
        "real_marker_count": len(real_markers),
        "clipped_markers": clipped_markers,
        "static_disagreement": static_disagreement,
    }
    return inp, out, meta


def generate_input() -> np.ndarray:
    inp, _, _ = _build_case()
    return inp


def generate_pair() -> tuple[np.ndarray, np.ndarray]:
    inp, out, _ = _build_case()
    return inp, out
