# Variant of dfadab01
# Dimension: D2 — 标记色与字形色的映射关系 (transform)
# Change: 固定全局映射：标记色2->字形色4（空心方块），3->1（菱形），5->6（对角块），8->7（X形），映射关系跨所有示例一致 → 相邻标记交换颜色
#
# Design analysis
# a) Concrete behavior:
#    - 形状字典不变: 2/3/5/8 仍然分别展开成原来的 4x4 外框、菱形、对角块、X 形掩码。
#    - 变化只发生在“涂什么颜色”这一步: 两个异色标记如果互为最近的曼哈顿邻居，就交换默认字形色。
#      例如红标记2默认应画黄框4、绿标记3默认应画蓝菱形1；若它们构成一对，则输出变成“红标记画黄框形状但用1涂色，
#      绿标记画蓝菱形形状但用4涂色”。
#    - 例子: (1,1) 处颜色2、(1,5) 处颜色3 是最近异色对。原题输出会是 2->4 和 3->1；变体则改成 2->1、3->4。
#      未配对单点仍然使用自己的默认颜色。
# b) Complexity calibration:
#    - 只引入一个新自由度: “是否与最近异色邻居成对”。不改字形、不改尺寸、不改背景。
#    - 为避免肉眼看不出差异，生成器始终放入至少两对已交换标记，并保留至少一个未配对单点，保证同一张图里同时出现
#      “交换色”和“默认色”两种分支。
# c) Implicit assumptions audited from the original main():
#    - 原版 `for marker_color, (glyph_color, mask) in GLYPHS.items()` 假设字形色是全局常量；现在必须先收集全部真实标记，
#      再根据空间配对决定每个标记的实际输出色。
#    - 原版 `_has_source_glyph(... glyph_color, mask)` 用默认色识别 legend label。这个假设仍然成立，因为 legend 继续展示
#      “默认映射”，而不是交换后的局部结果。
#    - 原版逐色扫描的处理顺序无关；现在若边扫边画会丢失配对信息，因此先做配对、后统一绘制。
# d) Placement legality:
#    - 生成器把 20x20 网格切成 10x10 象限，每个象限只放一个 pair / single / legend。
#    - 成对标记在同象限内相距 4 格或 5 格，保证它们是最近邻但对应的 4x4 字形不重叠。
#    - 不同象限之间最近距离至少 6，避免跨象限误配对或字形碰撞。
# e) Visual observability:
#    - 配对完全由可见空间关系决定，不依赖颜色 id 排序，也不依赖黑色数值。
#    - 同一图中会重复出现“近邻异色二元组 -> 颜色互换”的现象，并与远处单点的默认颜色形成对照。
# f) Visual cleanliness:
#    - 所有真实字形之间都保持不重叠；legend 单独占一个象限，不和真实标记混放。
#    - 输入里除单像素标记与可选 legend 外没有额外噪声，输出仍是干净的黑底字形阵列。
# g) Color inferability:
#    - 只使用原题已有的四种默认输出色 {4,1,6,7}，颜色种类不增加。
#    - 交换规则始终是“把对方的默认色拿来给自己的固定形状上色”，不会引入难以归纳的任意颜色表。
# h) Variant-specific metric used during validation:
#    - `swapped_markers`: 输出颜色不同于自己默认色的真实标记数量。生成器要求该值至少为 4（至少两对）。
#    - `default_markers`: 未配对且保持默认色的真实标记数量。生成器要求该值至少为 1。
# i) New solver assumptions and how generate_input() satisfies them:
#    - `main()` 采用“互为唯一最近的异色邻居才配对”的判定，以避免 tie 时的歧义。
#    - `generate_input()` 只构造这种无歧义布局；若采样结果让 `main(input)` 与意图输出不一致，则直接重采样。

from __future__ import annotations

import random

import numpy as np


GLYPHS: dict[int, tuple[int, np.ndarray]] = {
    2: (
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
    3: (
        1,
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
    5: (
        6,
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
    8: (
        7,
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
}

MARKER_COLORS = tuple(GLYPHS)
DEFAULT_GLYPH_COLORS = {marker_color: glyph_color for marker_color, (glyph_color, _mask) in GLYPHS.items()}
ALLOWED_COLORS = {0, *MARKER_COLORS, *DEFAULT_GLYPH_COLORS.values()}
CELL_ORIGINS = [(0, 0), (0, 10), (10, 0), (10, 10)]
PAIR_LAYOUTS = [
    ((1, 1), (1, 5)),
    ((1, 1), (5, 1)),
    ((5, 1), (5, 5)),
    ((1, 5), (5, 5)),
]
SINGLE_OFFSETS = [(1, 1), (1, 5), (5, 1), (5, 5)]


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


def _is_legend_label(g: np.ndarray, r: int, c: int, marker_color: int) -> bool:
    glyph_color, mask = GLYPHS[marker_color]
    return _has_source_glyph(g, r - 4, c - 4, glyph_color, mask)


def _collect_real_markers(g: np.ndarray) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    for marker_color in MARKER_COLORS:
        for r, c in np.argwhere(g == marker_color):
            rr = int(r)
            cc = int(c)
            if _is_legend_label(g, rr, cc, marker_color):
                continue
            markers.append((marker_color, rr, cc))
    markers.sort(key=lambda item: (item[1], item[2], item[0]))
    return markers


def _mutual_nearest_pairs(markers: list[tuple[int, int, int]]) -> dict[int, int]:
    nearest: list[list[int]] = [[] for _ in markers]
    nearest_dist: list[int | None] = [None] * len(markers)
    for i, (color_i, r_i, c_i) in enumerate(markers):
        for j, (color_j, r_j, c_j) in enumerate(markers):
            if i == j or color_i == color_j:
                continue
            dist = abs(r_i - r_j) + abs(c_i - c_j)
            best = nearest_dist[i]
            if best is None or dist < best:
                nearest_dist[i] = dist
                nearest[i] = [j]
            elif dist == best:
                nearest[i].append(j)

    pairs: dict[int, int] = {}
    used: set[int] = set()
    candidates: list[tuple[int, int, int]] = []
    for i, js in enumerate(nearest):
        if len(js) != 1:
            continue
        j = js[0]
        if len(nearest[j]) != 1 or nearest[j][0] != i:
            continue
        if i < j:
            candidates.append((int(nearest_dist[i]), i, j))

    candidates.sort()
    for _dist, i, j in candidates:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        pairs[i] = j
        pairs[j] = i
    return pairs


def _render_output_from_markers(
    shape: tuple[int, int],
    markers: list[tuple[int, int, int]],
    pair_map: dict[int, int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for i, (marker_color, r, c) in enumerate(markers):
        glyph_color = DEFAULT_GLYPH_COLORS[marker_color]
        if i in pair_map:
            partner_color = markers[pair_map[i]][0]
            glyph_color = DEFAULT_GLYPH_COLORS[partner_color]
        mask = GLYPHS[marker_color][1]
        _paint_clipped(out, r, c, glyph_color, mask)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each control pixel into its fixed 4x4 glyph, but let nearby different-color pairs exchange glyph colors.

    Input structure:
    - Black background with single-pixel markers in colors 2, 3, 5, and/or 8.
    - Some inputs may also include one legend entry: a 4x4 sample glyph in its default output color plus a same-type marker at the sample's lower-right corner.

    Rule:
    - Keep the original 4x4 shape dictionary fixed for each marker color, and still place each glyph with its top-left corner at the marker location, clipping at the border.
    - Ignore legend labels and erase legend glyphs. Among the remaining real markers, two different-color markers swap output colors only when they are each other's unique nearest different-color neighbor in Manhattan distance.
    - Any real marker that is not part of such a pair uses its own default glyph color.

    Key insight:
    - Marker color still chooses the glyph shape; the new reasoning step is spatial pairing, because local nearest-neighbor relationships now decide whether the glyph keeps its default color or borrows a neighbor's.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset(ALLOWED_COLORS)

    markers = _collect_real_markers(g)
    assert markers
    pair_map = _mutual_nearest_pairs(markers)
    return _render_output_from_markers(g.shape, markers, pair_map)


def generate_input() -> np.ndarray:
    for _attempt in range(400):
        inp = np.zeros((20, 20), dtype=int)
        cells = CELL_ORIGINS[:]
        random.shuffle(cells)

        num_pairs = random.choice([2, 3])
        num_singles = 1 if num_pairs == 3 else random.choice([1, 2])
        pair_cells = cells[:num_pairs]
        single_cells = cells[num_pairs : num_pairs + num_singles]
        free_cells = cells[num_pairs + num_singles :]

        while True:
            pair_specs = [tuple(random.sample(MARKER_COLORS, 2)) for _ in range(num_pairs)]
            single_color = random.choice(MARKER_COLORS)
            all_colors = set()
            for a, b in pair_specs:
                all_colors.add(a)
                all_colors.add(b)
            all_colors.add(single_color)
            if len(all_colors) >= 3:
                break

        markers: list[tuple[int, int, int]] = []
        intended_pairs: dict[int, int] = {}

        for pair_idx, (cell_r, cell_c) in enumerate(pair_cells):
            layout = random.choice(PAIR_LAYOUTS)
            colors = pair_specs[pair_idx]
            idx0 = len(markers)
            (dr0, dc0), (dr1, dc1) = layout
            markers.append((colors[0], cell_r + dr0, cell_c + dc0))
            markers.append((colors[1], cell_r + dr1, cell_c + dc1))
            intended_pairs[idx0] = idx0 + 1
            intended_pairs[idx0 + 1] = idx0

        for cell_r, cell_c in single_cells:
            dr, dc = random.choice(SINGLE_OFFSETS)
            markers.append((single_color, cell_r + dr, cell_c + dc))

        for marker_color, r, c in markers:
            inp[r, c] = marker_color

        if free_cells and random.random() < 0.8:
            cell_r, cell_c = free_cells[0]
            legend_marker_color = random.choice([marker_color for marker_color, _r, _c in markers])
            legend_glyph_color, legend_mask = GLYPHS[legend_marker_color]
            glyph_r = cell_r + 1
            glyph_c = cell_c + 1
            _paint_clipped(inp, glyph_r, glyph_c, legend_glyph_color, legend_mask)
            inp[glyph_r + 4, glyph_c + 4] = legend_marker_color

        expected = _render_output_from_markers(inp.shape, markers, intended_pairs)
        actual = main(inp)
        if not np.array_equal(actual, expected):
            continue

        swapped_markers = sum(
            DEFAULT_GLYPH_COLORS[marker_color] != DEFAULT_GLYPH_COLORS[markers[intended_pairs[i]][0]]
            for i, (marker_color, _r, _c) in enumerate(markers)
            if i in intended_pairs
        )
        default_markers = len(markers) - len(intended_pairs)
        if swapped_markers < 4 or default_markers < 1:
            continue

        default_output = _render_output_from_markers(inp.shape, markers, {})
        if np.array_equal(default_output, expected):
            continue

        return inp

    raise RuntimeError("failed to generate a valid dfadab01 D2 variant sample")
