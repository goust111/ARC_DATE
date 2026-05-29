# Combo variant of dfadab01 (index 5)
# combo_spec: D2/3+D3/2
# Name: 邻近色交换+对角像素对标记
#
# Design analysis
# a) Concrete behavior and trace:
#    - 每个逻辑标记不再是单像素，而是一个同色对角像素对，位于一个固定 5x5 盒子的两端；其中心点就是原题里
#      的 glyph 左上角锚点。
#    - 例子：颜色 2 的像素对在 (0,0) 与 (4,4)，颜色 3 的像素对在 (0,4) 与 (4,8)。二者分别解码成中心
#      (2,2) 与 (2,6)。原题会把单点 2 直接画成颜色 4 的方框、单点 3 直接画成颜色 1 的菱形；融合后要先恢复
#      这两个中心，再比较中心间曼哈顿距离，发现它们互为最近异色邻居，于是仍画各自的固定形状，但颜色互换成
#      2->1、3->4。
#    - 单标记分支也保留：例如一个远离其他标记的颜色 8 对角对会先解码到中心，再因没有互为最近异色伙伴而保留
#      默认颜色 7。这样同一张图里同时出现“先配对解码”与“近邻换色”两层推理。
# b) Complexity calibration:
#    - 为了让“对角像素对”足够稳定可见，生成器只使用固定 5x5 标记盒，而不是任意尺寸矩形；这仍然符合
#      “对角像素对确定中心”的核心规则，但避免少量样例下出现过多几何自由度。
#    - 为了让换色规律可观察，生成器强制至少放入两对已交换的异色标记和至少一个未交换单体；因此解题者必须同时
#      解释“为什么有些形状换色、有些没有”以及“为什么输入里两个像素才对应一个输出 glyph”。
# c) Hidden-assumption audit of the original solver:
#    - 原版 `np.argwhere(g == marker_color)` 默认“一像素=一标记”；融合后必须先把每种颜色的像素按对角关系两两配对，
#      否则会把一个逻辑标记误当成两个。
#    - 原版 `for marker_color, (glyph_color, mask) in GLYPHS.items()` 默认字形颜色是全局常量；融合后形状仍由
#      `marker_color` 决定，但颜色要等全部真实标记收集完成后，根据最近异色邻居关系再决定。
#    - 原版 `_has_source_glyph(g, r - 4, c - 4, glyph_color, mask)` 假设 `(r,c)` 是 legend label 的旧锚点；
#      现在 `(r,c)` 先由对角对解码得到，但 legend 仍展示默认映射，所以只要先解码中心再做同样检测即可。
#    - 原版按颜色边扫边画无副作用；融合后如果边解码边绘制，就无法知道某个标记是否需要与别的颜色交换，因此必须
#      改为“先解码所有真实标记 -> 再求配对 -> 最后统一绘制”。
# d) Placement legality:
#    - 生成器固定使用 20x20 画布并切成四个 10x10 象限。每个交换对独占一个象限，单体独占一个象限，legend 若出现
#      则占剩余空象限。
#    - 象限内两个中心只取相距 4 的水平或垂直布局，并让两者使用同一对角朝向，从而保证四个端点互不重合，且输出
#      的两个 4x4 glyph 只接触边界、不重叠。
#    - 不同象限中心之间最近距离至少 6，所以不会出现跨象限的最近邻反转，也不会产生同色像素的错误跨配对。
# e) Visual observability:
#    - 输入里所有真实标记都显式呈现为“同色两点成对角”，输出里却只出现一个 glyph；从 2-4 组样例中，人类可以
#      直接看出需要先取中点而不是把每个点单独展开。
#    - 同时，靠得最近的异色二元组会反复表现出“形状不变但颜色互借”，而远处单体保留默认色，形成清晰对照。
# f) Visual cleanliness:
#    - marker 颜色 {2,3,5,8} 与 legend/sample glyph 颜色 {1,4,6,7} 分离，角色不会混淆。
#    - 每个功能块被隔离到独立象限，避免同一区域叠加过多对象；legend 也不与真实标记混放。
# g) Color inferability:
#    - 输出颜色仍只来自原题四个默认 glyph 色 {4,1,6,7}，不引入新调色板。
#    - 颜色变化不是任意重映射，而是受可见空间结构约束：只有互为最近异色邻居的两个 decoded markers 互换默认色。
# h) Variant-specific metrics used during validation:
#    - `swapped_markers`: 实际输出里使用了“别人的默认色”的真实标记数，要求至少为 4（两对）。
#    - `default_markers`: 未参与交换而保持默认色的真实标记数，要求至少为 1。
#    - `swap_effect_cells`: 融合输出与“不做换色、只做对角解码”的输出之间的差异像素数，要求大于 0。
# i) New solver assumptions and how generate_input() satisfies them:
#    - `main()` 假设每个 marker-colored 像素恰好有一个同色对角伙伴，偏移为 `(±4, ±4)`；生成器只使用这类
#      固定 5x5 对角对，并在返回前再次用解码函数校验。
#    - `main()` 采用“互为唯一最近的异色邻居才交换”的判定来消除 tie；生成器通过象限分区保证这些最近邻关系
#      无歧义，并且用 `main(inp) == expected` 与 `swap_effect_cells > 0` 双重验证。

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

PAIR_RADIUS = 2
GRID_SHAPE = (20, 20)
CELL_ORIGINS = ((0, 0), (0, 10), (10, 0), (10, 10))
LEGEND_PROB = 0.8
GEN_RETRIES = 500

PAIR_DIAGONALS: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
    ((-PAIR_RADIUS, -PAIR_RADIUS), (PAIR_RADIUS, PAIR_RADIUS)),
    ((-PAIR_RADIUS, PAIR_RADIUS), (PAIR_RADIUS, -PAIR_RADIUS)),
)

PAIR_CLUSTER_LAYOUTS: tuple[tuple[tuple[int, int], tuple[int, int], int], ...] = (
    ((2, 2), (2, 6), 0),
    ((2, 2), (2, 6), 1),
    ((2, 2), (6, 2), 0),
    ((2, 2), (6, 2), 1),
    ((6, 2), (6, 6), 0),
    ((6, 2), (6, 6), 1),
    ((2, 6), (6, 6), 0),
    ((2, 6), (6, 6), 1),
)
SINGLE_CENTERS: tuple[tuple[int, int], ...] = ((2, 2), (2, 6), (6, 2), (6, 6))


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


def _pair_cells(center_r: int, center_c: int, diag_index: int) -> tuple[tuple[int, int], tuple[int, int]]:
    (dr1, dc1), (dr2, dc2) = PAIR_DIAGONALS[diag_index]
    return (center_r + dr1, center_c + dc1), (center_r + dr2, center_c + dc2)


def _decode_centers(g: np.ndarray, marker_color: int) -> list[tuple[int, int]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(g == marker_color)]
    if not coords:
        return []
    assert len(coords) % 2 == 0, "marker pixels must come in diagonal pairs"

    coord_set = set(coords)
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in coords:
        found: list[tuple[int, int]] = []
        for dr in (-2 * PAIR_RADIUS, 2 * PAIR_RADIUS):
            for dc in (-2 * PAIR_RADIUS, 2 * PAIR_RADIUS):
                q = (r + dr, c + dc)
                if q in coord_set:
                    found.append(q)
        uniq = sorted(set(found))
        assert len(uniq) <= 1, "marker pixel participates in multiple diagonal pairs"
        neighbors[(r, c)] = uniq

    centers: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for pt in sorted(coords):
        if pt in used:
            continue
        partner_list = neighbors[pt]
        assert len(partner_list) == 1, "marker pixel is missing its diagonal partner"
        partner = partner_list[0]
        assert neighbors[partner] == [pt], "pairing must be mutual"
        used.add(pt)
        used.add(partner)
        centers.append(((pt[0] + partner[0]) // 2, (pt[1] + partner[1]) // 2))
    return centers


def _is_legend_center(g: np.ndarray, marker_color: int, r: int, c: int) -> bool:
    glyph_color, mask = GLYPHS[marker_color]
    return _has_source_glyph(g, r - 4, c - 4, glyph_color, mask)


def _collect_real_markers(g: np.ndarray) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    for marker_color in MARKER_COLORS:
        for r, c in _decode_centers(g, marker_color):
            if _is_legend_center(g, marker_color, r, c):
                continue
            markers.append((marker_color, r, c))
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


def _render_output(
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


def _place_marker_pair(inp: np.ndarray, marker_color: int, center_r: int, center_c: int, diag_index: int) -> bool:
    cells = _pair_cells(center_r, center_c, diag_index)
    h, w = inp.shape
    for rr, cc in cells:
        if not (0 <= rr < h and 0 <= cc < w) or inp[rr, cc] != 0:
            return False
    for rr, cc in cells:
        inp[rr, cc] = marker_color
    return True


def _markers_match_expected(inp: np.ndarray, expected_markers: list[tuple[int, int, int]]) -> bool:
    try:
        actual = _collect_real_markers(inp)
    except AssertionError:
        return False
    return actual == sorted(expected_markers, key=lambda item: (item[1], item[2], item[0]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode diagonal pixel-pair markers into fixed glyphs, then let nearest different-color markers exchange glyph colors.

    Input structure:
    - Black background with control colors 2, 3, 5, and/or 8, but each logical marker is now a same-color diagonal pair spanning a 5x5 box rather than a single pixel.
    - Some inputs may also include one legend entry: a 4x4 sample glyph in the default output color plus a matching diagonal label pair whose decoded center sits four cells below-right of that sample.

    Rule:
    - First pair same-color pixels that differ by four rows and four columns; each midpoint is a recovered marker anchor.
    - Ignore decoded anchors that belong to a legend label, and erase the legend glyph itself in the output.
    - Marker color still chooses the 4x4 glyph shape, placed with its top-left corner at the recovered anchor and clipped at the border.
    - Among the remaining real markers, two different-color markers swap default glyph colors only when they are each other's unique nearest different-color neighbor in Manhattan distance; all others keep their own default color.

    Key insight:
    - Neither input pixel directly becomes a glyph: first recover one logical marker from a visible diagonal pair, then use spatial nearest-neighbor reasoning between those decoded markers to decide whether each glyph keeps its own color or borrows a neighbor's.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset(ALLOWED_COLORS)

    markers = _collect_real_markers(g)
    assert markers
    pair_map = _mutual_nearest_pairs(markers)
    return _render_output(g.shape, markers, pair_map)


def _try_add_legend(inp: np.ndarray, free_cell: tuple[int, int] | None, used_markers: list[int]) -> bool:
    if free_cell is None or not used_markers or random.random() >= LEGEND_PROB:
        return True

    cell_r, cell_c = free_cell
    marker_color = random.choice(used_markers)
    glyph_color, mask = GLYPHS[marker_color]
    glyph_r = cell_r + 1
    glyph_c = cell_c + 1
    label_center_r = cell_r + 5
    label_center_c = cell_c + 5
    label_cells = _pair_cells(label_center_r, label_center_c, 1)

    trial = inp.copy()
    for rr, cc in label_cells:
        if trial[rr, cc] != 0:
            return False
    for rr, cc in label_cells:
        trial[rr, cc] = marker_color
    for dr in range(4):
        for dc in range(4):
            if not bool(mask[dr, dc]):
                continue
            rr = glyph_r + dr
            cc = glyph_c + dc
            if trial[rr, cc] != 0:
                return False
            trial[rr, cc] = glyph_color

    inp[:, :] = trial
    return True


def _generate_once() -> tuple[np.ndarray, dict[str, int]] | None:
    inp = np.zeros(GRID_SHAPE, dtype=int)
    cells = list(CELL_ORIGINS)
    random.shuffle(cells)

    include_legend = random.random() < LEGEND_PROB
    pair_cells = cells[:2]
    single_cells = cells[2 : 3 if include_legend else 4]
    legend_cell = cells[3] if include_legend else None

    while True:
        pair_specs = [tuple(random.sample(MARKER_COLORS, 2)) for _ in pair_cells]
        single_colors = [random.choice(MARKER_COLORS) for _ in single_cells]
        used_colors = {color for pair_spec in pair_specs for color in pair_spec}
        used_colors.update(single_colors)
        if len(used_colors) >= 3:
            break

    markers: list[tuple[int, int, int]] = []
    intended_pairs: dict[int, int] = {}

    for pair_idx, (cell_r, cell_c) in enumerate(pair_cells):
        (a_local, b_local, diag_index) = random.choice(PAIR_CLUSTER_LAYOUTS)
        colors = pair_specs[pair_idx]
        idx0 = len(markers)
        centers = (
            (cell_r + a_local[0], cell_c + a_local[1]),
            (cell_r + b_local[0], cell_c + b_local[1]),
        )
        if not _place_marker_pair(inp, colors[0], centers[0][0], centers[0][1], diag_index):
            return None
        if not _place_marker_pair(inp, colors[1], centers[1][0], centers[1][1], diag_index):
            return None
        markers.append((colors[0], centers[0][0], centers[0][1]))
        markers.append((colors[1], centers[1][0], centers[1][1]))
        intended_pairs[idx0] = idx0 + 1
        intended_pairs[idx0 + 1] = idx0

    for cell_idx, (cell_r, cell_c) in enumerate(single_cells):
        center_r, center_c = random.choice(SINGLE_CENTERS)
        diag_index = random.randint(0, 1)
        marker_color = single_colors[cell_idx]
        center_r += cell_r
        center_c += cell_c
        if not _place_marker_pair(inp, marker_color, center_r, center_c, diag_index):
            return None
        markers.append((marker_color, center_r, center_c))

    if not _markers_match_expected(inp, markers):
        return None

    if not _try_add_legend(inp, legend_cell, [marker_color for marker_color, _r, _c in markers]):
        return None

    if not _markers_match_expected(inp, markers):
        return None

    expected = _render_output(inp.shape, markers, intended_pairs)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    default_output = _render_output(inp.shape, markers, {})
    swap_effect_cells = int(np.count_nonzero(expected != default_output))
    swapped_markers = sum(
        DEFAULT_GLYPH_COLORS[marker_color] != DEFAULT_GLYPH_COLORS[markers[intended_pairs[i]][0]]
        for i, (marker_color, _r, _c) in enumerate(markers)
        if i in intended_pairs
    )
    default_markers = len(markers) - len(intended_pairs)
    if swapped_markers < 4 or default_markers < 1 or swap_effect_cells <= 0:
        return None

    marker_pixels = int(np.count_nonzero(np.isin(inp, MARKER_COLORS)))
    legend_pairs = (marker_pixels - 2 * len(markers)) // 2
    stats = {
        "logical_markers": len(markers),
        "marker_pixels": marker_pixels,
        "swapped_markers": swapped_markers,
        "default_markers": default_markers,
        "swap_effect_cells": swap_effect_cells,
        "legend_pairs": legend_pairs,
    }
    return inp, stats


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _generate_once()
        if result is not None:
            inp, _stats = result
            return inp
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
