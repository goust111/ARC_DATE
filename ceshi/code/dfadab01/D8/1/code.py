# Variant of dfadab01
# Dimension: D8 — 字形字典中标记色与字形色的数值关系 (meta)
# Change: 标记色与字形色形成固定配对：2->4、3->1、5->6、8->7，配对关系为全局常量... → 字形色由相邻标记决定
"""
design_analysis:
  concretization:
    preserved:
      - 输入里的四种单像素标记仍然代表四种固定 4x4 字形轮廓。
      - 输出仍以标记像素作为字形左上角，越界部分继续裁剪。
      - 输入仍允许出现原题那种“4x4 字形 + 右下标签像素”的 legend，因此 solver 仍会跳过这种标签像素。
    changed:
      - 字形颜色不再由标记自身颜色直接决定。
      - 先用标记自身颜色选字形轮廓，再找“最近的另一个真实标记”的颜色，把那个邻居颜色代入原来的固定映射
        2->4, 3->1, 5->6, 8->7，得到当前字形应涂的颜色。
      - “最近”具体化为 Manhattan 距离；这是肉眼可数的空间关系，不依赖不可见排序。
    trace_example:
      original:
        - 标记 2 在 (1,1) 会直接画成颜色 4 的方框。
        - 标记 3 在 (1,6) 会直接画成颜色 1 的圆环。
      variant:
        - (1,1) 的 2 最近的是 (1,6) 的 3，所以它仍画方框，但颜色改取 3 的原映射色 1。
        - (1,6) 的 3 最近的是 (1,1) 的 2，所以它仍画圆环，但颜色改取 2 的原映射色 4。
      difference:
        - 原题是“自己的颜色决定自己的输出颜色”。
        - 变体是“自己的颜色决定形状，邻居的颜色决定输出颜色”。
  complexity_calibration:
    - 不能让所有样本都退化成“同色近邻”或“每种形状只出现一种输出颜色”，否则视觉上又像原题。
    - 生成器固定使用 2-3 个局部簇；簇内距离为 5 或 6，簇间距离明显更大，使最近邻关系清楚可见。
    - 每个样本强制满足：至少有一种标记颜色重复出现两次，并且这两次的最近邻颜色不同。
      因此相同字形会在同一任务里呈现至少两种不同输出颜色，D8 变化肉眼可见。
  implicit_assumption_review:
    original_main:
      - 原来的 `GLYPHS` 同时把“形状”和“输出颜色”绑在一起；D8 变化后这条前提失效，必须拆成
        `标记->形状` 和 `标记->原始映射色` 两层。
      - 原来的 `_has_source_glyph(..., glyph_color, mask)` 假设 legend 字形颜色由标签色唯一决定；
        变体后 legend 若存在，其字形颜色可能来自别的标记，因此只能检查“对应轮廓是否以任一合法字形色出现”。
      - 原来的 `_paint_clipped` 仍然安全，因为生成器继续保证真实标记之间的 4x4 输出不会互相覆盖。
    new_main:
      - 每个真实标记都必须至少有一个其他真实标记，生成器因此从不生成单独一个标记。
      - 若出现多个等距最近邻，solver 只接受它们颜色一致的情况；生成器直接避免这种平局。
      - legend 标签不能参与最近邻传色；solver 会先识别并剔除它们，生成器默认不放 legend 来保持规则信号干净。
  placement_legality:
    - 画布固定为 20x20，标记放在若干经过人工校准的模板位置上；这些位置之间至少相隔 5 个单元，
      所以 4x4 扩张后不会重叠，但仍会保留原题允许的边缘裁剪现象。
    - 模板经过筛选，保证每个标记的最近邻在 Manhattan 距离上唯一，且比第二近至少再近 5 格，
      因而 solver 不需要不可见的 tie-break。
  observability_and_cleanliness:
    - 同一张图里会重复出现同形异色的字形，足以推翻原题的固定配色假设。
    - 不使用按颜色 id 排序或“最左/最上”等隐藏规则来决定颜色；颜色只来自可见的最近邻关系。
    - 不生成 legend，避免静态样例字形与动态传色规则混在一起造成视觉噪声。
  generator_checks:
    - output-first：先采样标记位置与颜色，手工渲染期望输出，再验证 `main(input) == expected_output`。
    - 退化指标：
      - changed_fraction = 最近邻颜色与自身颜色不同的标记占比，要求至少 1/2。
      - repeated_shape_multicolor = 存在同一种标记色，其不同实例对应的最近邻颜色至少两种。
"""

from __future__ import annotations

import random

import numpy as np


GLYPH_MASKS: dict[int, np.ndarray] = {
    2: np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    3: np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=bool,
    ),
    5: np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=bool,
    ),
    8: np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
        ],
        dtype=bool,
    ),
}

PAINT_FROM_MARKER: dict[int, int] = {
    2: 4,
    3: 1,
    5: 6,
    8: 7,
}

MARKER_COLORS = tuple(GLYPH_MASKS.keys())
GLYPH_COLORS = tuple(PAINT_FROM_MARKER.values())
GRID_SIZE = 20

# Templates are laid out so that nearest-neighbor relations are local and obvious.
BASE_LAYOUTS: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (0, 5), (11, 11), (16, 11)),
    ((0, 0), (0, 5), (11, 0), (16, 0), (11, 11), (11, 16)),
    ((0, 0), (0, 5), (0, 11), (11, 11), (16, 11)),
    ((0, 11), (5, 11), (16, 0), (16, 5), (16, 11)),
    ((0, 0), (5, 0), (11, 0), (11, 11), (11, 16)),
)


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> bool:
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + 4 > h or c0 + 4 > w:
        return False
    patch = g[r0 : r0 + 4, c0 : c0 + 4]
    return bool(np.all(patch[mask] == color) and np.all(patch[~mask] != color))


def _is_legend_label(g: np.ndarray, r: int, c: int, marker_color: int) -> bool:
    mask = GLYPH_MASKS[marker_color]
    for glyph_color in GLYPH_COLORS:
        if _has_source_glyph(g, r - 4, c - 4, glyph_color, mask):
            return True
    return False


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


def _collect_real_markers(g: np.ndarray) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    marker_mask = np.isin(g, MARKER_COLORS)
    for r, c in np.argwhere(marker_mask):
        rr = int(r)
        cc = int(c)
        marker_color = int(g[rr, cc])
        if _is_legend_label(g, rr, cc, marker_color):
            continue
        markers.append((rr, cc, marker_color))
    return markers


def _nearest_neighbor_colors(markers: list[tuple[int, int, int]]) -> list[int]:
    assert len(markers) >= 2
    donor_colors: list[int] = []
    for i, (r, c, _) in enumerate(markers):
        distances: list[tuple[int, int]] = []
        for j, (rr, cc, _) in enumerate(markers):
            if i == j:
                continue
            distances.append((abs(r - rr) + abs(c - cc), j))
        min_dist = min(dist for dist, _ in distances)
        nearest_colors = {markers[j][2] for dist, j in distances if dist == min_dist}
        assert len(nearest_colors) == 1, "nearest-neighbor color must be unambiguous"
        donor_colors.append(next(iter(nearest_colors)))
    return donor_colors


def _render_output(shape: tuple[int, int], markers: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    donor_colors = _nearest_neighbor_colors(markers)
    for (r, c, marker_color), donor_marker_color in zip(markers, donor_colors, strict=True):
        glyph_color = PAINT_FROM_MARKER[donor_marker_color]
        _paint_clipped(out, r, c, glyph_color, GLYPH_MASKS[marker_color])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each marker into its fixed glyph shape, but tint it by the nearest other marker.

    Input structure:
    - Black background with single-pixel markers in colors 2, 3, 5, and/or 8.
    - Some grids may also include an old-style legend: a 4x4 sample glyph plus a matching label marker one step down-right from its bbox.

    Rule:
    - A marker's own color still selects which 4x4 glyph mask to draw.
    - The drawn color is not fixed: find the nearest other real marker by Manhattan distance, then convert that neighbor's marker color through the original palette mapping 2->4, 3->1, 5->6, 8->7.
    - Paint the glyph with its top-left corner at the marker position, clipping at the grid edge, and ignore legend label markers.

    Key insight:
    - Marker color now controls shape only; output color is an attribute transferred from the closest neighboring marker, so solving requires reading spatial proximity before applying the old color dictionary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({0, *MARKER_COLORS, *GLYPH_COLORS})

    markers = _collect_real_markers(g)
    assert len(markers) >= 2, "need at least two real markers for nearest-neighbor transfer"
    return _render_output(g.shape, markers)


def _transform_positions(
    positions: tuple[tuple[int, int], ...],
    transpose: bool,
    flip_rows: bool,
    flip_cols: bool,
) -> list[tuple[int, int]]:
    transformed: list[tuple[int, int]] = []
    for r, c in positions:
        rr, cc = (c, r) if transpose else (r, c)
        if flip_rows:
            rr = GRID_SIZE - 1 - rr
        if flip_cols:
            cc = GRID_SIZE - 1 - cc
        transformed.append((rr, cc))
    assert len(set(transformed)) == len(transformed)
    return transformed


def _nearest_indices(positions: list[tuple[int, int]]) -> list[int]:
    nearest: list[int] = []
    for i, (r, c) in enumerate(positions):
        dists = sorted(
            (abs(r - rr) + abs(c - cc), j)
            for j, (rr, cc) in enumerate(positions)
            if i != j
        )
        assert dists, "every marker needs a neighbor"
        best_dist = dists[0][0]
        assert len(dists) == 1 or best_dist < dists[1][0], "generator must avoid nearest-neighbor ties"
        nearest.append(dists[0][1])
    return nearest


def _sample_colors(nearest: list[int]) -> list[int]:
    n = len(nearest)
    for _ in range(400):
        colors = [random.choice(MARKER_COLORS) for _ in range(n)]
        if len(set(colors)) < 2:
            continue
        changed = sum(colors[i] != colors[nearest[i]] for i in range(n))
        if changed * 2 < n:
            continue
        repeated_shape_multicolor = False
        for own_color in MARKER_COLORS:
            idxs = [i for i, color in enumerate(colors) if color == own_color]
            if len(idxs) < 2:
                continue
            donor_marker_colors = {colors[nearest[i]] for i in idxs}
            if len(donor_marker_colors) >= 2:
                repeated_shape_multicolor = True
                break
        if not repeated_shape_multicolor:
            continue
        paint_colors = {PAINT_FROM_MARKER[colors[nearest[i]]] for i in range(n)}
        if len(paint_colors) < 2:
            continue
        return colors
    raise RuntimeError("failed to sample a non-degenerate color assignment")


def generate_input() -> np.ndarray:
    for _ in range(200):
        layout = random.choice(BASE_LAYOUTS)
        positions = _transform_positions(
            layout,
            transpose=bool(random.getrandbits(1)),
            flip_rows=bool(random.getrandbits(1)),
            flip_cols=bool(random.getrandbits(1)),
        )
        nearest = _nearest_indices(positions)
        colors = _sample_colors(nearest)

        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        markers = []
        for (r, c), marker_color in zip(positions, colors, strict=True):
            grid[r, c] = marker_color
            markers.append((r, c, marker_color))

        expected = _render_output(grid.shape, markers)
        actual = main(grid)
        assert np.array_equal(actual, expected)
        return grid

    raise RuntimeError("failed to construct a valid D8 variant input")
