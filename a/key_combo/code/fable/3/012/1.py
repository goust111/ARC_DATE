from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 800

Rect = tuple[int, int, int, int, int, int]  # color, r0, r1, c0, c1, area


def _contiguous_span(values: np.ndarray) -> tuple[int, int] | None:
    if values.size == 0:
        return None
    vals = sorted(set(map(int, values)))
    if vals != list(range(vals[0], vals[-1] + 1)):
        return None
    return vals[0], vals[-1]


def _parse_projected_rectangles(grid: np.ndarray) -> tuple[int, list[Rect]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    height, width = map(int, g.shape)

    for band in range(1, min(height, width)):
        if np.any(g[:band, :band] != BG):
            continue
        if np.any(g[band:, band:] != BG):
            continue

        top = g[:band, band:]
        left = g[band:, :band]
        top_colors = {int(x) for x in np.unique(top) if int(x) != BG}
        left_colors = {int(x) for x in np.unique(left) if int(x) != BG}
        if not top_colors or top_colors != left_colors or len(top_colors) != band:
            continue

        rects: list[Rect] = []
        ok = True
        for color in top_colors:
            top_coords = np.argwhere(top == color)
            left_coords = np.argwhere(left == color)
            if top_coords.size == 0 or left_coords.size == 0:
                ok = False
                break

            top_rows = set(map(int, top_coords[:, 0]))
            left_cols = set(map(int, left_coords[:, 1]))
            if len(top_rows) != 1 or len(left_cols) != 1:
                ok = False
                break

            col_span = _contiguous_span(top_coords[:, 1] + band)
            row_span = _contiguous_span(left_coords[:, 0] + band)
            if col_span is None or row_span is None:
                ok = False
                break

            c0, c1 = col_span
            r0, r1 = row_span
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            rects.append((int(color), int(r0), int(r1), int(c0), int(c1), int(area)))

        if not ok:
            continue
        assert len({rect[-1] for rect in rects}) == len(rects), "rectangle areas must be unique"
        return band, rects

    raise AssertionError("could not parse paired top/left projection bars")


def _intersection_area(a: Rect, b: Rect) -> int:
    _ca, ar0, ar1, ac0, ac1, _aa = a
    _cb, br0, br1, bc0, bc1, _ba = b
    h = min(ar1, br1) - max(ar0, br0) + 1
    w = min(ac1, bc1) - max(ac0, bc0) + 1
    return max(0, h) * max(0, w)


def _paint_intersections(base: np.ndarray, band: int, rects: list[Rect]) -> np.ndarray:
    out = np.array(base, dtype=int, copy=True)
    height, width = map(int, out.shape)

    for r in range(band, height):
        for c in range(band, width):
            covering = [
                rect
                for rect in rects
                if rect[1] <= r <= rect[2] and rect[3] <= c <= rect[4]
            ]
            if len(covering) < 2:
                continue

            min_area = min(rect[5] for rect in covering)
            winners = [rect for rect in covering if rect[5] == min_area]
            assert len(winners) == 1, "overlapping rectangles must have a unique smallest area"
            out[r, c] = winners[0][0]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    由顶边和左边的同色投影条带还原隐形矩形，只给矩形相交处涂上较小面积矩形的颜色。

    Input structure:
    - 黑色背景上，顶部投影带含若干同色水平连续条带，左侧投影带含对应的同色垂直连续条带。
    - 每种颜色恰好有一条顶部投影和一条左侧投影：顶部条带给出该色隐形矩形的列范围，左侧条带给出行范围。
    - 投影带外的内部区域在输入中全黑，隐形矩形本身从未被画出；不同颜色矩形的行列范围可以交叠。

    Rule:
    - 按同色顶部条带与左侧条带配对，构造轴对齐矩形（行范围 x 列范围），但不绘制矩形本体。
    - 对内部每个被至少两个隐形矩形覆盖的格子，填入覆盖它的矩形中面积最小者的颜色；三重及以上覆盖也取唯一最小面积者。
    - 顶部和左侧投影条带保持原样，未被两个以上矩形覆盖的背景保持黑色。

    Key insight:
    - 可见对象只是边缘投影；输出颜色来自隐形矩形之间的相交关系和面积排序，而不是来自条带自身的邻接或位置顺序。
    """
    g = np.asarray(grid, dtype=int)
    band, rects = _parse_projected_rectangles(g)
    return _paint_intersections(g, band, rects)


def _relative_intersection(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b
    h = min(ar1, br1) - max(ar0, br0) + 1
    w = min(ac1, bc1) - max(ac0, bc0) + 1
    return max(0, h) * max(0, w)


def _relative_cover_counts(
    rel_rects: list[tuple[int, int, int, int]],
    areas: list[int],
    colors: list[int],
    inner_h: int,
    inner_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((inner_h, inner_w), dtype=int)
    smallest_area = np.full((inner_h, inner_w), 10_000, dtype=int)
    owner = np.zeros((inner_h, inner_w), dtype=int)

    for rect, area, color in zip(rel_rects, areas, colors):
        r0, r1, c0, c1 = rect
        counts[r0 : r1 + 1, c0 : c1 + 1] += 1
        window = smallest_area[r0 : r1 + 1, c0 : c1 + 1]
        mask = area < window
        window[mask] = area
        owner_window = owner[r0 : r1 + 1, c0 : c1 + 1]
        owner_window[mask] = color
    return counts, owner


def _valid_relative_rectangles(
    rel_rects: list[tuple[int, int, int, int]],
    colors: list[int],
    inner_h: int,
    inner_w: int,
) -> bool:
    areas = [(r1 - r0 + 1) * (c1 - c0 + 1) for r0, r1, c0, c1 in rel_rects]
    if len(set(areas)) != len(areas):
        return False

    counts, owner = _relative_cover_counts(rel_rects, areas, colors, inner_h, inner_w)
    fill = counts >= 2
    if int(np.count_nonzero(counts >= 3)) < 2:
        return False
    if int(np.count_nonzero(counts == 2)) < 5:
        return False
    fill_count = int(np.count_nonzero(fill))
    if not (8 <= fill_count <= max(8, inner_h * inner_w // 4)):
        return False
    if len(set(map(int, owner[fill]))) < 2:
        return False

    pair_areas = [
        _relative_intersection(a, b)
        for i, a in enumerate(rel_rects)
        for b in rel_rects[i + 1 :]
    ]
    if not any(area == 0 for area in pair_areas):
        return False
    if not all(
        any(i != j and _relative_intersection(a, rel_rects[j]) > 0 for j in range(len(rel_rects)))
        for i, a in enumerate(rel_rects)
    ):
        return False

    mixed_dimension_comparison = False
    for i, a in enumerate(rel_rects):
        ah = a[1] - a[0] + 1
        aw = a[3] - a[2] + 1
        for j, b in enumerate(rel_rects[i + 1 :], start=i + 1):
            if _relative_intersection(a, b) == 0:
                continue
            bh = b[1] - b[0] + 1
            bw = b[3] - b[2] + 1
            if areas[i] < areas[j] and (ah > bh or aw > bw):
                mixed_dimension_comparison = True
            if areas[j] < areas[i] and (bh > ah or bw > aw):
                mixed_dimension_comparison = True
    return mixed_dimension_comparison


def _sample_cluster_rectangles(
    count: int,
    inner_h: int,
    inner_w: int,
) -> list[tuple[int, int, int, int]] | None:
    core_h = random.choice((2, 3))
    core_w = random.choice((2, 3))
    core_r0 = random.randint(4, inner_h - core_h - 5)
    core_c0 = random.randint(4, inner_w - core_w - 5)
    core_r1 = core_r0 + core_h - 1
    core_c1 = core_c0 + core_w - 1

    rects: list[tuple[int, int, int, int]] = []
    for _ in range(3):
        up = random.randint(1, min(5, core_r0))
        down = random.randint(1, min(5, inner_h - 1 - core_r1))
        left = random.randint(1, min(5, core_c0))
        right = random.randint(1, min(5, inner_w - 1 - core_c1))
        rects.append((core_r0 - up, core_r1 + down, core_c0 - left, core_c1 + right))

    for _ in range(count - 3):
        placed = False
        for _attempt in range(160):
            h = random.randint(3, 8)
            w = random.randint(3, 8)
            r0 = random.randint(0, inner_h - h)
            c0 = random.randint(0, inner_w - w)
            cand = (r0, r0 + h - 1, c0, c0 + w - 1)
            covers_core = not (
                cand[1] < core_r0 or cand[0] > core_r1 or cand[3] < core_c0 or cand[2] > core_c1
            )
            if covers_core:
                continue
            intersections = [_relative_intersection(cand, rect) for rect in rects]
            if sum(area > 0 for area in intersections) not in (1, 2):
                continue
            rects.append(cand)
            placed = True
            break
        if not placed:
            return None
    return rects


def _render_input(
    rel_rects: list[tuple[int, int, int, int]],
    colors: list[int],
    band: int,
    inner_h: int,
    inner_w: int,
) -> np.ndarray:
    grid = np.zeros((band + inner_h, band + inner_w), dtype=int)
    top_lanes = list(range(band))
    left_lanes = list(range(band))
    random.shuffle(top_lanes)
    random.shuffle(left_lanes)

    for rect, color, top_lane, left_lane in zip(rel_rects, colors, top_lanes, left_lanes):
        r0, r1, c0, c1 = rect
        grid[top_lane, band + c0 : band + c1 + 1] = int(color)
        grid[band + r0 : band + r1 + 1, left_lane] = int(color)
    return grid


def _make_attempt() -> np.ndarray | None:
    count = random.choice((4, 4, 5))
    band = count
    inner_h = random.randint(16, 24)
    inner_w = random.randint(16, 24)
    if band + inner_h > 30 or band + inner_w > 30:
        return None

    colors = random.sample(PALETTE, count)
    rel_rects = _sample_cluster_rectangles(count, inner_h, inner_w)
    if rel_rects is None:
        return None
    if not _valid_relative_rectangles(rel_rects, colors, inner_h, inner_w):
        return None

    grid = _render_input(rel_rects, colors, band, inner_h, inner_w)
    try:
        parsed_band, parsed = _parse_projected_rectangles(grid)
        expected = _paint_intersections(grid, parsed_band, parsed)
        actual = main(grid)
    except AssertionError:
        return None
    if parsed_band != band or not np.array_equal(actual, expected):
        return None
    if not np.array_equal(grid[:band, :band], np.zeros((band, band), dtype=int)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _make_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid phantom-rectangle projection instance")
