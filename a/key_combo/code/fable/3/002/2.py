from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

PALETTE_COLORS = (BLUE, RED, YELLOW, PINK, ORANGE, TEAL, MAROON)
STRUCTURE_COLORS = {BLACK, GREEN, GREY}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int, list[np.ndarray]]:
    H, W = mask.shape
    labels = np.zeros((H, W), dtype=int)
    components: list[np.ndarray] = [np.empty((0, 2), dtype=int)]
    label = 0

    for sr_raw, sc_raw in np.argwhere(mask):
        sr = int(sr_raw)
        sc = int(sc_raw)
        if labels[sr, sc] != 0:
            continue

        label += 1
        cells: list[tuple[int, int]] = []
        queue: deque[tuple[int, int]] = deque([(sr, sc)])
        labels[sr, sc] = label
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < H and 0 <= cc < W and bool(mask[rr, cc]) and labels[rr, cc] == 0:
                    labels[rr, cc] = label
                    queue.append((rr, cc))

        components.append(np.array(cells, dtype=int))
    return labels, label, components


def _find_separator(g: np.ndarray) -> tuple[str, int]:
    H, W = g.shape
    candidates: list[tuple[str, int]] = []
    for c in range(W):
        if bool(np.all(g[:, c] == GREY)):
            candidates.append(("vertical", c))
    for r in range(H):
        if bool(np.all(g[r, :] == GREY)):
            candidates.append(("horizontal", r))

    assert len(candidates) == 1, "expected one full grey separator line"
    return candidates[0]


def _palette_pixel_count(section: np.ndarray) -> int:
    return int(np.count_nonzero(~np.isin(section, list(STRUCTURE_COLORS))))


def _split_input(g: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice]]:
    axis, idx = _find_separator(g)
    H, W = g.shape
    if axis == "vertical":
        assert 0 < idx < W - 1, "vertical separator must split two nonempty sides"
        left = g[:, :idx]
        right = g[:, idx + 1 :]
        left_count = _palette_pixel_count(left)
        right_count = _palette_pixel_count(right)
        assert (left_count > 0) != (right_count > 0), "exactly one side must contain palette bars"
        if left_count > 0:
            return right, left, (slice(None), slice(idx + 1, None))
        return left, right, (slice(None), slice(0, idx))

    assert axis == "horizontal"
    assert 0 < idx < H - 1, "horizontal separator must split two nonempty sides"
    top = g[:idx, :]
    bottom = g[idx + 1 :, :]
    top_count = _palette_pixel_count(top)
    bottom_count = _palette_pixel_count(bottom)
    assert (top_count > 0) != (bottom_count > 0), "exactly one side must contain palette bars"
    if top_count > 0:
        return bottom, top, (slice(idx + 1, None), slice(None))
    return top, bottom, (slice(0, idx), slice(None))


def _parse_palette(legend: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    seen_colors: set[int] = set()
    colors = sorted(int(c) for c in np.unique(legend) if int(c) not in STRUCTURE_COLORS)
    assert colors, "legend must contain at least one colored bar"

    for color in colors:
        labels, count, components = _label_components(legend == color)
        del labels
        assert count == 1, "each palette color must form one bar"
        coords = components[1]
        rows = set(int(r) for r in coords[:, 0])
        assert len(rows) == 1, "palette bars must be horizontal"
        row = next(iter(rows))
        cols = sorted(int(c) for c in coords[:, 1])
        length = len(cols)
        assert cols == list(range(cols[0], cols[-1] + 1)), "palette bars must be solid runs"
        assert int(np.count_nonzero(legend[row, cols[0] : cols[-1] + 1] == color)) == length
        assert length not in mapping, "palette bar lengths must be unique"
        assert color not in seen_colors, "palette bar colors must be unique"
        mapping[length] = color
        seen_colors.add(color)

    return mapping


def _region_depths(main_area: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    assert set(map(int, np.unique(main_area))).issubset({BLACK, GREEN}), "main picture must be black and green only"
    H, W = main_area.shape
    labels, region_count, regions = _label_components(main_area == BLACK)
    del regions
    assert region_count >= 2, "expected at least one enclosed black region"

    boundary_labels: set[int] = set()
    boundary_labels.update(int(x) for x in labels[0, :] if int(x) > 0)
    boundary_labels.update(int(x) for x in labels[H - 1, :] if int(x) > 0)
    boundary_labels.update(int(x) for x in labels[:, 0] if int(x) > 0)
    boundary_labels.update(int(x) for x in labels[:, W - 1] if int(x) > 0)
    assert len(boundary_labels) == 1, "outside background must be one boundary-touching region"
    outside = next(iter(boundary_labels))

    _green_labels, contour_count, contours = _label_components(main_area == GREEN)
    assert contour_count >= 1, "expected at least one green closed contour"

    graph: dict[int, set[int]] = {i: set() for i in range(1, region_count + 1)}
    for contour_id in range(1, contour_count + 1):
        adjacent: set[int] = set()
        for r_raw, c_raw in contours[contour_id]:
            r = int(r_raw)
            c = int(c_raw)
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < H and 0 <= cc < W:
                    label = int(labels[rr, cc])
                    if label > 0:
                        adjacent.add(label)
        assert len(adjacent) == 2, "each green contour must separate exactly two black regions"
        a, b = tuple(adjacent)
        graph[a].add(b)
        graph[b].add(a)

    depths = {outside: 0}
    queue: deque[int] = deque([outside])
    while queue:
        region = queue.popleft()
        for neighbor in graph[region]:
            want = depths[region] + 1
            if neighbor in depths:
                assert abs(depths[neighbor] - depths[region]) == 1, "region nesting graph must be a tree-like map"
                continue
            depths[neighbor] = want
            queue.append(neighbor)

    assert set(depths) == set(range(1, region_count + 1)), "all black regions must be reachable through contours"
    return labels, depths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color every enclosed region by looking up its nesting depth in a length-coded palette.

    Input structure:
    - A black main picture contains one or more green closed contours, with some
      contours side by side and some nested inside others.
    - A full grey separator line cuts off one side of the grid as a legend area.
    - The legend contains distinct horizontal color bars whose lengths are all different.

    Rule:
    - Treat black connected components in the main picture as regions separated
      by the green contour components.
    - The boundary-touching black component has depth 0; crossing each green
      closed contour adds one level of nesting depth.
    - A legend bar of length k supplies the color for every enclosed black
      region at depth k; green contours, outside background, the separator, and
      the legend are preserved.

    Key insight:
    - The useful code is the bar length, not the bar position or the region
      size; the same topological depth receives the same color even when those
      regions have very different areas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    main_area, legend, main_slice = _split_input(g)
    palette = _parse_palette(legend)
    region_labels, depths = _region_depths(main_area)

    out = g.copy()
    main_out = out[main_slice]
    painted = False
    for label, depth in depths.items():
        if depth == 0:
            continue
        assert depth in palette, "palette must include every used nesting depth"
        main_out[region_labels == label] = int(palette[depth])
        painted = True

    assert painted, "expected at least one enclosed region to color"
    return out


# --------------- generator ---------------

GEN_RETRY_LIMIT = 500
GEN_PLACEMENT_RETRIES = 160
GEN_GROUP_COUNT_RANGE = (3, 5)
GEN_VERTICAL_MAIN_H_RANGE = (17, 26)
GEN_HORIZONTAL_MAIN_W_RANGE = (17, 27)


def _make_group(depth: int) -> tuple[list[tuple[int, int, int, int]], int, int]:
    if depth == 1:
        h = random.randint(5, 9)
        w = random.randint(5, 11)
        return [(0, 0, h - 1, w - 1)], h, w

    h = random.randint(3, 4)
    w = random.randint(3, 5)
    rects = [(0, 0, h - 1, w - 1)]
    for _ in range(depth - 1):
        top = 2 + int(random.random() < 0.25)
        bottom = 2 + int(random.random() < 0.25)
        left = 2 + int(random.random() < 0.25)
        right = 2 + int(random.random() < 0.25)
        rects = [(r0 + top, c0 + left, r1 + top, c1 + left) for r0, c0, r1, c1 in rects]
        h += top + bottom
        w += left + right
        rects.insert(0, (0, 0, h - 1, w - 1))
    return rects, h, w


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 2) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _draw_rect_outline(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = GREEN
    grid[r1, c0 : c1 + 1] = GREEN
    grid[r0 : r1 + 1, c0] = GREEN
    grid[r0 : r1 + 1, c1] = GREEN


def _place_group(
    main_area: np.ndarray,
    rel_rects: list[tuple[int, int, int, int]],
    height: int,
    width: int,
    occupied_outer: list[tuple[int, int, int, int]],
) -> bool:
    H, W = main_area.shape
    if height + 2 > H or width + 2 > W:
        return False

    for _ in range(GEN_PLACEMENT_RETRIES):
        r = random.randint(1, H - height - 1)
        c = random.randint(1, W - width - 1)
        outer = (r, c, r + height - 1, c + width - 1)
        if not all(_separated(outer, prev) for prev in occupied_outer):
            continue

        abs_rects = [
            (r + r0, c + c0, r + r1, c + c1)
            for r0, c0, r1, c1 in rel_rects
        ]
        if any(np.any(main_area[r0 : r1 + 1, c0 : c1 + 1] != BLACK) for r0, c0, r1, c1 in abs_rects):
            continue

        for box in abs_rects:
            _draw_rect_outline(main_area, box)
        occupied_outer.append(outer)
        return True
    return False


def _sample_group_depths(max_depth: int) -> list[int]:
    count = 3 if max_depth == 3 else random.randint(*GEN_GROUP_COUNT_RANGE)
    depths = [max_depth, 1, 1] if max_depth == 3 else [max_depth, 1]
    while len(depths) < count:
        if random.random() < 0.65:
            depths.append(1)
        else:
            depths.append(random.randint(2, max_depth))
    random.shuffle(depths)
    return depths


def _build_main_area(height: int, width: int, max_depth: int) -> np.ndarray | None:
    main_area = np.zeros((height, width), dtype=int)
    occupied: list[tuple[int, int, int, int]] = []
    for depth in _sample_group_depths(max_depth):
        rel_rects, h, w = _make_group(depth)
        if not _place_group(main_area, rel_rects, h, w, occupied):
            return None
    return main_area


def _ordered_bar_rows(legend_h: int, count: int) -> list[int] | None:
    if legend_h < count * 2 + 1:
        return None
    start = random.randint(1, legend_h - (count - 1) * 2 - 1)
    return [start + i * 2 for i in range(count)]


def _render_legend(shape: tuple[int, int], depth_colors: dict[int, int]) -> tuple[np.ndarray, list[int]] | None:
    h, w = shape
    legend = np.zeros((h, w), dtype=int)
    depths = sorted(depth_colors)
    order = depths[:]
    random.shuffle(order)
    if len(order) > 1 and order == depths:
        order = list(reversed(order))

    rows = _ordered_bar_rows(h, len(order))
    if rows is None:
        return None
    for row, depth in zip(rows, order):
        length = int(depth)
        if length > w:
            return None
        c0 = random.randint(0, w - length)
        legend[row, c0 : c0 + length] = int(depth_colors[depth])
    return legend, order


def _compose_grid(main_area: np.ndarray, legend: np.ndarray, side: str) -> np.ndarray:
    mh, mw = main_area.shape
    lh, lw = legend.shape
    if side == "left":
        assert lh == mh
        grid = np.zeros((mh, lw + 1 + mw), dtype=int)
        grid[:, :lw] = legend
        grid[:, lw] = GREY
        grid[:, lw + 1 :] = main_area
        return grid
    if side == "right":
        assert lh == mh
        grid = np.zeros((mh, mw + 1 + lw), dtype=int)
        grid[:, :mw] = main_area
        grid[:, mw] = GREY
        grid[:, mw + 1 :] = legend
        return grid
    if side == "top":
        assert lw == mw
        grid = np.zeros((lh + 1 + mh, mw), dtype=int)
        grid[:lh, :] = legend
        grid[lh, :] = GREY
        grid[lh + 1 :, :] = main_area
        return grid

    assert side == "bottom"
    assert lw == mw
    grid = np.zeros((mh + 1 + lh, mw), dtype=int)
    grid[:mh, :] = main_area
    grid[mh, :] = GREY
    grid[mh + 1 :, :] = legend
    return grid


def _expected_from_known_semantics(
    grid: np.ndarray,
    main_slice: tuple[slice, slice],
    depth_colors: dict[int, int],
) -> np.ndarray | None:
    main_area = grid[main_slice]
    try:
        labels, depths = _region_depths(main_area)
    except AssertionError:
        return None
    used_depths = {d for d in depths.values() if d > 0}
    if used_depths != set(depth_colors):
        return None

    depth1_areas = [int(np.count_nonzero(labels == label)) for label, depth in depths.items() if depth == 1]
    if len(depth1_areas) < 2 or len(set(depth1_areas)) < 2:
        return None

    expected = grid.copy()
    view = expected[main_slice]
    for label, depth in depths.items():
        if depth > 0:
            view[labels == label] = int(depth_colors[depth])
    return expected


def _main_slice_for_composed(
    side: str,
    main_shape: tuple[int, int],
    legend_shape: tuple[int, int],
) -> tuple[slice, slice]:
    mh, mw = main_shape
    lh, lw = legend_shape
    if side == "left":
        return slice(None), slice(lw + 1, lw + 1 + mw)
    if side == "right":
        return slice(None), slice(0, mw)
    if side == "top":
        return slice(lh + 1, lh + 1 + mh), slice(None)
    assert side == "bottom"
    return slice(0, mh), slice(None)


def _sample_dimensions(side: str, max_depth: int) -> tuple[int, int, int, int]:
    if side in ("left", "right"):
        legend_w = max_depth + 3
        min_main = 19 if max_depth == 3 else 17
        main_w = random.randint(min_main, 30 - legend_w - 1)
        main_h = random.randint(max(min_main, GEN_VERTICAL_MAIN_H_RANGE[0]), GEN_VERTICAL_MAIN_H_RANGE[1])
        legend_h = main_h
        return main_h, main_w, legend_h, legend_w

    legend_h = max_depth * 2 + 1
    min_main = 19 if max_depth == 3 else 17
    main_h = random.randint(min_main, 30 - legend_h - 1)
    main_w = random.randint(max(min_main, GEN_HORIZONTAL_MAIN_W_RANGE[0]), GEN_HORIZONTAL_MAIN_W_RANGE[1])
    legend_w = main_w
    return main_h, main_w, legend_h, legend_w


def _gen_attempt() -> np.ndarray | None:
    max_depth = random.choices([2, 3], weights=[10, 90], k=1)[0]
    side = random.choice(["left", "right", "top", "bottom"])
    main_h, main_w, legend_h, legend_w = _sample_dimensions(side, max_depth)

    main_area = _build_main_area(main_h, main_w, max_depth)
    if main_area is None:
        return None

    colors = random.sample(list(PALETTE_COLORS), max_depth)
    depth_colors = {depth: int(colors[depth - 1]) for depth in range(1, max_depth + 1)}
    rendered = _render_legend((legend_h, legend_w), depth_colors)
    if rendered is None:
        return None
    legend, bar_order = rendered
    if bar_order == sorted(bar_order):
        return None

    grid = _compose_grid(main_area, legend, side)
    if max(grid.shape) > 30:
        return None

    main_slice = _main_slice_for_composed(side, main_area.shape, legend.shape)
    expected = _expected_from_known_semantics(grid, main_slice, depth_colors)
    if expected is None:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-depth palette input")
