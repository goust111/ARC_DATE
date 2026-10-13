from __future__ import annotations

import random
from collections import Counter

import numpy as np


Corner = tuple[int, int]
Rectangle = dict[str, int]

_CORNER_ORDER = (
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _corner_map(shape: tuple[int, int]) -> dict[str, Corner]:
    height, width = map(int, shape)
    return {
        "top_left": (0, 0),
        "top_right": (0, width - 1),
        "bottom_left": (height - 1, 0),
        "bottom_right": (height - 1, width - 1),
    }


def _anchor_to_top_left(corner_name: str, anchor: Corner, h: int, w: int) -> Corner:
    r, c = map(int, anchor)
    if corner_name == "top_left":
        return r, c
    if corner_name == "top_right":
        return r, c - w + 1
    if corner_name == "bottom_left":
        return r - h + 1, c
    if corner_name == "bottom_right":
        return r - h + 1, c - w + 1
    raise ValueError(f"unknown corner {corner_name}")


def _next_anchor(corner_name: str, anchor: Corner, h: int, w: int) -> Corner:
    r, c = map(int, anchor)
    if corner_name == "top_left":
        return r + h - 1, c + w - 1
    if corner_name == "top_right":
        return r + h - 1, c - (w - 1)
    if corner_name == "bottom_left":
        return r - (h - 1), c + w - 1
    if corner_name == "bottom_right":
        return r - (h - 1), c - (w - 1)
    raise ValueError(f"unknown corner {corner_name}")


def _component_rectangles(grid: np.ndarray, bg: int, marker: Corner) -> list[Rectangle]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    mask = g != bg
    mask[marker] = False
    visited = np.zeros_like(mask, dtype=bool)
    rectangles: list[Rectangle] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or visited[r, c]:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            coords: list[Corner] = []
            visited[r, c] = True

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if visited[nr, nc] or not mask[nr, nc]:
                        continue
                    assert int(g[nr, nc]) == color, "foreground rectangles must be monochrome"
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0 = min(rows)
            r1 = max(rows)
            c0 = min(cols)
            c1 = max(cols)
            h = r1 - r0 + 1
            w = c1 - c0 + 1

            assert r1 == height - 1, "every rectangle must touch the bottom row"
            assert len(coords) == h * w, "every component must be a solid rectangle"
            assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle bbox must be solid"
            rectangles.append({"color": color, "r0": r0, "c0": c0, "h": h, "w": w})

    assert rectangles, "expected at least one foreground rectangle"
    left_edges = [rect["c0"] for rect in rectangles]
    assert len(set(left_edges)) == len(left_edges), "expected unique left edges"
    return rectangles


def _identify_marker_and_rectangles(grid: np.ndarray, bg: int) -> tuple[str, Corner, list[Rectangle]]:
    g = np.asarray(grid, dtype=int)
    corners = _corner_map(g.shape)
    candidates: list[tuple[str, Corner, list[Rectangle]]] = []

    for name, coord in corners.items():
        if int(g[coord]) == bg:
            continue
        try:
            rectangles = _component_rectangles(g, bg, coord)
        except AssertionError:
            continue
        marker_color = int(g[coord])
        if any(rect["color"] == marker_color for rect in rectangles):
            continue
        candidates.append((name, coord, rectangles))

    assert len(candidates) == 1, "expected a uniquely identifiable corner marker"
    return candidates[0]


def _ordered_rectangles(rectangles: list[Rectangle], corner_name: str) -> list[Rectangle]:
    reverse = corner_name.endswith("right")
    return sorted(rectangles, key=lambda rect: int(rect["c0"]), reverse=reverse)


def _paint_chain(shape: tuple[int, int], bg: int, corner_name: str, rectangles: list[Rectangle]) -> np.ndarray:
    height, width = map(int, shape)
    out = np.full((height, width), bg, dtype=int)
    anchor = _corner_map(shape)[corner_name]

    for rect in rectangles:
        color = int(rect["color"])
        h = int(rect["h"])
        w = int(rect["w"])
        r0, c0 = _anchor_to_top_left(corner_name, anchor, h, w)
        assert 0 <= r0 and 0 <= c0 and r0 + h <= height and c0 + w <= width, "packed chain must fit"
        out[r0 : r0 + h, c0 : c0 + w] = color
        anchor = _next_anchor(corner_name, anchor, h, w)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repack the bottom rectangles into a diagonal corner-overlap chain chosen by the single corner marker.

    Input structure:
    - The background is the most common color.
    - Several solid monochrome rectangles touch the bottom row and have distinct left edges.
    - Exactly one corner contains a single-pixel marker whose color is different from every rectangle color.

    Rule:
    - Clear the whole canvas back to the background and erase the marker.
    - Use the marker corner as the chain start, and sort rectangles by the distance from their original left edge to the marker's vertical side: left-corner markers read left-to-right, right-corner markers read right-to-left.
    - Place the first rectangle flush to that corner; each later rectangle moves one diagonal step toward the grid center by the previous rectangle's height minus one and width minus one, so adjacent rectangles share exactly one corner cell and the later one overwrites it.

    Key insight:
    - The marker does not just choose a location; its corner simultaneously determines the chain anchor, the inward diagonal direction, and the rectangle ordering direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    corner_name, _marker, rectangles = _identify_marker_and_rectangles(g, bg)
    ordered = _ordered_rectangles(rectangles, corner_name)
    return _paint_chain(g.shape, bg, corner_name, ordered)


_GEN_RETRY_LIMIT = 300
_GEN_OBJECT_COUNT_RANGE = (2, 6)
_GEN_RECT_HEIGHT_RANGE = (1, 6)
_GEN_RECT_WIDTH_RANGE = (1, 6)
_GEN_GAP_RANGE = (1, 4)
_GEN_MARGIN_RANGE = (1, 4)
_GEN_EXTRA_SPACE_RANGE = (0, 7)
_GEN_COLORS = tuple(range(10))


def _chain_span(lengths: list[int]) -> int:
    return 1 + sum(length - 1 for length in lengths)


def _gen_attempt() -> np.ndarray | None:
    n_rects = random.randint(*_GEN_OBJECT_COUNT_RANGE)
    heights = [random.randint(*_GEN_RECT_HEIGHT_RANGE) for _ in range(n_rects)]
    widths = [random.randint(*_GEN_RECT_WIDTH_RANGE) for _ in range(n_rects)]
    gaps = [random.randint(*_GEN_GAP_RANGE) for _ in range(max(0, n_rects - 1))]
    left_margin = random.randint(*_GEN_MARGIN_RANGE)
    right_margin = random.randint(*_GEN_MARGIN_RANGE)

    chain_h = _chain_span(heights)
    chain_w = _chain_span(widths)
    input_w = left_margin + sum(widths) + sum(gaps) + right_margin

    min_h = max(max(heights), chain_h)
    min_w = max(chain_w, input_w)
    if min_h > 30 or min_w > 30:
        return None

    height = min_h + random.randint(*_GEN_EXTRA_SPACE_RANGE)
    width = min_w + random.randint(*_GEN_EXTRA_SPACE_RANGE)
    if height > 30 or width > 30:
        return None

    bg = random.choice(_GEN_COLORS)
    usable = [color for color in _GEN_COLORS if color != bg]
    if len(usable) < n_rects + 1:
        return None
    chosen = random.sample(usable, n_rects + 1)
    marker_color = chosen[0]
    rect_colors = chosen[1:]
    corner_name = random.choice(_CORNER_ORDER)

    grid = np.full((height, width), bg, dtype=int)
    marker = _corner_map(grid.shape)[corner_name]
    grid[marker] = marker_color

    col = left_margin
    for idx, (rect_h, rect_w, color) in enumerate(zip(heights, widths, rect_colors)):
        r0 = height - rect_h
        grid[r0:height, col : col + rect_w] = color
        if idx < len(gaps):
            col += rect_w + gaps[idx]

    bg_count = int(np.count_nonzero(grid == bg))
    color_counts = Counter(map(int, grid.ravel()))
    if any(count >= bg_count for color, count in color_counts.items() if color != bg):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 8")
