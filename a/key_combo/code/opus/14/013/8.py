import random
from typing import Dict, List, Tuple

import numpy as np


BACKGROUND = 0
PALETTE = (1, 2, 4, 5)
HEIGHT_RANGE = (8, 15)
WIDTH_RANGE = (8, 15)
RECT_H_RANGE = (2, 4)
RECT_W_RANGE = (3, 5)
RECT_COUNT_RANGE = (2, 4)
MAX_ATTEMPTS = 500
CORNER_MODES = ("nw", "ne", "sw", "se")


Rect = Dict[str, int]


def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_cells(rect: Rect) -> List[Tuple[int, int]]:
    return [
        (r, c)
        for r in range(rect["r0"], rect["r1"] + 1)
        for c in range(rect["c0"], rect["c1"] + 1)
    ]


def _intersect_area(a: Rect, b: Rect) -> int:
    r0 = max(a["r0"], b["r0"])
    c0 = max(a["c0"], b["c0"])
    r1 = min(a["r1"], b["r1"])
    c1 = min(a["c1"], b["c1"])
    if r0 > r1 or c0 > c1:
        return 0
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _covers(rect: Rect, r: int, c: int) -> bool:
    return rect["r0"] <= r <= rect["r1"] and rect["c0"] <= c <= rect["c1"]


def _place_corner_overlap(lower: Rect, h: int, w: int, mode: str, ov_h: int, ov_w: int) -> Rect:
    if mode == "nw":
        r0 = lower["r0"] - (h - ov_h)
        c0 = lower["c0"] - (w - ov_w)
    elif mode == "ne":
        r0 = lower["r0"] - (h - ov_h)
        c0 = lower["c0"] + (lower["w"] - ov_w)
    elif mode == "sw":
        r0 = lower["r0"] + (lower["h"] - ov_h)
        c0 = lower["c0"] - (w - ov_w)
    elif mode == "se":
        r0 = lower["r0"] + (lower["h"] - ov_h)
        c0 = lower["c0"] + (lower["w"] - ov_w)
    else:
        raise ValueError(mode)
    return {"r0": r0, "c0": c0, "r1": r0 + h - 1, "c1": c0 + w - 1, "h": h, "w": w}


def _shift_rect(rect: Rect, dr: int, dc: int, color: int) -> Rect:
    return {
        "color": color,
        "r0": rect["r0"] + dr,
        "c0": rect["c0"] + dc,
        "r1": rect["r1"] + dr,
        "c1": rect["c1"] + dc,
        "h": rect["h"],
        "w": rect["w"],
    }


def _topological_order(graph: Dict[int, set]) -> List[int]:
    indegree = {node: 0 for node in graph}
    for node in graph:
        for nxt in graph[node]:
            indegree[nxt] += 1

    order: List[int] = []
    remaining = set(graph)
    while remaining:
        zeros = sorted(node for node in remaining if indegree[node] == 0)
        assert len(zeros) == 1, "depth order must be uniquely determined"
        node = zeros[0]
        remaining.remove(node)
        order.append(node)
        for nxt in graph[node]:
            indegree[nxt] -= 1
    return order


def _infer_rectangles(grid: np.ndarray) -> Tuple[Dict[int, Rect], List[int]]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(int(v) for v in np.unique(g) if int(v) != BACKGROUND)
    assert 2 <= len(colors) <= 4

    rects: Dict[int, Rect] = {}
    graph = {color: set() for color in colors}

    for color in colors:
        mask = g == color
        r0, c0, r1, c1 = _bbox_from_mask(mask)
        box = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(box != BACKGROUND), "rectangle footprint must stay foreground-filled"
        rects[color] = {"color": color, "r0": r0, "c0": c0, "r1": r1, "c1": c1, "h": r1 - r0 + 1, "w": c1 - c0 + 1}
        for other in colors:
            if other != color and np.any(box == other):
                graph[other].add(color)

    order = _topological_order(graph)
    ranks = {color: idx for idx, color in enumerate(order)}

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            covering = [color for color in colors if _covers(rects[color], r, c)]
            assert len(covering) <= 2, "expected at most two-layer overlap"
            if not covering:
                assert int(g[r, c]) == BACKGROUND
            else:
                top = min(covering, key=lambda color: ranks[color])
                assert int(g[r, c]) == top

    return rects, order


def _encoded_output(shape: Tuple[int, int], rects: Dict[int, Rect], order: List[int]) -> np.ndarray:
    height, width = shape
    out = np.zeros((2 * height, 2 * width), dtype=int)
    ranks = {color: idx for idx, color in enumerate(order)}
    colors = list(rects)

    for r in range(height):
        for c in range(width):
            covering = [color for color in colors if _covers(rects[color], r, c)]
            rr = 2 * r
            cc = 2 * c
            if not covering:
                continue
            if len(covering) == 1:
                out[rr : rr + 2, cc : cc + 2] = covering[0]
                continue
            covering.sort(key=lambda color: ranks[color])
            top, bottom = covering
            out[rr, cc] = top
            out[rr + 1, cc + 1] = top
            out[rr, cc + 1] = bottom
            out[rr + 1, cc] = bottom

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Double an occluded-rectangle grid and encode two-layer depth inside overlapped 2x2 blocks.

    Input structure:
    - The background is black.
    - There are 2 to 4 distinct-colored solid axis-aligned rectangles.
    - Lower rectangles may be partially hidden by upper rectangles, but each color's
      visible pixels still determine that rectangle's full bounding box.

    Rule:
    - For each non-background color, recover its full rectangular footprint as the
      bounding box of its visible pixels.
    - Infer depth order from bbox containment-by-replacement: if color A appears
      inside color B's bbox, then A is above B.
    - Create a grid with doubled height and width. A background cell becomes a
      black 2x2 block, a singly covered cell becomes a solid 2x2 block of that
      rectangle's color, and a doubly covered cell becomes a 2x2 depth code with
      the upper color on the main diagonal and the lower color on the anti-diagonal.

    Key insight:
    - The visible pixels are enough to recover both the full hidden rectangle
      footprints and the global depth order, and the doubled subpixel space is
      then used to show both layers at once in overlap cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    rects, order = _infer_rectangles(g)
    return _encoded_output(g.shape, rects, order)


def _render_input(shape: Tuple[int, int], rects_top_to_bottom: List[Rect]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for rect in reversed(rects_top_to_bottom):
        grid[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = rect["color"]
    return grid


def _edge_fully_visible(grid: np.ndarray, rect: Rect) -> bool:
    r0, c0, r1, c1 = rect["r0"], rect["c0"], rect["r1"], rect["c1"]
    color = rect["color"]
    return (
        np.all(grid[r0, c0 : c1 + 1] == color)
        or np.all(grid[r1, c0 : c1 + 1] == color)
        or np.all(grid[r0 : r1 + 1, c0] == color)
        or np.all(grid[r0 : r1 + 1, c1] == color)
    )


def _validate_scene(shape: Tuple[int, int], rects_top_to_bottom: List[Rect], grid: np.ndarray) -> None:
    expected = np.zeros(shape, dtype=int)
    coverage = np.zeros(shape, dtype=int)
    rank = {rect["color"]: idx for idx, rect in enumerate(rects_top_to_bottom)}

    for rect in reversed(rects_top_to_bottom):
        expected[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = rect["color"]
    assert np.array_equal(grid, expected)

    for rect in rects_top_to_bottom:
        for r, c in _rect_cells(rect):
            coverage[r, c] += 1
    assert coverage.max() == 2
    assert np.any(coverage == 2)

    for idx, rect in enumerate(rects_top_to_bottom):
        color = rect["color"]
        mask = grid == color
        bbox = _bbox_from_mask(mask)
        assert bbox == (rect["r0"], rect["c0"], rect["r1"], rect["c1"])
        box = grid[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1]
        assert np.all(box != BACKGROUND)
        assert _edge_fully_visible(grid, rect)
        if idx > 0:
            lower = rects_top_to_bottom[idx - 1]
            overlap = _intersect_area(rect, lower)
            assert overlap > 0
            assert overlap < rect["h"] * rect["w"]
            assert overlap < lower["h"] * lower["w"]

    inferred_rects, inferred_order = _infer_rectangles(grid)
    assert inferred_order == [rect["color"] for rect in rects_top_to_bottom]
    for rect in rects_top_to_bottom:
        inf = inferred_rects[rect["color"]]
        assert (inf["r0"], inf["c0"], inf["r1"], inf["c1"]) == (
            rect["r0"],
            rect["c0"],
            rect["r1"],
            rect["c1"],
        )

    solver_out = main(grid)
    expected_out = _encoded_output(shape, inferred_rects, inferred_order)
    assert np.array_equal(solver_out, expected_out)

    unique_cells = {rect["color"]: 0 for rect in rects_top_to_bottom}
    overlap_pairs = set()
    colors = [rect["color"] for rect in rects_top_to_bottom]
    rect_map = {rect["color"]: rect for rect in rects_top_to_bottom}
    for r in range(shape[0]):
        for c in range(shape[1]):
            covering = [color for color in colors if _covers(rect_map[color], r, c)]
            if len(covering) == 1:
                unique_cells[covering[0]] += 1
            elif len(covering) == 2:
                covering.sort(key=lambda color: rank[color])
                overlap_pairs.add(tuple(covering))
    assert all(count > 0 for count in unique_cells.values())
    assert overlap_pairs == {
        (rects_top_to_bottom[i]["color"], rects_top_to_bottom[i + 1]["color"])
        for i in range(len(rects_top_to_bottom) - 1)
    }


def _generate_relative_chain(count: int) -> List[Rect] | None:
    colors = random.sample(PALETTE, count)
    dims = [
        (
            random.randint(*RECT_H_RANGE),
            random.randint(*RECT_W_RANGE),
        )
        for _ in range(count)
    ]

    bottom_h, bottom_w = dims[-1]
    bottom = {"r0": 0, "c0": 0, "r1": bottom_h - 1, "c1": bottom_w - 1, "h": bottom_h, "w": bottom_w}
    bottom["color"] = colors[-1]
    rects_bottom_to_top = [bottom]

    for idx in range(count - 2, -1, -1):
        h, w = dims[idx]
        lower = rects_bottom_to_top[-1]
        placed = None
        attempts = 0
        while attempts < 80:
            attempts += 1
            ov_h = random.randint(1, min(h, lower["h"]) - 1)
            ov_w = random.randint(1, min(w, lower["w"]) - 1)
            overlap = ov_h * ov_w
            if overlap < 2:
                continue
            if overlap > int(0.5 * min(h * w, lower["h"] * lower["w"])):
                continue
            mode = random.choice(CORNER_MODES)
            candidate = _place_corner_overlap(lower, h, w, mode, ov_h, ov_w)
            if any(_intersect_area(candidate, earlier) > 0 for earlier in rects_bottom_to_top[:-1]):
                continue
            candidate["color"] = colors[idx]
            placed = candidate
            break
        if placed is None:
            return None
        rects_bottom_to_top.append(placed)

    return list(reversed(rects_bottom_to_top))


def _fit_chain_to_grid(rects_top_to_bottom: List[Rect]) -> Tuple[Tuple[int, int], List[Rect]] | None:
    min_r = min(rect["r0"] for rect in rects_top_to_bottom)
    min_c = min(rect["c0"] for rect in rects_top_to_bottom)
    max_r = max(rect["r1"] for rect in rects_top_to_bottom)
    max_c = max(rect["c1"] for rect in rects_top_to_bottom)
    span_h = max_r - min_r + 1
    span_w = max_c - min_c + 1

    min_h = max(HEIGHT_RANGE[0], span_h)
    min_w = max(WIDTH_RANGE[0], span_w)
    if min_h > HEIGHT_RANGE[1] or min_w > WIDTH_RANGE[1]:
        return None

    height = random.randint(min_h, HEIGHT_RANGE[1])
    width = random.randint(min_w, WIDTH_RANGE[1])
    top_margin = random.randint(0, height - span_h)
    left_margin = random.randint(0, width - span_w)
    dr = top_margin - min_r
    dc = left_margin - min_c

    shifted = [_shift_rect(rect, dr, dc, rect["color"]) for rect in rects_top_to_bottom]
    return (height, width), shifted


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        count = random.randint(*RECT_COUNT_RANGE)
        rects = _generate_relative_chain(count)
        if rects is None:
            continue
        fitted = _fit_chain_to_grid(rects)
        if fitted is None:
            continue
        shape, shifted = fitted
        grid = _render_input(shape, shifted)
        try:
            _validate_scene(shape, shifted, grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {MAX_ATTEMPTS} attempts")
