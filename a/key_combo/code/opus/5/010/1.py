from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MIN_SIDE = 4
MAX_SIDE = 8
GRID_H_RANGE = (18, 26)
GRID_W_RANGE = (18, 26)
RECT_COUNT_RANGE = (4, 6)
GEN_ATTEMPTS = 5000
PAIR_ATTEMPTS = 160


Rect = tuple[int, int, int, int]


def _bbox_coords(coords: np.ndarray) -> Rect:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_area(rect: Rect) -> int:
    r0, c0, r1, c1 = rect
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _rect_perimeter(rect: Rect) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        pts.append((r, c1))
    return pts


def _edge_cells(rect: Rect, edge: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    if edge == "top":
        return [(r0, c) for c in range(c0, c1 + 1)]
    if edge == "bottom":
        return [(r1, c) for c in range(c0, c1 + 1)]
    if edge == "left":
        return [(r, c0) for r in range(r0, r1 + 1)]
    if edge == "right":
        return [(r, c1) for r in range(r0, r1 + 1)]
    raise ValueError(f"unknown edge {edge}")


def _reconstruct_rectangles(grid: np.ndarray) -> dict[int, Rect]:
    rects: dict[int, Rect] = {}
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        coords = np.argwhere(grid == color)
        assert coords.size > 0, f"missing visible pixels for color {color}"
        rects[color] = _bbox_coords(coords)
    return rects


def _infer_occlusion_order(rects: dict[int, Rect], grid: np.ndarray) -> list[int]:
    cover_map: dict[tuple[int, int], list[int]] = defaultdict(list)
    for color, rect in rects.items():
        for rc in _rect_perimeter(rect):
            cover_map[rc].append(int(color))

    edges: dict[int, set[int]] = {color: set() for color in rects}
    indeg: dict[int, int] = {color: 0 for color in rects}
    for (r, c), colors in cover_map.items():
        if len(colors) < 2:
            continue
        visible = int(grid[r, c])
        assert visible in rects, "overlap pixel must display a foreground color"
        for lower in colors:
            lower = int(lower)
            if lower == visible or visible in edges[lower]:
                continue
            edges[lower].add(visible)
            indeg[visible] += 1

    ready = sorted(color for color, deg in indeg.items() if deg == 0)
    order: list[int] = []
    while ready:
        assert len(ready) == 1, "occlusion constraints must determine a unique total order"
        cur = ready.pop()
        order.append(cur)
        for nxt in sorted(edges[cur]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
                ready.sort()
    assert len(order) == len(rects), "occlusion graph must be acyclic"
    return order


def _draw_concentric_rings(colors: list[int]) -> np.ndarray:
    side = 2 * len(colors) - 1
    out = np.full((side, side), int(colors[0]), dtype=int)
    for layer, color in enumerate(colors[1:], start=1):
        out[layer : side - layer, layer : side - layer] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct occluded hollow rectangles, infer their layer order, and redraw their colors as concentric rings.

    Input structure:
    - The background is black (0).
    - Foreground colors each belong to one one-pixel-wide hollow rectangle frame.
    - The frames partially overlap, so only the topmost color is visible at shared perimeter pixels.
    - Every frame keeps enough visible pixels to recover its original bounding box.

    Rule:
    - For each visible foreground color, take the bounding box of its visible pixels and complete the full hollow rectangle perimeter for that box.
    - At every pixel covered by two or more reconstructed perimeters, the visible color is above the other colors covering that pixel.
    - Combine all such overlap constraints into the unique bottom-to-top occlusion order.
    - Create a square output of side 2N-1, where N is the number of rectangles, and paint concentric square rings from outermost to center using that bottom-to-top color order.

    Key insight:
    - The original rectangles are not copied directly: their partially visible perimeters first reveal the full rectangles, then overlap visibility reveals depth, and that depth order is reversed into outside-to-inside rings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    rects = _reconstruct_rectangles(g)
    order = _infer_occlusion_order(rects, g)
    return _draw_concentric_rings(order)


def _render_input(order: list[int], rects: dict[int, Rect], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for color in order:
        for r, c in _rect_perimeter(rects[color]):
            grid[r, c] = int(color)
    return grid


def _has_adjacent_visible_edges(grid: np.ndarray, color: int, rect: Rect) -> bool:
    pairs = [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    for e1, e2 in pairs:
        if all(int(grid[r, c]) == color for r, c in _edge_cells(rect, e1)) and all(
            int(grid[r, c]) == color for r, c in _edge_cells(rect, e2)
        ):
            return True
    return False


def _overlap_size(a: Rect, b: Rect) -> int:
    return len(set(_rect_perimeter(a)) & set(_rect_perimeter(b)))


def _heuristic_orders(rects: dict[int, Rect], grid: np.ndarray) -> list[list[int]]:
    visible_counts = {color: int(np.sum(grid == color)) for color in rects}
    areas = {color: _rect_area(rect) for color, rect in rects.items()}
    perimeters = {
        color: 2 * ((rect[2] - rect[0] + 1) + (rect[3] - rect[1] + 1)) - 4 for color, rect in rects.items()
    }
    return [
        sorted(rects, key=lambda color: rects[color][0]),
        sorted(rects, key=lambda color: rects[color][1]),
        sorted(rects, key=lambda color: rects[color][0] + rects[color][1]),
        sorted(rects, key=lambda color: areas[color]),
        sorted(rects, key=lambda color: perimeters[color]),
        sorted(rects, key=lambda color: visible_counts[color]),
    ]


def _semantically_valid(order: list[int], rects: dict[int, Rect], grid: np.ndarray) -> bool:
    for color in order:
        if not _has_adjacent_visible_edges(grid, color, rects[color]):
            return False
        vis_bbox = _bbox_coords(np.argwhere(grid == color))
        if vis_bbox != rects[color]:
            return False

    if any(not any(_overlap_size(rects[color], rects[other]) > 0 for other in order if other != color) for color in order):
        return False

    try:
        inferred = _infer_occlusion_order(_reconstruct_rectangles(grid), grid)
    except AssertionError:
        return False
    if inferred != order:
        return False

    for heuristic in _heuristic_orders(rects, grid):
        if heuristic == order or heuristic[::-1] == order:
            return False
    return True


def _choose_start_rect(h: int, w: int) -> Rect:
    rh = random.randint(5, min(MAX_SIDE, h - 6))
    rw = random.randint(5, min(MAX_SIDE, w - 6))
    r0 = random.randint(1, max(1, h // 4))
    c0 = random.randint(1, max(1, w // 4))
    return r0, c0, r0 + rh - 1, c0 + rw - 1


def _propose_next_rect(prev: Rect, h: int, w: int, mode: str) -> Rect | None:
    pr0, pc0, pr1, pc1 = prev
    for _ in range(PAIR_ATTEMPTS):
        rh = random.randint(MIN_SIDE, min(MAX_SIDE, h - 1))
        rw = random.randint(MIN_SIDE, min(MAX_SIDE, w - 1))
        if mode == "top":
            r0 = pr1
            if r0 + rh > h:
                continue
            c_low = max(1, pr0 - 99)  # temporary sentinel, overwritten below
            c_low = max(1, pc0 - rw + 2)
            c_high = min(w - rw - 1, pc1 - 1)
            if c_low > c_high:
                continue
            c0 = random.randint(c_low, c_high)
        else:
            c0 = pc1
            if c0 + rw > w:
                continue
            r_low = max(1, pr0 - rh + 2)
            r_high = min(h - rh - 1, pr1 - 1)
            if r_low > r_high:
                continue
            r0 = random.randint(r_low, r_high)
        rect = (int(r0), int(c0), int(r0 + rh - 1), int(c0 + rw - 1))
        if _overlap_size(prev, rect) >= 2:
            return rect
    return None


def _propose_rectangles(h: int, w: int, n: int) -> tuple[list[int], dict[int, Rect]] | None:
    colors = random.sample(PALETTE, n)
    order = colors[:]
    random.shuffle(order)

    rects: dict[int, Rect] = {order[0]: _choose_start_rect(h, w)}
    for idx in range(1, n):
        prev = rects[order[idx - 1]]
        modes = ["top", "left"]
        random.shuffle(modes)
        placed = None
        for mode in modes:
            placed = _propose_next_rect(prev, h, w, mode)
            if placed is not None:
                break
        if placed is None:
            return None
        rects[order[idx]] = placed
    return order, rects


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n = random.randint(*RECT_COUNT_RANGE)
    proposed = _propose_rectangles(h, w, n)
    if proposed is None:
        return None
    order, rects = proposed
    grid = _render_input(order, rects, (h, w))
    if not _semantically_valid(order, rects, grid):
        return None
    if not np.array_equal(main(grid), _draw_concentric_rings(order)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid occlusion-order ring instance")
