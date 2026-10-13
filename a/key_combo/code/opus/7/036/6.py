from __future__ import annotations

import random
from collections import deque

import numpy as np


MIN_SIDE = 4
YELLOW = 4
GREEN = 3

GRID_HEIGHT_RANGE = (17, 23)
GRID_WIDTH_RANGE = (17, 23)
RECT_COUNT_RANGE = (3, 5)
RECT_HEIGHT_RANGE = (4, 7)
RECT_WIDTH_RANGE = (4, 7)
MIN_ADJ_OVERLAP = 2
PALETTE = (0, 1, 2, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
MAX_RECT_PLACEMENT_ATTEMPTS = 250

Rect = tuple[int, int, int, int, int]


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    pref = np.pad(mask.astype(int), ((1, 0), (1, 0)), constant_values=0)
    return pref.cumsum(axis=0).cumsum(axis=1)


def _rect_sum(pref: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    return int(
        pref[r1 + 1, c1 + 1]
        - pref[r0, c1 + 1]
        - pref[r1 + 1, c0]
        + pref[r0, c0]
    )


def _build_color_prefixes(grid: np.ndarray) -> dict[int, np.ndarray]:
    return {int(color): _prefix_sum(grid == int(color)) for color in np.unique(grid)}


def _is_maximal_uniform_rect(
    pref: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> bool:
    width = c1 - c0 + 1
    height = r1 - r0 + 1
    if r0 > 0 and _rect_sum(pref, r0 - 1, c0, r0 - 1, c1) == width:
        return False
    if r1 + 1 < pref.shape[0] - 1 and _rect_sum(pref, r1 + 1, c0, r1 + 1, c1) == width:
        return False
    if c0 > 0 and _rect_sum(pref, r0, c0 - 1, r1, c0 - 1) == height:
        return False
    if c1 + 1 < pref.shape[1] - 1 and _rect_sum(pref, r0, c1 + 1, r1, c1 + 1) == height:
        return False
    return True


def _maximal_uniform_rectangles(grid: np.ndarray) -> list[Rect]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    prefixes = _build_color_prefixes(g)
    rects: list[Rect] = []
    for color, pref in prefixes.items():
        row_intervals = [(r0, r1) for r0 in range(h - MIN_SIDE + 1) for r1 in range(r0 + MIN_SIDE - 1, h)]
        col_intervals = [(c0, c1) for c0 in range(w - MIN_SIDE + 1) for c1 in range(c0 + MIN_SIDE - 1, w)]
        for r0, r1 in row_intervals:
            height = r1 - r0 + 1
            for c0, c1 in col_intervals:
                area = height * (c1 - c0 + 1)
                if _rect_sum(pref, r0, c0, r1, c1) != area:
                    continue
                if _is_maximal_uniform_rect(pref, r0, c0, r1, c1):
                    rects.append((r0, c0, r1, c1, color))
    rects.sort()
    return rects


def _overlap_1d(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _rectangles_adjacent(a: Rect, b: Rect) -> bool:
    ar0, ac0, ar1, ac1, _ = a
    br0, bc0, br1, bc1, _ = b

    if _overlap_1d(ar0, ar1, br0, br1):
        if ac1 < bc0:
            return bc0 - ac1 - 1 <= 1
        if bc1 < ac0:
            return ac0 - bc1 - 1 <= 1

    if _overlap_1d(ac0, ac1, bc0, bc1):
        if ar1 < br0:
            return br0 - ar1 - 1 <= 1
        if br1 < ar0:
            return ar0 - br1 - 1 <= 1

    return False


def _build_graph(rects: list[Rect]) -> dict[int, set[int]]:
    graph = {idx: set() for idx in range(len(rects))}
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if _rectangles_adjacent(rects[i], rects[j]):
                graph[i].add(j)
                graph[j].add(i)
    return graph


def _anchor_index(rects: list[Rect]) -> int:
    return min(
        range(len(rects)),
        key=lambda idx: (rects[idx][0] + rects[idx][1], rects[idx][0], rects[idx][1], idx),
    )


def _bipartite_coloring(graph: dict[int, set[int]], start: int) -> dict[int, int]:
    colors = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            expected = 1 - colors[node]
            if neighbor in colors:
                assert colors[neighbor] == expected, "rectangle adjacency graph must be bipartite"
                continue
            colors[neighbor] = expected
            queue.append(neighbor)
    assert len(colors) == len(graph), "rectangle adjacency graph must be connected"
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect the large hidden monochrome rectangles and recolor them by alternating graph parity.

    Input structure:
    - The grid is a noisy multicolor background with 3-5 hidden solid rectangles.
    - Each meaningful rectangle is axis-aligned, monochrome, maximal, and has height and width at least 4.
    - Rectangle adjacency is defined by horizontal or vertical proximity: overlapping spans with a gap of at most one cell.

    Rule:
    - Find every maximal same-color rectangle whose height and width are both at least 4.
    - Build a graph on those rectangles, connecting two rectangles when they are horizontally or vertically adjacent with gap 0 or 1.
    - Choose the rectangle whose top-left corner is closest to the grid origin and color it yellow; then color graph neighbors green and continue alternating.
    - Recolor only the detected rectangles; leave all noise pixels unchanged.

    Key insight:
    - The output colors are determined by the bipartite structure of the rectangle adjacency graph, so rectangle detection must happen before recoloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rects = _maximal_uniform_rectangles(g)
    assert rects, "expected at least one maximal hidden rectangle"

    graph = _build_graph(rects)
    parity = _bipartite_coloring(graph, _anchor_index(rects))

    out = g.copy()
    occupied = np.zeros(g.shape, dtype=bool)
    for idx, (r0, c0, r1, c1, color) in enumerate(rects):
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert patch.shape[0] >= MIN_SIDE and patch.shape[1] >= MIN_SIDE
        assert np.all(patch == color)
        assert not np.any(occupied[r0 : r1 + 1, c0 : c1 + 1])
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        out[r0 : r1 + 1, c0 : c1 + 1] = YELLOW if parity[idx] == 0 else GREEN
    return out


def _interval_overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _rectangles_overlap(a: Rect, b: Rect) -> bool:
    ar0, ac0, ar1, ac1, _ = a
    br0, bc0, br1, bc1, _ = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _sample_adjacent_rect(
    existing: list[tuple[Rect, int]],
    parent_idx: int,
    child_color: int,
    child_parity: int,
    h: int,
    w: int,
    grid_h: int,
    grid_w: int,
) -> Rect | None:
    parent = existing[parent_idx][0]
    pr0, pc0, pr1, pc1, _ = parent
    tries = list(range(32))
    random.shuffle(tries)
    for _ in tries:
        side = random.choice(("up", "down", "left", "right"))
        gap = random.choice((0, 1))
        if side in ("left", "right"):
            lo = max(0, pr0 - h + MIN_ADJ_OVERLAP)
            hi = min(grid_h - h, pr1 - MIN_ADJ_OVERLAP + 1)
            if lo > hi:
                continue
            r0 = random.randint(lo, hi)
            c0 = pc0 - gap - w if side == "left" else pc1 + gap + 1
        else:
            lo = max(0, pc0 - w + MIN_ADJ_OVERLAP)
            hi = min(grid_w - w, pc1 - MIN_ADJ_OVERLAP + 1)
            if lo > hi:
                continue
            c0 = random.randint(lo, hi)
            r0 = pr0 - gap - h if side == "up" else pr1 + gap + 1

        r1 = r0 + h - 1
        c1 = c0 + w - 1
        if r0 < 0 or c0 < 0 or r1 >= grid_h or c1 >= grid_w:
            continue

        candidate = (r0, c0, r1, c1, child_color)
        if any(_rectangles_overlap(candidate, rect) for rect, _ in existing):
            continue

        neighbors = [idx for idx, (rect, _) in enumerate(existing) if _rectangles_adjacent(candidate, rect)]
        if parent_idx not in neighbors:
            continue
        if any(existing[idx][1] == child_parity for idx in neighbors):
            continue
        return candidate
    return None


def _place_rectangles(grid_h: int, grid_w: int, count: int) -> tuple[list[Rect], list[int]] | None:
    colors = random.sample(PALETTE, count)
    root_h = random.randint(*RECT_HEIGHT_RANGE)
    root_w = random.randint(*RECT_WIDTH_RANGE)
    if root_h > grid_h or root_w > grid_w:
        return None

    root_r0 = random.randint(0, grid_h - root_h)
    root_c0 = random.randint(0, grid_w - root_w)
    root = (root_r0, root_c0, root_r0 + root_h - 1, root_c0 + root_w - 1, colors[0])
    placed: list[tuple[Rect, int]] = [(root, 0)]
    used_gaps: list[int] = []

    for idx in range(1, count):
        child_h = random.randint(*RECT_HEIGHT_RANGE)
        child_w = random.randint(*RECT_WIDTH_RANGE)
        success = False
        parent_order = list(range(len(placed)))
        random.shuffle(parent_order)
        for parent_idx in parent_order:
            rect = _sample_adjacent_rect(
                placed,
                parent_idx,
                colors[idx],
                1 - placed[parent_idx][1],
                child_h,
                child_w,
                grid_h,
                grid_w,
            )
            if rect is None:
                continue
            parent = placed[parent_idx][0]
            if _overlap_1d(rect[0], rect[2], parent[0], parent[2]):
                gap = max(rect[1], parent[1]) - min(rect[3], parent[3]) - 1
            else:
                gap = max(rect[0], parent[0]) - min(rect[2], parent[2]) - 1
            used_gaps.append(gap)
            placed.append((rect, 1 - placed[parent_idx][1]))
            success = True
            break
        if not success:
            return None

    rects = [rect for rect, _ in placed]
    sums = sorted(rect[0] + rect[1] for rect in rects)
    if len(sums) >= 2 and sums[0] == sums[1]:
        return None
    return rects, used_gaps


def _forbidden_colors(rects: list[Rect], shape: tuple[int, int]) -> list[list[set[int]]]:
    h, w = shape
    forbidden = [[set() for _ in range(w)] for _ in range(h)]
    rect_mask = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1, _ in rects:
        rect_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    for r0, c0, r1, c1, color in rects:
        for r in range(max(0, r0 - 1), min(h, r1 + 2)):
            for c in range(max(0, c0 - 1), min(w, c1 + 2)):
                if not rect_mask[r, c]:
                    forbidden[r][c].add(color)
    return forbidden


def _fill_noise(rects: list[Rect], shape: tuple[int, int]) -> np.ndarray | None:
    h, w = shape
    grid = np.empty(shape, dtype=int)
    forbidden = _forbidden_colors(rects, shape)
    rect_mask = np.zeros(shape, dtype=bool)

    for r0, c0, r1, c1, color in rects:
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
        rect_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    for r in range(h):
        for c in range(w):
            if rect_mask[r, c]:
                continue
            choices = [color for color in PALETTE if color not in forbidden[r][c]]
            if not choices:
                return None
            grid[r, c] = random.choice(choices)
    return grid


def _expected_output(grid: np.ndarray, rects: list[Rect]) -> np.ndarray:
    graph = _build_graph(rects)
    parity = _bipartite_coloring(graph, _anchor_index(rects))
    out = grid.copy()
    for idx, (r0, c0, r1, c1, _) in enumerate(rects):
        out[r0 : r1 + 1, c0 : c1 + 1] = YELLOW if parity[idx] == 0 else GREEN
    return out


def _valid_instance(grid: np.ndarray, rects: list[Rect], used_gaps: list[int]) -> bool:
    detected = _maximal_uniform_rectangles(grid)
    if detected != sorted(rects):
        return False

    graph = _build_graph(detected)
    if any(len(neighbors) == 0 for neighbors in graph.values()):
        return False
    try:
        parity = _bipartite_coloring(graph, _anchor_index(detected))
    except AssertionError:
        return False

    if set(parity.values()) != {0, 1}:
        return False
    if 1 not in used_gaps:
        return False

    expected = _expected_output(grid, detected)
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_HEIGHT_RANGE)
    grid_w = random.randint(*GRID_WIDTH_RANGE)
    rect_count = random.randint(*RECT_COUNT_RANGE)

    placed = _place_rectangles(grid_h, grid_w, rect_count)
    if placed is None:
        return None
    rects, used_gaps = placed

    grid = _fill_noise(rects, (grid_h, grid_w))
    if grid is None:
        return None
    if not _valid_instance(grid, rects, used_gaps):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle 6 input")
