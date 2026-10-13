from __future__ import annotations

from collections import deque
import random

import numpy as np


def _foreground_colors(g: np.ndarray) -> list[int]:
    colors = [int(color) for color in np.unique(g) if int(color) != 0]
    assert 3 <= len(colors) <= 6, "expected 3-6 foreground colors"
    return colors


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _contains(box: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _unique_topological_order(colors: list[int], edges: dict[int, set[int]]) -> list[int]:
    indegree = {color: 0 for color in colors}
    for src in colors:
        for dst in edges[src]:
            indegree[dst] += 1

    ready = deque(sorted(color for color in colors if indegree[color] == 0))
    order: list[int] = []
    while ready:
        assert len(ready) == 1, "layer order must be uniquely determined"
        color = ready.popleft()
        order.append(color)
        for dst in sorted(edges[color]):
            indegree[dst] -= 1
            if indegree[dst] == 0:
                ready.append(dst)

    assert len(order) == len(colors), "layer relation graph must be acyclic"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct each color's full rectangle and output horizontal area bands by layer order.

    Input structure
    - Background is black.
    - Each foreground color comes from one solid rectangle.
    - Rectangles overlap, and only the top visible color is shown at each cell.

    Rule
    - Infer each rectangle as the bounding box of its visible pixels.
    - If a visible pixel of color A lies inside rectangle B's bbox, then A must be above B.
    - Combine all such relations into the unique bottom-to-top stacking order.
    - Output one row per color; row width equals the rectangle's full area, padded with black
      to the maximum area across all colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors = _foreground_colors(g)
    bboxes = {color: _bbox(g == color) for color in colors}
    areas = {}
    for color, (r0, c0, r1, c1) in bboxes.items():
        areas[color] = int((r1 - r0 + 1) * (c1 - c0 + 1))

    edges = {color: set() for color in colors}
    for upper in colors:
        visible = np.argwhere(g == upper)
        for r, c in visible:
            rr, cc = int(r), int(c)
            for lower in colors:
                if lower != upper and _contains(bboxes[lower], rr, cc):
                    edges[lower].add(upper)

    order = _unique_topological_order(colors, edges)
    width = max(areas.values())
    out = np.zeros((len(order), width), dtype=int)
    for row, color in enumerate(order):
        out[row, : areas[color]] = int(color)
    return out


# --------------- generator ---------------

COLOR_POOL = tuple(range(1, 10))
GRID_MIN = 12
GRID_MAX = 22
RECT_COUNT_RANGE = (3, 6)
TOP_RECT_SIDE_RANGE = (3, 8)
STEP_RANGE = (1, 2)
MARGIN_RANGE = (0, 4)
MAX_GENERATION_ATTEMPTS = 400


def _sample_steps(count: int) -> list[int]:
    return [random.randint(*STEP_RANGE) for _ in range(count - 1)]


def _prefix_sums(steps: list[int]) -> list[int]:
    total = 0
    out = [0]
    for step in steps:
        total += step
        out.append(total)
    return out


def _sample_geometry(count: int) -> tuple[list[tuple[int, int, int, int]], tuple[int, int]] | None:
    row_steps = _sample_steps(count)
    col_steps = _sample_steps(count)
    row_offsets = _prefix_sums(row_steps)
    col_offsets = _prefix_sums(col_steps)

    top_h = random.randint(*TOP_RECT_SIDE_RANGE)
    top_w = random.randint(*TOP_RECT_SIDE_RANGE)
    bottom_h = top_h + row_offsets[-1]
    bottom_w = top_w + col_offsets[-1]

    inner_h = bottom_h
    inner_w = bottom_w
    total_h = inner_h + random.randint(*MARGIN_RANGE) + random.randint(*MARGIN_RANGE)
    total_w = inner_w + random.randint(*MARGIN_RANGE) + random.randint(*MARGIN_RANGE)
    if not (GRID_MIN <= total_h <= GRID_MAX and GRID_MIN <= total_w <= GRID_MAX):
        return None

    top_margin = random.randint(0, total_h - inner_h)
    left_margin = random.randint(0, total_w - inner_w)
    bottom_r = top_margin + bottom_h - 1
    right_c = left_margin + bottom_w - 1

    rects: list[tuple[int, int, int, int]] = []
    for roff, coff in zip(row_offsets, col_offsets):
        r0 = top_margin + roff
        c0 = left_margin + coff
        rects.append((r0, c0, bottom_r, right_c))

    return rects, (total_h, total_w)


def _render_stacked(
    shape: tuple[int, int],
    colors: list[int],
    rects: list[tuple[int, int, int, int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for color, (r0, c0, r1, c1) in zip(colors, rects):
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return grid


def _expected_output(colors: list[int], rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    areas = [int((r1 - r0 + 1) * (c1 - c0 + 1)) for r0, c0, r1, c1 in rects]
    width = max(areas)
    out = np.zeros((len(colors), width), dtype=int)
    for row, (color, area) in enumerate(zip(colors, areas)):
        out[row, :area] = int(color)
    return out


def _apply_random_flip(grid: np.ndarray) -> np.ndarray:
    out = grid
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*RECT_COUNT_RANGE)
    geometry = _sample_geometry(count)
    if geometry is None:
        return None
    rects, shape = geometry
    colors = random.sample(COLOR_POOL, count)

    inp = _render_stacked(shape, colors, rects)
    inp = _apply_random_flip(inp)
    expected = _expected_output(colors, rects)

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
