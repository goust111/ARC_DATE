from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Generator ranges.
MIN_COLORS = 3
MAX_COLORS = 6
MIN_ARM = 1
MAX_ARM = 4
MIN_EXTRA_EDGES = 0
MAX_EXTRA_EDGES = 3
PLACEMENT_MARGIN = 1
MAX_PLACE_ATTEMPTS = 80
MAX_GENERATION_ATTEMPTS = 400


def _crossing_edges(grid: np.ndarray, bg: int) -> set[tuple[int, int]]:
    """Read visible-over-hidden priority relations from orthogonal crossings."""
    h, w = grid.shape
    edges: set[tuple[int, int]] = set()
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center = int(grid[r, c])
            if center == bg:
                continue

            left = int(grid[r, c - 1])
            right = int(grid[r, c + 1])
            up = int(grid[r - 1, c])
            down = int(grid[r + 1, c])

            horiz = left if left == right and left != bg else None
            vert = up if up == down and up != bg else None
            if horiz is None or vert is None or horiz == vert:
                continue

            if center == horiz:
                edges.add((horiz, vert))
            elif center == vert:
                edges.add((vert, horiz))
            else:
                raise AssertionError(f"crossing center does not match either path color at {(r, c)}")
    return edges


def _unique_priority_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    """Topologically sort the visible-over-hidden relations and require a unique order."""
    higher_to_lower = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}
    for higher, lower in edges:
        if higher == lower:
            continue
        if lower in higher_to_lower[higher]:
            continue
        higher_to_lower[higher].add(lower)
        indegree[lower] += 1

    order: list[int] = []
    available = [color for color in colors if indegree[color] == 0]
    while available:
        if len(available) != 1:
            raise AssertionError("color priority is ambiguous")
        color = available.pop()
        order.append(color)
        for lower in higher_to_lower[color]:
            indegree[lower] -= 1
            if indegree[lower] == 0:
                available.append(lower)

    if len(order) != len(colors):
        raise AssertionError("color priority relations must form an acyclic total order")
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the global color priority from path crossings and output it as a single-column bar.

    Input structure:
    - The background is black.
    - Foreground colors form one-pixel-wide orthogonal paths made of horizontal
      and vertical line segments.
    - Some path pairs cross at single pixels, and each crossing visibly shows
      only the higher-priority color.

    Rule:
    - Leave non-crossing path pixels conceptually unchanged.
    - At each crossing, compare the visible center color with the perpendicular
      continuation visible above/below or left/right to recover which color is
      higher and which is hidden.
    - Combine all crossing constraints into one global priority order and output
      that order as a one-column list from highest priority to lowest.

    Key insight:
    - A valid crossing is locally recognizable because one color continues
      straight through the center while a different color continues straight
      across the perpendicular axis, so every crossing contributes a directed
      visible-over-hidden ordering edge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors = list(dict.fromkeys(int(value) for value in g.ravel() if int(value) != BG))
    assert colors, "expected at least one foreground color"
    if len(colors) == 1:
        return np.array([[colors[0]]], dtype=int)

    edges = _crossing_edges(g, BG)
    assert edges, "expected at least one informative crossing"
    order = _unique_priority_order(colors, edges)
    return np.array(order, dtype=int).reshape(-1, 1)


def _draw_crossing(
    grid: np.ndarray,
    center: tuple[int, int],
    higher: int,
    lower: int,
    higher_axis: str,
    left: int,
    right: int,
    up: int,
    down: int,
) -> None:
    r, c = center
    grid[r, c - left : c + right + 1] = higher if higher_axis == "h" else lower
    grid[r - up : r + down + 1, c] = higher if higher_axis == "v" else lower
    grid[r, c] = higher


def _reserve_box(
    occupied: np.ndarray,
    center: tuple[int, int],
    left: int,
    right: int,
    up: int,
    down: int,
    margin: int,
) -> bool:
    r, c = center
    h, w = occupied.shape
    r0 = r - up - margin
    r1 = r + down + margin
    c0 = c - left - margin
    c1 = c + right + margin
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
        return False
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True
    return True


def _candidate_edges(order: list[int]) -> list[tuple[int, int]]:
    edges = [(order[i], order[i + 1]) for i in range(len(order) - 1)]
    extras = [(order[i], order[j]) for i in range(len(order)) for j in range(i + 2, len(order))]
    random.shuffle(extras)
    extra_count = min(len(extras), random.randint(MIN_EXTRA_EDGES, MAX_EXTRA_EDGES))
    edges.extend(extras[:extra_count])
    random.shuffle(edges)
    return edges


def _grid_shape(n_crossings: int) -> tuple[int, int]:
    side_min = max(12, int(np.ceil(np.sqrt(40 * n_crossings))))
    side_max = min(30, side_min + 8)
    return random.randint(side_min, side_max), random.randint(side_min, side_max)


def _place_crossings(order: list[int]) -> np.ndarray | None:
    edges = _candidate_edges(order)
    height, width = _grid_shape(len(edges))
    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    for higher, lower in edges:
        placed = False
        for _ in range(MAX_PLACE_ATTEMPTS):
            left = random.randint(MIN_ARM, MAX_ARM)
            right = random.randint(MIN_ARM, MAX_ARM)
            up = random.randint(MIN_ARM, MAX_ARM)
            down = random.randint(MIN_ARM, MAX_ARM)
            r = random.randint(up + PLACEMENT_MARGIN, height - down - PLACEMENT_MARGIN - 1)
            c = random.randint(left + PLACEMENT_MARGIN, width - right - PLACEMENT_MARGIN - 1)
            if not _reserve_box(occupied, (r, c), left, right, up, down, PLACEMENT_MARGIN):
                continue

            higher_axis = random.choice(["h", "v"])
            _draw_crossing(grid, (r, c), higher, lower, higher_axis, left, right, up, down)
            placed = True
            break
        if not placed:
            return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_colors = random.randint(MIN_COLORS, MAX_COLORS)
    order = random.sample(PALETTE, n_colors)
    grid = _place_crossings(order)
    if grid is None:
        return None

    expected = np.array(order, dtype=int).reshape(-1, 1)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid crossing-priority input")
