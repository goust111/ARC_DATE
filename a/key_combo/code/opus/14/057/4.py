from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6

GENERATE_ATTEMPTS = 500
GRID_MIN = 15
GRID_MAX = 24
LENGTH_MIN = 5
LENGTH_MAX = 10
PAIR_COUNT_WEIGHTS = {2: 3, 3: 4}
TRIGGER_COUNT_WEIGHTS = {
    2: {1: 1, 2: 4},
    3: {1: 1, 2: 3, 3: 3},
}


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    assert dr != 0 and abs(dr) == abs(dc), "blue endpoints must define a 45-degree diagonal"
    sr = _sign(dr)
    sc = _sign(dc)
    return [(r0 + k * sr, c0 + k * sc) for k in range(abs(dr) + 1)]


def _pair_orientation(a: tuple[int, int], b: tuple[int, int]) -> str:
    return "\\" if (b[1] - a[1]) == (b[0] - a[0]) else "/"


def _perp_dirs(orientation: str) -> list[tuple[int, int]]:
    if orientation == "\\":
        return [(-1, 1), (1, -1)]
    return [(-1, -1), (1, 1)]


def _weighted_choice(weights: dict[int, int]) -> int:
    items = sorted(weights.items())
    total = sum(weight for _, weight in items)
    pick = random.randint(1, total)
    acc = 0
    for value, weight in items:
        acc += weight
        if pick <= acc:
            return value
    raise AssertionError("unreachable weighted choice")


def _find_blue_pairs(grid: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    blue_points = [tuple(map(int, rc)) for rc in np.argwhere(grid == BLUE)]
    assert len(blue_points) >= 4 and len(blue_points) % 2 == 0

    groups: list[list[tuple[int, int]]] = []
    by_backslash: dict[int, list[tuple[int, int]]] = defaultdict(list)
    by_slash: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in blue_points:
        by_backslash[r - c].append((r, c))
        by_slash[r + c].append((r, c))

    for coords in by_backslash.values():
        if len(coords) == 2:
            groups.append(sorted(coords))
    for coords in by_slash.values():
        if len(coords) == 2:
            groups.append(sorted(coords))

    usage = defaultdict(int)
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for a, b in groups:
        usage[a] += 1
        usage[b] += 1
        pairs.append((a, b))

    assert len(pairs) * 2 == len(blue_points), "blue pairing must be unique and complete"
    assert all(usage[pt] == 1 for pt in blue_points), "each blue endpoint must belong to exactly one pair"
    return pairs


def _render_line_network(grid: np.ndarray, *, include_rays: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = np.where(g == BLUE, BLUE, BLACK)
    pairs = _find_blue_pairs(g)

    if include_rays:
        for a, b in pairs:
            path = _line_cells(a, b)
            orientation = _pair_orientation(a, b)
            triggers = [(r, c) for r, c in path[1:-1] if int(g[r, c]) == MAGENTA]
            assert len(triggers) <= 1, "each blue diagonal may have at most one trigger"
            for tr, tc in triggers:
                for dr, dc in _perp_dirs(orientation):
                    r, c = tr, tc
                    while _in_bounds(out.shape, r, c):
                        out[r, c] = MAGENTA
                        r += dr
                        c += dc

    for a, b in pairs:
        path = _line_cells(a, b)
        rows, cols = np.array(path, dtype=int).T
        out[rows, cols] = BLUE
    return out


def _analyze_regions(line_grid: np.ndarray):
    g = np.asarray(line_grid, dtype=int)
    seen = np.zeros(g.shape, dtype=bool)
    regions = []
    h, w = g.shape

    for r0 in range(h):
        for c0 in range(w):
            if int(g[r0, c0]) != BLACK or seen[r0, c0]:
                continue

            q = deque([(r0, c0)])
            seen[r0, c0] = True
            cells = []
            touches_edge = False

            while q:
                r, c = q.popleft()
                cells.append((r, c))
                if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                    touches_edge = True
                for nr, nc in _neighbors4(r, c):
                    if not _in_bounds(g.shape, nr, nc):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != BLACK:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            boundary_colors = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) != BLACK:
                        boundary_colors.add(int(g[nr, nc]))

            regions.append(
                {
                    "cells": cells,
                    "touches_edge": touches_edge,
                    "boundary_colors": boundary_colors,
                }
            )
    return regions


def _fill_color(boundary_colors: set[int]) -> int:
    assert boundary_colors in ({BLUE}, {MAGENTA}, {BLUE, MAGENTA})
    if boundary_colors == {BLUE}:
        return RED
    if boundary_colors == {MAGENTA}:
        return GREEN
    return YELLOW


def _paint_filled_output(line_grid: np.ndarray) -> np.ndarray:
    out = np.asarray(line_grid, dtype=int).copy()
    for region in _analyze_regions(out):
        if region["touches_edge"]:
            continue
        color = _fill_color(region["boundary_colors"])
        rows, cols = np.array(region["cells"], dtype=int).T
        out[rows, cols] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect blue diagonal pairs, split magenta triggers into perpendicular rays, then flood-fill enclosed black regions by boundary-color composition.

    Input structure:
    - Black background contains 2-3 blue endpoint pairs and 1-3 magenta trigger pixels.
    - Each blue pair lies on one strict 45-degree diagonal, and each trigger lies strictly inside one such blue-to-blue diagonal.
    - Different blue diagonals do not cross; triggers only mark where perpendicular rays should be emitted.

    Rule:
    - For each blue pair, draw the full blue diagonal segment between the endpoints.
    - For each magenta trigger on that segment, extend a magenta diagonal ray in both perpendicular 45-degree directions until the grid boundary.
    - Blue has higher priority than magenta at every overlap, so the completed blue segments overwrite ray cells they pass through.
    - Find every black 4-connected region that does not touch the grid edge, inspect the set of adjacent non-black boundary colors, and fill it red for blue-only boundaries, green for magenta-only boundaries, or yellow for mixed blue-magenta boundaries.

    Key insight:
    - The output depends on a dependency chain: blue pair completion defines the base lattice, triggers rotate that lattice into perpendicular beam lines, and only after those regions are formed can the boundary-color set determine which fill color each enclosed pocket receives.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    assert set(map(int, np.unique(g))).issubset({BLACK, BLUE, MAGENTA}), "input may only use colors 0, 1, 6"

    line_grid = _render_line_network(g, include_rays=True)
    return _paint_filled_output(line_grid)


def _candidate_input(
    h: int,
    w: int,
    pair_count: int,
    trigger_indices: list[int],
    orientation: str,
    length: int,
    base_r: int,
    base_c: int,
    trigger_steps: dict[int, int],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for idx in range(pair_count):
        r0 = base_r + 2 * idx
        if orientation == "\\":
            c0 = base_c
            a = (r0, c0)
            b = (r0 + length, c0 + length)
            trigger = (r0 + trigger_steps[idx], c0 + trigger_steps[idx]) if idx in trigger_steps else None
        else:
            c0 = base_c
            a = (r0, c0)
            b = (r0 + length, c0 - length)
            trigger = (r0 + trigger_steps[idx], c0 - trigger_steps[idx]) if idx in trigger_steps else None

        grid[a] = BLUE
        grid[b] = BLUE
        if trigger is not None:
            grid[trigger] = MAGENTA
    return grid


def _blue_paths_do_not_cross(grid: np.ndarray) -> bool:
    pairs = _find_blue_pairs(grid)
    occupied: set[tuple[int, int]] = set()
    for a, b in pairs:
        path = set(_line_cells(a, b))
        if occupied & path:
            return False
        occupied |= path
    return True


def _input_is_unambiguous(grid: np.ndarray) -> bool:
    try:
        pairs = _find_blue_pairs(grid)
    except AssertionError:
        return False

    if not _blue_paths_do_not_cross(grid):
        return False

    magenta_points = [tuple(map(int, rc)) for rc in np.argwhere(grid == MAGENTA)]
    memberships = defaultdict(int)
    for a, b in pairs:
        path = set(_line_cells(a, b)[1:-1])
        line_trigger_count = 0
        for pt in magenta_points:
            if pt in path:
                memberships[pt] += 1
                line_trigger_count += 1
        if line_trigger_count > 1:
            return False
    return magenta_points and all(memberships[pt] == 1 for pt in magenta_points)


def _validate_generated_input(grid: np.ndarray) -> bool:
    if not _input_is_unambiguous(grid):
        return False

    line_grid = _render_line_network(grid, include_rays=True)
    no_ray_grid = _render_line_network(grid, include_rays=False)

    enclosed = [reg for reg in _analyze_regions(line_grid) if not reg["touches_edge"]]
    if len(enclosed) < 2:
        return False

    boundary_sets = {tuple(sorted(reg["boundary_colors"])) for reg in enclosed}
    if (MAGENTA,) not in boundary_sets and (BLUE, MAGENTA) not in boundary_sets:
        return False

    trigger_count = int(np.count_nonzero(grid == MAGENTA))
    if trigger_count >= 2:
        if not {(BLUE,), (MAGENTA,), (BLUE, MAGENTA)} <= {
            tuple(sorted(reg["boundary_colors"])) for reg in enclosed
        }:
            return False
    else:
        if not {(BLUE,), (BLUE, MAGENTA)} <= {
            tuple(sorted(reg["boundary_colors"])) for reg in enclosed
        }:
            return False

    filled_full = _paint_filled_output(line_grid)
    filled_no_rays = _paint_filled_output(no_ray_grid)
    if np.array_equal(filled_full, filled_no_rays):
        return False

    # The completed blue lattice itself must matter, not only the original sparse endpoints.
    if np.array_equal(filled_full, grid):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    pair_count = _weighted_choice(PAIR_COUNT_WEIGHTS)
    trigger_count = _weighted_choice(TRIGGER_COUNT_WEIGHTS[pair_count])
    orientation = random.choice(["\\", "/"])

    max_length = min(LENGTH_MAX, h - (2 * pair_count + 3), w - 5)
    if orientation == "/":
        max_length = min(max_length, w - 5)
    if max_length < LENGTH_MIN:
        return None
    length = random.randint(LENGTH_MIN, max_length)

    max_base_r = h - (length + 2 * (pair_count - 1)) - 3
    if orientation == "\\":
        max_base_c = w - length - 3
        if max_base_c < 2:
            return None
        base_c = random.randint(2, max_base_c)
    else:
        min_base_c = length + 2
        max_base_c = w - 3
        if min_base_c > max_base_c:
            return None
        base_c = random.randint(min_base_c, max_base_c)

    if max_base_r < 2:
        return None
    base_r = random.randint(2, max_base_r)

    trigger_indices = sorted(random.sample(range(pair_count), trigger_count))
    trigger_steps = {idx: random.randint(1, length - 1) for idx in trigger_indices}
    grid = _candidate_input(h, w, pair_count, trigger_indices, orientation, length, base_r, base_c, trigger_steps)

    return grid if _validate_generated_input(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-partition flood-fill input")
