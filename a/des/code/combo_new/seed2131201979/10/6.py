from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np


BG = 0
CROSS = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey crossing so the orthogonal polylines alternate visible overpasses like a weave.

    Input structure:
    - The background is black.
    - Each non-grey foreground color forms one one-pixel-wide orthogonal polyline whose two endpoints lie on the grid edge.
    - Grey pixels mark crossings where exactly two polylines share a cell, and each crossing is a straight vertical-over-horizontal intersection rather than a corner.

    Rule:
    - For every colored polyline, pick the lexicographically earlier endpoint as its start and trace the path to the other endpoint.
    - Record the grey crossings encountered along that path; the 1st, 3rd, 5th, ... crossings are overpasses for that line, while the 2nd, 4th, 6th, ... are underpasses.
    - At each grey crossing, recolor the cell with the color of the line that is on top there.
    - If both lines mark the same crossing as overpasses, keep the one whose chosen start endpoint is earlier; all non-crossing cells stay unchanged.

    Key insight:
    - The grey cells are not a separate object class to preserve; they are missing line colors, and reconstructing the ordered path of each line is enough to recover the alternating over-under weaving pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert 2 <= height <= 30 and 2 <= width <= 30

    occupied = g != BG
    grey_cells = [tuple(map(int, coord)) for coord in np.argwhere(g == CROSS)]
    for r, c in grey_cells:
        neighbors = _occupied_neighbors(occupied, r, c)
        assert len(neighbors) == 4, f"crossing {(r, c)} must have four occupied neighbors"

    line_colors = [int(color) for color in np.unique(g) if int(color) not in (BG, CROSS)]
    assert line_colors, "expected at least one colored polyline"

    line_starts: Dict[int, Tuple[int, int]] = {}
    crossing_orders: Dict[Tuple[int, int], Dict[int, int]] = defaultdict(dict)

    for color in line_colors:
        path, crossings, start = _trace_line(g, color)
        line_starts[color] = start
        for order, cell in enumerate(crossings):
            assert color not in crossing_orders[cell], "a line should not visit the same crossing twice"
            crossing_orders[cell][color] = order

    assert set(crossing_orders) == set(grey_cells), "every grey cell must belong to exactly two traced lines"

    out = g.copy()
    for cell in grey_cells:
        visits = crossing_orders[cell]
        assert len(visits) == 2, f"crossing {cell} must be shared by exactly two lines"
        top_lines = [color for color, order in visits.items() if order % 2 == 0]
        if len(top_lines) == 1:
            chosen = top_lines[0]
        else:
            assert len(top_lines) == 2, f"crossing {cell} has no visible overpass"
            chosen = min(top_lines, key=lambda color: line_starts[color])
        out[cell] = chosen

    return out


def _trace_line(grid: np.ndarray, color: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], Tuple[int, int]]:
    occupied = grid != BG
    coords = [tuple(map(int, coord)) for coord in np.argwhere(grid == color)]
    assert coords, f"missing cells for color {color}"

    endpoints = []
    for r, c in coords:
        neighbors = _occupied_neighbors(occupied, r, c)
        assert len(neighbors) in (1, 2), f"line color {color} has invalid degree at {(r, c)}"
        if len(neighbors) == 1:
            endpoints.append((r, c))

    assert len(endpoints) == 2, f"line color {color} must have exactly two endpoints"
    start = min(endpoints)

    path: List[Tuple[int, int]] = []
    crossings: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    prev: Tuple[int, int] | None = None
    cur = start

    while True:
        assert cur not in seen, f"line color {color} revisits {cur}"
        seen.add(cur)
        path.append(cur)
        if int(grid[cur]) == CROSS:
            crossings.append(cur)

        nxt = _next_step(grid, color, cur, prev)
        if nxt is None:
            break
        prev, cur = cur, nxt

    visited_color_cells = {cell for cell in path if int(grid[cell]) == color}
    assert visited_color_cells == set(coords), f"tracing color {color} left unvisited cells"
    return path, crossings, start


def _next_step(
    grid: np.ndarray,
    color: int,
    cur: Tuple[int, int],
    prev: Tuple[int, int] | None,
) -> Tuple[int, int] | None:
    occupied = grid != BG
    r, c = cur
    neighbors = _occupied_neighbors(occupied, r, c)

    if prev is None:
        assert len(neighbors) == 1, "start endpoint must have a single neighbor"
        return neighbors[0]

    if int(grid[r, c]) == CROSS:
        dr = r - prev[0]
        dc = c - prev[1]
        nxt = (r + dr, c + dc)
        assert _in_bounds(grid.shape, *nxt), f"crossing {(r, c)} exits the grid for color {color}"
        assert nxt in neighbors, f"crossing {(r, c)} is not straight for color {color}"
        return nxt

    candidates = [neighbor for neighbor in neighbors if neighbor != prev]
    if not candidates:
        return None
    assert len(candidates) == 1, f"line color {color} branches at {(r, c)}"
    return candidates[0]


def _occupied_neighbors(occupied: np.ndarray, r: int, c: int) -> List[Tuple[int, int]]:
    height, width = occupied.shape
    result = []
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width and occupied[nr, nc]:
            result.append((nr, nc))
    return result


def _in_bounds(shape: Tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid weaving input after {GEN_RETRIES} attempts")


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(12, 24)
    width = rng.randint(12, 24)

    max_turns = min(3, max(0, min(height - 6, width - 6) // 3))
    turn_count = rng.randint(0, max_turns)
    segment_count = turn_count + 1

    columns = _sample_spaced_values(2, width - 3, segment_count, 3, rng)
    turn_rows = _sample_spaced_values(2, height - 3, turn_count, 3, rng)
    if columns is None or turn_rows is None:
        return None
    turn_rows = sorted(turn_rows)

    available_rows = _available_spoke_rows(height, turn_rows)
    if len(available_rows) < 2:
        return None

    max_spokes = min(6, len(available_rows), len(COLOR_POOL) - 1)
    if max_spokes < 2:
        return None
    spoke_count = rng.randint(2, max_spokes)
    spoke_rows = sorted(rng.sample(available_rows, spoke_count))

    colors = rng.sample(COLOR_POOL, spoke_count + 1)
    trunk_color = colors[0]
    spoke_colors = colors[1:]

    grid = np.full((height, width), BG, dtype=int)
    trunk_path = _build_trunk_path(height, columns, turn_rows)
    if trunk_path is None:
        return None
    if not _paint_path(grid, trunk_path, trunk_color):
        return None

    for row, color in zip(spoke_rows, spoke_colors):
        for col in range(width):
            cell = int(grid[row, col])
            if cell == BG:
                grid[row, col] = color
            elif cell == trunk_color:
                grid[row, col] = CROSS
            else:
                return None
        if int(np.count_nonzero(grid[row, :] == CROSS)) != 1:
            return None

    crossings = [tuple(map(int, coord)) for coord in np.argwhere(grid == CROSS)]
    if len(crossings) != spoke_count:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None

    visible_at_crossings = [int(out[r, c]) for r, c in sorted(crossings)]
    if trunk_color not in visible_at_crossings:
        return None
    if all(color == trunk_color for color in visible_at_crossings):
        return None

    return grid


def _build_trunk_path(
    height: int,
    columns: Sequence[int],
    turn_rows: Sequence[int],
) -> List[Tuple[int, int]] | None:
    assert len(columns) == len(turn_rows) + 1
    path = [(0, int(columns[0]))]
    current_row = 0
    current_col = int(columns[0])

    for turn_row, next_col in zip(turn_rows, columns[1:]):
        if turn_row <= current_row:
            return None
        for row in range(current_row + 1, int(turn_row) + 1):
            path.append((row, current_col))
        step = 1 if next_col > current_col else -1
        for col in range(current_col + step, int(next_col) + step, step):
            path.append((int(turn_row), col))
        current_row = int(turn_row)
        current_col = int(next_col)

    for row in range(current_row + 1, height):
        path.append((row, current_col))

    if path[-1][0] != height - 1:
        return None
    if len(path) != len(set(path)):
        return None
    return path


def _paint_path(grid: np.ndarray, path: Sequence[Tuple[int, int]], color: int) -> bool:
    for r, c in path:
        if int(grid[r, c]) != BG:
            return False
        grid[r, c] = color
    return True


def _available_spoke_rows(height: int, turn_rows: Sequence[int]) -> List[int]:
    available: List[int] = []
    prev = 0
    for stop in list(turn_rows) + [height - 1]:
        available.extend(range(prev + 1, int(stop)))
        prev = int(stop)
    return available


def _sample_spaced_values(
    low: int,
    high: int,
    count: int,
    gap: int,
    rng: random.Random,
) -> List[int] | None:
    if count == 0:
        return []
    if high < low:
        return None

    candidates = list(range(low, high + 1))
    rng.shuffle(candidates)
    chosen: List[int] = []
    for value in candidates:
        if all(abs(value - other) >= gap for other in chosen):
            chosen.append(value)
            if len(chosen) == count:
                return chosen
    return None
