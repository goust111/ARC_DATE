from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PATH_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (12, 20)
GRID_W_RANGE = (12, 20)
PATH_COUNT_CHOICES = (1, 2, 2, 3)
SEGMENT_COUNT_RANGE = (2, 5)
SEGMENT_LENGTH_CHOICES = (2, 3, 4)
GEN_RETRIES = 400


Coord = tuple[int, int]


def _neighbors4(cell: Coord) -> list[Coord]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _trace_path_from_cells(cells: set[Coord]) -> list[Coord]:
    if not cells:
        raise ValueError("expected at least one path cell")

    adjacency: dict[Coord, list[Coord]] = {cell: [] for cell in cells}
    for cell in cells:
        for neighbor in _neighbors4(cell):
            if neighbor in cells:
                adjacency[cell].append(neighbor)

    if any(len(neighbors) not in (1, 2) for neighbors in adjacency.values()):
        raise ValueError("expected a simple open path")

    endpoints = sorted(cell for cell, neighbors in adjacency.items() if len(neighbors) == 1)
    if len(endpoints) != 2:
        raise ValueError("expected exactly two endpoints")

    path = [endpoints[0]]
    prev: Coord | None = None
    cur = endpoints[0]
    while True:
        nxt = [neighbor for neighbor in adjacency[cur] if neighbor != prev]
        if not nxt:
            break
        if len(nxt) != 1:
            raise ValueError("branching path")
        prev, cur = cur, nxt[0]
        path.append(cur)

    if len(path) != len(cells):
        raise ValueError("disconnected path")
    return path


def _marker_positions(path: list[Coord]) -> list[Coord]:
    markers: list[Coord] = []
    for i in range(1, len(path) - 1):
        prev = path[i - 1]
        turn = path[i]
        nxt = path[i + 1]
        prev_dir = (turn[0] - prev[0], turn[1] - prev[1])
        next_dir = (nxt[0] - turn[0], nxt[1] - turn[1])
        if prev_dir == next_dir:
            continue
        if prev_dir[0] and next_dir[0]:
            continue
        if prev_dir[1] and next_dir[1]:
            continue

        # The examples fix the diagonal using the walking direction:
        # right->down => (+0,+1) and (+1,+0) maps to (-1,+1),
        # down->left => (+1,+0) and (+0,-1) maps to (+1,-1).
        mr = turn[0] + prev_dir[0] - next_dir[0]
        mc = turn[1] + prev_dir[1] + next_dir[1]
        markers.append((mr, mc))
    return markers


def _segment_lengths(path: list[Coord]) -> list[int]:
    if len(path) < 2:
        return []

    lengths: list[int] = []
    cur_dir = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    cur_len = 2
    for i in range(2, len(path)):
        step = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        if step == cur_dir:
            cur_len += 1
        else:
            lengths.append(cur_len)
            cur_dir = step
            cur_len = 2
    lengths.append(cur_len)
    return lengths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct each colored path and place grey markers at its outer turns.

    The turn markers depend on walking direction, so the open path is traversed
    from its lexicographically smallest endpoint as a canonical tie-break.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    colors = sorted(int(color) for color in np.unique(g) if int(color) not in (BLACK, GREY))
    for color in colors:
        cells = {tuple(coord) for coord in np.argwhere(g == color)}
        path = _trace_path_from_cells(cells)
        for mr, mc in _marker_positions(path):
            if 0 <= mr < g.shape[0] and 0 <= mc < g.shape[1] and out[mr, mc] in (BLACK, GREY):
                out[mr, mc] = GREY

    return out


def _perpendicular_dirs(direction: Coord) -> list[Coord]:
    if direction[0]:
        return [(0, -1), (0, 1)]
    return [(-1, 0), (1, 0)]


def _try_extend_segment(
    path: list[Coord],
    direction: Coord,
    length: int,
    height: int,
    width: int,
    blocked: set[Coord],
) -> list[Coord] | None:
    own = set(path)
    new_cells: list[Coord] = []
    r0, c0 = path[-1]
    dr, dc = direction

    for step in range(1, length):
        cell = (r0 + dr * step, c0 + dc * step)
        if not (0 <= cell[0] < height and 0 <= cell[1] < width):
            return None
        if cell in own or cell in blocked or cell in new_cells:
            return None

        prev = path[-1] if step == 1 else new_cells[-1]
        neighbors = {
            neighbor
            for neighbor in _neighbors4(cell)
            if neighbor in own or neighbor in blocked or neighbor in new_cells
        }
        if neighbors != {prev}:
            return None
        new_cells.append(cell)

    return path + new_cells


def _grow_path(
    path: list[Coord],
    prev_dir: Coord | None,
    seg_index: int,
    seg_total: int,
    height: int,
    width: int,
    blocked: set[Coord],
) -> list[Coord] | None:
    if seg_index == seg_total:
        return path

    directions = DIRS4[:] if prev_dir is None else _perpendicular_dirs(prev_dir)
    random.shuffle(directions)
    lengths = list(SEGMENT_LENGTH_CHOICES)
    random.shuffle(lengths)

    for direction in directions:
        for length in lengths:
            extended = _try_extend_segment(path, direction, length, height, width, blocked)
            if extended is None:
                continue
            result = _grow_path(
                extended,
                direction,
                seg_index + 1,
                seg_total,
                height,
                width,
                blocked,
            )
            if result is not None:
                return result
    return None


def _build_path(height: int, width: int, blocked: set[Coord], seg_total: int) -> tuple[list[Coord], set[Coord]] | None:
    starts = [(r, c) for r in range(height) for c in range(width) if (r, c) not in blocked]
    random.shuffle(starts)

    for start in starts:
        if any(neighbor in blocked for neighbor in _neighbors4(start)):
            continue

        candidate = _grow_path([start], None, 0, seg_total, height, width, blocked)
        if candidate is None:
            continue

        cells = set(candidate)
        path = _trace_path_from_cells(cells)
        lengths = _segment_lengths(path)
        if not lengths or any(length not in SEGMENT_LENGTH_CHOICES for length in lengths):
            continue

        marker_positions = _marker_positions(path)
        marker_set = set(marker_positions)
        if len(marker_positions) != len(marker_set):
            continue

        valid = True
        for marker in marker_positions:
            mr, mc = marker
            if not (0 <= mr < height and 0 <= mc < width):
                valid = False
                break
            if marker in blocked or marker in cells:
                valid = False
                break
            if any(neighbor in blocked or neighbor in cells for neighbor in _neighbors4(marker)):
                valid = False
                break

        if valid:
            return path, marker_set

    return None


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    path_count = random.choice(PATH_COUNT_CHOICES)
    colors = random.sample(PATH_COLORS, path_count)

    grid = np.full((height, width), BLACK, dtype=int)
    expected = grid.copy()
    blocked: set[Coord] = set()

    for color in colors:
        seg_total = random.randint(*SEGMENT_COUNT_RANGE)
        built = _build_path(height, width, blocked, seg_total)
        if built is None:
            return None
        path, markers = built

        for r, c in path:
            grid[r, c] = color
            expected[r, c] = color
        for r, c in markers:
            expected[r, c] = GREY

        blocked.update(path)
        blocked.update(markers)

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for the path-and-corner-marker puzzle."""
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
