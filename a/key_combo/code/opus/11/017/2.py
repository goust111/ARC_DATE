from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG_COLOR = 0
LINE_COLOR = 5
PATH_COLOR = 8
MAX_GENERATION_ATTEMPTS = 400


def _most_common(values: np.ndarray) -> int:
    return int(Counter(map(int, values.reshape(-1))).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _separator_pair(size: int, full_indices: list[int]) -> tuple[int, int]:
    candidates = []
    for i, a in enumerate(full_indices):
        for b in full_indices[i + 1 :]:
            if a == b:
                continue
            left = a
            middle = b - a - 1
            right = size - b - 1
            if left > 0 and left == middle == right:
                candidates.append((a, b))
    assert len(candidates) == 1, candidates
    return candidates[0]


def _detect_structure(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = _most_common(g)

    full_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] != bg)]
    full_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] != bg)]
    row_seps = _separator_pair(g.shape[0], full_rows)
    col_seps = _separator_pair(g.shape[1], full_cols)

    line_samples = np.concatenate(
        [
            g[row_seps[0], :],
            g[row_seps[1], :],
            g[:, col_seps[0]],
            g[:, col_seps[1]],
        ]
    )
    line_color = _most_common(line_samples[line_samples != bg])
    path_colors = sorted(set(map(int, np.unique(g))) - {bg, line_color})
    assert len(path_colors) == 1, path_colors
    return bg, int(line_color), int(path_colors[0]), row_seps, col_seps


def _trace_path(mask: np.ndarray) -> list[tuple[int, int]]:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    assert coords
    pset = set(coords)
    degrees = {}
    for r, c in coords:
        degrees[(r, c)] = sum((nr, nc) in pset for nr, nc in _neighbors4(r, c))

    endpoints = sorted([pt for pt in coords if degrees[pt] == 1])
    assert len(endpoints) == 2, endpoints
    start = endpoints[0]

    order = []
    prev = None
    cur = start
    while True:
        order.append(cur)
        nxts = [pt for pt in _neighbors4(*cur) if pt in pset and pt != prev]
        if not nxts:
            break
        assert len(nxts) == 1, nxts
        prev, cur = cur, nxts[0]

    assert len(order) == len(coords)
    return order


def _raw_cell_id(r: int, c: int, row_seps: tuple[int, int], col_seps: tuple[int, int]):
    if r in row_seps or c in col_seps:
        return None
    rb = 0 if r < row_seps[0] else 1 if r < row_seps[1] else 2
    cb = 0 if c < col_seps[0] else 1 if c < col_seps[1] else 2
    return rb, cb


def _path_cell_ids(
    order: list[tuple[int, int]],
    row_seps: tuple[int, int],
    col_seps: tuple[int, int],
) -> list[tuple[int, int]]:
    cell_ids = [_raw_cell_id(r, c, row_seps, col_seps) for r, c in order]
    for i, cell_id in enumerate(cell_ids):
        if cell_id is not None:
            continue
        assigned = None
        for j in range(i + 1, len(cell_ids)):
            if cell_ids[j] is not None:
                assigned = cell_ids[j]
                break
        if assigned is None:
            for j in range(i - 1, -1, -1):
                if cell_ids[j] is not None:
                    assigned = cell_ids[j]
                    break
        assert assigned is not None
        cell_ids[i] = assigned
    return cell_ids


def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten a grid-crossing snake path into a single column colored by cell-entry order.

    Input structure:
    - A black background contains two full horizontal and two full vertical separator lines that split the canvas into a 3x3 grid of equal square cells.
    - The separator lines are grey except where a cyan 4-connected simple path overwrites them while crossing between cells.
    - The path has exactly two endpoints and spans at least three different cells.

    Rule:
    - Recover the equal 3x3 cell partition from the full separator rows and columns.
    - Trace the cyan path from the top-most endpoint, breaking ties by the left-most endpoint.
    - Assign each path pixel to a cell; pixels on separator lines belong to the next non-separator cell reached later in the traversal.
    - Give the first visited cell color 1, the next new cell color 2, and so on; output the recolored path as an N×1 column in traversal order.

    Key insight:
    - The output color of each path pixel depends on two structures at once: the snake traversal order and the equal-cell partition that the snake is entering when it touches separator lines.
    """
    g = np.asarray(grid, dtype=int)
    _bg, _line_color, path_color, row_seps, col_seps = _detect_structure(g)
    order = _trace_path(g == path_color)
    cell_ids = _path_cell_ids(order, row_seps, col_seps)

    color_by_cell: dict[tuple[int, int], int] = {}
    next_color = 1
    out = []
    for cell_id in cell_ids:
        if cell_id not in color_by_cell:
            color_by_cell[cell_id] = next_color
            next_color += 1
        out.append(color_by_cell[cell_id])

    return np.array(out, dtype=int).reshape(-1, 1)


def _cell_origin(cell: tuple[int, int], cell_size: int) -> tuple[int, int]:
    return cell[0] * (cell_size + 1), cell[1] * (cell_size + 1)


def _cell_neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    nbrs = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            nbrs.append((nr, nc))
    return nbrs


def _sample_cell_walk(length: int) -> list[tuple[int, int]] | None:
    for _ in range(80):
        start = (random.randrange(3), random.randrange(3))
        walk = [start]
        while len(walk) < length:
            nbrs = [cell for cell in _cell_neighbors(walk[-1]) if cell not in walk]
            if not nbrs:
                break
            random.shuffle(nbrs)
            walk.append(nbrs[0])
        if len(walk) == length:
            return walk
    return None


def _direction(a: tuple[int, int], b: tuple[int, int]) -> str:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    mapping = {
        (-1, 0): "U",
        (1, 0): "D",
        (0, -1): "L",
        (0, 1): "R",
    }
    return mapping[(dr, dc)]


def _opposite(direction: str) -> str:
    return {"U": "D", "D": "U", "L": "R", "R": "L"}[direction]


def _pick_offset(cell_size: int) -> int:
    inner = list(range(1, cell_size - 1))
    if inner and random.random() < 0.75:
        return random.choice(inner)
    return random.randrange(cell_size)


def _boundary_point(direction: str, offset: int, cell_size: int) -> tuple[int, int]:
    if direction == "U":
        return 0, offset
    if direction == "D":
        return cell_size - 1, offset
    if direction == "L":
        return offset, 0
    return offset, cell_size - 1


def _orth_line(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a == b:
        return [a]
    if a[0] == b[0]:
        step = 1 if b[1] > a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] > a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    raise ValueError("line must be orthogonal")


def _polyline(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    path = [points[0]]
    for pt in points[1:]:
        segment = _orth_line(path[-1], pt)
        path.extend(segment[1:])
    return path


def _valid_simple_path(points: list[tuple[int, int]], cell_size: int) -> bool:
    if not points:
        return False
    if len(set(points)) != len(points):
        return False
    for r, c in points:
        if not (0 <= r < cell_size and 0 <= c < cell_size):
            return False
    for a, b in zip(points, points[1:]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            return False
    return True


def _local_paths(start: tuple[int, int], end: tuple[int, int], cell_size: int) -> list[list[tuple[int, int]]]:
    if start == end:
        return [[start]]

    raw_candidates = []
    raw_candidates.append([start, end])
    raw_candidates.append([start, (start[0], end[1]), end])
    raw_candidates.append([start, (end[0], start[1]), end])

    if start[1] != end[1]:
        for rr in range(cell_size):
            if rr in {start[0], end[0]}:
                continue
            raw_candidates.append([start, (rr, start[1]), (rr, end[1]), end])
    if start[0] != end[0]:
        for cc in range(cell_size):
            if cc in {start[1], end[1]}:
                continue
            raw_candidates.append([start, (start[0], cc), (end[0], cc), end])

    candidates = []
    seen = set()
    for control_points in raw_candidates:
        try:
            path = _polyline(control_points)
        except ValueError:
            continue
        key = tuple(path)
        if key in seen or not _valid_simple_path(path, cell_size):
            continue
        seen.add(key)
        candidates.append(path)

    candidates.sort(key=len)
    return candidates


def _choose_local_path(
    start: tuple[int, int],
    end: tuple[int, int],
    cell_size: int,
    *,
    prefer_short: bool = False,
):
    candidates = _local_paths(start, end, cell_size)
    if not candidates:
        return None
    if prefer_short:
        return random.choice(candidates[: max(1, (len(candidates) + 1) // 2)])
    if random.random() < 0.5:
        return random.choice(candidates[-max(1, len(candidates) // 2) :])
    return random.choice(candidates)


def _pick_start_or_end(cell_size: int, other: tuple[int, int]) -> tuple[int, int]:
    pts = [(r, c) for r in range(cell_size) for c in range(cell_size) if (r, c) != other]
    pts.sort(key=lambda pt: abs(pt[0] - other[0]) + abs(pt[1] - other[1]), reverse=True)
    return random.choice(pts[: max(1, min(len(pts), 6))])


def _bridge_segment(
    cell: tuple[int, int],
    direction: str,
    offset_a: int,
    offset_b: int,
    cell_size: int,
) -> list[tuple[int, int]]:
    r0, c0 = _cell_origin(cell, cell_size)
    if direction == "R":
        return _orth_line((r0 + offset_a, c0 + cell_size), (r0 + offset_b, c0 + cell_size))
    if direction == "L":
        return _orth_line((r0 + offset_a, c0 - 1), (r0 + offset_b, c0 - 1))
    if direction == "D":
        return _orth_line((r0 + cell_size, c0 + offset_a), (r0 + cell_size, c0 + offset_b))
    return _orth_line((r0 - 1, c0 + offset_a), (r0 - 1, c0 + offset_b))


def _globalize(local_path: list[tuple[int, int]], cell: tuple[int, int], cell_size: int) -> list[tuple[int, int]]:
    r0, c0 = _cell_origin(cell, cell_size)
    return [(r0 + r, c0 + c) for r, c in local_path]


def _is_induced_chain(points: list[tuple[int, int]]) -> bool:
    if len(set(points)) != len(points):
        return False
    pset = set(points)
    degs = []
    for r, c in points:
        deg = sum((nr, nc) in pset for nr, nc in _neighbors4(r, c))
        degs.append(deg)
    return degs.count(1) == 2 and all(deg in {1, 2} for deg in degs)


def _grid_from_path(points: list[tuple[int, int]], cell_size: int) -> np.ndarray:
    side = 3 * cell_size + 2
    row_seps = [cell_size, 2 * cell_size + 1]
    col_seps = [cell_size, 2 * cell_size + 1]
    grid = np.full((side, side), BG_COLOR, dtype=int)
    grid[row_seps[0], :] = LINE_COLOR
    grid[row_seps[1], :] = LINE_COLOR
    grid[:, col_seps[0]] = LINE_COLOR
    grid[:, col_seps[1]] = LINE_COLOR
    for r, c in points:
        grid[r, c] = PATH_COLOR
    return grid


def _gen_attempt() -> np.ndarray | None:
    cell_size = 4 if random.random() < 0.75 else 5
    walk_len = random.choice([3, 4] if cell_size == 5 else [3, 4, 4])
    cells = _sample_cell_walk(walk_len)
    if cells is None:
        return None

    entries: list[tuple[int, int] | None] = [None] * len(cells)
    exits: list[tuple[int, int] | None] = [None] * len(cells)
    bridges: list[list[tuple[int, int]]] = []

    for i in range(len(cells) - 1):
        direction = _direction(cells[i], cells[i + 1])
        offset_a = _pick_offset(cell_size)
        offset_b = _pick_offset(cell_size)
        exits[i] = _boundary_point(direction, offset_a, cell_size)
        entries[i + 1] = _boundary_point(_opposite(direction), offset_b, cell_size)
        bridges.append(_bridge_segment(cells[i], direction, offset_a, offset_b, cell_size))

    assert exits[0] is not None and entries[-1] is not None
    start_point = _pick_start_or_end(cell_size, exits[0])
    end_point = _pick_start_or_end(cell_size, entries[-1])
    prefer_short = cell_size == 5 or len(cells) >= 4

    local_segments = []
    for i, cell in enumerate(cells):
        if i == 0:
            local = _choose_local_path(start_point, exits[i], cell_size, prefer_short=prefer_short)
        elif i == len(cells) - 1:
            local = _choose_local_path(entries[i], end_point, cell_size, prefer_short=prefer_short)
        else:
            if entries[i] == exits[i]:
                return None
            local = _choose_local_path(entries[i], exits[i], cell_size, prefer_short=prefer_short)
        if local is None:
            return None
        local_segments.append(_globalize(local, cell, cell_size))

    points = list(local_segments[0])
    for i in range(len(bridges)):
        points.extend(bridges[i])
        points.extend(local_segments[i + 1])

    if not (12 <= len(points) <= 20):
        return None
    if not _is_induced_chain(points):
        return None

    grid = _grid_from_path(points, cell_size)
    row_seps = [cell_size, 2 * cell_size + 1]
    col_seps = [cell_size, 2 * cell_size + 1]
    full_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] != BG_COLOR)]
    full_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] != BG_COLOR)]
    if full_rows != row_seps or full_cols != col_seps:
        return None

    path_order = _trace_path(grid == PATH_COLOR)
    cell_ids = _path_cell_ids(path_order, (row_seps[0], row_seps[1]), (col_seps[0], col_seps[1]))
    if len(dict.fromkeys(cell_ids)) < 3:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != (len(points), 1):
        return None
    if len(set(map(int, out[:, 0]))) < 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
