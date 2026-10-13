from __future__ import annotations

import random

import numpy as np


BG = 0
LOOKUP_COLORS = (1, 2, 3, 4)
WALL = 5
PARTICLE = 6
GEN_ATTEMPTS = 500

Coord = tuple[int, int]
Band = tuple[int, int]


def _full_gray_indices(grid: np.ndarray, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == WALL)]
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == WALL)]


def _bands(size: int, cuts: list[int]) -> list[Band]:
    cuts = sorted(int(x) for x in cuts)
    starts = [0] + [cut + 1 for cut in cuts]
    ends = cuts + [size]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _cell_bands(grid: np.ndarray) -> tuple[list[Band], list[Band]]:
    row_cuts = _full_gray_indices(grid, axis=0)
    col_cuts = _full_gray_indices(grid, axis=1)
    assert len(row_cuts) == 2 and len(col_cuts) == 2, "expected a 3x3 divider grid"
    row_bands = _bands(grid.shape[0], row_cuts)
    col_bands = _bands(grid.shape[1], col_cuts)
    assert len(row_bands) == 3 and len(col_bands) == 3
    row_sizes = {r1 - r0 for r0, r1 in row_bands}
    col_sizes = {c1 - c0 for c0, c1 in col_bands}
    assert len(row_sizes) == 1 and len(col_sizes) == 1, "all cells must be equal-sized"
    return row_bands, col_bands


def _lookup_mapping(cell: np.ndarray) -> dict[int, int]:
    entries: list[int] = []
    for r in range(cell.shape[0]):
        for c in range(cell.shape[1]):
            value = int(cell[r, c])
            if value not in (BG, WALL):
                entries.append(value)
    assert 3 <= len(entries) <= 4, "lookup cell must contain 3-4 colored pixels"
    assert len(set(entries)) == len(entries), "lookup colors must be unique"
    return {idx + 1: color for idx, color in enumerate(entries)}


def _start_direction(start: Coord, shape: tuple[int, int]) -> Coord:
    r, c = start
    h, w = shape
    candidates: list[Coord] = []
    if c == 0:
        candidates.append((0, 1))
    if r == 0:
        candidates.append((1, 0))
    if c == w - 1:
        candidates.append((0, -1))
    if r == h - 1:
        candidates.append((-1, 0))
    assert candidates, "particle must be on a cell edge"
    for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        if (dr, dc) not in candidates:
            continue
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            return dr, dc
    raise AssertionError("particle must face inward")


def _trace_path(cell: np.ndarray) -> list[Coord]:
    starts = np.argwhere(cell == PARTICLE)
    assert starts.shape[0] == 1, "each maze cell must contain exactly one particle"
    start = (int(starts[0, 0]), int(starts[0, 1]))
    dr, dc = _start_direction(start, cell.shape)
    path = [start]
    r, c = start
    while True:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < cell.shape[0] and 0 <= nc < cell.shape[1]):
            break
        if int(cell[nr, nc]) == WALL:
            break
        path.append((int(nr), int(nc)))
        r, c = int(nr), int(nc)
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each straight particle corridor by indexing the lookup cell with its travel length.

    Input structure:
    - A single-pixel gray divider splits the grid into a 3x3 array of equal cells.
    - The top-left cell is a lookup cell containing 3-4 isolated colored pixels on black.
    - Every other cell contains black corridor space, 1-3 internal gray wall pixels, and one magenta particle on the cell edge.

    Rule:
    - Read the lookup cell in row-major order and number its non-black pixels as 1, 2, 3, ...
    - In each non-lookup cell, send the magenta particle straight inward; if its next step would hit a gray wall or leave the cell, it stops immediately without turning.
    - Measure the particle path length in cells, counting both the start pixel and the final occupied pixel.
    - Use that length as a lookup index and repaint the entire traversed path with the corresponding lookup color.
    - Leave the lookup cell, gray dividers, and gray internal walls unchanged.

    Key insight:
    - The maze cell does not encode a color directly: its geometry only encodes a stopping distance, and that distance becomes meaningful only after the separate lookup cell converts length into color.
    """
    g = np.asarray(grid, dtype=int)
    row_bands, col_bands = _cell_bands(g)
    lookup = g[row_bands[0][0] : row_bands[0][1], col_bands[0][0] : col_bands[0][1]]
    mapping = _lookup_mapping(lookup)

    out = g.copy()
    for rr in range(3):
        for cc in range(3):
            if rr == 0 and cc == 0:
                continue
            r0, r1 = row_bands[rr]
            c0, c1 = col_bands[cc]
            cell = g[r0:r1, c0:c1]
            path = _trace_path(cell)
            length = len(path)
            assert length in mapping, "trajectory length must exist in lookup table"
            color = int(mapping[length])
            for r, c in path:
                out[r0 + r, c0 + c] = color
    return out


def _random_lookup_positions(size: int, count: int) -> list[Coord]:
    all_cells = [(r, c) for r in range(size) for c in range(size)]
    for _ in range(100):
        pts = random.sample(all_cells, count)
        if len({r for r, _ in pts}) >= 2 and len({c for _, c in pts}) >= 2:
            return sorted(pts)
    return sorted(random.sample(all_cells, count))


def _pick_start(size: int, kind: str) -> Coord:
    mid_choices = list(range(1, size - 1))
    if kind == "left":
        return random.choice(mid_choices), 0
    if kind == "top":
        return 0, random.choice(mid_choices)
    if kind == "right":
        return random.choice(mid_choices), size - 1
    if kind == "bottom":
        return size - 1, random.choice(mid_choices)
    assert kind == "corner"
    return random.choice([(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)])


def _wall_rows_cols_ok(size: int, walls: set[Coord]) -> bool:
    row_counts = [0] * size
    col_counts = [0] * size
    for r, c in walls:
        row_counts[r] += 1
        col_counts[c] += 1
    return max(row_counts) < size and max(col_counts) < size


def _build_maze_cell(size: int, length: int, kind: str) -> dict:
    assert 1 <= length <= size
    start = _pick_start(size, kind)
    dr, dc = _start_direction(start, (size, size))
    path = [(start[0] + dr * step, start[1] + dc * step) for step in range(length)]
    assert all(0 <= r < size and 0 <= c < size for r, c in path)

    main_wall: Coord | None = None
    if length < size:
        main_wall = (start[0] + dr * length, start[1] + dc * length)
        assert 0 <= main_wall[0] < size and 0 <= main_wall[1] < size

    blocked = set(path)
    if main_wall is not None:
        blocked.add(main_wall)
    free_cells = [(r, c) for r in range(size) for c in range(size) if (r, c) not in blocked]

    min_walls = 1
    base_walls = 1 if main_wall is not None else 0
    total_walls = random.randint(max(min_walls, base_walls), 3)
    extra_needed = total_walls - base_walls

    extras: list[Coord] | None = None
    for _ in range(120):
        trial = random.sample(free_cells, extra_needed)
        wall_set = set(trial)
        if main_wall is not None:
            wall_set.add(main_wall)
        if _wall_rows_cols_ok(size, wall_set):
            extras = trial
            break
    assert extras is not None

    cell = np.zeros((size, size), dtype=int)
    cell[start] = PARTICLE
    if main_wall is not None:
        cell[main_wall] = WALL
    for r, c in extras:
        cell[r, c] = WALL

    traced = _trace_path(cell)
    assert traced == [(int(r), int(c)) for r, c in path]
    return {
        "grid": cell,
        "path": path,
        "start": start,
        "kind": kind,
        "length": length,
        "wall_count": int(np.count_nonzero(cell == WALL)),
    }


def _maze_specs(size: int) -> list[dict[str, int | str]]:
    specs: list[dict[str, int | str]] = [
        {"kind": "left", "length": 1},
        {"kind": "top", "length": min(2, size)},
        {"kind": "right", "length": size},
        {"kind": "bottom", "length": random.randint(1, size)},
        {"kind": "corner", "length": random.randint(1, size)},
    ]

    present = {int(spec["length"]) for spec in specs}
    for length in range(1, size + 1):
        if length not in present:
            specs.append({"kind": random.choice(["left", "top", "right", "bottom", "corner"]), "length": length})
    while len(specs) < 8:
        specs.append(
            {
                "kind": random.choice(["left", "top", "right", "bottom", "corner"]),
                "length": random.randint(1, size),
            }
        )
    random.shuffle(specs)
    return specs[:8]


def _validate_semantics(
    grid: np.ndarray,
    expected: np.ndarray,
    planned_lookup: dict[int, int],
    planned_lengths: dict[tuple[int, int], int],
) -> bool:
    row_bands, col_bands = _cell_bands(grid)
    assert len(_full_gray_indices(grid, axis=0)) == 2
    assert len(_full_gray_indices(grid, axis=1)) == 2

    lookup = grid[row_bands[0][0] : row_bands[0][1], col_bands[0][0] : col_bands[0][1]]
    if _lookup_mapping(lookup) != planned_lookup:
        return False

    recovered_lengths: dict[tuple[int, int], int] = {}
    boundary_stop_seen = False
    wall_stop_seen = False
    corner_seen = False
    for rr in range(3):
        for cc in range(3):
            r0, r1 = row_bands[rr]
            c0, c1 = col_bands[cc]
            cell = grid[r0:r1, c0:c1]
            if rr == 0 and cc == 0:
                if np.count_nonzero(cell == WALL) != 0 or np.count_nonzero(cell == PARTICLE) != 0:
                    return False
                continue
            if np.count_nonzero(cell == PARTICLE) != 1:
                return False
            wall_count = int(np.count_nonzero(cell == WALL))
            if not (1 <= wall_count <= 3):
                return False
            path = _trace_path(cell)
            length = len(path)
            recovered_lengths[(rr, cc)] = length
            if length != planned_lengths[(rr, cc)]:
                return False
            start = tuple(int(x) for x in np.argwhere(cell == PARTICLE)[0])
            if start in {(0, 0), (0, cell.shape[1] - 1), (cell.shape[0] - 1, 0), (cell.shape[0] - 1, cell.shape[1] - 1)}:
                corner_seen = True
            dr, dc = _start_direction(start, cell.shape)
            end = path[-1]
            nr, nc = end[0] + dr, end[1] + dc
            if 0 <= nr < cell.shape[0] and 0 <= nc < cell.shape[1]:
                if int(cell[nr, nc]) != WALL:
                    return False
                wall_stop_seen = True
            else:
                boundary_stop_seen = True

    if recovered_lengths != planned_lengths:
        return False
    if not boundary_stop_seen or not wall_stop_seen or not corner_seen:
        return False
    return np.array_equal(main(grid), expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        size = random.choice([3, 4])
        grid = np.zeros((3 * size + 2, 3 * size + 2), dtype=int)
        separators = [size, 2 * size + 1]
        for idx in separators:
            grid[idx, :] = WALL
            grid[:, idx] = WALL

        row_bands, col_bands = _cell_bands(grid)

        lookup_colors = random.sample(list(LOOKUP_COLORS), size)
        lookup_positions = _random_lookup_positions(size, size)
        lookup = np.zeros((size, size), dtype=int)
        planned_lookup: dict[int, int] = {}
        for idx, (pos, color) in enumerate(zip(lookup_positions, lookup_colors), start=1):
            lookup[pos] = int(color)
            planned_lookup[idx] = int(color)
        lr0, lr1 = row_bands[0]
        lc0, lc1 = col_bands[0]
        grid[lr0:lr1, lc0:lc1] = lookup

        specs = _maze_specs(size)
        maze_positions = [(r, c) for r in range(3) for c in range(3) if (r, c) != (0, 0)]
        planned_lengths: dict[tuple[int, int], int] = {}
        expected = grid.copy()
        wall_histogram = {}

        for (rr, cc), spec in zip(maze_positions, specs):
            built = _build_maze_cell(size, int(spec["length"]), str(spec["kind"]))
            planned_lengths[(rr, cc)] = int(built["length"])
            wall_histogram[(rr, cc)] = int(built["wall_count"])
            r0, r1 = row_bands[rr]
            c0, c1 = col_bands[cc]
            grid[r0:r1, c0:c1] = built["grid"]
            expected[r0:r1, c0:c1] = built["grid"]
            color = planned_lookup[int(built["length"])]
            for r, c in built["path"]:
                expected[r0 + r, c0 + c] = color

        if len(set(planned_lengths.values())) < size:
            continue
        if len(set(wall_histogram.values())) == 1:
            continue
        if not _validate_semantics(grid, expected, planned_lookup, planned_lengths):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
