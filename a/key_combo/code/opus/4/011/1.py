from __future__ import annotations

import random

import numpy as np


BG = 0
OBSTACLE = 5
SOURCE_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

GEN_ATTEMPTS = 800
GRID_H_RANGE = (12, 16)
GRID_W_EXTRA_RANGE = (5, 9)
SOURCE_COUNT_OPTIONS = [2, 3]
OBSTACLE_COUNT_RANGE = (4, 7)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _source_info(grid: np.ndarray) -> list[dict[str, int]]:
    h, _w = grid.shape
    cols = [int(c) for c in np.flatnonzero((grid[h - 1] != BG) & (grid[h - 1] != OBSTACLE))]
    sources = [{"row": h - 1, "col": c, "color": int(grid[h - 1, c])} for c in cols]
    return sources


def _trace_path(obstacles: np.ndarray, source_col: int) -> tuple[list[tuple[int, int]], int]:
    h, w = obstacles.shape
    r, c = h - 2, int(source_col)
    path: list[tuple[int, int]] = []
    deflections = 0
    while 0 <= r < h and 0 <= c < w:
        if obstacles[r, c]:
            break
        path.append((r, c))
        nr, nc = r - 1, c + 1
        if nr < 0 or nc >= w:
            break
        if obstacles[nr, nc]:
            deflections += 1
            scan = c + 1
            while scan < w and obstacles[r, scan]:
                scan += 1
            if scan >= w:
                break
            c = scan
            continue
        r, c = nr, nc
    return path, deflections


def _paths_from_grid(grid: np.ndarray) -> tuple[list[dict[str, int]], list[list[tuple[int, int]]], list[int]]:
    obstacles = grid == OBSTACLE
    sources = _source_info(grid)
    paths: list[list[tuple[int, int]]] = []
    deflections: list[int] = []
    for src in sources:
        path, count = _trace_path(obstacles, src["col"])
        paths.append(path)
        deflections.append(count)
    return sources, paths, deflections


def _stamp_cells(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = shape
    cells: list[tuple[int, int]] = []
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if 0 <= rr < h and 0 <= cc < w:
                cells.append((rr, cc))
    return cells


def _render_with_owners(
    grid: np.ndarray,
    sources: list[dict[str, int]],
    paths: list[list[tuple[int, int]]],
) -> tuple[np.ndarray, np.ndarray, int]:
    out = grid.copy()
    h, w = grid.shape
    immutable = grid != BG
    owners = np.full((h, w), -1, dtype=int)
    overwrite_count = 0
    order = sorted(range(len(sources)), key=lambda i: (len(paths[i]), sources[i]["col"]))
    for idx in order:
        color = sources[idx]["color"]
        for r, c in paths[idx]:
            for rr, cc in _stamp_cells(grid.shape, r, c):
                if grid[rr, cc] == OBSTACLE or immutable[rr, cc]:
                    continue
                if owners[rr, cc] >= 0 and owners[rr, cc] != idx:
                    overwrite_count += 1
                owners[rr, cc] = idx
                out[rr, cc] = color
    return out, owners, overwrite_count


def _render_output(
    grid: np.ndarray,
    sources: list[dict[str, int]],
    paths: list[list[tuple[int, int]]],
) -> np.ndarray:
    out, _owners, _overwrites = _render_with_owners(grid, sources, paths)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp 3x3 color blocks along bottom-launched diagonal rays that sidestep grey obstacles to the right.

    Input structure:
    - The grid is mostly black background with 2 or 3 distinct source pixels on the bottom edge.
    - Grey single-pixel obstacles sit in the interior and never occupy the leftmost column, rightmost column, or bottom row.
    - Each source emits a ray starting one cell above it; the ray moves on the grid itself and records every visited cell.

    Rule:
    - For each source, trace a ray from the cell directly above it. The normal step is one row up and one column right.
    - If that diagonal target is a grey obstacle, stay on the current row and slide right until a non-grey cell is found or the ray exits the right edge; then continue the diagonal stepping from there.
    - Compute every source path length, process sources from shorter path to longer path, breaking ties by bottom-row column from left to right.
    - For each visited path cell, stamp a centered 3x3 square in the source color onto the output, but only onto cells that were background or earlier stamp paint; never overwrite grey obstacles or the original bottom-row source pixels.

    Key insight:
    - Obstacles do two jobs at once: they bend the rays into different path lengths and also survive as holes inside the accumulated stamp fields, so overlap ownership can only be resolved by following the deflected paths first and then applying the length-based stamp priority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 5 and w >= 5, "grid is too small for the puzzle mechanics"

    sources, paths, _deflections = _paths_from_grid(g)
    assert 2 <= len(sources) <= 3, "expected 2 or 3 bottom-edge sources"
    assert len({src["color"] for src in sources}) == len(sources), "source colors must be distinct"

    cols = [src["col"] for src in sources]
    assert all(cols[i + 1] - cols[i] >= 3 for i in range(len(cols) - 1)), "sources must be spaced apart"

    return _render_output(g, sources, paths)


def _build_grid(
    h: int,
    w: int,
    source_cols: list[int],
    source_colors: list[int],
    obstacles: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for r, c in obstacles:
        grid[r, c] = OBSTACLE
    for c, color in zip(source_cols, source_colors):
        grid[h - 1, c] = color
    return grid


def _paths_from_layout(
    h: int,
    w: int,
    source_cols: list[int],
    obstacles: set[tuple[int, int]],
) -> tuple[list[list[tuple[int, int]]], list[int]]:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in obstacles:
        mask[r, c] = True
    paths: list[list[tuple[int, int]]] = []
    deflections: list[int] = []
    for col in source_cols:
        path, count = _trace_path(mask, col)
        paths.append(path)
        deflections.append(count)
    return paths, deflections


def _natural_paths(h: int, w: int, source_cols: list[int]) -> list[list[tuple[int, int]]]:
    return _paths_from_layout(h, w, source_cols, set())[0]


def _stamp_footprint(
    shape: tuple[int, int],
    path: list[tuple[int, int]],
    immutable: np.ndarray,
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r, c in path:
        for rr, cc in _stamp_cells(shape, r, c):
            if not immutable[rr, cc]:
                cells.add((rr, cc))
    return cells


def _raw_stamp_footprint(shape: tuple[int, int], path: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r, c in path:
        cells.update(_stamp_cells(shape, r, c))
    return cells


def _is_valid_obstacle_cell(
    cell: tuple[int, int],
    h: int,
    w: int,
    source_cols: list[int],
    obstacles: set[tuple[int, int]],
) -> bool:
    r, c = cell
    if not (0 <= r < h - 1 and 1 <= c < w - 1):
        return False
    if (r, c) in obstacles:
        return False
    if r == h - 1 and c in source_cols:
        return False
    for rr, cc in obstacles:
        if abs(rr - r) + abs(cc - c) == 1:
            return False
    return True


def _path_obstacle_candidates(
    h: int,
    w: int,
    source_cols: list[int],
    obstacles: set[tuple[int, int]],
    source_index: int,
) -> list[tuple[int, int]]:
    paths, _counts = _paths_from_layout(h, w, source_cols, obstacles)
    path = paths[source_index]
    if len(path) < 4:
        return []
    candidates: list[tuple[int, int]] = []
    for step_idx, (r, c) in enumerate(path[:-1]):
        if step_idx == 0 or step_idx >= len(path) - 2:
            continue
        cell = (r - 1, c + 1)
        if r <= 1:
            continue
        if _is_valid_obstacle_cell(cell, h, w, source_cols, obstacles):
            candidates.append(cell)
    candidates.sort(key=lambda rc: (-rc[0], rc[1]))
    return candidates


def _try_add_deflection_obstacles(
    h: int,
    w: int,
    source_cols: list[int],
    desired_deflections: list[int],
) -> set[tuple[int, int]] | None:
    obstacles: set[tuple[int, int]] = set()
    for source_index, need in enumerate(desired_deflections):
        for _ in range(need):
            base_paths, base_counts = _paths_from_layout(h, w, source_cols, obstacles)
            choices = _path_obstacle_candidates(h, w, source_cols, obstacles, source_index)
            if not choices:
                return None
            placed = False
            weighted = sorted(choices, key=lambda rc: (-rc[0], random.random()))
            for cell in weighted:
                new_obstacles = set(obstacles)
                new_obstacles.add(cell)
                new_paths, new_counts = _paths_from_layout(h, w, source_cols, new_obstacles)
                if new_counts[source_index] != base_counts[source_index] + 1:
                    continue
                if any(len(path) < 4 for path in new_paths):
                    continue
                if new_paths[source_index] == base_paths[source_index]:
                    continue
                obstacles = new_obstacles
                placed = True
                break
            if not placed:
                return None
    return obstacles


def _same_paths(a: list[list[tuple[int, int]]], b: list[list[tuple[int, int]]]) -> bool:
    return len(a) == len(b) and all(pa == pb for pa, pb in zip(a, b))


def _add_safe_extra_obstacles(
    h: int,
    w: int,
    source_cols: list[int],
    source_colors: list[int],
    obstacles: set[tuple[int, int]],
    target_total: int,
) -> set[tuple[int, int]] | None:
    base_grid = _build_grid(h, w, source_cols, source_colors, obstacles)
    sources, paths, _counts = _paths_from_grid(base_grid)
    union_footprint: set[tuple[int, int]] = set()
    path_cells = {cell for path in paths for cell in path}
    for path in paths:
        union_footprint |= _raw_stamp_footprint((h, w), path)

    candidates = [
        cell
        for cell in union_footprint
        if cell not in path_cells and _is_valid_obstacle_cell(cell, h, w, source_cols, obstacles)
    ]
    random.shuffle(candidates)

    current = set(obstacles)
    for cell in candidates:
        if len(current) >= target_total:
            break
        tentative = set(current)
        tentative.add(cell)
        new_paths, _new_counts = _paths_from_layout(h, w, source_cols, tentative)
        if _same_paths(new_paths, paths):
            current = tentative

    if len(current) < target_total:
        return None
    return current


def _choose_source_cols(h: int, w: int, count: int) -> list[int] | None:
    headroom = random.randint(1, 3)
    max_last = w - h + 1 - headroom
    if max_last < 0:
        return None

    gaps = [random.choice([3, 4]) for _ in range(count - 1)]
    if gaps:
        gaps[random.randrange(len(gaps))] = 3
    total_gap = sum(gaps)
    first_max = max_last - total_gap
    if first_max < 0:
        return None

    first_col = random.randint(0, first_max)
    cols = [first_col]
    for gap in gaps:
        cols.append(cols[-1] + gap)
    return cols


def _choose_deflection_plan(count: int) -> list[int]:
    if count == 2:
        return [random.choice([2, 3]), random.choice([0, 1])]

    plan = [random.choice([1, 2]), random.choice([0, 1]), random.choice([0, 1, 2])]
    if sum(plan) < 2:
        plan[random.randrange(count)] += 1
    if len(set(plan)) == 1:
        plan[0] += 1
    return plan


def _semantic_check(grid: np.ndarray) -> bool:
    h, w = grid.shape
    if not (h <= 30 and w <= 30):
        return False

    sources = _source_info(grid)
    if not (2 <= len(sources) <= 3):
        return False
    if len({src["color"] for src in sources}) != len(sources):
        return False

    source_cols = [src["col"] for src in sources]
    if any(source_cols[i + 1] - source_cols[i] < 3 for i in range(len(source_cols) - 1)):
        return False

    obstacle_cells = [tuple(map(int, cell)) for cell in np.argwhere(grid == OBSTACLE)]
    if not (OBSTACLE_COUNT_RANGE[0] <= len(obstacle_cells) <= OBSTACLE_COUNT_RANGE[1]):
        return False
    if any(r == h - 1 or c == 0 or c == w - 1 for r, c in obstacle_cells):
        return False
    for i, (r0, c0) in enumerate(obstacle_cells):
        for r1, c1 in obstacle_cells[i + 1 :]:
            if abs(r0 - r1) + abs(c0 - c1) == 1:
                return False

    natural_paths = _natural_paths(h, w, source_cols)
    natural_lengths = [len(path) for path in natural_paths]
    if len(set(natural_lengths)) != 1:
        return False

    sources, paths, deflections = _paths_from_grid(grid)
    lengths = [len(path) for path in paths]
    if any(length < 4 for length in lengths):
        return False
    if sum(deflections) < 2 or max(deflections) == 0:
        return False

    immutable = grid != BG
    footprints = [_stamp_footprint(grid.shape, path, immutable) for path in paths]
    overlap_pairs = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            overlap = footprints[i] & footprints[j]
            if overlap and lengths[i] != lengths[j]:
                overlap_pairs.append((i, j, overlap))
    if not overlap_pairs:
        return False

    output, owners, overwrite_count = _render_with_owners(grid, sources, paths)
    if overwrite_count == 0:
        return False
    if np.array_equal(output, grid):
        return False

    raw_union: set[tuple[int, int]] = set()
    for path in paths:
        raw_union |= _raw_stamp_footprint(grid.shape, path)
    if not any(cell in raw_union for cell in obstacle_cells):
        return False

    overlap_has_longer_winner = False
    for i, j, overlap in overlap_pairs:
        longer = i if lengths[i] > lengths[j] else j
        if any(owners[r, c] == longer for r, c in overlap):
            overlap_has_longer_winner = True
            break
    if not overlap_has_longer_winner:
        return False

    try:
        solved = main(grid)
    except Exception:
        return False
    return np.array_equal(solved, output)


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = h + random.randint(*GRID_W_EXTRA_RANGE)
    source_count = random.choice(SOURCE_COUNT_OPTIONS)
    source_cols = _choose_source_cols(h, w, source_count)
    if source_cols is None:
        return None

    source_colors = random.sample(SOURCE_PALETTE, source_count)
    desired_deflections = _choose_deflection_plan(source_count)

    obstacles = _try_add_deflection_obstacles(h, w, source_cols, desired_deflections)
    if obstacles is None:
        return None

    target_total = random.randint(max(len(obstacles), OBSTACLE_COUNT_RANGE[0]), OBSTACLE_COUNT_RANGE[1])
    obstacles = _add_safe_extra_obstacles(h, w, source_cols, source_colors, obstacles, target_total)
    if obstacles is None:
        return None

    grid = _build_grid(h, w, source_cols, source_colors, obstacles)
    if _semantic_check(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-stamp instance")
