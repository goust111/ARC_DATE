from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# Generator bounds.
GRID_MIN_SIZE = 10
GRID_MAX_SIZE = 20
PATH_LEN_MIN = 10
PATH_LEN_MAX = 24
SEGMENT_COUNT_MIN = 3
SEGMENT_COUNT_MAX = 7
SEGMENT_STEPS_MIN = 1
SEGMENT_STEPS_MAX = 6
CHECKPOINT_COUNT_MIN = 2
CHECKPOINT_COUNT_MAX = 5
CHECKPOINT_INDEX_GAP = 2
CHECKPOINT_SELECTION_ATTEMPTS = 200
MAX_GENERATION_ATTEMPTS = 400


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _ordered_path(cells: set[Coord]) -> list[Coord]:
    degrees = {
        cell: sum((neighbor in cells) for neighbor in _neighbors4(*cell))
        for cell in cells
    }
    endpoints = sorted(cell for cell, degree in degrees.items() if degree == 1)
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    assert all(degree in (1, 2) for degree in degrees.values()), "path must not branch"

    ordered: list[Coord] = []
    prev = None
    cur = endpoints[0]
    while True:
        ordered.append(cur)
        next_cells = [
            neighbor
            for neighbor in _neighbors4(*cur)
            if neighbor in cells and neighbor != prev
        ]
        if not next_cells:
            break
        assert len(next_cells) == 1, "path traversal must be unique"
        prev, cur = cur, next_cells[0]

    assert len(ordered) == len(cells), "path traversal must visit every path cell"
    return ordered


def _find_path(grid: np.ndarray, bg: int) -> tuple[int, list[Coord]]:
    candidates: list[tuple[int, list[Coord]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        if coords.shape[0] < 2:
            continue
        cells = {tuple(map(int, coord)) for coord in coords}
        try:
            path = _ordered_path(cells)
        except AssertionError:
            continue
        candidates.append((color, path))

    assert len(candidates) == 1, "expected exactly one single-color path"
    return candidates[0]


def _checkpoint_hits(
    grid: np.ndarray,
    bg: int,
    path_color: int,
    path: list[Coord],
) -> list[tuple[int, int]]:
    path_index = {cell: idx for idx, cell in enumerate(path)}
    hits: list[tuple[int, int]] = []
    seen_colors: set[int] = set()
    seen_indices: set[int] = set()

    for r, c in np.argwhere((grid != bg) & (grid != path_color)):
        r = int(r)
        c = int(c)
        color = int(grid[r, c])
        assert color not in seen_colors, "checkpoint colors must be distinct"
        seen_colors.add(color)

        neighbor_indices = [
            path_index[neighbor]
            for neighbor in _neighbors4(r, c)
            if neighbor in path_index
        ]
        assert neighbor_indices, "checkpoint must touch the path"
        hit_index = min(neighbor_indices)
        assert hit_index not in seen_indices, "checkpoint encounter order must be unambiguous"
        seen_indices.add(hit_index)
        hits.append((hit_index, color))

    assert hits, "expected at least one checkpoint"
    hits.sort()
    return hits


def _paint_segments(
    grid: np.ndarray,
    path: list[Coord],
    hits: list[tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()
    segment = 0
    current_color = hits[0][1]
    for idx, (r, c) in enumerate(path):
        while segment + 1 < len(hits) and idx > hits[segment][0]:
            segment += 1
            current_color = hits[segment][1]
        out[r, c] = current_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a single thin path by projecting checkpoint colors onto consecutive path segments in encounter order.

    Input structure:
    - The background is the dominant color.
    - One foreground color forms a single 4-connected one-pixel-wide path with exactly two endpoints and no branches.
    - Several other foreground pixels are isolated checkpoints with distinct colors; each checkpoint sits off the path but 4-adjacent to it.

    Rule:
    - Choose the path endpoint with the smaller row, breaking ties by the smaller column, as the start.
    - Walk the path from that start endpoint to the other endpoint and record checkpoints in the order they are first encountered by 4-adjacency.
    - Recolor the path so the start-to-first-checkpoint segment uses the first checkpoint color, each interval up to the next checkpoint uses that next checkpoint color, and the tail after the last checkpoint keeps the last checkpoint color.
    - Leave checkpoint pixels and background pixels unchanged.

    Key insight:
    - The off-path colored pixels are not part of the path; they are ordered side markers whose encounter sequence determines a piecewise recoloring of the unchanged path geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _mode_color(g)
    path_color, path = _find_path(g, bg)
    hits = _checkpoint_hits(g, bg, path_color, path)
    return _paint_segments(g, path, hits)


def _sample_run_lengths(n_segments: int, total_steps: int) -> list[int]:
    runs: list[int] = []
    remaining = total_steps
    for i in range(n_segments):
        slots = n_segments - i - 1
        lo = max(SEGMENT_STEPS_MIN, remaining - SEGMENT_STEPS_MAX * slots)
        hi = min(SEGMENT_STEPS_MAX, remaining - SEGMENT_STEPS_MIN * slots)
        assert lo <= hi
        runs.append(random.randint(lo, hi))
        remaining -= runs[-1]
    random.shuffle(runs)
    return runs


def _sample_run_plan(h: int, w: int) -> list[int]:
    interior_area = (h - 2) * (w - 2)
    target_len_max = min(PATH_LEN_MAX, interior_area // 2)
    assert PATH_LEN_MIN <= target_len_max
    target_len = random.randint(PATH_LEN_MIN, target_len_max)
    total_steps = target_len - 1
    valid_counts = [
        count
        for count in range(SEGMENT_COUNT_MIN, SEGMENT_COUNT_MAX + 1)
        if SEGMENT_STEPS_MIN * count <= total_steps <= SEGMENT_STEPS_MAX * count
    ]
    assert valid_counts
    return _sample_run_lengths(random.choice(valid_counts), total_steps)


def _segment_directions(direction: Coord | None) -> list[Coord]:
    if direction is None:
        dirs = list(DIRS)
    else:
        dr, dc = direction
        dirs = [direction, (-dc, dr), (dc, -dr)]
    random.shuffle(dirs)
    return dirs


def _extend_segment(
    path: list[Coord],
    used: set[Coord],
    direction: Coord,
    steps: int,
    h: int,
    w: int,
) -> tuple[list[Coord], set[Coord]] | None:
    dr, dc = direction
    new_path = list(path)
    new_used = set(used)
    r, c = new_path[-1]

    for _ in range(steps):
        nr, nc = r + dr, c + dc
        if not (1 <= nr < h - 1 and 1 <= nc < w - 1):
            return None
        if (nr, nc) in new_used:
            return None
        if any((ar, ac) in new_used and (ar, ac) != (r, c) for ar, ac in _neighbors4(nr, nc)):
            return None

        new_path.append((nr, nc))
        new_used.add((nr, nc))
        r, c = nr, nc

    return new_path, new_used


def _build_path(h: int, w: int, run_lengths: list[int]) -> list[Coord] | None:
    starts = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(starts)

    def search(
        segment_idx: int,
        prev_dir: Coord | None,
        path: list[Coord],
        used: set[Coord],
    ) -> list[Coord] | None:
        if segment_idx == len(run_lengths):
            return path
        for direction in _segment_directions(prev_dir):
            extended = _extend_segment(path, used, direction, run_lengths[segment_idx], h, w)
            if extended is None:
                continue
            found = search(segment_idx + 1, direction, extended[0], extended[1])
            if found is not None:
                return found
        return None

    for start in starts:
        found = search(0, None, [start], {start})
        if found is not None:
            return found
    return None


def _checkpoint_candidates(path: list[Coord], shape: tuple[int, int]) -> dict[int, list[Coord]]:
    h, w = shape
    path_index = {cell: idx for idx, cell in enumerate(path)}
    candidates: dict[int, list[Coord]] = {}

    for idx, (r, c) in enumerate(path):
        if idx in (0, len(path) - 1):
            continue
        for nr, nc in _neighbors4(r, c):
            if not (0 < nr < h - 1 and 0 < nc < w - 1):
                continue
            if (nr, nc) in path_index:
                continue
            adjacent_indices = [
                path_index[neighbor]
                for neighbor in _neighbors4(nr, nc)
                if neighbor in path_index
            ]
            if len(adjacent_indices) == 1 and adjacent_indices[0] == idx:
                candidates.setdefault(idx, []).append((nr, nc))

    return candidates


def _checkpoint_positions_ok(chosen: list[Coord], candidate: Coord) -> bool:
    return all(abs(candidate[0] - r) + abs(candidate[1] - c) > 1 for r, c in chosen)


def _choose_checkpoints(path: list[Coord], shape: tuple[int, int]) -> list[tuple[int, Coord]] | None:
    candidates = _checkpoint_candidates(path, shape)
    eligible_indices = sorted(candidates)
    feasible_counts = [
        count
        for count in range(CHECKPOINT_COUNT_MIN, min(CHECKPOINT_COUNT_MAX, len(eligible_indices)) + 1)
    ]
    if not feasible_counts:
        return None

    for _ in range(CHECKPOINT_SELECTION_ATTEMPTS):
        count = random.choice(feasible_counts)
        chosen_indices = sorted(random.sample(eligible_indices, count))
        if any(b - a < CHECKPOINT_INDEX_GAP for a, b in zip(chosen_indices, chosen_indices[1:])):
            continue

        chosen: list[tuple[int, Coord]] = []
        occupied: list[Coord] = []
        ok = True
        for idx in chosen_indices:
            options = list(candidates[idx])
            random.shuffle(options)
            placed = None
            for cell in options:
                if _checkpoint_positions_ok(occupied, cell):
                    placed = cell
                    break
            if placed is None:
                ok = False
                break
            chosen.append((idx, placed))
            occupied.append(placed)

        if ok:
            return chosen

    return None


def _render_input(
    h: int,
    w: int,
    bg: int,
    path_color: int,
    path: list[Coord],
    checkpoints: list[tuple[int, Coord, int]],
) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    for r, c in path:
        grid[r, c] = path_color
    for _idx, (r, c), color in checkpoints:
        grid[r, c] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    w = random.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    run_lengths = _sample_run_plan(h, w)
    raw_path = _build_path(h, w, run_lengths)
    if raw_path is None:
        return None

    path = _ordered_path(set(raw_path))
    chosen = _choose_checkpoints(path, (h, w))
    if chosen is None:
        return None

    bg = random.randrange(10)
    path_color = random.choice([color for color in range(10) if color != bg])
    checkpoint_colors = random.sample(
        [color for color in range(10) if color not in {bg, path_color}],
        len(chosen),
    )
    checkpoints = [
        (idx, cell, color)
        for color, (idx, cell) in zip(checkpoint_colors, chosen)
    ]
    grid = _render_input(h, w, bg, path_color, path, checkpoints)
    if np.count_nonzero(grid != bg) * 2 >= grid.size:
        return None

    hits = sorted((idx, color) for idx, _cell, color in checkpoints)
    expected = _paint_segments(grid, path, hits)
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
    raise RuntimeError("generate_input: failed to build a valid input")
