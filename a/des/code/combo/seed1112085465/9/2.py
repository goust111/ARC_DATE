from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]


BG_COLOR = 7


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _neighbors8(r: int, c: int) -> tuple[Coord, ...]:
    return (
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    )


def _find_components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[Coord]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)

    return components


def _ordered_endpoints(coords: list[Coord]) -> tuple[Coord, Coord]:
    cells = set(coords)
    assert len(cells) == len(coords), "path coordinates must be unique"

    degrees = {
        cell: sum((neighbor in cells) for neighbor in _neighbors4(*cell))
        for cell in cells
    }
    endpoints = sorted(cell for cell, degree in degrees.items() if degree == 1)

    assert len(endpoints) == 2, "each path must have exactly two endpoints"
    assert all(degree in (1, 2) for degree in degrees.values()), "paths must not branch or cycle"
    return endpoints[0], endpoints[1]


def _swap_path_endpoints(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    components = _find_components(grid != BG_COLOR)
    assert components, "expected at least one non-background path"

    for coords in components:
        first, second = _ordered_endpoints(coords)
        c1 = int(grid[first])
        c2 = int(grid[second])
        assert c1 != BG_COLOR and c2 != BG_COLOR
        out[first] = c2
        out[second] = c1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the endpoint colors of every disconnected 4-connected foreground path.

    Input structure:
    - The background is orange.
    - Non-background cells form several one-pixel-wide 4-connected paths.
    - Each path has exactly two endpoints and no branches.

    Rule:
    - Find the two endpoints of each path.
    - Exchange the colors at those two endpoint cells.
    - Leave every non-endpoint path cell and every background cell unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _swap_path_endpoints(g)


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (10, 22)
GRID_WIDTH_RANGE = (10, 22)
PATH_COUNT_RANGE = (1, 4)
PATH_LEN_MIN = 2
PATH_LEN_MAX = 15
TURN_MAX = 4
SEGMENT_LEN_MIN = 1
SEGMENT_LEN_MAX = 6
INNER_MARGIN = 1
COLOR_POOL = [0, 1, 2, 3, 4, 5, 6, 8, 9]
MAX_GENERATION_ATTEMPTS = 400
MAX_PATH_BUILD_ATTEMPTS = 80


def _sample_run_lengths(n_segments: int, total_steps: int) -> list[int]:
    runs = []
    remaining = total_steps
    for i in range(n_segments):
        slots = n_segments - i - 1
        lo = max(SEGMENT_LEN_MIN, remaining - SEGMENT_LEN_MAX * slots)
        hi = min(SEGMENT_LEN_MAX, remaining - SEGMENT_LEN_MIN * slots)
        runs.append(random.randint(lo, hi))
        remaining -= runs[-1]
    random.shuffle(runs)
    return runs


def _sample_run_plan(path_len: int) -> list[int]:
    total_steps = path_len - 1
    if total_steps == 1:
        return [1]

    valid_turns = [
        turns
        for turns in range(0, min(TURN_MAX, total_steps - 1) + 1)
        if SEGMENT_LEN_MIN * (turns + 1) <= total_steps <= SEGMENT_LEN_MAX * (turns + 1)
    ]
    turns = random.choice(valid_turns)
    return _sample_run_lengths(turns + 1, total_steps)


def _segment_directions(prev_dir: Coord | None) -> list[Coord]:
    if prev_dir is None:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        dr, dc = prev_dir
        dirs = [(-dc, dr), (dc, -dr)]
    random.shuffle(dirs)
    return dirs


def _valid_step(
    nr: int,
    nc: int,
    prev: Coord,
    used: set[Coord],
    occupied: set[Coord],
    h: int,
    w: int,
) -> bool:
    if not (INNER_MARGIN <= nr < h - INNER_MARGIN and INNER_MARGIN <= nc < w - INNER_MARGIN):
        return False
    if (nr, nc) in used or (nr, nc) in occupied:
        return False

    prev_r, prev_c = prev
    for ar, ac in _neighbors4(nr, nc):
        if (ar, ac) == (prev_r, prev_c):
            continue
        if (ar, ac) in used or (ar, ac) in occupied:
            return False

    for ar, ac in _neighbors8(nr, nc):
        if (ar, ac) in occupied:
            return False

    return True


def _extend_segment(
    path: list[Coord],
    used: set[Coord],
    occupied: set[Coord],
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
        if not _valid_step(nr, nc, (r, c), new_used, occupied, h, w):
            return None
        new_path.append((nr, nc))
        new_used.add((nr, nc))
        r, c = nr, nc

    return new_path, new_used


def _build_path(h: int, w: int, occupied: set[Coord], run_lengths: list[int]) -> list[Coord] | None:
    starts = [
        (r, c)
        for r in range(INNER_MARGIN, h - INNER_MARGIN)
        for c in range(INNER_MARGIN, w - INNER_MARGIN)
        if (r, c) not in occupied and all((nr, nc) not in occupied for nr, nc in _neighbors8(r, c))
    ]
    random.shuffle(starts)

    def search(
        seg_i: int,
        prev_dir: Coord | None,
        path: list[Coord],
        used: set[Coord],
    ) -> list[Coord] | None:
        if seg_i == len(run_lengths):
            return path

        for direction in _segment_directions(prev_dir):
            extended = _extend_segment(path, used, occupied, direction, run_lengths[seg_i], h, w)
            if extended is None:
                continue
            found = search(seg_i + 1, direction, extended[0], extended[1])
            if found is not None:
                return found
        return None

    for start in starts:
        found = search(0, None, [start], {start})
        if found is not None:
            return found
    return None


def _sample_path_colors(length: int) -> list[int]:
    palette_size = random.randint(2, min(5, len(COLOR_POOL)))
    palette = random.sample(COLOR_POOL, palette_size)
    first, last = random.sample(palette, 2)
    if length == 2:
        return [first, last]

    interior = [random.choice(palette) for _ in range(length - 2)]
    if length >= 4 and all(color == first for color in interior):
        swap_options = [color for color in palette if color != first]
        interior[random.randrange(len(interior))] = random.choice(swap_options)
    return [first, *interior, last]


def _paint_path(grid: np.ndarray, path: list[Coord], colors: list[int]) -> None:
    for (r, c), color in zip(path, colors):
        grid[r, c] = int(color)


def _validate_instance(grid: np.ndarray) -> bool:
    if grid.ndim != 2:
        return False
    if not np.any(grid == BG_COLOR):
        return False

    components = _find_components(grid != BG_COLOR)
    if not components:
        return False

    try:
        for coords in components:
            first, second = _ordered_endpoints(coords)
            if int(grid[first]) == int(grid[second]):
                return False
    except AssertionError:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    grid = np.full((h, w), BG_COLOR, dtype=int)

    max_paths = min(PATH_COUNT_RANGE[1], max(1, (h * w) // 45))
    path_count = random.randint(PATH_COUNT_RANGE[0], max_paths)
    occupied: set[Coord] = set()
    built_paths: list[list[Coord]] = []

    for _ in range(path_count):
        area_budget = max(2, (h - 2) * (w - 2) // max(2, path_count))
        path_len_hi = min(PATH_LEN_MAX, area_budget)
        path_len = random.randint(PATH_LEN_MIN, max(PATH_LEN_MIN, path_len_hi))

        path = None
        for _ in range(MAX_PATH_BUILD_ATTEMPTS):
            run_lengths = _sample_run_plan(path_len)
            path = _build_path(h, w, occupied, run_lengths)
            if path is not None:
                break
        if path is None:
            return None

        colors = _sample_path_colors(len(path))
        _paint_path(grid, path, colors)
        occupied.update(path)
        built_paths.append(path)

    if not built_paths or not _validate_instance(grid):
        return None

    expected = grid.copy()
    for path in built_paths:
        first, second = sorted((path[0], path[-1]))
        expected[first], expected[second] = expected[second], expected[first]

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
    raise RuntimeError("failed to generate a valid input")
