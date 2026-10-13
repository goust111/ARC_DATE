from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg, neighbors4
import random


Coord = tuple[int, int]


def _ordered_path(coords: np.ndarray) -> list[Coord]:
    cells = {tuple(map(int, coord)) for coord in coords}
    assert len(cells) == len(coords), "path coordinates must be unique"

    degrees = {
        cell: sum((neighbor in cells) for neighbor in neighbors4(*cell))
        for cell in cells
    }
    endpoints = sorted(cell for cell, degree in degrees.items() if degree == 1)
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    assert all(degree in (1, 2) for degree in degrees.values()), "path must not branch"

    path = []
    prev = None
    cur = endpoints[0]
    while True:
        path.append(cur)
        next_cells = [
            neighbor
            for neighbor in neighbors4(*cur)
            if neighbor in cells and neighbor != prev
        ]
        if not next_cells:
            break
        assert len(next_cells) == 1, "path traversal must be unique"
        prev, cur = cur, next_cells[0]

    assert len(path) == len(cells), "path traversal must visit every foreground pixel"
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse the sequence of colors along the single 4-connected foreground path while keeping the path shape fixed.

    Input structure:
    - The background is the dominant color.
    - All non-background pixels form one multicolor 4-connected one-pixel-wide path.
    - The path has exactly two endpoints and no branches.

    Rule:
    - Walk the path from one endpoint to the other.
    - Replace the colors on those same path cells by the original path colors in reverse order.
    - Leave every background pixel unchanged.

    Key insight:
    - Nothing moves geometrically; only the color string written on the path is reversed between the endpoints.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    assert len(components) == 1, "expected exactly one foreground path"

    path = _ordered_path(components[0]["coords"])
    values = [int(g[r, c]) for r, c in path]

    out = g.copy()
    for (r, c), value in zip(path, reversed(values)):
        out[r, c] = value
    return out

# --------------- generator (att2, o5x) ---------------

# Grid and path geometry.
GRID_SHAPE = (10, 10)
BG_COLOR = 8
INTERIOR_MIN = 1
INTERIOR_MAX = 8
PATH_LEN_MIN, PATH_LEN_MAX = 8, 28
TURN_COUNT_MIN, TURN_COUNT_MAX = 2, 8
SEGMENT_STEPS_MIN, SEGMENT_STEPS_MAX = 1, 7

# Direction and color sampling.
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
PATH_COLORS = [1, 2, 4, 5, 6, 7, 9]
PALETTE_SIZE_MIN, PALETTE_SIZE_MAX = 6, 7

# Generation retry budget.
MAX_GENERATION_ATTEMPTS = 300


def _turn_dirs(direction: tuple[int, int] | None) -> list[tuple[int, int]]:
    if direction is None:
        dirs = list(DIRS)
    else:
        dr, dc = direction
        dirs = [(-dc, dr), (dc, -dr)]
    random.shuffle(dirs)
    return dirs


def _sample_run_lengths(n_segments: int, total_steps: int) -> list[int]:
    runs = []
    remaining = total_steps
    for i in range(n_segments):
        slots = n_segments - i - 1
        lo = max(SEGMENT_STEPS_MIN, remaining - SEGMENT_STEPS_MAX * slots)
        hi = min(SEGMENT_STEPS_MAX, remaining - SEGMENT_STEPS_MIN * slots)
        runs.append(random.randint(lo, hi))
        remaining -= runs[-1]
    random.shuffle(runs)
    return runs


def _sample_run_plan() -> list[int]:
    target_len = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)
    total_steps = target_len - 1
    valid_turns = [
        turns
        for turns in range(TURN_COUNT_MIN, TURN_COUNT_MAX + 1)
        if SEGMENT_STEPS_MIN * (turns + 1) <= total_steps <= SEGMENT_STEPS_MAX * (turns + 1)
    ]
    turns = random.choice(valid_turns)
    return _sample_run_lengths(turns + 1, total_steps)


def _extend_segment(
    path: list[tuple[int, int]],
    used: set[tuple[int, int]],
    direction: tuple[int, int],
    steps: int,
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    dr, dc = direction
    new_path = list(path)
    new_used = set(used)
    r, c = new_path[-1]

    for _ in range(steps):
        nr, nc = r + dr, c + dc
        if not (INTERIOR_MIN <= nr <= INTERIOR_MAX and INTERIOR_MIN <= nc <= INTERIOR_MAX):
            return None
        if (nr, nc) in new_used:
            return None
        if any((ar, ac) in new_used and (ar, ac) != (r, c) for ar, ac in neighbors4(nr, nc)):
            return None

        new_path.append((nr, nc))
        new_used.add((nr, nc))
        r, c = nr, nc

    return new_path, new_used


def _build_path(run_lengths: list[int]) -> list[tuple[int, int]] | None:
    starts = [(r, c) for r in range(INTERIOR_MIN, INTERIOR_MAX + 1) for c in range(INTERIOR_MIN, INTERIOR_MAX + 1)]
    random.shuffle(starts)

    def search(
        seg_i: int,
        prev_dir: tuple[int, int] | None,
        path: list[tuple[int, int]],
        used: set[tuple[int, int]],
    ) -> list[tuple[int, int]] | None:
        if seg_i == len(run_lengths):
            return path
        for direction in _turn_dirs(prev_dir):
            extended = _extend_segment(path, used, direction, run_lengths[seg_i])
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


def _color_sequence(n: int) -> list[int]:
    palette = random.sample(PATH_COLORS, random.randint(PALETTE_SIZE_MIN, PALETTE_SIZE_MAX))
    seq = palette + [random.choice(palette) for _ in range(n - len(palette))]
    random.shuffle(seq)
    if seq == seq[::-1]:
        for i in range(1, n):
            if seq[i] != seq[0]:
                seq[0], seq[i] = seq[i], seq[0]
                break
    return seq


def _paint_path(grid: np.ndarray, path: list[Coord], colors: list[int]) -> None:
    for (r, c), color in zip(path, colors):
        grid[r, c] = color


def _passes_oracle(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        return bool(np.array_equal(main(grid), expected))
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    path = _build_path(_sample_run_plan())
    if path is None:
        return None
    colors = _color_sequence(len(path))
    grid = np.full(GRID_SHAPE, BG_COLOR, dtype=int)
    _paint_path(grid, path, colors)
    expected = grid.copy()
    _paint_path(expected, path, colors[::-1])
    if not _passes_oracle(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 5792cb4d input")
