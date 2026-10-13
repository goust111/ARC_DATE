from __future__ import annotations

import random

import numpy as np


PRODUCTIVE_BRANCHES = ((1, 1), (1, 2), (2, 1), (2, 2))
VANISH_BRANCHES = ((0, 1), (1, 0), (0, 2), (2, 0))
ALL_BRANCHES = PRODUCTIVE_BRANCHES + VANISH_BRANCHES
MAX_GENERATION_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 2 and w >= 2, "expected interior plus two headers"

    interior = g[:-1, :-1]
    bg = _dominant_color(interior)
    right_header = g[:-1, -1]
    bottom_header = g[-1, :-1]

    out = g.copy()
    out[:-1, :-1] = bg

    for r, c in np.argwhere(interior != bg):
        color = int(interior[r, c])
        rows = np.flatnonzero(right_header == color)
        cols = np.flatnonzero(bottom_header == color)
        if rows.size and cols.size:
            out[np.ix_(rows, cols)] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reposition each interior colored pixel onto the Cartesian product of its same-colored right-header rows and bottom-header columns.

    Input structure:
    - The rightmost column and bottom row are headers; the remaining cells form the interior.
    - The interior is mostly one background color with several colored single pixels.
    - A header color may appear zero, one, or two times on each side, and the bottom-right corner is just the shared corner cell.

    Rule:
    - Keep both headers unchanged and clear the whole interior to background.
    - For each non-background interior pixel of color c, collect every interior row whose right header cell is c and every interior column whose bottom header cell is c.
    - Paint color c into every intersection of those rows and columns; if either occurrence set is empty, that pixel disappears.

    Key insight:
    - A pixel does not move by direction or distance from its source position; its destination pattern is determined only by the Cartesian product of same-color occurrences in the two headers.
    """
    return _transform(grid)


def _sample_branches() -> list[tuple[int, int]]:
    count = random.randint(3, 4)
    branches = [random.choice(PRODUCTIVE_BRANCHES), random.choice(VANISH_BRANCHES)]
    while len(branches) < count:
        branches.append(random.choice(ALL_BRANCHES))
    if not any(max(r_count, c_count) == 2 for r_count, c_count in branches):
        branches[-1] = random.choice(((1, 2), (2, 1), (2, 2), (0, 2), (2, 0)))
    random.shuffle(branches)
    return branches


def _assign_occurrences(length: int, colors: list[int], counts: list[int]) -> dict[int, list[int]]:
    slots = random.sample(range(length), sum(counts))
    random.shuffle(slots)
    out: dict[int, list[int]] = {}
    start = 0
    for color, count in zip(colors, counts):
        out[color] = sorted(slots[start : start + count])
        start += count
    return out


def _pick_source_cells(
    interior_shape: tuple[int, int],
    colors: list[int],
    targets: dict[int, set[tuple[int, int]]],
) -> dict[int, tuple[int, int]] | None:
    interior_h, interior_w = interior_shape
    blocked = set().union(*targets.values()) if targets else set()
    chosen: dict[int, tuple[int, int]] = {}

    for color in random.sample(colors, len(colors)):
        candidates = [
            (r, c)
            for r in range(interior_h)
            for c in range(interior_w)
            if (r, c) not in blocked
            and all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in chosen.values())
        ]
        random.shuffle(candidates)
        if not candidates:
            return None
        chosen[color] = candidates[0]

    return chosen


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, 9)
    palette = [color for color in range(10) if color != bg]

    branches = _sample_branches()
    colors = random.sample(palette, len(branches))
    right_counts = [branch[0] for branch in branches]
    bottom_counts = [branch[1] for branch in branches]

    min_h = max(6, sum(right_counts) + 4)
    min_w = max(6, sum(bottom_counts) + 4)
    if min_h > 29 or min_w > 29:
        return None

    interior_h = random.randint(min_h, min(18, 29))
    interior_w = random.randint(min_w, min(18, 29))
    grid = np.full((interior_h + 1, interior_w + 1), bg, dtype=int)

    right_rows = _assign_occurrences(interior_h, colors, right_counts)
    bottom_cols = _assign_occurrences(interior_w, colors, bottom_counts)

    for color in colors:
        for row in right_rows[color]:
            grid[row, -1] = color
        for col in bottom_cols[color]:
            grid[-1, col] = color

    targets: dict[int, set[tuple[int, int]]] = {}
    for color in colors:
        rows = right_rows[color]
        cols = bottom_cols[color]
        targets[color] = {(r, c) for r in rows for c in cols}

    sources = _pick_source_cells((interior_h, interior_w), colors, targets)
    if sources is None:
        return None

    for color, (r, c) in sources.items():
        grid[r, c] = color

    productive = [color for color in colors if targets[color]]
    vanish = [color for color in colors if not targets[color]]
    if not productive or not vanish:
        return None

    out = _transform(grid)
    if np.array_equal(out, grid):
        return None
    if not np.array_equal(out[:-1, -1], grid[:-1, -1]):
        return None
    if not np.array_equal(out[-1, :-1], grid[-1, :-1]):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
