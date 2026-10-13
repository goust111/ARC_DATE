from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8]

GEN_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_axis(grid: np.ndarray) -> tuple[str, int]:
    row_hits = [int(r) for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    col_hits = [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]

    if len(row_hits) + len(col_hits) != 1:
        raise AssertionError("expected exactly one full grey separator row or column")
    if col_hits:
        return "vertical", col_hits[0]
    return "horizontal", row_hits[0]


def _reflect(axis_kind: str, axis_idx: int, r: int, c: int) -> tuple[int, int]:
    if axis_kind == "vertical":
        return r, 2 * axis_idx - c
    return 2 * axis_idx - r, c


def _component_from(passable: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = passable.shape
    sr, sc = start
    if not (0 <= sr < h and 0 <= sc < w) or not bool(passable[sr, sc]):
        raise AssertionError("seed must land inside a fillable chamber")

    q = deque([(sr, sc)])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bool(passable[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return list(seen)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror one-sided maroon chamber walls and seed pixels across the grey separator,
    then flood-fill each original and mirrored chamber with the seed color.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.size == 0:
        raise AssertionError("expected a non-empty 2D grid")

    h, w = g.shape
    axis_kind, axis_idx = _find_axis(g)

    out = g.copy()
    maroon_pixels = [(int(r), int(c)) for r, c in np.argwhere(g == MAROON)]
    for r, c in maroon_pixels:
        rr, cc = _reflect(axis_kind, axis_idx, r, c)
        if not (0 <= rr < h and 0 <= cc < w):
            raise AssertionError("mirrored wall left the grid")
        if int(out[rr, cc]) not in (BLACK, MAROON):
            raise AssertionError("mirrored wall collided with a non-wall object")
        out[rr, cc] = MAROON

    seed_specs: list[tuple[int, int, int]] = []
    for r, c in np.argwhere((g != BLACK) & (g != GREY) & (g != MAROON)):
        rr = int(r)
        cc = int(c)
        color = int(g[rr, cc])
        seed_specs.append((rr, cc, color))
        mr, mc = _reflect(axis_kind, axis_idx, rr, cc)
        if not (0 <= mr < h and 0 <= mc < w):
            raise AssertionError("mirrored seed left the grid")
        if int(out[mr, mc]) in (GREY, MAROON):
            raise AssertionError("mirrored seed landed on a barrier")
        seed_specs.append((mr, mc, color))

    barrier = (out == GREY) | (out == MAROON)
    passable = ~barrier
    filled = out.copy()

    for sr, sc, color in seed_specs:
        for rr, cc in _component_from(passable, (sr, sc)):
            filled[rr, cc] = color

    return filled


def _build_vertical(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    side_width = rng.randint(2, 4)
    axis_col = side_width
    width = 2 * side_width + 1

    chamber_count = rng.randint(3, min(6, len(SEED_COLORS)))
    chamber_heights = [rng.randint(1, 4) for _ in range(chamber_count)]
    height = sum(chamber_heights) + chamber_count - 1

    wall_side = rng.choice(["left", "right"])
    grid = np.full((height, width), BLACK, dtype=int)
    grid[:, axis_col] = GREY

    intervals: list[tuple[int, int]] = []
    wall_rows: list[int] = []
    cursor = 0
    for idx, chamber_height in enumerate(chamber_heights):
        start = cursor
        end = cursor + chamber_height - 1
        intervals.append((start, end))
        cursor = end + 1
        if idx < chamber_count - 1:
            wall_rows.append(cursor)
            if wall_side == "left":
                grid[cursor, :axis_col] = MAROON
            else:
                grid[cursor, axis_col + 1 :] = MAROON
            cursor += 1

    seed_count = rng.randint(2, min(chamber_count, len(SEED_COLORS)))
    chosen = rng.sample(range(chamber_count), seed_count)
    rng.shuffle(chosen)
    left_count = rng.randint(1, seed_count - 1)
    left_seeded = set(chosen[:left_count])
    colors = rng.sample(SEED_COLORS, seed_count)

    expected = grid.copy()
    if wall_side == "left":
        for row in wall_rows:
            expected[row, axis_col + 1 :] = MAROON
    else:
        for row in wall_rows:
            expected[row, :axis_col] = MAROON

    for chamber_idx, color in zip(chosen, colors):
        r0, r1 = intervals[chamber_idx]
        seed_left = chamber_idx in left_seeded
        seed_col = rng.randint(0, axis_col - 1) if seed_left else rng.randint(axis_col + 1, width - 1)
        seed_row = rng.randint(r0, r1)
        grid[seed_row, seed_col] = color
        expected[r0 : r1 + 1, :axis_col] = color
        expected[r0 : r1 + 1, axis_col + 1 :] = color

    return grid, expected


def _build_horizontal(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    side_height = rng.randint(2, 4)
    axis_row = side_height
    height = 2 * side_height + 1

    chamber_count = rng.randint(3, min(6, len(SEED_COLORS)))
    chamber_widths = [rng.randint(1, 4) for _ in range(chamber_count)]
    width = sum(chamber_widths) + chamber_count - 1

    wall_side = rng.choice(["top", "bottom"])
    grid = np.full((height, width), BLACK, dtype=int)
    grid[axis_row, :] = GREY

    intervals: list[tuple[int, int]] = []
    wall_cols: list[int] = []
    cursor = 0
    for idx, chamber_width in enumerate(chamber_widths):
        start = cursor
        end = cursor + chamber_width - 1
        intervals.append((start, end))
        cursor = end + 1
        if idx < chamber_count - 1:
            wall_cols.append(cursor)
            if wall_side == "top":
                grid[:axis_row, cursor] = MAROON
            else:
                grid[axis_row + 1 :, cursor] = MAROON
            cursor += 1

    seed_count = rng.randint(2, min(chamber_count, len(SEED_COLORS)))
    chosen = rng.sample(range(chamber_count), seed_count)
    rng.shuffle(chosen)
    top_count = rng.randint(1, seed_count - 1)
    top_seeded = set(chosen[:top_count])
    colors = rng.sample(SEED_COLORS, seed_count)

    expected = grid.copy()
    if wall_side == "top":
        for col in wall_cols:
            expected[axis_row + 1 :, col] = MAROON
    else:
        for col in wall_cols:
            expected[:axis_row, col] = MAROON

    for chamber_idx, color in zip(chosen, colors):
        c0, c1 = intervals[chamber_idx]
        seed_top = chamber_idx in top_seeded
        seed_row = rng.randint(0, axis_row - 1) if seed_top else rng.randint(axis_row + 1, height - 1)
        seed_col = rng.randint(c0, c1)
        grid[seed_row, seed_col] = color
        expected[:axis_row, c0 : c1 + 1] = color
        expected[axis_row + 1 :, c0 : c1 + 1] = color

    return grid, expected


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    if rng.random() < 0.5:
        grid, expected = _build_vertical(rng)
    else:
        grid, expected = _build_horizontal(rng)

    colors, counts = np.unique(grid, return_counts=True)
    if int(colors[np.argmax(counts)]) != BLACK:
        return None

    seed_mask = (grid != BLACK) & (grid != GREY) & (grid != MAROON)
    if int(seed_mask.sum()) < 2:
        return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if np.array_equal(solved, grid):
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
