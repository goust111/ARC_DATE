from __future__ import annotations

import random

import numpy as np


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _wall_color(grid: np.ndarray, bg: int) -> int:
    foreground = grid[grid != bg]
    values, counts = np.unique(foreground, return_counts=True)
    assert values.size >= 1, "expected wall color"
    return int(values[int(np.argmax(counts))])


def _high_count_lines(counts: np.ndarray) -> list[int]:
    unique = sorted(set(int(x) for x in counts.tolist()))
    assert len(unique) >= 2, "need at least two count levels"
    gaps = [unique[i + 1] - unique[i] for i in range(len(unique) - 1)]
    split = int(np.argmax(gaps))
    threshold = 0.5 * (unique[split] + unique[split + 1])
    lines = [i for i, value in enumerate(counts.tolist()) if float(value) > threshold]
    assert len(lines) >= 3, "expected outer border and at least one internal separator"
    return lines


def _constant_step(lines: list[int]) -> int:
    diffs = np.diff(lines)
    assert diffs.size > 0, "need multiple lattice lines"
    step = int(diffs[0])
    assert step > 1 and np.all(diffs == step), "grid lines must be equally spaced"
    return step


def _detect_lattice(grid: np.ndarray, wall: int) -> tuple[list[int], list[int]]:
    row_counts = np.count_nonzero(grid == wall, axis=1)
    col_counts = np.count_nonzero(grid == wall, axis=0)

    rows = _high_count_lines(row_counts)
    cols = _high_count_lines(col_counts)
    step_r = _constant_step(rows)
    step_c = _constant_step(cols)

    assert rows[0] == 0 and rows[-1] == grid.shape[0] - 1
    assert cols[0] == 0 and cols[-1] == grid.shape[1] - 1
    assert rows == list(range(0, grid.shape[0], step_r))
    assert cols == list(range(0, grid.shape[1], step_c))

    wall_mask = grid == wall
    line_mask = np.zeros_like(wall_mask, dtype=bool)
    line_mask[rows, :] = True
    line_mask[:, cols] = True
    assert np.all(~wall_mask | line_mask), "wall pixels must lie on the room lattice"
    return rows, cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed color into the union of row/column projections of wall gaps in its room.

    Input structure
    - Background is black.
    - Grey horizontal and vertical lines partition the grid into equal-sized rooms.
    - Each room wall segment may contain black gaps.
    - Some rooms contain exactly one non-black, non-grey seed pixel; the other rooms are empty.

    Rule
    - For every seeded room, inspect the gap positions on its top/bottom/left/right walls.
    - Fill every interior cell whose row hits a gap on the left or right wall, or whose
      column hits a gap on the top or bottom wall.
    - The seed color is used for the whole projected union, and the seed itself remains.
    - Empty rooms stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _majority_color(g)
    wall = _wall_color(g, bg)
    rows, cols = _detect_lattice(g, wall)

    out = g.copy()
    seed_mask = (g != bg) & (g != wall)
    assert not np.any(seed_mask[rows, :]) and not np.any(seed_mask[:, cols]), "seeds must stay inside rooms"

    for i in range(len(rows) - 1):
        for j in range(len(cols) - 1):
            r0, r1 = int(rows[i]), int(rows[i + 1])
            c0, c1 = int(cols[j]), int(cols[j + 1])
            assert r1 - r0 >= 2 and c1 - c0 >= 2

            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            assert not np.any(interior == wall), "walls cannot enter room interiors"

            seeds = np.argwhere((interior != bg) & (interior != wall))
            assert seeds.shape[0] <= 1, "each room must have at most one seed"
            if seeds.shape[0] == 0:
                assert np.all(interior == bg), "empty rooms must contain only background"
                continue

            sr, sc = map(int, seeds[0])
            color = int(interior[sr, sc])
            seed_only = np.zeros_like(interior, dtype=bool)
            seed_only[sr, sc] = True
            assert np.all((interior == bg) | seed_only), "room interior must contain exactly one seed"

            row_hits = (g[r0 + 1 : r1, c0] != wall) | (g[r0 + 1 : r1, c1] != wall)
            col_hits = (g[r0, c0 + 1 : c1] != wall) | (g[r1, c0 + 1 : c1] != wall)
            fill_mask = row_hits[:, None] | col_hits[None, :]
            out[r0 + 1 : r1, c0 + 1 : c1][fill_mask] = color

    return out


# --------------- generator ---------------

BG = 0
WALL = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

ROOM_ROWS_RANGE = (2, 4)
ROOM_COLS_RANGE = (2, 5)
INTERIOR_MIN = 2
GRID_MAX = 22
MAX_SEEDED_ROOMS = 6
MAX_GENERATE_ATTEMPTS = 400


def _sample_room_shape() -> tuple[int, int, int, int] | None:
    room_rows = random.randint(*ROOM_ROWS_RANGE)
    room_cols = random.randint(*ROOM_COLS_RANGE)

    max_inner_h = (GRID_MAX - 1) // room_rows - 1
    max_inner_w = (GRID_MAX - 1) // room_cols - 1
    if max_inner_h < INTERIOR_MIN or max_inner_w < INTERIOR_MIN:
        return None

    inner_h = random.randint(INTERIOR_MIN, max_inner_h)
    inner_w = random.randint(INTERIOR_MIN, max_inner_w)
    return room_rows, room_cols, inner_h, inner_w


def _sample_gap_mask(length: int) -> np.ndarray:
    assert length >= 2
    max_gaps = length - 1
    coin = random.random()
    if coin < 0.25:
        gap_count = 0
    elif coin < 0.75:
        gap_count = 1
    else:
        gap_count = random.randint(2, max_gaps) if max_gaps >= 2 else 1
    gap_count = min(gap_count, max_gaps)

    mask = np.zeros(length, dtype=bool)
    if gap_count > 0:
        idx = random.sample(range(length), gap_count)
        mask[idx] = True
    return mask


def _room_fill_mask(
    h_gaps: list[list[np.ndarray]],
    v_gaps: list[list[np.ndarray]],
    room_r: int,
    room_c: int,
) -> np.ndarray:
    top = h_gaps[room_r][room_c]
    bottom = h_gaps[room_r + 1][room_c]
    left = v_gaps[room_r][room_c]
    right = v_gaps[room_r][room_c + 1]
    row_hits = left | right
    col_hits = top | bottom
    return row_hits[:, None] | col_hits[None, :]


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    shape = _sample_room_shape()
    if shape is None:
        raise AssertionError("invalid sampled room shape")
    room_rows, room_cols, inner_h, inner_w = shape

    height = room_rows * (inner_h + 1) + 1
    width = room_cols * (inner_w + 1) + 1
    row_lines = list(range(0, height, inner_h + 1))
    col_lines = list(range(0, width, inner_w + 1))

    grid = np.full((height, width), BG, dtype=int)
    grid[row_lines, :] = WALL
    grid[:, col_lines] = WALL

    h_gaps = [
        [_sample_gap_mask(inner_w) for _ in range(room_cols)]
        for _ in range(room_rows + 1)
    ]
    v_gaps = [
        [_sample_gap_mask(inner_h) for _ in range(room_cols + 1)]
        for _ in range(room_rows)
    ]

    for k, row in enumerate(row_lines):
        for j in range(room_cols):
            c0 = col_lines[j] + 1
            c1 = col_lines[j + 1]
            grid[row, c0:c1][h_gaps[k][j]] = BG

    for i in range(room_rows):
        for k, col in enumerate(col_lines):
            r0 = row_lines[i] + 1
            r1 = row_lines[i + 1]
            grid[r0:r1, col][v_gaps[i][k]] = BG

    candidates: list[tuple[int, int]] = []
    for i in range(room_rows):
        for j in range(room_cols):
            if np.any(_room_fill_mask(h_gaps, v_gaps, i, j)):
                candidates.append((i, j))
    assert candidates, "need at least one room with a visible projection"

    total_rooms = room_rows * room_cols
    max_seedable = min(len(candidates), MAX_SEEDED_ROOMS, total_rooms)
    min_seedable = 1
    if total_rooms > 1:
        max_seedable = min(max_seedable, total_rooms - 1)
    seed_count = random.randint(min_seedable, max_seedable)
    seeded_rooms = random.sample(candidates, seed_count)

    colors = random.sample(SEED_COLORS, seed_count)
    for color, (i, j) in zip(colors, seeded_rooms, strict=False):
        r0, r1 = row_lines[i] + 1, row_lines[i + 1]
        c0, c1 = col_lines[j] + 1, col_lines[j + 1]
        sr = random.randint(r0, r1 - 1)
        sc = random.randint(c0, c1 - 1)
        grid[sr, sc] = color

    out = grid.copy()
    for color, (i, j) in zip(colors, seeded_rooms, strict=False):
        r0, r1 = row_lines[i] + 1, row_lines[i + 1]
        c0, c1 = col_lines[j] + 1, col_lines[j + 1]
        fill_mask = _room_fill_mask(h_gaps, v_gaps, i, j)
        out[r0:r1, c0:c1][fill_mask] = color

    return grid, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        inp, expected = _build_case()
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid sample")
