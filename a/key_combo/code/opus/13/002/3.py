from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
DISTRACTOR_POOL = (1, 6, 7, 8, 9)
GEN_ATTEMPTS = 400

_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (0, 2)),
)


def _find_axis_row(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.all(grid == RED, axis=1))
    assert len(rows) == 1, "expected a unique full-width red mirror axis"
    return int(rows[0])


def _mirror_top_down(grid: np.ndarray) -> tuple[np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    axis = _find_axis_row(g)
    for r in range(axis):
        dst = 2 * axis - r
        if dst < g.shape[0]:
            out[dst, :] = g[r, :]
    out[axis, :] = BLACK
    return out, axis


def _bicolor_candidates(grid: np.ndarray) -> list[tuple[tuple[int, int], np.ndarray, np.ndarray, np.ndarray]]:
    colors = sorted(set(map(int, np.unique(grid))) - {BLACK})
    candidates: list[tuple[tuple[int, int], np.ndarray, np.ndarray, np.ndarray]] = []
    for a, b in itertools.combinations(colors, 2):
        mask = (grid == a) | (grid == b)
        if not mask.any():
            continue
        rows = mask.any(axis=1)
        cols = mask.any(axis=0)
        sub = grid[np.ix_(rows, cols)]
        if sub.size == 0:
            continue
        if not np.any(sub == a) or not np.any(sub == b):
            continue
        if np.all((sub == a) | (sub == b)):
            candidates.append(((int(a), int(b)), rows, cols, sub.copy()))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the area above a full-width red axis downward, erase the axis, then extract the unique two-color row-column product image.

    Input structure:
    - Black background with one full-width red horizontal line that acts as a mirror axis.
    - Green and yellow pixels appear above and below the axis, plus a few distractor pixels in other colors.
    - In the raw input, no two-color combination already fills the Cartesian product of its active rows and active columns.

    Rule:
    - Find the unique red row spanning the whole grid.
    - Copy every row above it to its mirrored row below it, clipping any destination row that would fall outside the grid, and then erase the red row to black.
    - In the processed grid, enumerate every non-background color pair and keep the unique pair whose pixels exactly fill every cell of its active rows by its active columns.
    - Delete all other rows and columns and return the induced two-color subgrid.

    Key insight:
    - The red row is not a separator to count or remove directly; it is the mirror axis that creates the only valid two-color row-column product image.
    """
    processed, _axis = _mirror_top_down(np.asarray(grid, dtype=int))
    candidates = _bicolor_candidates(processed)
    assert len(candidates) == 1, "expected exactly one valid two-color product after mirroring"
    _pair, rows, cols, sub = candidates[0]
    assert rows.any() and cols.any()
    return sub


def _make_pattern(height: int, width: int) -> np.ndarray:
    mode = random.randrange(4)
    rr, cc = np.indices((height, width))
    if mode == 0:
        mask = (rr + cc + random.randrange(2)) % 2 == 0
    elif mode == 1:
        period = random.choice((2, 3))
        mask = ((cc + random.randrange(period)) % period) == 0
    elif mode == 2:
        period = random.choice((2, 3))
        mask = ((rr + random.randrange(period)) % period) == 0
    else:
        mask = np.random.rand(height, width) < random.uniform(0.35, 0.65)

    pattern = np.where(mask, GREEN, YELLOW).astype(int)
    if np.all(pattern == GREEN) or np.all(pattern == YELLOW):
        pattern[0, 0] = GREEN
        pattern[-1, -1] = YELLOW
    return pattern


def _safe_origin_limits(length: int, span: int) -> range:
    return range(0, length - span + 1)


def _transform_shape(shape: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    cells = np.array(shape, dtype=int)
    if random.random() < 0.5:
        cells[:, 1] *= -1
    k = random.randrange(4)
    for _ in range(k):
        cells = np.column_stack((-cells[:, 1], cells[:, 0]))
    cells -= cells.min(axis=0)
    return [tuple(map(int, rc)) for rc in cells]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    ]


def _place_distractor(
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    band_rows: range,
) -> bool:
    cells = _transform_shape(random.choice(_SHAPES))
    span_r = max(r for r, _ in cells) + 1
    span_c = max(c for _, c in cells) + 1
    rows = [r for r in band_rows if r + span_r <= grid.shape[0]]
    cols = list(_safe_origin_limits(grid.shape[1], span_c))
    if not rows or not cols:
        return False

    for _ in range(80):
        r0 = random.choice(rows)
        c0 = random.choice(cols)
        placed = [(r0 + dr, c0 + dc) for dr, dc in cells]
        if any(blocked[r, c] for r, c in placed):
            continue
        halo = set(placed)
        for r, c in placed:
            halo.update((nr, nc) for nr, nc in _neighbors8(r, c) if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1])
        if any(blocked[r, c] for r, c in halo):
            continue
        for r, c in placed:
            grid[r, c] = color
        for r, c in halo:
            blocked[r, c] = True
        return True
    return False


def _expected_output(top_pattern: np.ndarray) -> np.ndarray:
    return np.vstack([top_pattern, top_pattern[::-1]])


def _raw_has_product(grid: np.ndarray) -> bool:
    return bool(_bicolor_candidates(np.asarray(grid, dtype=int)))


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(11, 18)
    width = random.randint(9, 17)

    axis = random.randint(3, height - 4)
    low_top = max(0, 2 * axis - (height - 1))
    max_k = axis - low_top
    if max_k < 2:
        return None

    k = random.randint(2, min(4, max_k))
    top_start = random.randint(low_top, axis - k)
    out_w = random.randint(3, min(8, width))
    left = random.randint(0, width - out_w)

    top_pattern = _make_pattern(k, out_w)
    expected = _expected_output(top_pattern)
    grid = np.full((height, width), BLACK, dtype=int)
    grid[axis, :] = RED

    top_rows = list(range(top_start, top_start + k))
    bottom_rows = [2 * axis - r for r in top_rows]
    target_cols = list(range(left, left + out_w))

    grid[np.ix_(top_rows, target_cols)] = top_pattern
    for idx, row in enumerate(top_rows):
        dst = 2 * axis - row
        keep = random.randint(1, out_w - 1)
        cols = sorted(random.sample(target_cols, keep))
        src_slice = top_pattern[idx, np.array(cols) - left]
        grid[dst, cols] = src_slice

    blocked = np.zeros_like(grid, dtype=bool)
    blocked[axis, :] = True
    blocked[top_rows, :] = True
    blocked[bottom_rows, :] = True
    blocked[np.ix_(top_rows, target_cols)] = True
    blocked[np.ix_(bottom_rows, target_cols)] = True

    color_count = random.randint(1, 3)
    distractor_colors = random.sample(DISTRACTOR_POOL, color_count)
    component_count = random.randint(4, 9)
    bands: list[range] = []
    if top_start > 0:
        bands.append(range(0, top_start))
    if top_start + k < axis:
        bands.append(range(top_start + k, axis))
    if axis + 1 < min(bottom_rows):
        bands.append(range(axis + 1, min(bottom_rows)))
    if max(bottom_rows) + 1 < height:
        bands.append(range(max(bottom_rows) + 1, height))
    if not bands:
        bands = [range(0, axis), range(axis + 1, height)]

    placed = 0
    for _ in range(component_count):
        band = random.choice(bands)
        color = random.choice(distractor_colors)
        placed += int(_place_distractor(grid, blocked, color, band))
    if placed < 3:
        return None

    if _raw_has_product(grid):
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        attempt = _build_attempt()
        if attempt is not None:
            grid, _expected = attempt
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
