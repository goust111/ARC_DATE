from __future__ import annotations

import numpy as np
from numpy.random import default_rng


RNG = default_rng()
MAX_GENERATION_ATTEMPTS = 300
GRID_SIZE_RANGE = (10, 24)
AXIS_MARGIN = 3
BG_COLOR = 0
SEEDS_PER_ACTIVE_QUADRANT = (1, 3)
QUADRANTS = (0, 1, 2, 3)  # TL, TR, BL, BR
ARC_COLORS = tuple(range(10))


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_cross(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    h, w = grid.shape

    row_candidates: list[tuple[int, int, int, int]] = []
    for r in range(h):
        values, counts = np.unique(grid[r, :], return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist(), strict=False):
            if value == bg or count != w - 1:
                continue
            off_cols = np.flatnonzero(grid[r, :] != value)
            if off_cols.size != 1:
                continue
            c = int(off_cols[0])
            row_candidates.append((r, c, int(value), int(grid[r, c])))

    col_candidates: list[tuple[int, int, int, int]] = []
    for c in range(w):
        values, counts = np.unique(grid[:, c], return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist(), strict=False):
            if value == bg or count != h - 1:
                continue
            off_rows = np.flatnonzero(grid[:, c] != value)
            if off_rows.size != 1:
                continue
            r = int(off_rows[0])
            col_candidates.append((r, c, int(value), int(grid[r, c])))

    matches = [
        row
        for row in row_candidates
        for col in col_candidates
        if row == col and row[2] != row[3]
    ]
    assert len(matches) == 1, "expected exactly one marked cross"
    return matches[0]


def _stamp_cells(r: int, c: int, axis_r: int, axis_c: int) -> tuple[tuple[int, int], ...]:
    assert r != axis_r and c != axis_c
    dr = 1 if r < axis_r else -1
    dc = 1 if c < axis_c else -1
    return ((r, c), (r + dr, c), (r, c + dc))


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    axis_r, axis_c, _axis_color, _cross_color = _find_cross(g, bg)

    out = g.copy()
    axis_mask = np.zeros_like(g, dtype=bool)
    axis_mask[axis_r, :] = True
    axis_mask[:, axis_c] = True

    targets = np.argwhere((g != bg) & ~axis_mask)
    assert targets.size > 0, "expected at least one off-axis target pixel"

    for r0, c0 in targets:
        r = int(r0)
        c = int(c0)
        color = int(g[r, c])
        for rr, cc in _stamp_cells(r, c, axis_r, axis_c):
            if rr == axis_r or cc == axis_c:
                continue
            existing = int(out[rr, cc])
            assert existing in (bg, color), "stamps must not create color conflicts"
            out[rr, cc] = color

    out[axis_r, :] = g[axis_r, :]
    out[:, axis_c] = g[:, axis_c]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each off-axis target pixel into a quadrant-directed L stamp oriented toward a marked cross.

    Input structure:
    - The background is black.
    - One non-background color fills exactly one whole row and one whole column, forming a crosshair.
    - The cross intersection is a different special color.
    - Additional foreground pixels are isolated target pixels away from the marked row and column.

    Rule:
    - Keep the crosshair row, crosshair column, and special intersection unchanged.
    - Treat the intersection as splitting the grid into four quadrants.
    - For each target pixel, paint a 3-cell L shape in the same color: top-left targets extend right and down, top-right extend left and down, bottom-left extend right and up, and bottom-right extend left and up.
    - Each L stamp includes the original target pixel plus its two inward neighbors toward the intersection.

    Key insight:
    - The full row/column marker is not copied itself; it defines the quadrant of each target pixel, and that quadrant determines the inward-facing L orientation.
    """
    return _transform(grid)


def _candidate_cells(h: int, w: int, axis_r: int, axis_c: int, quadrant: int) -> list[tuple[int, int]]:
    if quadrant == 0:
        rows = range(0, axis_r - 1)
        cols = range(0, axis_c - 1)
    elif quadrant == 1:
        rows = range(0, axis_r - 1)
        cols = range(axis_c + 2, w)
    elif quadrant == 2:
        rows = range(axis_r + 2, h)
        cols = range(0, axis_c - 1)
    else:
        rows = range(axis_r + 2, h)
        cols = range(axis_c + 2, w)
    return [(r, c) for r in rows for c in cols]


def _sample_quadrants() -> list[int]:
    mask = int(RNG.integers(1, 1 << len(QUADRANTS)))
    return [quadrant for quadrant in QUADRANTS if mask & (1 << quadrant)]


def _gen_attempt() -> np.ndarray | None:
    h = int(RNG.integers(GRID_SIZE_RANGE[0], GRID_SIZE_RANGE[1] + 1))
    w = int(RNG.integers(GRID_SIZE_RANGE[0], GRID_SIZE_RANGE[1] + 1))

    if h < 2 * AXIS_MARGIN + 2 or w < 2 * AXIS_MARGIN + 2:
        return None

    axis_r = int(RNG.integers(AXIS_MARGIN, h - AXIS_MARGIN))
    axis_c = int(RNG.integers(AXIS_MARGIN, w - AXIS_MARGIN))

    grid = np.full((h, w), BG_COLOR, dtype=int)
    axis_color, cross_color = RNG.choice(np.array(ARC_COLORS[1:]), size=2, replace=False).tolist()
    grid[axis_r, :] = axis_color
    grid[:, axis_c] = axis_color
    grid[axis_r, axis_c] = cross_color

    available_colors = [color for color in ARC_COLORS if color not in {BG_COLOR, axis_color, cross_color}]
    occupied = np.zeros((h, w), dtype=bool)
    occupied[axis_r, :] = True
    occupied[:, axis_c] = True

    active_quadrants = _sample_quadrants()
    total_targets = 0
    for quadrant in active_quadrants:
        candidates = _candidate_cells(h, w, axis_r, axis_c, quadrant)
        if not candidates:
            return None
        RNG.shuffle(candidates)

        want = int(
            RNG.integers(
                SEEDS_PER_ACTIVE_QUADRANT[0],
                SEEDS_PER_ACTIVE_QUADRANT[1] + 1,
            )
        )
        placed = 0
        for r, c in candidates:
            stamp = _stamp_cells(r, c, axis_r, axis_c)
            if any(occupied[rr, cc] for rr, cc in stamp):
                continue
            grid[r, c] = int(RNG.choice(available_colors))
            for rr, cc in stamp:
                occupied[rr, cc] = True
            placed += 1
            total_targets += 1
            if placed == want:
                break
        if placed != want:
            return None

    if total_targets == 0:
        return None

    out = _transform(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
