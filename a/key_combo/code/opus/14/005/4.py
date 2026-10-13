from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 5
LEGEND_POOL = (1, 2, 3, 4, 6, 7)
MARKER = 8
AXIS = 9
MAX_GENERATION_ATTEMPTS = 400


def _colors_present(grid: np.ndarray) -> list[int]:
    return sorted(int(x) for x in np.unique(grid))


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int, int]:
    h, w = map(int, grid.shape)
    candidates: list[tuple[int, int, int, int, int]] = []
    for color in _colors_present(grid):
        full_rows = np.where(np.all(grid == color, axis=1))[0]
        full_cols = np.where(np.all(grid == color, axis=0))[0]
        if len(full_rows) != 2 or len(full_cols) != 2:
            continue
        top, bottom = map(int, full_rows)
        left, right = map(int, full_cols)
        if bottom - top <= 1 or right - left <= 1:
            continue
        if top == 0 or left == 0 or bottom == h - 1 or right == w - 1:
            continue
        candidates.append((int(color), top, bottom, left, right))

    assert len(candidates) == 1, "expected one rectangular frame"
    return candidates[0]


def _find_cross(interior: np.ndarray, bg: int) -> tuple[int, int, int]:
    candidates: list[tuple[int, int, int]] = []
    for color in _colors_present(interior):
        if color == bg:
            continue
        full_rows = np.where(np.all(interior == color, axis=1))[0]
        full_cols = np.where(np.all(interior == color, axis=0))[0]
        if len(full_rows) == 1 and len(full_cols) == 1:
            candidates.append((int(color), int(full_rows[0]), int(full_cols[0])))

    assert len(candidates) == 1, "expected one full-row/full-column cross"
    return candidates[0]


def _quadrant_index(r: int, c: int, axis_r: int, axis_c: int) -> int:
    if r < axis_r and c < axis_c:
        return 0
    if r < axis_r and c > axis_c:
        return 1
    if r > axis_r and c < axis_c:
        return 2
    if r > axis_r and c > axis_c:
        return 3
    raise AssertionError("axis cells do not belong to a quadrant")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the framed cross interior, mirror sparse markers across the cross, and recolor them by quadrant legend.

    Input structure:
    - A grey rectangular frame is encoded by two full rows and two full columns, with one-cell padding to the grid edge.
    - The four diagonal cells outside the frame corners are distinct legend pixels for the four quadrants.
    - Inside the frame, a maroon full row and full column form a centered cross, and teal marker pixels appear away from the cross in one or more quadrants.

    Rule:
    - Crop away the frame and everything outside it, keeping only the framed interior.
    - Use the full-row/full-column cross as both mirror axes: every non-background, non-cross marker is copied to its reflections across the row axis, the column axis, and both.
    - Recolor each final marker by the legend color of the quadrant it lies in, while keeping the cross and background unchanged.

    Key insight:
    - The cross has a dual role: it is simultaneously the geometric symmetry axes that create the missing markers and the boundary that determines which corner legend color each completed marker must receive.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)

    _frame_color, top, bottom, left, right = _find_frame(g)
    interior = g[top + 1 : bottom, left + 1 : right]
    out_h, out_w = map(int, interior.shape)
    assert out_h > 0 and out_w > 0, "frame must enclose a non-empty interior"

    cross_color, axis_r, axis_c = _find_cross(interior, bg)
    assert axis_r * 2 + 1 == out_h, "cross row must split the interior into equal top/bottom halves"
    assert axis_c * 2 + 1 == out_w, "cross column must split the interior into equal left/right halves"

    legend_coords = [
        (top - 1, left - 1),
        (top - 1, right + 1),
        (bottom + 1, left - 1),
        (bottom + 1, right + 1),
    ]
    for r, c in legend_coords:
        assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1], "legend cells must sit just outside the frame corners"
    legend_colors = [int(g[r, c]) for r, c in legend_coords]
    assert len(set(legend_colors)) == 4, "quadrant legends must be distinct"
    assert all(color not in {bg, cross_color} for color in legend_colors), "legends must differ from background and cross colors"

    markers = np.argwhere((interior != bg) & (interior != cross_color))
    assert markers.size > 0, "expected at least one marker away from the cross"

    out = np.full_like(interior, bg)
    out[interior == cross_color] = cross_color

    for r0, c0 in markers:
        r = int(r0)
        c = int(c0)
        assert r != axis_r and c != axis_c, "markers must not touch the cross"
        reflected_rows = (r, 2 * axis_r - r)
        reflected_cols = (c, 2 * axis_c - c)
        for rr in reflected_rows:
            for cc in reflected_cols:
                q = _quadrant_index(int(rr), int(cc), axis_r, axis_c)
                color = legend_colors[q]
                existing = int(out[rr, cc])
                assert existing in (bg, color), "mirrored markers must not create color conflicts"
                out[rr, cc] = color

    return out


def _quadrant_cells(half_h: int, half_w: int, quadrant: int) -> list[tuple[int, int]]:
    if quadrant < 2:
        rows = range(0, half_h)
    else:
        rows = range(half_h + 1, 2 * half_h + 1)
    if quadrant % 2 == 0:
        cols = range(0, half_w)
    else:
        cols = range(half_w + 1, 2 * half_w + 1)
    return [(r, c) for r in rows for c in cols]


def _structured_sample(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    chosen = random.sample(cells, count)
    if count < 2:
        return chosen

    rows = {r for r, _ in chosen}
    cols = {c for _, c in chosen}
    if len(rows) > 1 and len(cols) > 1:
        return chosen

    for _ in range(30):
        chosen = random.sample(cells, count)
        rows = {r for r, _ in chosen}
        cols = {c for _, c in chosen}
        if len(rows) > 1 and len(cols) > 1:
            return chosen
    return chosen


def _build_input(
    half_h: int,
    half_w: int,
    source_quadrant: int,
    marker_cells: list[tuple[int, int]],
    legend_colors: list[int],
) -> np.ndarray:
    interior_h = 2 * half_h + 1
    interior_w = 2 * half_w + 1
    grid = np.full((interior_h + 4, interior_w + 4), BG, dtype=int)

    top, bottom = 1, interior_h + 2
    left, right = 1, interior_w + 2
    grid[top, :] = FRAME
    grid[bottom, :] = FRAME
    grid[:, left] = FRAME
    grid[:, right] = FRAME

    grid[0, 0] = legend_colors[0]
    grid[0, -1] = legend_colors[1]
    grid[-1, 0] = legend_colors[2]
    grid[-1, -1] = legend_colors[3]

    interior = grid[top + 1 : bottom, left + 1 : right]
    interior[half_h, :] = AXIS
    interior[:, half_w] = AXIS
    for r, c in marker_cells:
        assert _quadrant_index(r, c, half_h, half_w) == source_quadrant
        interior[r, c] = MARKER

    return grid


def _gen_attempt() -> np.ndarray | None:
    half_h = random.randint(2, 6)
    half_w = random.randint(2, 6)
    area = half_h * half_w
    if area < 2:
        return None

    marker_count = random.randint(2, min(4, area))
    source_quadrant = random.randrange(4)
    legend_colors = random.sample(list(LEGEND_POOL), 4)
    cells = _quadrant_cells(half_h, half_w, source_quadrant)
    marker_cells = _structured_sample(cells, marker_count)
    grid = _build_input(half_h, half_w, source_quadrant, marker_cells, legend_colors)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None

    expected_colors = {AXIS, *legend_colors}
    if not expected_colors.issubset(set(map(int, np.unique(out)))):
        return None

    if np.count_nonzero(out == AXIS) != out.shape[0] + out.shape[1] - 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 4")
