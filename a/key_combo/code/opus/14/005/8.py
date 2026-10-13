from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
LATTICE = 8
CAVITY_OFFSETS = (
    (1, 2),
    (1, 3),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
    (4, 2),
    (4, 3),
)
MARKER_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
TILE_COUNT_RANGE = (4, 5)
MAX_MARKERS = 3
GEN_RETRIES = 400


def _build_lattice(n_tile_rows: int, n_tile_cols: int) -> np.ndarray:
    h = 5 * n_tile_rows + 1
    w = 5 * n_tile_cols + 1
    grid = np.full((h, w), LATTICE, dtype=int)
    rr, cc = np.mgrid[:h, :w]
    dr = rr % 5
    dc = cc % 5
    corner_mask = (dr == 0) & (dc == 0)
    cavity_mask = np.zeros((h, w), dtype=bool)
    for ro, co in CAVITY_OFFSETS:
        cavity_mask |= (dr == ro) & (dc == co)
    grid[corner_mask | cavity_mask] = BACKGROUND
    return grid


def _find_cross_axes(grid: np.ndarray) -> tuple[int, int]:
    rows = np.where(np.all(grid == LATTICE, axis=1))[0]
    cols = np.where(np.all(grid == LATTICE, axis=0))[0]
    assert len(rows) == 1 and len(cols) == 1, "expected one full teal row and one full teal column"
    return int(rows[0]), int(cols[0])


def _marker_pixels(grid: np.ndarray, axis_r: int, axis_c: int) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    for r, c in np.argwhere((grid != BACKGROUND) & (grid != LATTICE)):
        rr = int(r)
        cc = int(c)
        color = int(grid[rr, cc])
        assert rr < axis_r and cc < axis_c, "markers must lie in the upper-left quadrant"
        assert rr % 5 == 0 and cc % 5 == 0, "markers must occupy lattice corner gaps"
        markers.append((rr, cc, color))
    assert markers, "expected at least one colored marker"
    return markers


def _valid_anchor(shape: tuple[int, int], anchor_r: int, anchor_c: int) -> bool:
    h, w = shape
    if anchor_r < 0 or anchor_c < 0 or anchor_r + 4 >= h or anchor_c + 4 >= w:
        return False
    return anchor_r % 5 == 0 and anchor_c % 5 == 0


def _fill_cavity(grid: np.ndarray, anchor_r: int, anchor_c: int, color: int) -> None:
    for dr, dc in CAVITY_OFFSETS:
        grid[anchor_r + dr, anchor_c + dc] = int(color)


def _mirrored_anchors(anchor_r: int, anchor_c: int, axis_r: int, axis_c: int) -> list[tuple[int, int]]:
    row_reflect = 2 * axis_r - anchor_r - 5
    col_reflect = 2 * axis_c - anchor_c - 5
    return [
        (anchor_r, anchor_c),
        (row_reflect, anchor_c),
        (anchor_r, col_reflect),
        (row_reflect, col_reflect),
    ]


def _oracle_from_input(grid: np.ndarray) -> np.ndarray:
    axis_r, axis_c = _find_cross_axes(grid)
    out = np.asarray(grid, dtype=int).copy()
    for anchor_r, anchor_c, color in _marker_pixels(out, axis_r, axis_c):
        out[anchor_r, anchor_c] = BACKGROUND
        for rr, cc in _mirrored_anchors(anchor_r, anchor_c, axis_r, axis_c):
            if _valid_anchor(out.shape, rr, cc):
                _fill_cavity(out, rr, cc, color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each marker-designated lattice cavity, then mirror that cavity color across the embedded cross axes.

    Input structure:
    - The grid is a 5x5-period teal lattice whose corner gaps and enclosed chambers are black.
    - Exactly one full teal row and one full teal column form a cross aligned to lattice skeleton lines.
    - One to three non-black, non-teal marker pixels sit at upper-left corner gaps of lattice tiles.

    Rule:
    - For each marker, erase the marker back to black and fill the adjacent down-right chamber with the marker color.
    - Treat the full teal row and full teal column as the two symmetry axes.
    - Mirror the filled chamber into the other quadrants by reflecting the chamber center across the row axis, the column axis, and both axes; fill only mirrored positions that correspond to valid lattice chambers.
    - Leave the teal lattice and the cross axes unchanged.

    Key insight:
    - The colored corner pixels do not mark standalone dots: each one selects a chamber in the lattice, and the unique full-row/full-column cross tells you how that chamber pattern propagates to the other quadrants.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _oracle_from_input(g)


def _sample_axis_tile(n_tiles: int) -> int:
    return random.randint(1, n_tiles - 1)


def _valid_source_tiles(n_tile_rows: int, n_tile_cols: int, axis_tile_r: int, axis_tile_c: int) -> list[tuple[int, int]]:
    valid = []
    for tr in range(axis_tile_r):
        for tc in range(axis_tile_c):
            mr = 2 * axis_tile_r - tr - 1
            mc = 2 * axis_tile_c - tc - 1
            if mr < n_tile_rows and mc < n_tile_cols:
                valid.append((tr, tc))
    return valid


def _choose_marker_count(tile_count: int) -> int:
    hi = min(MAX_MARKERS, tile_count)
    weights = [1 if k == 1 else 2 for k in range(1, hi + 1)]
    return random.choices(range(1, hi + 1), weights=weights, k=1)[0]


def _choose_marker_colors(marker_count: int) -> list[int]:
    distinct = random.randint(1, min(3, marker_count))
    palette = random.sample(MARKER_COLORS, distinct)
    colors = palette[:]
    while len(colors) < marker_count:
        colors.append(random.choice(palette))
    random.shuffle(colors)
    return colors


def _render_input(
    n_tile_rows: int,
    n_tile_cols: int,
    axis_tile_r: int,
    axis_tile_c: int,
    markers: list[tuple[int, int, int]],
) -> np.ndarray:
    grid = _build_lattice(n_tile_rows, n_tile_cols)
    axis_r = 5 * axis_tile_r
    axis_c = 5 * axis_tile_c
    grid[axis_r, :] = LATTICE
    grid[:, axis_c] = LATTICE
    for tr, tc, color in markers:
        grid[5 * tr, 5 * tc] = int(color)
    return grid


def _manual_output(
    n_tile_rows: int,
    n_tile_cols: int,
    axis_tile_r: int,
    axis_tile_c: int,
    markers: list[tuple[int, int, int]],
) -> np.ndarray:
    out = _build_lattice(n_tile_rows, n_tile_cols)
    axis_r = 5 * axis_tile_r
    axis_c = 5 * axis_tile_c
    out[axis_r, :] = LATTICE
    out[:, axis_c] = LATTICE
    for tr, tc, color in markers:
        for rr, cc in _mirrored_anchors(5 * tr, 5 * tc, axis_r, axis_c):
            if _valid_anchor(out.shape, rr, cc):
                _fill_cavity(out, rr, cc, color)
    return out


def _gen_attempt() -> np.ndarray | None:
    n_tile_rows = random.randint(*TILE_COUNT_RANGE)
    n_tile_cols = random.randint(*TILE_COUNT_RANGE)
    axis_tile_r = _sample_axis_tile(n_tile_rows)
    axis_tile_c = _sample_axis_tile(n_tile_cols)
    source_tiles = _valid_source_tiles(n_tile_rows, n_tile_cols, axis_tile_r, axis_tile_c)
    if not source_tiles:
        return None

    marker_count = _choose_marker_count(len(source_tiles))
    chosen_tiles = random.sample(source_tiles, marker_count)
    colors = _choose_marker_colors(marker_count)
    markers = [(tr, tc, color) for (tr, tc), color in zip(chosen_tiles, colors)]

    inp = _render_input(n_tile_rows, n_tile_cols, axis_tile_r, axis_tile_c, markers)
    expected = _manual_output(n_tile_rows, n_tile_cols, axis_tile_r, axis_tile_c, markers)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lattice-cross symmetry input")
