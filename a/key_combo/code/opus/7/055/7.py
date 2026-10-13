from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
ARC_COLORS = tuple(range(10))
NON_SPECIAL_COLORS = tuple(c for c in ARC_COLORS if c not in (BLACK, BLUE))
MAX_GEN_ATTEMPTS = 500


def _guide_rows(grid: np.ndarray) -> list[int]:
    return [
        int(r)
        for r in range(grid.shape[0])
        if int(grid[r, 0]) not in (BLACK, BLUE) and np.all(grid[r, :] == int(grid[r, 0]))
    ]


def _blue_positions(grid: np.ndarray) -> np.ndarray:
    return np.argwhere(grid == BLUE)


def _erased_block_columns(n: int, blue_cols: list[int], width: int) -> set[int]:
    erased: set[int] = set()
    for col in blue_cols:
        erased.update(range(int(col), min(int(n), int(col + width))))
    return erased


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the input sprite across its guide row band, then erase encoded block columns.

    Input structure:
    - The input is an n x n square tile with black background.
    - Exactly one row is fully filled by one non-black, non-blue color; this is the guide row.
    - One or more blue marker pixels lie outside the guide row, and their column positions encode block columns.

    Rule:
    - Find the unique solid guide row and collect the column positions of all blue pixels.
    - Replace every blue pixel with black to obtain the clean tile.
    - Create an n x n block canvas of size (n^2) x (n^2), and copy the clean tile into every block of the guide row band.
    - Let K be the number of blue pixels. For each encoded column c, erase the tiled copies in block columns c through c+K-1, clipped by the right edge.

    Key insight:
    - The tile is self-describing: blue pixels are metadata, with their columns selecting which tiled copies disappear and their count setting each erased run's width.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height == width, "expected a square tile"

    n = height
    assert n > 0 and n * n <= 30, "input tile must expand to at most 30x30"

    guides = _guide_rows(g)
    assert len(guides) == 1, "expected exactly one solid guide row"
    guide_row = guides[0]

    blue_pts = _blue_positions(g)
    assert len(blue_pts) >= 1, "expected at least one blue marker pixel"
    assert np.all(blue_pts[:, 0] != guide_row), "blue markers must stay off the guide row"

    marker_width = int(len(blue_pts))
    blue_cols = sorted({int(c) for _r, c in blue_pts})
    assert len(blue_cols) == marker_width, "blue marker columns must be distinct"

    clean = g.copy()
    clean[clean == BLUE] = BLACK

    out = np.zeros((n * n, n * n), dtype=int)
    row_slice = slice(guide_row * n, (guide_row + 1) * n)
    for block_col in range(n):
        col_slice = slice(block_col * n, (block_col + 1) * n)
        out[row_slice, col_slice] = clean

    for block_col in _erased_block_columns(n, blue_cols, marker_width):
        col_slice = slice(block_col * n, (block_col + 1) * n)
        out[row_slice, col_slice] = BLACK

    return out


def _paint_mirrored_pairs(
    grid: np.ndarray,
    rows: list[int],
    colors: list[int],
    rng: random.Random,
) -> None:
    n = grid.shape[0]
    half_span = (n + 1) // 2
    offset = rng.randrange(half_span)
    for idx, row in enumerate(rows):
        left = int((offset + idx) % half_span)
        right = int(n - 1 - left)
        grid[row, left] = colors[idx % len(colors)]
        if right != left:
            grid[row, right] = colors[(idx + 1) % len(colors)]
        elif n >= 4:
            alt = int((left + 2) % n)
            grid[row, alt] = colors[(idx + 1) % len(colors)]


def _paint_stair_runs(
    grid: np.ndarray,
    rows: list[int],
    colors: list[int],
    rng: random.Random,
) -> None:
    n = grid.shape[0]
    run_len = 2 if n == 4 else rng.choice([2, 3])
    span = n - run_len + 1
    offset = rng.randrange(span)
    for idx, row in enumerate(rows):
        start = int((offset + idx) % span)
        for delta in range(run_len):
            grid[row, start + delta] = colors[(idx + delta) % len(colors)]


def _paint_checker_knots(
    grid: np.ndarray,
    rows: list[int],
    colors: list[int],
    rng: random.Random,
) -> None:
    n = grid.shape[0]
    phase = rng.randrange(2)
    for idx, row in enumerate(rows):
        cols = [c for c in range(n) if (c + idx + phase) % 2 == 0]
        if n == 5 and len(cols) == 3 and rng.random() < 0.5:
            cols.pop()
        for j, col in enumerate(cols):
            grid[row, col] = colors[(idx + j) % len(colors)]
        if len(cols) == 1:
            alt = int((cols[0] + 2) % n)
            grid[row, alt] = colors[(idx + 1) % len(colors)]


def _paint_clean_tile(
    n: int,
    guide_row: int,
    guide_color: int,
    decor_colors: list[int],
    rng: random.Random,
) -> np.ndarray:
    grid = np.zeros((n, n), dtype=int)
    grid[guide_row, :] = guide_color

    rows = [r for r in range(n) if r != guide_row]
    style = rng.choice(
        [_paint_mirrored_pairs, _paint_stair_runs, _paint_checker_knots]
    )
    style(grid, rows, decor_colors, rng)

    extra_cells = [(r, c) for r in rows for c in range(n) if grid[r, c] == BLACK]
    rng.shuffle(extra_cells)
    for idx, (r, c) in enumerate(extra_cells[: rng.randint(1, 2)]):
        grid[r, c] = decor_colors[(idx + r + c) % len(decor_colors)]

    return grid


def _clean_tile_is_valid(tile: np.ndarray) -> bool:
    guides = _guide_rows(tile)
    if len(guides) != 1:
        return False

    guide_row = guides[0]
    guide_color = int(tile[guide_row, 0])
    used_decor = {
        int(tile[r, c])
        for r in range(tile.shape[0])
        for c in range(tile.shape[1])
        if r != guide_row and int(tile[r, c]) not in (BLACK, guide_color)
    }
    if len(used_decor) < 2:
        return False

    for r in range(tile.shape[0]):
        if r == guide_row:
            continue
        nonzero = int(np.count_nonzero(tile[r, :]))
        if nonzero == 0 or nonzero == tile.shape[1]:
            return False
    return True


def _marker_column_choices(
    clean_tile: np.ndarray,
    guide_row: int,
    marker_count: int,
) -> tuple[list[tuple[int, ...]], dict[int, list[int]]]:
    n = clean_tile.shape[0]
    black_rows_by_col: dict[int, list[int]] = {}
    usable_cols: list[int] = []
    for col in range(n):
        rows = [r for r in range(n) if r != guide_row and int(clean_tile[r, col]) == BLACK]
        if rows:
            usable_cols.append(col)
            black_rows_by_col[col] = rows

    choices: list[tuple[int, ...]] = []
    for cols in itertools.combinations(usable_cols, marker_count):
        erased = _erased_block_columns(n, list(cols), marker_count)
        if 1 <= len(erased) <= n - 2:
            choices.append(tuple(int(c) for c in cols))
    return choices, black_rows_by_col


def _place_markers(
    clean_tile: np.ndarray,
    guide_row: int,
    marker_cols: tuple[int, ...],
    black_rows_by_col: dict[int, list[int]],
    rng: random.Random,
) -> np.ndarray:
    tile = np.array(clean_tile, copy=True)
    for col in marker_cols:
        row = rng.choice(black_rows_by_col[int(col)])
        tile[row, int(col)] = BLUE
    assert np.all(tile[guide_row, :] != BLUE)
    return tile


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.choice([4, 5])
    guide_row = rng.randrange(n)
    guide_color = rng.choice(NON_SPECIAL_COLORS)
    decor_colors = rng.sample([c for c in NON_SPECIAL_COLORS if c != guide_color], k=3)

    clean_tile = _paint_clean_tile(n, guide_row, guide_color, decor_colors, rng)
    if not _clean_tile_is_valid(clean_tile):
        return None

    marker_count = rng.choice([1, 2])
    marker_choices, black_rows_by_col = _marker_column_choices(clean_tile, guide_row, marker_count)
    if not marker_choices:
        return None

    marker_cols = rng.choice(marker_choices)
    tile = _place_markers(clean_tile, guide_row, marker_cols, black_rows_by_col, rng)

    try:
        out = main(tile)
    except AssertionError:
        return None

    if out.ndim != 2 or max(out.shape) > 30:
        return None

    erased = _erased_block_columns(n, list(marker_cols), marker_count)
    if len(erased) > n - 2:
        return None

    return tile


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
