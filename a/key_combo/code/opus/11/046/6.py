from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

ARRAY_N = 3
TILE_SIZE = 4
SEP = 1
FRAME_SIZE = ARRAY_N * TILE_SIZE + (ARRAY_N + 1) * SEP  # 16
LOCAL_CENTER = (1, 1)
MAX_GEN_ATTEMPTS = 300


def _separator_lines_1d(values: np.ndarray) -> list[int]:
    lines: list[int] = []
    for idx in range(values.shape[0]):
        line = values[idx]
        if np.count_nonzero(line == GREY) >= FRAME_SIZE and np.all(
            (line == BG) | (line == GREY)
        ):
            lines.append(idx)
    return lines


def _locate_array(grid: np.ndarray) -> tuple[list[int], list[int]]:
    row_lines = _separator_lines_1d(grid)
    col_lines = _separator_lines_1d(grid.T)
    assert len(row_lines) == 4 and len(col_lines) == 4, "expected one framed 3x3 tile array"
    assert all(row_lines[i + 1] - row_lines[i] == TILE_SIZE + SEP for i in range(3))
    assert all(col_lines[i + 1] - col_lines[i] == TILE_SIZE + SEP for i in range(3))
    return row_lines, col_lines


def _tile_bounds(row_lines: list[int], col_lines: list[int], tr: int, tc: int) -> tuple[int, int]:
    return row_lines[tr] + 1, col_lines[tc] + 1


def _hole_local(block: np.ndarray) -> tuple[int, int]:
    holes = np.argwhere(block == BG)
    assert holes.shape == (1, 2), "each tile must contain exactly one black hole"
    return int(holes[0, 0]), int(holes[0, 1])


def _validate_tile(block: np.ndarray, tr: int, tc: int) -> tuple[int, int]:
    assert block.shape == (TILE_SIZE, TILE_SIZE)
    hr, hc = _hole_local(block)
    assert (hr, hc) == (tr, tc), "hole must encode the tile array coordinate"

    expected = np.full((TILE_SIZE, TILE_SIZE), BLUE, dtype=int)
    expected[hr, hc] = BG
    if (tr, tc) != LOCAL_CENTER:
        expected[LOCAL_CENTER] = RED

    assert np.array_equal(block, expected), "unexpected tile contents"
    return hr, hc


def _ray_counts(grid: np.ndarray) -> np.ndarray:
    row_lines, col_lines = _locate_array(grid)
    counts = np.zeros(grid.shape, dtype=int)

    for tr in range(ARRAY_N):
        for tc in range(ARRAY_N):
            r0, c0 = _tile_bounds(row_lines, col_lines, tr, tc)
            block = grid[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE]
            hr, hc = _validate_tile(block, tr, tc)
            dr = LOCAL_CENTER[0] - hr
            dc = LOCAL_CENTER[1] - hc
            if dr == 0 and dc == 0:
                continue

            r = r0 + LOCAL_CENTER[0]
            c = c0 + LOCAL_CENTER[1]
            while True:
                r += dr
                c += dc
                if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
                    break
                if grid[r, c] == GREY:
                    counts[r, c] += 1

    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor separator-lattice pixels hit by hole-directed rays from the one-hole tiles.

    Input structure
    The grid has a black background containing a framed 3x3 array of 4x4 blue
    tiles. A one-pixel grey corridor network separates the tiles and also forms
    the array's outer frame. Each tile has exactly one black hole whose local
    position equals the tile's array coordinate; every non-central tile also has
    a red launcher at local position (1,1), while the central tile's hole
    occupies that same center cell.

    Rule
    Read each tile's hole offset relative to the launcher center. For every
    non-central tile, emit a ray from the local center in the opposite direction
    of that offset, stepping by the corresponding row/column delta across the
    whole grid. Only grey corridor pixels touched by exactly one ray turn green;
    grey corridor pixels touched by two or more rays turn yellow. Tile interiors
    and all non-grey pixels remain unchanged.

    Key insight
    The hole is not just an identity label for the tile position: its offset
    from the red center becomes a direction vector, and the output is determined
    by where those directional rays overlap on the shared separator lattice.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    counts = _ray_counts(g)
    out = g.copy()
    out[(g == GREY) & (counts == 1)] = GREEN
    out[(g == GREY) & (counts >= 2)] = YELLOW
    return out


def _build_base_grid() -> np.ndarray:
    grid = np.full((FRAME_SIZE, FRAME_SIZE), BG, dtype=int)

    lines = [0, 5, 10, 15]
    for r in lines:
        grid[r, :] = GREY
    for c in lines:
        grid[:, c] = GREY

    for tr in range(ARRAY_N):
        for tc in range(ARRAY_N):
            r0 = lines[tr] + 1
            c0 = lines[tc] + 1
            grid[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE] = BLUE
            grid[r0 + tr, c0 + tc] = BG
            if (tr, tc) != LOCAL_CENTER:
                grid[r0 + LOCAL_CENTER[0], c0 + LOCAL_CENTER[1]] = RED

    return grid


def _pad_grid(grid: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
    out = np.full((grid.shape[0] + top + bottom, grid.shape[1] + left + right), BG, dtype=int)
    out[top : top + grid.shape[0], left : left + grid.shape[1]] = grid
    return out


def _is_non_degenerate(inp: np.ndarray, out: np.ndarray) -> bool:
    if inp.shape != out.shape or inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if not np.array_equal(inp[(inp != GREY)], out[(inp != GREY)]):
        return False

    changed = inp != out
    if np.count_nonzero(changed) != 18:
        return False
    if np.count_nonzero(out == GREEN) != 8:
        return False
    if np.count_nonzero(out == YELLOW) != 10:
        return False
    if not np.all(inp[changed] == GREY):
        return False

    counts = _ray_counts(inp)
    if np.count_nonzero((inp == GREY) & (counts == 1)) != 8:
        return False
    if np.count_nonzero((inp == GREY) & (counts >= 2)) != 10:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    base = _build_base_grid()
    top = random.randint(0, 4)
    bottom = random.randint(0, 4)
    left = random.randint(0, 4)
    right = random.randint(0, 4)
    inp = _pad_grid(base, top, bottom, left, right)

    try:
        out = main(inp)
    except Exception:
        return None

    if not _is_non_degenerate(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
