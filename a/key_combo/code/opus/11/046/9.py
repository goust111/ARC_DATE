import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
TEAL = 8
MAROON = 9

PALETTE_COLORS = (BLUE, RED, GREEN, YELLOW, ORANGE, TEAL)
CELL_SIZE = 3
GRID_BLOCK = 11
DIVIDER_ROWS = (3, 7)
DIVIDER_COLS = (3, 7)
SEPARATOR_ROW = 11
GEN_RETRIES = 400
MIN_VARIANT_VISIBLE_RINGS = 2


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _cell_slices() -> tuple[tuple[slice, slice], ...]:
    row_starts = (0, 4, 8)
    col_starts = (0, 4, 8)
    return tuple(
        (slice(r0, r0 + CELL_SIZE), slice(c0, c0 + CELL_SIZE))
        for r0 in row_starts
        for c0 in col_starts
    )


def _extract_layout(grid: np.ndarray) -> tuple[int, int, list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h > GRID_BLOCK

    separator_rows = [
        r for r in range(h) if int(g[r, 0]) == MAGENTA and np.all(g[r] == MAGENTA)
    ]
    assert len(separator_rows) == 1
    separator_row = int(separator_rows[0])
    assert separator_row >= GRID_BLOCK

    grey_box = _bbox(g[:separator_row] == GREY)
    assert grey_box is not None
    r0, c0, r1, c1 = grey_box
    assert (r0, r1 - r0 + 1) == (0, GRID_BLOCK)
    assert c1 - c0 + 1 == GRID_BLOCK

    block = g[r0 : r1 + 1, c0 : c1 + 1]
    full_grey_rows = [r for r in range(GRID_BLOCK) if np.all(block[r] == GREY)]
    full_grey_cols = [c for c in range(GRID_BLOCK) if np.all(block[:, c] == GREY)]
    assert full_grey_rows == list(DIVIDER_ROWS)
    assert full_grey_cols == list(DIVIDER_COLS)

    cells = []
    for rs, cs in _cell_slices():
        cell = block[rs, cs]
        assert cell.shape == (CELL_SIZE, CELL_SIZE)
        cells.append(cell.copy())
    return c0, separator_row, cells


def _guide_palette(cells: list[np.ndarray]) -> tuple[np.ndarray, int]:
    counts = [
        int(np.count_nonzero((cell != BLACK) & (cell != GREY))) for cell in cells
    ]
    min_count = min(counts)
    assert min_count == 3
    assert counts.count(min_count) == 1
    idx = counts.index(min_count)
    guide = cells[idx]
    assert guide.shape == (CELL_SIZE, CELL_SIZE)
    return guide.copy(), idx


def _seed_position(grid: np.ndarray, separator_row: int) -> tuple[int, int]:
    seed_coords = np.argwhere(grid[separator_row + 1 :] == MAROON)
    assert seed_coords.shape == (1, 2)
    sr = int(seed_coords[0, 0]) + separator_row + 1
    sc = int(seed_coords[0, 1])
    return sr, sc


def _render_rings(base: np.ndarray, guide: np.ndarray, separator_row: int, sr: int, sc: int) -> np.ndarray:
    out = np.array(base, dtype=int, copy=True)
    rows, cols = np.indices(out.shape)
    distance = np.maximum(np.abs(rows - sr), np.abs(cols - sc))
    lower_mask = rows > separator_row

    for d in range(1, 9):
        color = int(guide[d // 3, d % 3])
        ring_mask = lower_mask & (distance == d)
        if color == BLACK:
            out[ring_mask] = BLACK
        else:
            out[ring_mask] = color
    out[sr, sc] = MAROON
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Read the uniquely sparse divided cell as a row-major ring palette and expand it around the lower seed pixel.

    Input structure:
    - The grid has black background, a magenta full-width separator row, and above it one 3x3 divided block.
    - Inside that upper block, grey one-pixel divider lines split a 11x11 area into nine 3x3 cells.
    - Exactly one cell contains 3 non-black non-grey pixels; the other eight cells each contain 4.
    - Below the separator there is exactly one maroon seed pixel on otherwise black background.

    Rule:
    - Detect the grey divided 3x3 block and extract its nine 3x3 cells.
    - Find the uniquely sparsest cell and treat its 3x3 pixels as a palette matrix.
    - Flatten that palette in row-major order so position (i, j) maps to Chebyshev distance d = 3*i + j.
    - Keep the upper area unchanged; below the separator, color every pixel at distance d>0 from the seed with that palette entry, leaving black-coded rings blank, and keep the center pixel maroon.

    Key insight:
    - The sparse guide cell is not copied spatially; instead its 2D pixel positions are reinterpreted as ordered ring indices for a concentric square expansion.
    """
    g = np.asarray(grid, dtype=int)
    x0, separator_row, cells = _extract_layout(g)
    del x0
    guide, _ = _guide_palette(cells)
    sr, sc = _seed_position(g, separator_row)
    return _render_rings(g, guide, separator_row, sr, sc)


GUIDE_PATTERNS = (
    ((0, 1), (1, 0), (2, 2)),  # d = 1,3,8
    ((0, 1), (1, 2), (2, 0)),  # d = 1,5,6
    ((0, 2), (1, 0), (2, 1)),  # d = 2,3,7
    ((0, 2), (1, 1), (2, 0)),  # d = 2,4,6
)


def _random_guide_cell() -> np.ndarray:
    cell = np.zeros((CELL_SIZE, CELL_SIZE), dtype=int)
    pattern = random.choice(GUIDE_PATTERNS)
    colors = random.sample(PALETTE_COLORS, 3)
    for (r, c), color in zip(pattern, colors):
        cell[r, c] = int(color)
    return cell


def _random_dense_cell() -> np.ndarray:
    cell = np.zeros((CELL_SIZE, CELL_SIZE), dtype=int)
    positions = random.sample([(r, c) for r in range(CELL_SIZE) for c in range(CELL_SIZE)], 4)
    colors = random.choices(PALETTE_COLORS, k=4)
    for (r, c), color in zip(positions, colors):
        cell[r, c] = int(color)
    return cell


def _place_top_block(grid: np.ndarray, left: int, guide_index: int, guide: np.ndarray) -> None:
    block = np.zeros((GRID_BLOCK, GRID_BLOCK), dtype=int)
    block[DIVIDER_ROWS[0], :] = GREY
    block[DIVIDER_ROWS[1], :] = GREY
    block[:, DIVIDER_COLS[0]] = GREY
    block[:, DIVIDER_COLS[1]] = GREY

    cells = []
    for idx in range(9):
        cells.append(guide.copy() if idx == guide_index else _random_dense_cell())

    for idx, (rs, cs) in enumerate(_cell_slices()):
        block[rs, cs] = cells[idx]

    block[DIVIDER_ROWS[0], :] = GREY
    block[DIVIDER_ROWS[1], :] = GREY
    block[:, DIVIDER_COLS[0]] = GREY
    block[:, DIVIDER_COLS[1]] = GREY
    grid[:GRID_BLOCK, left : left + GRID_BLOCK] = block


def _candidate_seed_positions(h: int, w: int) -> list[tuple[int, int]]:
    lower_top = SEPARATOR_ROW + 1
    candidates = []
    center_r = (lower_top + h - 1) / 2.0
    center_c = (w - 1) / 2.0
    for r in range(lower_top + 4, h - 4):
        for c in range(4, w - 4):
            score = abs(r - center_r) + abs(c - center_c)
            candidates.append((score, r, c))
    candidates.sort(key=lambda item: item[0])
    return [(r, c) for _score, r, c in candidates]


def _visible_ring_count(out: np.ndarray, guide: np.ndarray, separator_row: int, sr: int, sc: int) -> int:
    rows, cols = np.indices(out.shape)
    lower_mask = rows > separator_row
    distance = np.maximum(np.abs(rows - sr), np.abs(cols - sc))
    count = 0
    for d in range(1, 9):
        color = int(guide[d // 3, d % 3])
        if color == BLACK:
            continue
        if np.any(lower_mask & (distance == d) & (out == color)):
            count += 1
    return count


def _ring_signature(
    guide: np.ndarray, separator_row: int, sr: int, sc: int, shape: tuple[int, int]
) -> tuple[tuple[int, int], ...]:
    h, w = shape
    sig = []
    for d in range(1, 9):
        color = int(guide[d // 3, d % 3])
        if color == BLACK:
            continue
        cells = 0
        for r in range(separator_row + 1, h):
            for c in range(w):
                if max(abs(r - sr), abs(c - sc)) == d:
                    cells += 1
        if cells:
            sig.append((d, color))
    return tuple(sig)


def _check_input_structure(inp: np.ndarray, out: np.ndarray) -> None:
    x0, separator_row, cells = _extract_layout(inp)
    assert separator_row == SEPARATOR_ROW
    guide, guide_index = _guide_palette(cells)
    assert 0 <= x0 <= inp.shape[1] - GRID_BLOCK
    assert 0 <= guide_index < 9
    assert int(np.count_nonzero(guide)) == 3
    assert guide[0, 0] == BLACK

    counts = [int(np.count_nonzero((cell != BLACK) & (cell != GREY))) for cell in cells]
    assert Counter(counts)[3] == 1
    assert Counter(counts)[4] == 8

    sr, sc = _seed_position(inp, separator_row)
    assert min(sr - (separator_row + 1), inp.shape[0] - 1 - sr, sc, inp.shape[1] - 1 - sc) >= 4
    assert np.count_nonzero(inp[separator_row + 1 :] == MAROON) == 1
    assert np.all(inp[separator_row + 1 :][inp[separator_row + 1 :] != MAROON] == BLACK)
    assert np.array_equal(out[: separator_row + 1], inp[: separator_row + 1])

    oracle = _render_rings(inp, guide, separator_row, sr, sc)
    assert np.array_equal(out, oracle)
    assert _visible_ring_count(out, guide, separator_row, sr, sc) >= MIN_VARIANT_VISIBLE_RINGS

    sparse_alt = []
    for idx, cell in enumerate(cells):
        if idx == guide_index:
            continue
        alt = _render_rings(inp, cell, separator_row, sr, sc)
        sparse_alt.append(np.array_equal(alt, out))
    assert not any(sparse_alt)

    sig = _ring_signature(guide, separator_row, sr, sc, inp.shape)
    assert len(sig) >= MIN_VARIANT_VISIBLE_RINGS


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(21, 29)
    w = random.randint(13, 25)
    if GRID_BLOCK > w:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[SEPARATOR_ROW, :] = MAGENTA

    left = random.randint(0, w - GRID_BLOCK)
    guide_index = random.randrange(9)
    guide = _random_guide_cell()
    _place_top_block(grid, left, guide_index, guide)

    candidates = _candidate_seed_positions(h, w)
    if not candidates:
        return None
    band = candidates[: max(4, len(candidates) // 3)]
    sr, sc = random.choice(band)
    grid[sr, sc] = MAROON

    try:
        out = main(grid)
        _check_input_structure(grid, out)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid input for puzzle 9")
