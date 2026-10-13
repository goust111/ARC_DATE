import numpy as np


def _placeholder_suffix(g: np.ndarray) -> tuple[int, int, int]:
    H, W = g.shape
    candidates: list[tuple[int, int, int]] = []
    for color in np.unique(g):
        color = int(color)
        rows_all = np.all(g == color, axis=1)
        cols_all = np.all(g == color, axis=0)
        if not rows_all[-1] or not cols_all[-1]:
            continue

        r_cut = H
        while r_cut > 0 and rows_all[r_cut - 1]:
            r_cut -= 1
        c_cut = W
        while c_cut > 0 and cols_all[c_cut - 1]:
            c_cut -= 1
        if r_cut == 0 or c_cut == 0:
            continue

        expected = np.zeros((H, W), dtype=bool)
        expected[r_cut:, :] = True
        expected[:, c_cut:] = True
        if np.array_equal(g == color, expected):
            candidates.append((color, r_cut, c_cut))

    assert len(candidates) == 1
    return candidates[0]


def _minimal_periods(block: np.ndarray) -> tuple[int, int]:
    h, w = block.shape

    row_period = None
    for period in range(1, h + 1):
        if all(block[r, c] == block[r % period, c] for r in range(h) for c in range(w)):
            row_period = period
            break

    col_period = None
    for period in range(1, w + 1):
        if all(block[r, c] == block[r, c % period] for r in range(h) for c in range(w)):
            col_period = period
            break

    assert row_period is not None and col_period is not None
    assert row_period < h and col_period < w
    return int(row_period), int(col_period)


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the repeating pattern through the placeholder suffix, then shift it one column left.

    Input structure
    - The grid contains a single placeholder color forming the union of bottom suffix rows and right suffix columns.
    - The remaining top-left block is a repeated rectangular color tile.

    Rule
    - Infer the minimal tile from the top-left block and extend it over the full input shape.
    - Replace each cell by the color one column to its right in that completed repeating pattern.

    Key insight
    - The placeholder region is not background to preserve; it hides the continuation of the same tile, and all visible tile colors advance by the same horizontal step.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape

    placeholder, r_cut, c_cut = _placeholder_suffix(g)
    block = g[:r_cut, :c_cut]
    assert placeholder not in set(map(int, block.flat))

    row_period, col_period = _minimal_periods(block)
    tile = block[:row_period, :col_period]
    assert len(set(map(int, tile.flat))) >= 2

    out = np.empty_like(g)
    for r in range(H):
        for c in range(W):
            out[r, c] = tile[r % row_period, (c + 1) % col_period]
    return out

# --------------- generator (o5x) ---------------

GRID_SIZE_RANGE = (6, 21)  # Square canvas size; randint high is exclusive.
TILE_WIDTH_RANGE = (2, 5)  # Tile column period; randint high is exclusive.
COLOR_POOL = np.arange(1, 10)  # Tile colors plus one placeholder.
ROW_SHIFT, OUTPUT_COL_SHIFT = -1, 1  # Row shift and output column lookahead.
MIN_SUFFIX, MAX_SUFFIX = 1, 9  # Hidden suffix row/col bounds.
GENERATION_RETRIES = 300  # Oracle-checked generation attempts.

def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.randint(*GRID_SIZE_RANGE))
    tile_w = int(np.random.randint(*TILE_WIDTH_RANGE))

    colors = np.random.choice(COLOR_POOL, size=tile_w + 1, replace=False)
    row0 = colors[:tile_w].astype(int)
    placeholder = int(colors[tile_w])
    tile = np.vstack([row0, np.roll(row0, ROW_SHIFT)])

    rr = np.arange(n)[:, None] % tile.shape[0]
    cc = np.arange(n)[None, :] % tile_w
    out = tile[rr, (cc + OUTPUT_COL_SHIFT) % tile_w]

    base = tile[rr, cc]
    max_suffix_rows = min(MAX_SUFFIX, n - tile.shape[0] - 1)
    max_suffix_cols = min(MAX_SUFFIX, n - tile_w - 1)
    suffix_rows = int(np.random.randint(MIN_SUFFIX, max_suffix_rows + 1))
    suffix_cols = int(np.random.randint(MIN_SUFFIX, max_suffix_cols + 1))
    r_cut = n - suffix_rows
    c_cut = n - suffix_cols

    inp = np.full((n, n), placeholder, dtype=int)
    inp[:r_cut, :c_cut] = base[:r_cut, :c_cut]

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError('failed to generate valid input')
