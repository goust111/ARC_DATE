import numpy as np

from arc_utils import get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """Repeat the foreground row band vertically to fill the whole grid.

    Input structure
    - The input is a 2D grid with a background color and one contiguous horizontal
      band of rows containing all non-background pixels.
    - Rows outside that band are entirely background.

    Rule
    - Use the full-width foreground band as a vertical tile.
    - Repeat that tile upward and downward over the original canvas, keeping the
      original band row aligned with its input row position.

    Key insight
    - The sparse input is a phase marker for a vertical row pattern, not a set of
      separate objects to move or recolor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    bg = int(get_bg(g))
    active_rows = np.any(g != bg, axis=1)
    rows = np.flatnonzero(active_rows)
    assert rows.size > 0, "grid must contain a foreground row band"

    r0 = int(rows[0])
    r1 = int(rows[-1])
    assert np.array_equal(rows, np.arange(r0, r1 + 1)), "foreground rows must form one contiguous band"
    assert np.all(g[:r0] == bg) and np.all(g[r1 + 1 :] == bg), "rows outside the band must be background"

    tile = g[r0 : r1 + 1, :]
    tile_h = tile.shape[0]
    row_indices = (np.arange(g.shape[0]) - r0) % tile_h
    return tile[row_indices].copy()

# --------------- generator (o5x) ---------------

# Canvas/band geometry; randint high values are exclusive.
HEIGHT_BOUNDS = (10, 16)
WIDTH_BOUNDS = (12, 20)
TILE_HEIGHT = 3
BAND_TOP_PAD = 3
BAND_BOTTOM_PAD = 3

# Motif, color, and accepted foreground-count bounds.
BACKGROUND_COLOR = 0
COLOR_BOUNDS = (1, 10)
PERIOD_BOUNDS = (2, 7)
FG_TOTAL_MIN_NUM = 12
FG_TOTAL_MAX_NUM = 21
FG_TOTAL_DEN = 10

# Oracle-checked retry limit.
MAX_ATTEMPTS = 300

def _gen_row_count(bits: int, period: int, width: int) -> int:
    base = width // period
    rem = width % period
    return sum(base + (1 if j < rem else 0) for j in range(period) if (bits >> j) & 1)


def _gen_has_period(bits_rows: tuple[int, int, int], period: int, shift: int) -> bool:
    for bits in bits_rows:
        for j in range(period):
            if ((bits >> j) & 1) != ((bits >> ((j + shift) % period)) & 1):
                return False
    return True


def _gen_min_period(bits_rows: tuple[int, int, int], period: int) -> int:
    for shift in range(1, period):
        if _gen_has_period(bits_rows, period, shift):
            return shift
    return period


def _gen_tile(width: int, color: int, period: int) -> np.ndarray:
    row_options = [(bits, _gen_row_count(bits, period, width)) for bits in range(1, 1 << period)]
    lo = (FG_TOTAL_MIN_NUM * width + FG_TOTAL_DEN - 1) // FG_TOTAL_DEN
    hi = (FG_TOTAL_MAX_NUM * width) // FG_TOTAL_DEN
    by_total: dict[int, list[tuple[int, int, int]]] = {}

    for bits0, count0 in row_options:
        for bits1, count1 in row_options:
            partial = count0 + count1
            for bits2, count2 in row_options:
                total = partial + count2
                bits_rows = (bits0, bits1, bits2)
                if lo <= total <= hi and _gen_min_period(bits_rows, period) == period:
                    by_total.setdefault(total, []).append(bits_rows)

    if not by_total:
        raise ValueError("no periodic tile choices")

    totals = sorted(by_total)
    target = totals[int(np.random.randint(0, len(totals)))]
    bits_rows = by_total[target][int(np.random.randint(0, len(by_total[target])))]
    cols = np.arange(width) % period
    mask = ((np.asarray(bits_rows)[:, None] >> cols) & 1).astype(bool)
    tile = np.zeros((TILE_HEIGHT, width), dtype=int)
    tile[mask] = color
    return tile


def _gen_attempt() -> np.ndarray | None:
    height = int(np.random.randint(*HEIGHT_BOUNDS))
    width = int(np.random.randint(*WIDTH_BOUNDS))
    color = int(np.random.randint(*COLOR_BOUNDS))
    period = int(np.random.randint(*PERIOD_BOUNDS))
    r0 = int(np.random.randint(BAND_TOP_PAD, height - TILE_HEIGHT - BAND_BOTTOM_PAD + 1))

    tile = _gen_tile(width, color, period)
    row_indices = (np.arange(height) - r0) % tile.shape[0]
    expected = tile[row_indices].copy()

    inp = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    inp[r0 : r0 + tile.shape[0], :] = expected[r0 : r0 + tile.shape[0], :]

    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 8eb1be9a input")
