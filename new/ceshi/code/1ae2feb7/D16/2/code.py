# Variant of 1ae2feb7
# Dimension: D16 — 分隔条颜色的语义作用 (symbol_instructions)
# Change: 分隔条颜色仅作为分隔标志，不携带额外语义信息... → 分隔条像素交替颜色作为行级开关

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for the D16 variant
# ----------------------------------
# a) Concrete behavior and trace
#    The original task has one vertical separator column whose non-background cells all share one color.
#    Bars live entirely on one side of that column, and every bar row extends periodically onto the other side.
#    In this variant the separator column is full-height and alternates between two colors, e.g. A/B/A/B/...
#    The color itself now matters: rows whose separator pixel matches the top separator color A are enabled,
#    and rows whose separator pixel is the other color B are skipped.
#    Small trace:
#      input row r=2:  [111|A|000000]  -> enabled, so output paints positions 0,3,6,... after the separator
#      input row r=3:  [ 22|B|000000]  -> skipped, so output keeps the extension side blank on that row
#    The periodic paint order inside an enabled row is unchanged: nearest bar first, then farther bars, never overwrite.
#
# b) Complexity calibration
#    If rows with separator color B never carried bars, the puzzle would visually collapse back into the old task.
#    So the generator must include both:
#      - enabled bar rows (they extend), and
#      - skipped bar rows (they keep the target side blank despite having bars).
#    The separator uses exactly two colors, repeated many times down one column, so the new signal is visible and low-entropy.
#
# c) Hidden-assumption audit against the original main()
#    - Old _find_sep_col assumed "all non-bg cells in the separator column have the same color". That is now false.
#      New solver instead looks for a full-height column with exactly two colors alternating every row.
#    - Old code processed every row that contained bars. That would be wrong now, because some bar rows must be skipped.
#      New code gates bar collection by the separator color in that row.
#    - The old optional separator gap would make the alternating code and the "top color is active" rule harder to read.
#      The generator therefore keeps the separator continuous with no background gaps.
#
# d) Placement legality
#    Bars still occupy only one side of the separator and remain horizontal monochrome runs.
#    Rows containing bars are spaced by at least one empty row, matching the base task's readability.
#    The generator chooses at least one enabled bar row and at least one skipped bar row, then validates with main().
#
# e) Visual observability
#    Humans can see one vertical column alternating between two colors. Comparing rows with bars reveals:
#      same kind of bars on rows of separator color A extend, rows of separator color B do not.
#    The active/inactive distinction is therefore visible from position and repetition, not from color id ordering.
#
# f) Visual cleanliness
#    The separator stays one-cell wide, bars remain on one side only, and skipped rows simply leave the opposite side empty.
#    No extra markers are introduced, so the new clue is localized and readable rather than cluttered.
#
# g) Color inference
#    Separator colors are disjoint from bar colors. There are only two separator colors.
#    The active color is defined structurally as "the separator color on the top row", so main() can infer it
#    from a single grid, and train/test examples can vary the actual color ids without ambiguity.


def _find_sep_col(g: np.ndarray, bg: int) -> tuple[int | None, int | None]:
    """Find a full-height alternating 2-color separator column; return (col_idx, active_color)."""
    h, w = g.shape
    for c in range(w):
        col = g[:, c]
        if np.any(col == bg):
            continue
        vals = list(map(int, col))
        uniq = set(vals)
        if len(uniq) != 2:
            continue
        if all(vals[r] != vals[r + 1] for r in range(h - 1)):
            return c, vals[0]
    return None, None


def _iter_runs_1d(row: np.ndarray, c_lo: int, c_hi: int, bg: int):
    c = c_lo
    while c < c_hi:
        if row[c] == bg:
            c += 1
            continue
        col = int(row[c])
        c0 = c
        while c < c_hi and row[c] == col:
            c += 1
        yield c0, c - 1, col


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)
    sep_col, active_sep_color = _find_sep_col(g, bg)
    if sep_col is None or active_sep_color is None:
        return g.copy()

    out = g.copy()

    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < w else 0
    bars_on_left = left_mass >= right_mass

    bars = []
    for r in range(h):
        if int(g[r, sep_col]) != active_sep_color:
            continue
        if bars_on_left:
            for c0, c1, col in _iter_runs_1d(g[r, :], 0, sep_col, bg=bg):
                length = c1 - c0 + 1
                dist = (sep_col - 1) - c1
                bars.append((dist, r, c0, c1, length, col))
        else:
            for c0, c1, col in _iter_runs_1d(g[r, :], sep_col + 1, w, bg=bg):
                length = c1 - c0 + 1
                dist = c0 - (sep_col + 1)
                bars.append((dist, r, c0, c1, length, col))

    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    for _, r, _c0, _c1, length, col in bars:
        if length <= 0:
            continue
        if bars_on_left:
            for cc in range(sep_col + 1, w, length):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col
        else:
            for cc in range(sep_col - 1, -1, -length):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col

    return out


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 2, 14

PALETTE_MIN, PALETTE_MAX = 5, 7

BAR_ROWS_MIN, BAR_ROWS_MAX = 3, 5

P_TWO_BARS = 0.45
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5

MAX_ATTEMPTS = 400


def _choose_bar_rows(rng: np.random.Generator, h: int, count: int) -> list[int] | None:
    """Pick spaced rows that include both separator parities."""
    for _ in range(200):
        rows = sorted(map(int, rng.choice(h, size=count, replace=False)))
        if any(rows[i + 1] - rows[i] < 2 for i in range(count - 1)):
            continue
        if len({r % 2 for r in rows}) < 2:
            continue
        return rows
    return None


def _sample_palette(
    rng: np.random.Generator, min_size: int
) -> tuple[tuple[int, int], list[int]]:
    """Pick the two separator colors and a disjoint bar-color palette."""
    sep_colors = tuple(map(int, rng.choice(np.arange(1, 10), size=2, replace=False)))
    bar_pool = [c for c in range(1, 10) if c not in sep_colors]
    lo = max(PALETTE_MIN, min_size)
    hi = min(PALETTE_MAX, len(bar_pool))
    if lo > hi:
        raise ValueError("palette bounds are infeasible")
    size = int(rng.integers(lo, hi + 1))
    palette = list(map(int, rng.choice(bar_pool, size=size, replace=False)))
    return sep_colors, palette


def _sample_row_bars(
    rng: np.random.Generator,
    palette: list[int],
    primary: int,
    bar_side_len: int,
) -> tuple[list[tuple[int, int]], int]:
    """Return (bars, sep_gap) where bars is ordered near->far."""
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap

    if capacity <= 1 or rng.random() >= P_TWO_BARS:
        length = int(rng.integers(1, capacity + 1))
        return [(length, primary)], sep_gap

    near_len = int(rng.integers(1, capacity))
    far_len = int(rng.integers(1, capacity - near_len + 1))
    others = [c for c in palette if c != primary]
    colors = [primary, int(rng.choice(others))]
    if rng.random() < P_SWAP_COLORS:
        colors.reverse()
    return [(near_len, colors[0]), (far_len, colors[1])], sep_gap


def _place_bars(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
    sep_gap: int,
) -> None:
    if bars_on_left:
        cursor = sep_col - sep_gap
        for length, color in bars:
            out[row, cursor - length : cursor] = color
            cursor -= length
    else:
        cursor = sep_col + 1 + sep_gap
        for length, color in bars:
            out[row, cursor : cursor + length] = color
            cursor += length


def _paint_extension(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
) -> None:
    w = out.shape[1]
    for length, color in bars:
        if bars_on_left:
            positions = range(sep_col + 1, w, length)
        else:
            positions = range(sep_col - 1, -1, -length)
        for cc in positions:
            if out[row, cc] == 0:
                out[row, cc] = color


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(H_MIN, H_MAX + 1))
    w = int(rng.integers(W_MIN, W_MAX + 1))

    lo = max(BAR_SIDE_MIN, w - 1 - EXT_SIDE_MAX)
    hi = min(BAR_SIDE_MAX, w - 1 - EXT_SIDE_MIN)
    if lo > hi:
        return None
    bar_side_len = int(rng.integers(lo, hi + 1))

    bars_on_left = bool(rng.integers(0, 2) == 0)
    sep_col = bar_side_len if bars_on_left else (w - 1 - bar_side_len)

    max_rows = min(BAR_ROWS_MAX, (h + 1) // 2)
    if max_rows < BAR_ROWS_MIN:
        return None
    bar_row_count = int(rng.integers(BAR_ROWS_MIN, max_rows + 1))
    bar_rows = _choose_bar_rows(rng, h, bar_row_count)
    if bar_rows is None:
        return None

    enabled_rows = [r for r in bar_rows if r % 2 == 0]
    skipped_rows = [r for r in bar_rows if r % 2 == 1]
    if not enabled_rows or not skipped_rows:
        return None

    sep_colors, palette = _sample_palette(rng, min_size=bar_row_count)

    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        out[r, sep_col] = sep_colors[r % 2]

    row_primary = list(map(int, rng.permutation(palette)[:bar_row_count]))
    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for idx, row in enumerate(bar_rows):
        bars, sep_gap = _sample_row_bars(rng, palette, row_primary[idx], bar_side_len)
        row_bars_map[row] = bars
        _place_bars(out, row, sep_col, bars_on_left, bars, sep_gap)

    for row in enabled_rows:
        _paint_extension(out, row, sep_col, bars_on_left, row_bars_map[row])

    inp = out.copy()
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0

    if get_bg(inp) != 0:
        return None
    present = colors_present(inp, exclude=[0])
    if sep_colors[0] not in present or sep_colors[1] not in present:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
