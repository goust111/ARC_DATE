# Variant of 1ae2feb7
# Dimension: D12 — bar_side方向 (meta)
# Change: 色条在分隔条的左侧（或右侧均可），但所有色条在同一侧... → 色条在分隔条两侧

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for D12
# a) Concrete behavior
#    - The separator is still one nearly-solid vertical monochrome column.
#    - Bars now appear on both sides of that separator within the same puzzle.
#    - To keep the visual rule readable, each active row belongs to exactly one side:
#      left-row = bars only on the left, output dots only on the right;
#      right-row = bars only on the right, output dots only on the left.
#    - Small trace:
#      original row: [11|sep|000000] -> paint rightward every 2 => [11|sep|101010]
#      variant adds independent rows such as [000000|sep|333] -> paint leftward every 3
#      => [300300|sep|333].
#      The full grid therefore contains both left-driven and right-driven rows.
# b) Complexity calibration
#    - Both sides are guaranteed to appear in every generated grid, so the change is visible.
#    - Each row still has only 1-2 adjacent bars, so the local rule stays simple.
#    - Total active rows stay sparse with empty rows between them, preventing clutter.
# c) Old main() assumptions that break under D12
#    - "Decide which side contains the bars" by comparing left/right mass: invalid now.
#    - "Collect bars only from that side": misses the opposite-side rows entirely.
#    - "Paint only one target half": invalid because both halves can receive output.
#    - The no-overwrite rule itself still holds and remains safe.
# d) Placement legality
#    - Separator column keeps >= H-1 cells of one color, so separator detection still works.
#    - Left and right widths are each constrained to at least 4 cells, keeping room for
#      bars and visible periodic dots.
#    - Active rows are split into left-side rows and right-side rows; rows are disjoint,
#      so generator never places source bars on both sides of the same row.
# e) Visual observability
#    - In 2-3 examples a human can see that some rows radiate rightward and some leftward.
#    - No hidden ordering by color id is used; only geometric proximity to separator matters.
# f) Cleanliness
#    - Independent row assignment avoids bars and output dots competing within one row.
#    - Empty spacer rows keep the two side-families easy to parse.
# g) Color inference
#    - Colors are copied directly from bars to their periodic dots; there is no remapping.
# Post-change solver assumptions checked against generator
#    - `_find_sep_col` assumes one nearly-solid monochrome column: generator guarantees it.
#    - Per-row run extraction assumes horizontal monochrome runs: generator uses exactly that.
#    - Distance ordering assumes nearer bars should paint first on their own side: generator
#      places 1-2 runs on a side, optionally with a 1-cell separator gap, so this is valid.


def _find_sep_col(g: np.ndarray, bg: int) -> tuple[int | None, int | None]:
    h, w = g.shape
    for c in range(w):
        col = g[:, c]
        vals = col[col != bg]
        if vals.size >= h - 1 and (vals == vals[0]).all():
            return c, int(vals[0])
    return None, None


def _iter_runs_1d(row: np.ndarray, c_lo: int, c_hi: int, bg: int):
    c = c_lo
    while c < c_hi:
        if int(row[c]) == bg:
            c += 1
            continue
        col = int(row[c])
        c0 = c
        while c < c_hi and int(row[c]) == col:
            c += 1
        yield c0, c - 1, col


def _paint_from_side(
    out: np.ndarray,
    g: np.ndarray,
    sep_col: int,
    bg: int,
    *,
    source_side: str,
) -> None:
    h, w = g.shape
    bars = []
    for r in range(h):
        if source_side == "left":
            for c0, c1, col in _iter_runs_1d(g[r, :], 0, sep_col, bg=bg):
                length = c1 - c0 + 1
                dist = (sep_col - 1) - c1
                bars.append((dist, r, c0, length, col))
        else:
            for c0, c1, col in _iter_runs_1d(g[r, :], sep_col + 1, w, bg=bg):
                length = c1 - c0 + 1
                dist = c0 - (sep_col + 1)
                bars.append((dist, r, c0, length, col))

    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    for _, r, _c0, length, col in bars:
        if length <= 0:
            continue
        if source_side == "left":
            for cc in range(sep_col + 1, w, length):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col
        else:
            for cc in range(sep_col - 1, -1, -length):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    out = g.copy()
    _paint_from_side(out, g, sep_col, bg, source_side="left")
    _paint_from_side(out, g, sep_col, bg, source_side="right")
    return out


# --------------- generator (D12 bilateral variant) ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

SIDE_MIN, SIDE_MAX = 4, 10

PALETTE_MIN, PALETTE_MAX = 5, 8

ACTIVE_TOTAL_MIN, ACTIVE_TOTAL_MAX = 4, 5
SIDE_ROWS_MIN = 2

P_TWO_BARS = 0.45
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5
P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 300


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    bar_pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(bar_pool)) + 1))
    palette = list(map(int, rng.choice(bar_pool, size=size, replace=False)))
    return sep_color, palette


def _sample_row_bars(
    rng: np.random.Generator,
    palette: list[int],
    primary: int,
    side_len: int,
) -> tuple[list[tuple[int, int]], int]:
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = side_len - sep_gap
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
    side: str,
    bars: list[tuple[int, int]],
    sep_gap: int,
) -> None:
    if side == "left":
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
    side: str,
    bars: list[tuple[int, int]],
) -> None:
    w = out.shape[1]
    for length, color in bars:
        if side == "left":
            positions = range(sep_col + 1, w, length)
        else:
            positions = range(sep_col - 1, -1, -length)
        for cc in positions:
            if int(out[row, cc]) == 0:
                out[row, cc] = color


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(H_MIN, H_MAX + 1))
    w = int(rng.integers(W_MIN, W_MAX + 1))

    lo = max(SIDE_MIN, w - 1 - SIDE_MAX)
    hi = min(SIDE_MAX, w - 1 - SIDE_MIN)
    if lo > hi:
        return None
    left_width = int(rng.integers(lo, hi + 1))
    right_width = w - 1 - left_width
    sep_col = left_width

    sep_color, palette = _sample_palette(rng)

    max_active = min(ACTIVE_TOTAL_MAX, (h + 1) // 2)
    if max_active < ACTIVE_TOTAL_MIN:
        return None
    active_total = int(rng.integers(ACTIVE_TOTAL_MIN, max_active + 1))
    left_rows_count = int(rng.integers(SIDE_ROWS_MIN, active_total - SIDE_ROWS_MIN + 1))
    right_rows_count = active_total - left_rows_count

    active_rows = _choose_active_rows(rng, h, active_total)
    left_rows = set(map(int, rng.choice(active_rows, size=left_rows_count, replace=False)))
    right_rows = [r for r in active_rows if r not in left_rows]
    inactive_rows = [r for r in range(h) if r not in set(active_rows)]

    out = np.zeros((h, w), dtype=int)
    out[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        out[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_total]))
    row_bars_map: dict[int, tuple[str, list[tuple[int, int]]]] = {}
    for idx, row in enumerate(active_rows):
        side = "left" if row in left_rows else "right"
        side_len = left_width if side == "left" else right_width
        bars, sep_gap = _sample_row_bars(rng, palette, row_primary[idx], side_len)
        row_bars_map[row] = (side, bars)
        _place_bars(out, row, sep_col, side, bars, sep_gap)

    for row, (side, bars) in row_bars_map.items():
        _paint_extension(out, row, sep_col, side, bars)

    inp = out.copy()
    for row, (side, _bars) in row_bars_map.items():
        if side == "left":
            inp[row, sep_col + 1 :] = 0
        else:
            inp[row, :sep_col] = 0

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if not left_rows or not right_rows:
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
