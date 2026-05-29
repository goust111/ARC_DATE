# Variant of 1ae2feb7
# Dimension: D2 — 轨迹涂色模式 (trajectory)
# Change: periodic_pattern——以间隔=色条像素数在ext_side上涂单色点（位置0, length, 2*len... → 间歇跳跃（涂一跳一）

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for this D2 variant
# -----------------------------------
# a) Concrete behavior and trace
#    Original task: each bar of length L paints the ext side at periodic hits
#    {0, L, 2L, 3L, ...} measured from the separator outward, with nearer bars
#    processed first and without overwriting.
#    Variant: keep the same separator, same bar parsing, same processing order,
#    same no-overwrite rule, but gate the trajectory with a second cadence:
#    only even-numbered periodic hits survive. So the painted positions become
#    {0, 2L, 4L, ...}; the odd hits {L, 3L, 5L, ...} stay blank unless some
#    farther bar later fills them.
#    Small trace on a left-bar row with a single length-3 bar:
#    ext cells relative to separator are [0 1 2 3 4 5 6 7]
#    original paints indices 0,3,6
#    variant paints indices 0,6
#    visible difference: the middle periodic dot at index 3 disappears.
#
# b) Complexity calibration
#    The rule changes only the trajectory cadence, not the objects. This keeps
#    the task close to the seed while still making the output pattern visibly
#    harder: humans must notice both the bar-length period and the alternating
#    "paint one / skip one" rhythm. To avoid over-simplification, generation
#    still allows 1-2 bars per active row, both separator sides, separator gaps,
#    and multiple bar lengths/colors.
#
# c) Hidden assumptions audit for main()
#    - _find_sep_col(): unchanged assumption; separator remains a mostly full
#      monochrome vertical column.
#    - left/right mass test: unchanged; bars still all live on one side.
#    - _iter_runs_1d(): unchanged; bars are still monochrome horizontal runs.
#    - bars.sort(key=(dist,row,col)): unchanged; priority is still nearest-first.
#    - Original paint loops assumed every periodic hit should be colored. That
#      assumption is invalid now. The only required code change is the stride:
#      from length to 2*length. No-overwrite remains valid and necessary.
#
# d) Placement legality
#    Generation stays output-first. Bars are still packed from the separator
#    outward on one side, never overlap, and ext-side painting only writes into
#    empty ext cells. Clearing the ext side recovers a legal input that the new
#    solver can parse exactly as before.
#
# e) Visual observability
#    A degenerate sample would look identical to the original periodic rule if
#    every row had only one visible periodic hit. To prevent that, the generator
#    forces one witness row to contain a single bar whose length L satisfies
#    ext_side_len > L, guaranteeing an odd periodic slot exists and stays blank
#    in the variant but would be colored in the original task.
#
# f) Visual cleanliness
#    The output remains row-local: no cross-row interference, no new markers,
#    no extra colors. The only visual difference is a cleaner alternating sparse
#    trajectory on affected rows, which is easy to compare against the dense
#    original periodic continuation.
#
# g) Color inference
#    No color remapping is introduced. Colors still identify bars directly, so
#    the human only needs to infer a spatial cadence, not an arbitrary palette
#    rule.
#
# 9) New solver assumptions and generator support
#    The updated solver newly assumes that blank odd periodic slots are allowed
#    and meaningful, rather than evidence of missing paint. The generator
#    enforces this by explicitly simulating the skip-pattern output and by
#    rejecting any sample whose variant output accidentally equals the original
#    periodic output.


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
        if row[c] == bg:
            c += 1
            continue
        col = int(row[c])
        c0 = c
        while c < c_hi and row[c] == col:
            c += 1
        yield c0, c - 1, col


def _paint_extension(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
    *,
    bg: int,
    intermittent_skip: bool,
) -> None:
    """Paint ext-side dots using either the original or variant cadence."""
    w = out.shape[1]
    for length, color in bars:
        if length <= 0:
            continue
        step = length * (2 if intermittent_skip else 1)
        if bars_on_left:
            positions = range(sep_col + 1, w, step)
        else:
            positions = range(sep_col - 1, -1, -step)
        for cc in positions:
            if int(out[row, cc]) == bg:
                out[row, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)

    # Separator structure is unchanged in the variant.
    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    out = g.copy()

    # Bar side vs extension side is still identified by where most non-bg mass is.
    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < w else 0
    bars_on_left = left_mass >= right_mass

    bars = []
    for r in range(h):
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

    # Priority stays nearest-first; only the trajectory cadence changes.
    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    for _, r, _c0, _c1, length, col in bars:
        _paint_extension(
            out,
            r,
            sep_col,
            bars_on_left,
            [(length, col)],
            bg=bg,
            intermittent_skip=True,
        )

    return out


# --------------- generator (variant-aware output-first) ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 2, 14

PALETTE_MIN, PALETTE_MAX = 5, 8

ACTIVE_MIN, ACTIVE_MAX = 2, 5

P_TWO_BARS = 0.45
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5
P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 300


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    """Pick `count` row indices in [0, h) with pairwise spacing >= 2."""
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    """Pick separator color and a disjoint bar-color palette."""
    sep_color = int(rng.integers(1, 10))
    bar_pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(bar_pool)) + 1))
    palette = list(map(int, rng.choice(bar_pool, size=size, replace=False)))
    return sep_color, palette


def _sample_row_bars(
    rng: np.random.Generator,
    palette: list[int],
    primary: int,
    bar_side_len: int,
    *,
    force_witness: bool,
    ext_side_len: int,
) -> tuple[list[tuple[int, int]], int]:
    """Return (bars, sep_gap), with one forced witness row for observability."""
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap

    if force_witness:
        max_visible_len = min(capacity, ext_side_len - 1)
        if max_visible_len < 1:
            return [], sep_gap
        length = int(rng.integers(1, max_visible_len + 1))
        return [(length, primary)], sep_gap

    two_bars = rng.random() < P_TWO_BARS
    if not two_bars:
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
    """Paint bars on the bar side, stacking from nearest-to-separator outward."""
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


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(H_MIN, H_MAX + 1))
    w = int(rng.integers(W_MIN, W_MAX + 1))

    lo = max(BAR_SIDE_MIN, w - 1 - EXT_SIDE_MAX)
    hi = min(BAR_SIDE_MAX, w - 1 - EXT_SIDE_MIN)
    if lo > hi:
        return None
    bar_side_len = int(rng.integers(lo, hi + 1))
    ext_side_len = w - 1 - bar_side_len

    bars_on_left = bool(rng.integers(0, 2) == 0)
    sep_col = bar_side_len if bars_on_left else (w - 1 - bar_side_len)

    sep_color, palette = _sample_palette(rng)

    max_active = min(ACTIVE_MAX, (h + 1) // 2)
    active_count = int(rng.integers(ACTIVE_MIN, max_active + 1))
    active_rows = _choose_active_rows(rng, h, active_count)
    inactive_rows = [r for r in range(h) if r not in set(active_rows)]
    witness_row = active_rows[int(rng.integers(0, len(active_rows)))]

    out = np.zeros((h, w), dtype=int)
    out[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        out[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for idx, row in enumerate(active_rows):
        bars, sep_gap = _sample_row_bars(
            rng,
            palette,
            row_primary[idx],
            bar_side_len,
            force_witness=(row == witness_row),
            ext_side_len=ext_side_len,
        )
        if not bars:
            return None
        row_bars_map[row] = bars
        _place_bars(out, row, sep_col, bars_on_left, bars, sep_gap)

    for row, bars in row_bars_map.items():
        _paint_extension(
            out,
            row,
            sep_col,
            bars_on_left,
            bars,
            bg=0,
            intermittent_skip=True,
        )

    inp = out.copy()
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0

    periodic_out = inp.copy()
    for row, bars in row_bars_map.items():
        _paint_extension(
            periodic_out,
            row,
            sep_col,
            bars_on_left,
            bars,
            bg=0,
            intermittent_skip=False,
        )

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if np.array_equal(periodic_out, out):
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
