# Combo variant of 1ae2feb7 (index 30)
# combo_spec: D9/2+D16/2
# Name: 有缺口且交替颜色的分隔条

from __future__ import annotations

import numpy as np

from arc_utils import get_bg
from arc_utils import get_bg as _bg_color

# -----------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior and trace
#    Original task:
#      - one almost-solid 1-color separator column
#      - every bar row on one side emits periodic paint to the blank opposite side
#    Combo task:
#      - the separator column is sparse: some rows have a separator pixel, some are
#        true holes
#      - visible separator pixels alternate between two colors as you scan top->down
#      - the first visible separator color is the active phase A; the next visible
#        separator pixel is color B and is inactive; then A again; etc.
#      - a row emits only if two conditions both hold:
#          1) the separator cell exists on that row
#          2) that separator cell has the active phase color A
#    Small trace, bars on the left:
#      - row with bar `333|A|000000`   -> emit at offsets 0,3,6,...
#      - row with bar `444|B|000000`   -> separator exists but skip entirely
#      - row with bar `555|0|000000`   -> separator hole, also skip entirely
#    The periodic paint rule itself is unchanged; only the row-activation logic is
#    fused from the physical hole signal and the color-phase signal.
#
# b) Complexity calibration
#    The generator forces all three bar-row categories to appear in the same input:
#      - active separator color rows that do paint
#      - inactive separator color rows that do not paint
#      - hole rows that do not paint
#    This prevents collapse into either single variant. The separator keeps only two
#    colors and 4-6 visible pixels so the phase pattern is repeated enough to read.
#
# c) Hidden-assumption audit against the original main()
#    - Original `_find_sep_col` expected one mostly solid monochrome column.
#      Broken here: the combo separator has many holes and two colors.
#    - Original main processed every bar row.
#      Broken here: rows with inactive color or a hole must be skipped.
#    - Original solver could infer the separator from "almost full height".
#      Broken here: we instead exploit a different invariant guaranteed by the
#      generator: the extension side is completely blank in the input, so the true
#      separator is the unique non-empty column bordering that blank half-plane.
#    - New solver assumption introduced: visible separator pixels alternate in color
#      when read top->down, and the first visible color is the active phase.
#      The generator constructs exactly that phase pattern.
#
# d) Placement legality
#    - Bars stay entirely on one side of the separator position.
#    - The blank extension side is cleared completely in the input.
#    - Bar rows are spaced by at least one empty row, so rows do not merge visually.
#    - Extra separator-only rows may exist, but hole bar rows are never reused as
#      separator rows, so the "missing separator" cue stays physically real.
#
# e) Visual observability
#    A human can compare bar rows and notice:
#      - some rows with a visible separator pixel still fail to emit, and those rows
#        share the same separator color
#      - some other bar rows fail because the separator pixel is literally absent
#    Because both failure modes recur in the same examples, the fused rule is
#    recoverable from 2-4 demonstrations without using hidden numeric color order.
#
# f) Visual cleanliness
#    The only overlaid structures are the standard bars and one 1-cell separator
#    column. No extra markers are added. Holes appear as blank cells in the same
#    column, and the output only fills the already-empty extension side, keeping the
#    scene easy to parse.
#
# g) Change inferability and anti-degeneracy
#    The two separator colors are disjoint from bar colors. Their semantic roles are
#    structural, not based on color ids: the topmost visible separator color is
#    always the active phase. Generator validation also checks that:
#      - at least one inactive-color bar row exists
#      - at least one hole bar row exists
#      - both categories would have produced non-empty extension paint if enabled
#    Therefore the generated samples cannot quietly reduce to the original task or to
#    only one of the two constituent single-variant behaviors.
# -----------------------------------------------------------------------------


def _find_sep_info(g: np.ndarray, bg: int) -> tuple[int | None, int | None, bool | None]:
    """Locate the sparse alternating separator and infer its active color."""
    h, w = g.shape
    for c in range(w):
        col = g[:, c]
        present_rows = np.flatnonzero(col != bg)
        if len(present_rows) < 3 or len(present_rows) >= h:
            continue

        left_blank = c > 0 and np.all(g[:, :c] == bg)
        right_blank = c + 1 < w and np.all(g[:, c + 1 :] == bg)
        if right_blank and not left_blank:
            bars_on_left = True
        elif left_blank and not right_blank:
            bars_on_left = False
        else:
            continue

        vals = [int(col[r]) for r in present_rows]
        if len(set(vals)) != 2:
            continue
        if any(vals[i] == vals[i + 1] for i in range(len(vals) - 1)):
            continue
        return c, vals[0], bars_on_left
    return None, None, None


def _iter_runs_1d(row: np.ndarray, c_lo: int, c_hi: int, bg: int):
    c = c_lo
    while c < c_hi:
        if row[c] == bg:
            c += 1
            continue
        color = int(row[c])
        c0 = c
        while c < c_hi and row[c] == color:
            c += 1
        yield c0, c - 1, color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overview:
        A broken vertical separator alternates between two colors. A bar row extends
        periodically only when its separator cell both exists and matches the active
        phase color.

    Input structure:
        One side of the separator contains 1-2 horizontal monochrome bars on several
        spaced rows. The opposite side is entirely background in the input. The
        separator column itself contains holes and visible pixels in two alternating
        colors.

    Rule:
        Find the separator column. Read its visible pixels from top to bottom; the
        first visible color is active and the alternating second color is inactive.
        For each row whose separator cell exists and has the active color, copy that
        row's bars onto the blank side with the original periodic nearest-first,
        no-overwrite rule. Rows with the inactive separator color or a hole do
        nothing.

    Key insight:
        Row activation is controlled by a conjunction of two cues: separator geometry
        (present vs hole) and separator phase color (active vs inactive).
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape

    bg = _bg_color(g)
    sep_col, active_sep_color, bars_on_left = _find_sep_info(g, bg)
    if sep_col is None or active_sep_color is None or bars_on_left is None:
        return g.copy()

    out = g.copy()

    bars = []
    for r in range(g.shape[0]):
        if int(g[r, sep_col]) != active_sep_color:
            continue
        if bars_on_left:
            for c0, c1, col in _iter_runs_1d(g[r, :], 0, sep_col, bg=bg):
                length = c1 - c0 + 1
                dist = (sep_col - 1) - c1
                bars.append((dist, r, length, col))
        else:
            for c0, c1, col in _iter_runs_1d(g[r, :], sep_col + 1, w, bg=bg):
                length = c1 - c0 + 1
                dist = c0 - (sep_col + 1)
                bars.append((dist, r, length, col))

    bars.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    for _, r, length, col in bars:
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

BAR_ROWS_MIN, BAR_ROWS_MAX = 4, 5
SEP_ROWS_MIN, SEP_ROWS_MAX = 4, 6
PALETTE_MIN, PALETTE_MAX = 5, 7

P_TWO_BARS = 0.45
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5

MAX_ATTEMPTS = 500


def _choose_spaced_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    """Pick `count` row indices with pairwise spacing >= 2."""
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(
    rng: np.random.Generator, min_size: int
) -> tuple[tuple[int, int], list[int]]:
    """Pick two separator colors and a disjoint bar-color palette."""
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
    """Return bars as (length, color) near->far, plus a 0/1 gap to the separator."""
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


def _extension_paint_count(
    width: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
) -> int:
    probe = np.zeros((1, width), dtype=int)
    _paint_extension(probe, 0, sep_col, bars_on_left, bars)
    if bars_on_left:
        return int(np.count_nonzero(probe[0, sep_col + 1 :]))
    return int(np.count_nonzero(probe[0, :sep_col]))


def _choose_separator_layout(
    rng: np.random.Generator,
    h: int,
    bar_rows: list[int],
    hole_bar_rows: set[int],
) -> tuple[list[int], list[int], list[int]] | None:
    visible_bar_rows = sorted(r for r in bar_rows if r not in hole_bar_rows)
    first_visible = visible_bar_rows[0]
    extra_candidates = [r for r in range(first_visible + 1, h) if r not in bar_rows]
    sep_total_lo = max(SEP_ROWS_MIN, len(visible_bar_rows))
    sep_total_hi = min(SEP_ROWS_MAX, h - 2)
    if sep_total_lo > sep_total_hi:
        return None

    for _ in range(200):
        sep_total = int(rng.integers(sep_total_lo, sep_total_hi + 1))
        extra_count = sep_total - len(visible_bar_rows)
        if extra_count > len(extra_candidates):
            continue

        extras: list[int]
        if extra_count == 0:
            extras = []
        else:
            extras = sorted(map(int, rng.choice(extra_candidates, size=extra_count, replace=False)))
        sep_rows = sorted(visible_bar_rows + extras)
        if sep_rows[0] != first_visible:
            continue

        active_bar_rows = []
        inactive_bar_rows = []
        visible_bar_row_set = set(visible_bar_rows)
        for idx, row in enumerate(sep_rows):
            if row not in visible_bar_row_set:
                continue
            if idx % 2 == 0:
                active_bar_rows.append(row)
            else:
                inactive_bar_rows.append(row)

        if len(active_bar_rows) >= 2 and len(inactive_bar_rows) >= 1:
            return sep_rows, active_bar_rows, inactive_bar_rows
    return None


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

    max_bar_rows = min(BAR_ROWS_MAX, (h + 1) // 2)
    if max_bar_rows < BAR_ROWS_MIN:
        return None
    bar_row_count = int(rng.integers(BAR_ROWS_MIN, max_bar_rows + 1))
    bar_rows = _choose_spaced_rows(rng, h, bar_row_count)

    max_hole_bar_rows = max(1, bar_row_count - 3)
    hole_bar_count = int(rng.integers(1, max_hole_bar_rows + 1))
    hole_bar_rows = set(map(int, rng.choice(bar_rows, size=hole_bar_count, replace=False)))

    layout = _choose_separator_layout(rng, h, bar_rows, hole_bar_rows)
    if layout is None:
        return None
    sep_rows, active_bar_rows, inactive_bar_rows = layout

    sep_colors, palette = _sample_palette(rng, min_size=bar_row_count)

    out = np.zeros((h, w), dtype=int)
    for idx, row in enumerate(sep_rows):
        out[row, sep_col] = sep_colors[idx % 2]

    row_primary = list(map(int, rng.permutation(palette)[:bar_row_count]))
    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for idx, row in enumerate(bar_rows):
        bars, sep_gap = _sample_row_bars(rng, palette, row_primary[idx], bar_side_len)
        row_bars_map[row] = bars
        _place_bars(out, row, sep_col, bars_on_left, bars, sep_gap)

    for row in active_bar_rows:
        _paint_extension(out, row, sep_col, bars_on_left, row_bars_map[row])

    inp = out.copy()
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0

    hole_effect = sum(
        _extension_paint_count(w, sep_col, bars_on_left, row_bars_map[row])
        for row in bar_rows
        if row in hole_bar_rows
    )
    color_effect = sum(
        _extension_paint_count(w, sep_col, bars_on_left, row_bars_map[row])
        for row in inactive_bar_rows
    )

    if get_bg(inp) != 0:
        return None
    if hole_effect <= 0 or color_effect <= 0:
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
