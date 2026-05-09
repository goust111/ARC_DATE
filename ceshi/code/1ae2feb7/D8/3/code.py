# Variant of 1ae2feb7
# Dimension: D8 — 间隔计算规则 (meta)
# Change: interval = 色条像素数（bar_length）——直接以色条的水平长度作为周期性涂点的步长... → 间隔 = 各色条长度的GCD

from __future__ import annotations

from math import gcd

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for the D8 variant (kept in code so generation/solver stay aligned)
# -----------------------------------------------------------------------------
# a) Concrete behavior and trace
# - Input still has one vertical separator column and 2-5 active rows. Each active row
#   has 1-2 horizontal monochrome bars on exactly one side of the separator.
# - Output still paints dots on the empty side, starting from the cell immediately
#   across the separator, and still processes bars nearest-to-separator first with
#   "do not overwrite".
# - The only rule change is the step selection on a row:
#   original: each bar uses its own length as step.
#   variant: every bar on that row uses the GCD of all bar lengths on that row.
# - Small trace, bars on the left, row bars near->far = [(4, red), (6, blue)]:
#   original:
#     red paints x = 0,4,8,... ; blue paints x = 0,6,12,... but cannot overwrite red,
#     so blue only survives where red missed.
#   variant:
#     gcd(4,6)=2, so both bars target x = 0,2,4,6,... ; red is processed first and
#     occupies every target, so blue often contributes no visible extension.
#   This makes the change visible: denser cadence, and multi-bar rows can collapse to
#   the nearest color even though the farther bar still matters through the GCD.
#
# b) Complexity calibration
# - If generation allowed mostly single bars or pairs like (3,6) where gcd == near_len,
#   many samples would look identical to the original task. To prevent that, the
#   generator deliberately creates at least one "non-trivial GCD row" where:
#     gcd(lengths) > 1 and gcd(lengths) < each length.
# - Valid length pairs are still few and human-readable, so the task stays inferable.
#
# c) Hidden assumptions in the original main()
# - _find_sep_col assumes the separator is the unique near-full column of one color.
#   This remains valid; generation still guarantees it.
# - _iter_runs_1d assumes bars are contiguous monochrome runs. Still valid.
# - The original processing loop used "step = length" per bar. That assumption breaks
#   under D8; step must become a row property, not a bar property.
# - Sorting by distance to separator is still valid because overwrite priority remains
#   nearest-first. Shared row step does not change the priority rule.
#
# d) Placement legality
# - Bars are still placed on one side only, contiguous, within a bar-side width that
#   leaves a non-empty extension side.
# - Two-bar rows are generated with total length <= available side width, so they never
#   overlap or cross the separator. Optional separator gap remains at most 1 cell.
# - The extension side is always empty in the input; output is produced by simulation.
#
# e) Visual observability
# - The generator enforces at least one row where the variant output differs from the
#   original per-bar-step output on the exact same input.
# - Non-trivial GCD rows are chosen so the extension side is long enough to show at
#   least three beats of the shared step, making the denser cadence visible.
#
# f) Visual cleanliness
# - Only one separator, one bar side, and sparse active rows are kept from the original
#   task, so the output stays easy to parse. Overlap happens only conceptually through
#   no-overwrite; the final pixels remain a clean single layer.
#
# g) Color inference
# - No color-id arithmetic is used. Colors only label bars. The changed behavior is
#   entirely geometric: shared spacing derived from visible bar lengths.
#
# h) Reference-task calibration
# - The source task 16de56c4 was consulted as the baseline for introducing a simple
#   arithmetic period (a common step derived from multiple marks) without adding extra
#   object types or hidden metadata. Here the arithmetic operation is GCD on lengths.
#
# i) New solver assumptions introduced by the variant
# - Row step is computed from all runs on that row, so generation must ensure the row's
#   visible runs are exactly the intended bars. This is guaranteed by using 1-2 runs,
#   contiguous when there are 2, with distinct colors so runs do not merge.
# - Because a shared step can let the nearest bar absorb every landing spot, farther
#   bars may be invisible on the extension side. Generation therefore ensures at least
#   one such row exists while keeping the farther bar visible on the bar side.


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


def _gcd_lengths(lengths: list[int]) -> int:
    step = 0
    for length in lengths:
        step = length if step == 0 else gcd(step, length)
    return step


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)
    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    out = g.copy()

    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < w else 0
    bars_on_left = left_mass >= right_mass

    row_bars: dict[int, list[tuple[int, int, int, int, int]]] = {}
    for r in range(h):
        runs = []
        if bars_on_left:
            for c0, c1, col in _iter_runs_1d(g[r, :], 0, sep_col, bg=bg):
                length = c1 - c0 + 1
                dist = (sep_col - 1) - c1
                runs.append((dist, c0, c1, length, col))
        else:
            for c0, c1, col in _iter_runs_1d(g[r, :], sep_col + 1, w, bg=bg):
                length = c1 - c0 + 1
                dist = c0 - (sep_col + 1)
                runs.append((dist, c0, c1, length, col))
        if runs:
            row_bars[r] = runs

    bars = []
    row_steps = {}
    for r, runs in row_bars.items():
        row_steps[r] = _gcd_lengths([length for _dist, _c0, _c1, length, _col in runs])
        for dist, c0, c1, length, col in runs:
            bars.append((dist, r, c0, c1, length, col))

    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    for _dist, r, _c0, _c1, _length, col in bars:
        step = row_steps[r]
        if step <= 0:
            continue
        if bars_on_left:
            for cc in range(sep_col + 1, w, step):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col
        else:
            for cc in range(sep_col - 1, -1, -step):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col

    return out


# --------------- generator for D8 variant ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

# Raised minimum widths are a targeted D8 adaptation: non-trivial GCD rows need
# enough bar-side room to contain pairs like (2,4) and enough extension room to
# display the denser shared cadence.
BAR_SIDE_MIN, BAR_SIDE_MAX = 6, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 5, 14

PALETTE_MIN, PALETTE_MAX = 5, 8
ACTIVE_MIN, ACTIVE_MAX = 2, 5

P_TWO_BARS = 0.7
P_NONTRIVIAL_TWO_BAR = 0.7
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5
P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 400


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    bar_pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(bar_pool)) + 1))
    palette = list(map(int, rng.choice(bar_pool, size=size, replace=False)))
    return sep_color, palette


def _valid_nontrivial_pairs(capacity: int, ext_side_len: int) -> list[tuple[int, int]]:
    pairs = []
    for near_len in range(2, capacity):
        for far_len in range(2, capacity - near_len + 1):
            step = gcd(near_len, far_len)
            if step <= 1:
                continue
            if step >= near_len or step >= far_len:
                continue
            if ext_side_len < 2 * step + 1:
                continue
            pairs.append((near_len, far_len))
    return pairs


def _sample_row_bars(
    rng: np.random.Generator,
    palette: list[int],
    primary: int,
    bar_side_len: int,
    ext_side_len: int,
    *,
    force_nontrivial: bool = False,
) -> tuple[list[tuple[int, int]], int, bool] | None:
    """Return (bars, sep_gap, is_nontrivial_gcd_row), bars ordered near->far."""
    gap_choices = [0, 1] if force_nontrivial else ([1, 0] if rng.random() < P_SEP_GAP else [0, 1])
    non_primary = [c for c in palette if c != primary]
    if not non_primary:
        return None

    for sep_gap in gap_choices:
        capacity = bar_side_len - sep_gap
        if capacity <= 0:
            continue

        nontrivial_pairs = _valid_nontrivial_pairs(capacity, ext_side_len)
        use_two_bars = force_nontrivial or (rng.random() < P_TWO_BARS and capacity >= 2)
        if not use_two_bars:
            length = int(rng.integers(1, capacity + 1))
            return [(length, primary)], sep_gap, False

        want_nontrivial = force_nontrivial or (
            nontrivial_pairs and rng.random() < P_NONTRIVIAL_TWO_BAR
        )
        if want_nontrivial and nontrivial_pairs:
            near_len, far_len = nontrivial_pairs[int(rng.integers(0, len(nontrivial_pairs)))]
            colors = [primary, int(rng.choice(non_primary))]
            if rng.random() < P_SWAP_COLORS:
                colors.reverse()
            return [(near_len, colors[0]), (far_len, colors[1])], sep_gap, True

        if capacity < 2:
            continue
        near_len = int(rng.integers(1, capacity))
        far_len = int(rng.integers(1, capacity - near_len + 1))
        colors = [primary, int(rng.choice(non_primary))]
        if rng.random() < P_SWAP_COLORS:
            colors.reverse()
        is_nontrivial = gcd(near_len, far_len) > 1 and gcd(near_len, far_len) < min(near_len, far_len)
        return [(near_len, colors[0]), (far_len, colors[1])], sep_gap, is_nontrivial

    return None


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


def _paint_extension_variant(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
) -> None:
    w = out.shape[1]
    step = _gcd_lengths([length for length, _color in bars])
    for _length, color in bars:
        if bars_on_left:
            positions = range(sep_col + 1, w, step)
        else:
            positions = range(sep_col - 1, -1, -step)
        for cc in positions:
            if out[row, cc] == 0:
                out[row, cc] = color


def _paint_extension_original(
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
    ext_side_len = w - 1 - bar_side_len

    bars_on_left = bool(rng.integers(0, 2) == 0)
    sep_col = bar_side_len if bars_on_left else (w - 1 - bar_side_len)

    sep_color, palette = _sample_palette(rng)

    max_active = min(ACTIVE_MAX, (h + 1) // 2)
    active_count = int(rng.integers(ACTIVE_MIN, max_active + 1))
    active_rows = _choose_active_rows(rng, h, active_count)
    inactive_rows = [r for r in range(h) if r not in set(active_rows)]

    inp = np.zeros((h, w), dtype=int)
    inp[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        inp[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    forced_row = active_rows[int(rng.integers(0, len(active_rows)))]

    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    nontrivial_rows = 0
    for idx, row in enumerate(active_rows):
        sample = _sample_row_bars(
            rng,
            palette,
            row_primary[idx],
            bar_side_len,
            ext_side_len,
            force_nontrivial=(row == forced_row),
        )
        if sample is None:
            return None
        bars, sep_gap, is_nontrivial = sample
        row_bars_map[row] = bars
        nontrivial_rows += int(is_nontrivial)
        _place_bars(inp, row, sep_col, bars_on_left, bars, sep_gap)

    out_variant = inp.copy()
    out_original = inp.copy()
    for row, bars in row_bars_map.items():
        _paint_extension_variant(out_variant, row, sep_col, bars_on_left, bars)
        _paint_extension_original(out_original, row, sep_col, bars_on_left, bars)

    if nontrivial_rows == 0:
        return None
    if np.array_equal(out_variant, out_original):
        return None
    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if not np.array_equal(main(inp), out_variant):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
