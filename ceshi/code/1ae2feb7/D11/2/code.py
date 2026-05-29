# Variant of 1ae2feb7
# Dimension: D11 — 涂色起始偏移 (trajectory)
# Change: 从位置0开始涂色（ext_side紧邻分隔条的第一个格子），序列为0, length, 2*length, ...... → 不同色条的涂色起始点交错排列

from __future__ import annotations

# Design analysis
# - Concrete behavior:
#   The input still contains one vertical separator column and 1-2 horizontal monochrome bars
#   on the same side of that separator. The output still extends each active row onto the
#   opposite side by painting periodic dots and still processes bars nearest-to-separator first
#   without overwriting existing color. The only intended change is the trajectory start offset
#   for rows that contain multiple bars: within one row, the first processed bar starts at ext
#   position 0, while the second processed bar starts at floor(length/2). With at most two bars
#   per row in this generator, that creates an alternating stripe phase shift instead of both
#   colors aligning on the separator-adjacent cell.
# - Small trace:
#   Suppose the separator is at column 5, bars are on the left, and a row has near bar
#   length 3 color A plus far bar length 4 color B. Original rule paints A at ext offsets
#   0,3,6,... and B at 0,4,8,..., skipping already-filled cells. Variant paints A at 0,3,6,...
#   but paints B at 2,6,10,... because floor(4/2)=2. The order and no-overwrite rule stay the
#   same; only the initial phase of the second bar changes.
# - Complexity calibration:
#   The new effect is only visible when a row has at least two bars and the second bar's half-
#   length offset is non-zero. The generator therefore guarantees at least one two-bar row per
#   sample and requires the second bar length to be >=2 on such rows. This keeps the task close
#   to the source but prevents degeneration back to the original visual pattern.
# - Implicit assumptions in the original solver:
#   `_find_sep_col`, the left/right mass heuristic, run extraction, and nearest-first sorting
#   remain valid because geometry is unchanged. The original paint loop implicitly assumed every
#   bar starts at ext offset 0; that is the only assumption we must break. Because bars are
#   globally sorted across rows, the solver must now also track each row's local bar index to
#   know whether that row's current bar is the first or second one.
# - Placement legality:
#   Bars still occupy only the bar side, remain contiguous, do not overlap, and stay within
#   `bar_side_len`. The generator samples bar lengths so total occupied width fits even when
#   a separator gap exists. Output is still generated first, then ext side is cleared to form
#   the input, and the result is validated by the solver.
# - Visual observability / neatness:
#   No hidden property such as color id participates in ordering. The only control variable is
#   the visible order of bars by proximity to the separator. Active rows stay separated by at
#   least one empty row, so the staggered extension remains readable instead of becoming a dense
#   overlay. The validator rejects samples where the new rule accidentally matches the original.
# - New solver assumptions after the change:
#   For any row, bars must be discoverable in nearest-to-farthest order so the local index is
#   meaningful; the generator guarantees this by placing bars contiguously from the separator
#   outward. The ext side must be wide enough that at least one staggered row differs from the
#   old rule; the generator checks that directly before accepting a sample.

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color


def _find_sep_col(g, bg):
    h, w = g.shape
    for c in range(w):
        col = g[:, c]
        vals = col[col != bg]
        if vals.size >= h - 1 and (vals == vals[0]).all():
            return c, int(vals[0])
    return None, None


def _iter_runs_1d(row, c_lo, c_hi, bg):
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


def _bar_offset(length: int, bar_index: int) -> int:
    """Alternate the phase per row: 0, floor(length/2), 0, floor(length/2), ..."""
    return 0 if bar_index % 2 == 0 else length // 2


def _collect_bars(g: np.ndarray, sep_col: int, bars_on_left: bool, bg: int):
    bars = []
    h, w = g.shape
    for r in range(h):
        if bars_on_left:
            runs = _iter_runs_1d(g[r, :], 0, sep_col, bg=bg)
            for c0, c1, col in runs:
                length = c1 - c0 + 1
                dist = (sep_col - 1) - c1
                bars.append((dist, r, c0, c1, length, col))
        else:
            runs = _iter_runs_1d(g[r, :], sep_col + 1, w, bg=bg)
            for c0, c1, col in runs:
                length = c1 - c0 + 1
                dist = c0 - (sep_col + 1)
                bars.append((dist, r, c0, c1, length, col))
    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    return bars


def _paint_bar(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    length: int,
    color: int,
    bar_index: int,
    bg: int,
) -> None:
    if length <= 0:
        return
    offset = _bar_offset(length, bar_index)
    w = out.shape[1]
    if bars_on_left:
        start = sep_col + 1 + offset
        positions = range(start, w, length)
    else:
        start = sep_col - 1 - offset
        positions = range(start, -1, -length)
    for cc in positions:
        if int(out[row, cc]) == bg:
            out[row, cc] = color


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

    bars = _collect_bars(g, sep_col, bars_on_left, bg)

    row_bar_counts: dict[int, int] = {}
    for _, r, _c0, _c1, length, col in bars:
        bar_index = row_bar_counts.get(r, 0)
        _paint_bar(out, r, sep_col, bars_on_left, length, col, bar_index, bg)
        row_bar_counts[r] = bar_index + 1

    return out


# --------------- generator (variant D11) ---------------

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
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    bar_pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(bar_pool)) + 1))
    palette = list(map(int, rng.choice(bar_pool, size=size, replace=False)))
    return sep_color, palette


def _sample_two_bar_lengths(rng: np.random.Generator, capacity: int) -> tuple[int, int]:
    near_len = int(rng.integers(1, capacity - 1))
    far_len = int(rng.integers(2, capacity - near_len + 1))
    return near_len, far_len


def _sample_row_bars(
    rng: np.random.Generator,
    palette: list[int],
    primary: int,
    bar_side_len: int,
    *,
    force_two_bars: bool = False,
) -> tuple[list[tuple[int, int]], int]:
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap

    if force_two_bars and capacity < 3:
        sep_gap = 0
        capacity = bar_side_len

    can_use_two_bars = capacity >= 3
    two_bars = force_two_bars or (can_use_two_bars and rng.random() < P_TWO_BARS)

    if not two_bars:
        length = int(rng.integers(1, capacity + 1))
        return [(length, primary)], sep_gap

    near_len, far_len = _sample_two_bar_lengths(rng, capacity)
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
    *,
    staggered: bool,
) -> None:
    for bar_index, (length, color) in enumerate(bars):
        offset = _bar_offset(length, bar_index) if staggered else 0
        if bars_on_left:
            start = sep_col + 1 + offset
            positions = range(start, out.shape[1], length)
        else:
            start = sep_col - 1 - offset
            positions = range(start, -1, -length)
        for cc in positions:
            if out[row, cc] == 0:
                out[row, cc] = color


def _outputs_differ_from_original(
    inp: np.ndarray,
    row_bars_map: dict[int, list[tuple[int, int]]],
    sep_col: int,
    bars_on_left: bool,
) -> bool:
    new_out = inp.copy()
    old_out = inp.copy()
    for row, bars in row_bars_map.items():
        _paint_extension(new_out, row, sep_col, bars_on_left, bars, staggered=True)
        _paint_extension(old_out, row, sep_col, bars_on_left, bars, staggered=False)
    return not np.array_equal(new_out, old_out)


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

    sep_color, palette = _sample_palette(rng)

    max_active = min(ACTIVE_MAX, (h + 1) // 2)
    active_count = int(rng.integers(ACTIVE_MIN, max_active + 1))
    active_rows = _choose_active_rows(rng, h, active_count)
    inactive_rows = [r for r in range(h) if r not in set(active_rows)]

    out = np.zeros((h, w), dtype=int)
    out[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        out[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    forced_row = int(rng.choice(active_rows))
    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for idx, row in enumerate(active_rows):
        bars, sep_gap = _sample_row_bars(
            rng,
            palette,
            row_primary[idx],
            bar_side_len,
            force_two_bars=(row == forced_row),
        )
        row_bars_map[row] = bars
        _place_bars(out, row, sep_col, bars_on_left, bars, sep_gap)

    for row, bars in row_bars_map.items():
        _paint_extension(out, row, sep_col, bars_on_left, bars, staggered=True)

    inp = out.copy()
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if not any(len(bars) >= 2 and bars[1][0] >= 2 for bars in row_bars_map.values()):
        return None
    if not _outputs_differ_from_original(inp, row_bars_map, sep_col, bars_on_left):
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
