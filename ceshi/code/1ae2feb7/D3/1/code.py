# Variant of 1ae2feb7
# Dimension: D3 — 射线源对象类型 (src)
# Change: 水平单色色条——分隔条一侧的水平连续同色像素段，每行1-2个色条，紧邻分隔条排列... → 垂直色条

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color


# Design analysis
# ===============
# a) Concrete behavior and trace
#    Original task:
#    - A near-full vertical separator column splits the board into a bar side and an extension side.
#    - Each active row contains 1-2 horizontal monochrome bars. Their widths define horizontal paint periods.
#    - Bars are processed nearest-to-separator first; each bar paints dots across the opposite side on the same row,
#      stepping by its width and never overwriting earlier paint.
#    D3 variant:
#    - Transpose the layout. The separator becomes a horizontal row, and source objects become vertical bars placed
#      in active columns entirely above or entirely below that row.
#    - Each active column contains 1-2 vertical monochrome bars stacked from the separator outward.
#    - A bar of height k now paints along the same column on the opposite side at rows
#      sep+1, sep+1+k, sep+1+2k, ... (or symmetrically upward), again nearest-first with no overwrite.
#    Example trace on a top-bar layout:
#    - Input column c has a 2-cell bar touching the separator and a 3-cell bar above it.
#    - Original logic would have scanned rows and projected horizontally; that is now wrong because width no longer
#      encodes the rule.
#    - Variant logic scans column c, measures heights [2, 3], sorts by separator distance [0, 2], then paints
#      bottom-side cells every 2 rows first and fills remaining background cells every 3 rows second.
#
# b) Complexity calibration
#    - Keep 1-2 bars per active column and 2-5 active columns, mirroring the original complexity.
#    - Keep occasional 1-cell separator gaps and occasional 2-bar columns so the "nearest first" rule remains visible.
#    - Swap canvas aspect ratio to favor tall layouts (15-20 rows x 8-13 cols); otherwise vertical periodicity would
#      be visually cramped and the D3 change would be less apparent.
#
# c) Implicit-assumption audit of original main()
#    - `_find_sep_col` assumed the separator is vertical. This fails completely once the separator is horizontal,
#      so it must become `_find_sep_row`.
#    - Row scanning via `_iter_runs_1d(g[r, :], ...)` assumed source objects are horizontal runs. In the variant the
#      same helper is still useful, but only on column vectors `g[:, c]`.
#    - `left_mass/right_mass` assumed bar-side choice is left vs right. This becomes `top_mass/bottom_mass`.
#    - The original bar tuples stored row id and width-derived step. In the variant they must store column id and
#      height-derived step; otherwise the projection direction and period are both wrong.
#    - The horizontal painting loops wrote to `out[r, cc]`. In the variant they must write to `out[rr, c]`. The
#      no-overwrite guard remains valid because bars still need nearest-first priority.
#
# d) Placement legality
#    - Generator places all bars on one side of the separator row only.
#    - Each active column is separated by at least one empty column, matching the original "empty row" spacing after
#      transposition and reducing visual collisions.
#    - In a 2-bar column, the bars are adjacent to each other and fit within the available bar-side height after an
#      optional 1-cell gap to the separator.
#    - Output is built first, then the extension side is cleared to form the input, and `main(input) == output` is
#      enforced before accepting a sample.
#
# e) Visual observability
#    - The variant is visible immediately: the separator is horizontal, sources are vertical, and output growth happens
#      up/down within columns rather than left/right within rows.
#    - Periods are encoded by bar heights, which are directly countable in the input.
#
# f) Visual cleanliness
#    - Only one projection axis is active per sample, and columns are spaced out, so outputs remain legible.
#    - The no-overwrite rule prevents dense multi-color overpainting within one column.
#
# g) Color inference
#    - No hidden color arithmetic is used. Colors are copied directly from source bars to their projected dots.
#    - Separator color stays distinct from all bar colors to keep object roles obvious.
#
# Solver assumptions introduced by the variant and guaranteed by the generator
#    - There is exactly one salient horizontal separator row whose non-background cells are all one color and whose
#      coverage is at least W-1 cells.
#    - All source bars lie entirely on one side of that row.
#    - Source objects are vertical monochrome runs when viewed column-wise on the bar side.
#    - The extension side is empty in the input, so nearest-first projection without overwrite is well-defined.


def _find_sep_row(g: np.ndarray, bg: int) -> tuple[int | None, int | None]:
    h, w = g.shape
    for r in range(h):
        row = g[r, :]
        vals = row[row != bg]
        if vals.size >= w - 1 and (vals == vals[0]).all():
            return r, int(vals[0])
    return None, None


def _iter_runs_1d(vec: np.ndarray, lo: int, hi: int, bg: int):
    i = lo
    while i < hi:
        if int(vec[i]) == bg:
            i += 1
            continue
        col = int(vec[i])
        i0 = i
        while i < hi and int(vec[i]) == col:
            i += 1
        yield i0, i - 1, col


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)

    sep_row, _sep_color = _find_sep_row(g, bg)
    if sep_row is None:
        return g.copy()

    out = g.copy()

    top_mass = int(np.sum(g[:sep_row, :] != bg)) if sep_row > 0 else 0
    bottom_mass = int(np.sum(g[sep_row + 1 :, :] != bg)) if sep_row + 1 < h else 0
    bars_on_top = top_mass >= bottom_mass

    bars: list[tuple[int, int, int, int, int, int]] = []
    for c in range(w):
        col_vec = g[:, c]
        if bars_on_top:
            for r0, r1, color in _iter_runs_1d(col_vec, 0, sep_row, bg=bg):
                length = r1 - r0 + 1
                dist = (sep_row - 1) - r1
                bars.append((dist, c, r0, r1, length, color))
        else:
            for r0, r1, color in _iter_runs_1d(col_vec, sep_row + 1, h, bg=bg):
                length = r1 - r0 + 1
                dist = r0 - (sep_row + 1)
                bars.append((dist, c, r0, r1, length, color))

    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    for _, c, _r0, _r1, length, color in bars:
        if length <= 0:
            continue
        if bars_on_top:
            for rr in range(sep_row + 1, h, length):
                if int(out[rr, c]) == bg:
                    out[rr, c] = color
        else:
            for rr in range(sep_row - 1, -1, -length):
                if int(out[rr, c]) == bg:
                    out[rr, c] = color

    return out


# --------------- generator ---------------

# Canvas size bounds (inclusive ranges). Swapped to make the transposed layout visually clear.
H_MIN, H_MAX = 15, 20
W_MIN, W_MAX = 8, 13

# Bar-side height bounds (clamped so ext_side stays in [EXT_SIDE_MIN, EXT_SIDE_MAX]).
BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 2, 14

# Palette sizes.
PALETTE_MIN, PALETTE_MAX = 5, 8

# Active-column counts.
ACTIVE_MIN, ACTIVE_MAX = 2, 5

# Per-column bar structure probabilities.
P_TWO_BARS = 0.45
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5

# Separator gap (a single terminal-column cell omitted from separator row).
P_SEP_ROW_GAP = 0.2

# Retry budget for generation.
MAX_ATTEMPTS = 300


def _choose_active_cols(rng: np.random.Generator, w: int, count: int) -> list[int]:
    """Pick `count` column indices in [0, w) with pairwise spacing >= 2."""
    base = np.sort(rng.choice(w - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    bar_pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(bar_pool)) + 1))
    palette = list(map(int, rng.choice(bar_pool, size=size, replace=False)))
    return sep_color, palette


def _sample_col_bars(
    rng: np.random.Generator,
    palette: list[int],
    primary: int,
    bar_side_len: int,
) -> tuple[list[tuple[int, int]], int]:
    """Return (bars, sep_gap) where bars is a list of (height, color) ordered near->far."""
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap
    two_bars = capacity >= 2 and rng.random() < P_TWO_BARS

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
    col: int,
    sep_row: int,
    bars_on_top: bool,
    bars: list[tuple[int, int]],
    sep_gap: int,
) -> None:
    """Paint bars on the bar side, stacking from nearest-to-separator outward."""
    if bars_on_top:
        cursor = sep_row - sep_gap
        for length, color in bars:
            out[cursor - length : cursor, col] = color
            cursor -= length
    else:
        cursor = sep_row + 1 + sep_gap
        for length, color in bars:
            out[cursor : cursor + length, col] = color
            cursor += length


def _paint_extension(
    out: np.ndarray,
    col: int,
    sep_row: int,
    bars_on_top: bool,
    bars: list[tuple[int, int]],
) -> None:
    """Simulate the periodic extension onto the extension side (nearest-first, no overwrite)."""
    h = out.shape[0]
    for length, color in bars:
        if bars_on_top:
            positions = range(sep_row + 1, h, length)
        else:
            positions = range(sep_row - 1, -1, -length)
        for rr in positions:
            if int(out[rr, col]) == 0:
                out[rr, col] = color


def _bar_heights(bar_map: dict[int, list[tuple[int, int]]]) -> list[int]:
    return [length for bars in bar_map.values() for length, _color in bars]


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(H_MIN, H_MAX + 1))
    w = int(rng.integers(W_MIN, W_MAX + 1))

    lo = max(BAR_SIDE_MIN, h - 1 - EXT_SIDE_MAX)
    hi = min(BAR_SIDE_MAX, h - 1 - EXT_SIDE_MIN)
    if lo > hi:
        return None
    bar_side_len = int(rng.integers(lo, hi + 1))

    bars_on_top = bool(rng.integers(0, 2) == 0)
    sep_row = bar_side_len if bars_on_top else (h - 1 - bar_side_len)

    sep_color, palette = _sample_palette(rng)

    max_active = min(ACTIVE_MAX, (w + 1) // 2)
    active_count = int(rng.integers(ACTIVE_MIN, max_active + 1))
    active_cols = _choose_active_cols(rng, w, active_count)
    active_set = set(active_cols)
    inactive_cols = [c for c in range(w) if c not in active_set]

    out = np.zeros((h, w), dtype=int)
    out[sep_row, :] = sep_color
    if inactive_cols and rng.random() < P_SEP_ROW_GAP:
        gap_col = inactive_cols[int(rng.integers(0, len(inactive_cols)))]
        out[sep_row, gap_col] = 0

    col_primary = list(map(int, rng.permutation(palette)[:active_count]))
    col_bars_map: dict[int, list[tuple[int, int]]] = {}
    for idx, col in enumerate(active_cols):
        bars, sep_gap = _sample_col_bars(rng, palette, col_primary[idx], bar_side_len)
        col_bars_map[col] = bars
        _place_bars(out, col, sep_row, bars_on_top, bars, sep_gap)

    if max(_bar_heights(col_bars_map), default=0) < 2:
        return None

    for col, bars in col_bars_map.items():
        _paint_extension(out, col, sep_row, bars_on_top, bars)

    inp = out.copy()
    if bars_on_top:
        inp[sep_row + 1 :, :] = 0
    else:
        inp[:sep_row, :] = 0

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
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
