# Variant of 1ae2feb7
# Dimension: D1 — 射线路径方向 (path)
# Change: straight_ray——从分隔条出发，沿水平方向向对侧网格边缘延伸（垂直于分隔条方向的直线射线）... → 垂直方向射线
#
# Design analysis for this variant
# a) Concrete behavior:
#    The whole task is rotated 90 degrees. The original has a vertical separator
#    column and horizontal bars on one side; this variant has a horizontal
#    separator row and vertical bars on one side. Each bar keeps its original
#    rule: its pixel count is the period, and the bar paints dots along the
#    extension side starting immediately across the separator. Example:
#    if a 3-cell vertical bar sits above the separator in column c, then the
#    output paints column c at rows sep+1, sep+4, sep+7, ... unless those cells
#    were already claimed by a nearer bar in the same column. With two stacked
#    bars in one column, the one closer to the separator is processed first.
# b) Complexity calibration:
#    I keep the original 1-2 bars per active line structure, but lines become
#    columns. Active columns still have spacing >= 2 so the separator row is
#    visually obvious and bars do not merge into a blob. Canvas aspect is also
#    rotated (taller than wide) so the direction change is visible at a glance.
# c) Hidden-assumption audit of main():
#    - `_find_sep_col` assumed the separator is vertical. It must become a
#      row-based detector.
#    - Side-mass detection assumed left vs right. It must become above vs below.
#    - Iteration over bars assumed horizontal runs inside rows. It must become
#      vertical runs inside columns.
#    - Distance-to-separator used column offsets. It must use row offsets.
#    - Extension loops stepped over columns. They must step over rows.
#    - The "do not overwrite" rule remains unchanged; this is still essential
#      when two bars share a column.
# d) Placement legality:
#    The generator chooses a separator row so bar-side height is 3-12 and
#    extension-side height is 2-14, mirroring the original numeric constraints.
#    Active columns are spaced by at least one empty column. Within each active
#    column, 1-2 bars are stacked contiguously from the separator outward, with
#    an optional 1-cell gap only at the separator-facing end.
# e) Visual observability:
#    Inputs now show a long horizontal divider plus several vertical bars above
#    or below it, while outputs add vertical dotted rays in matching columns.
#    This differs visibly from the original horizontal propagation and should be
#    inferable from 2-3 examples without relying on color IDs.
# f) Visual cleanliness:
#    Only one kind of interaction happens on the extension side: dots in the
#    same column. Overlap can happen only between bars from the same column, and
#    nearest-first plus non-overwrite keeps the result structured rather than
#    noisy.
# g) Color inference:
#    Colors are preserved. No hidden color remapping is introduced; colors only
#    identify bars and their propagated dots.
# h) Variant-specific validation target:
#    The intended distinguishing feature is column-wise vertical extension from a
#    horizontal separator. On these inputs the original solver would normally do
#    nothing because there is no qualifying vertical separator column.
# i) New solver assumptions and generator guarantees:
#    - Assumption: exactly one near-full monochrome separator row exists.
#      Guarantee: generator paints one separator row and bars are too sparse to
#      imitate such a row.
#    - Assumption: all bars lie on one side of the separator.
#      Guarantee: generator clears the whole extension side in the input.
#    - Assumption: per-column bars are contiguous runs.
#      Guarantee: generator stacks runs without internal gaps.

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color


def _find_sep_row(g: np.ndarray, bg: int) -> tuple[int | None, int | None]:
    h, w = g.shape
    for r in range(h):
        row = g[r, :]
        vals = row[row != bg]
        if vals.size >= w - 1 and (vals == vals[0]).all():
            return r, int(vals[0])
    return None, None


def _iter_runs_1d(line: np.ndarray, i_lo: int, i_hi: int, bg: int):
    i = i_lo
    while i < i_hi:
        if line[i] == bg:
            i += 1
            continue
        col = int(line[i])
        i0 = i
        while i < i_hi and line[i] == col:
            i += 1
        yield i0, i - 1, col


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)

    # Horizontal separator row: non-bg cells are almost full-row and monochrome.
    sep_row, _sep_color = _find_sep_row(g, bg)
    if sep_row is None:
        return g.copy()

    out = g.copy()

    # Decide whether the bars live above or below the separator.
    top_mass = int(np.sum(g[:sep_row, :] != bg)) if sep_row > 0 else 0
    bottom_mass = int(np.sum(g[sep_row + 1 :, :] != bg)) if sep_row + 1 < h else 0
    bars_on_top = top_mass >= bottom_mass

    bars = []
    for c in range(w):
        col_vec = g[:, c]
        if bars_on_top:
            for r0, r1, col in _iter_runs_1d(col_vec, 0, sep_row, bg=bg):
                length = r1 - r0 + 1
                dist = (sep_row - 1) - r1  # 0 if bar touches separator from above
                bars.append((dist, c, r0, r1, length, col))
        else:
            for r0, r1, col in _iter_runs_1d(col_vec, sep_row + 1, h, bg=bg):
                length = r1 - r0 + 1
                dist = r0 - (sep_row + 1)  # 0 if bar touches separator from below
                bars.append((dist, c, r0, r1, length, col))

    # Process bars from nearest to farthest from the separator; do not overwrite.
    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    for _, c, _r0, _r1, length, col in bars:
        step = length
        if step <= 0:
            continue
        if bars_on_top:
            for rr in range(sep_row + 1, h, step):
                if int(out[rr, c]) == bg:
                    out[rr, c] = col
        else:
            for rr in range(sep_row - 1, -1, -step):
                if int(out[rr, c]) == bg:
                    out[rr, c] = col

    return out


# --------------- generator ---------------

# Rotated canvas bounds (inclusive ranges).
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

# Separator gap (one omitted cell in the separator row).
P_SEP_ROW_GAP = 0.2

# Retry budget for generation.
MAX_ATTEMPTS = 300


def _choose_active_indices(rng: np.random.Generator, size: int, count: int) -> list[int]:
    """Pick `count` indices in [0, size) with pairwise spacing >= 2."""
    base = np.sort(rng.choice(size - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    """Pick separator color and a disjoint bar-color palette."""
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
    """Return (bars, sep_gap) with bars listed from nearest-to-separator outward."""
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
    """Simulate the periodic vertical extension on the extension side."""
    h = out.shape[0]
    for length, color in bars:
        if bars_on_top:
            positions = range(sep_row + 1, h, length)
        else:
            positions = range(sep_row - 1, -1, -length)
        for rr in positions:
            if out[rr, col] == 0:
                out[rr, col] = color


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(H_MIN, H_MAX + 1))
    w = int(rng.integers(W_MIN, W_MAX + 1))

    # bar_side_len constrained so ext_side_len = h - 1 - bar_side_len is valid.
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
    active_cols = _choose_active_indices(rng, w, active_count)
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
