# Variant of 1ae2feb7
# Dimension: D6 — 输出网格尺寸 (grid_shape)
# Change: same——输出与输入同尺寸，仅在ext_side区域新增涂色内容... → 裁剪为仅ext_side

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for the D6 variant
# a) Concretization and behavior trace
#    - Original task: the input contains a vertical separator column, monochrome
#      horizontal bars on one side, and an all-background ext_side on the other.
#      The solver keeps the whole grid and paints periodic dots on ext_side.
#    - Variant: the painting rule is unchanged, but the final output is only the
#      painted ext_side rectangle. The separator column and the entire bar_side are
#      removed from the output. Height stays the same; width becomes ext_side_len.
#    - Small trace, bars on the left:
#        input row: 111102000000
#        original full output row: 111102100010
#        variant output row:             100010
#      The changed stage is purely the final projection from "full solved grid" to
#      "solved ext_side subgrid".
# b) Complexity calibration
#    - The crop must be visually obvious, so the generator keeps the original bar
#      logic but requires ext_side to be at least 4 columns wide. This preserves
#      enough periodic structure for humans to infer the rule after noticing the
#      output is a sub-rectangle of the input.
#    - No extra branches are introduced: the only global branch remains whether the
#      bars lie to the left or right of the separator.
# c) Implicit-assumption audit against the original main()
#    - `_find_sep_col`: still valid; the separator is still a nearly full vertical
#      monochrome column inside the input.
#    - `bars_on_left = left_mass >= right_mass`: still valid because the input still
#      contains bars on exactly one side and the ext_side is cleared in the input.
#    - run collection and nearest-first sorting: unchanged; the crop does not alter
#      how bars are parsed or prioritized.
#    - the original `return out` relied on "output keeps separator and bars". That
#      assumption is no longer valid, so the new solver must crop after painting.
# d) Placement legality
#    - Generator still constructs a full valid original-style scene first, with one
#      separator column, bars placed fully on one side, and a disjoint ext_side.
#    - Because cropping happens only after solving, no object overlap rule changes.
# e) Visual observability
#    - In 2-4 examples, the human sees that every output has the same height as the
#      input but a smaller width equal to the blank side only. The periodic dots are
#      still present, so the crop is inferable from repeated examples.
# f) Visual cleanliness
#    - The output contains only the relevant painted target region, so it is cleaner
#      than the original full-grid output. No new overlays are introduced.
# g) Color-rule audit
#    - No color remapping is added. Colors stay tied to the original bars, which is
#      directly visible and avoids hidden color-id logic.
# h) Reference-task note
#    - The prompt names `e87109e9` as a tight-crop inspiration, but no readable
#      solver file for that task was present in the allowed workspace paths during
#      this turn. The variant therefore uses the minimal tight-crop mechanism
#      described in the prompt itself: solve normally, then return the ext_side crop.
# i) New solver assumptions introduced by the crop
#    - The solver now assumes the separator is not on the outer border, so ext_side
#      has positive width. The generator enforces ext_side_len in [4, 14].
#    - The solver assumes the target region is the side opposite the bars; the
#      generator preserves the original "bars only on one side" invariant.


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


def _crop_ext_side(full: np.ndarray, sep_col: int, bars_on_left: bool) -> np.ndarray:
    if bars_on_left:
        return full[:, sep_col + 1 :].copy()
    return full[:, :sep_col].copy()


def _solve_full(grid: np.ndarray) -> tuple[np.ndarray, int | None, bool | None, int]:
    g = np.asarray(grid, dtype=int)

    bg = _bg_color(g)
    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy(), None, None, bg

    out = g.copy()

    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < g.shape[1] else 0
    bars_on_left = left_mass >= right_mass

    bars = []
    for r in range(g.shape[0]):
        if bars_on_left:
            for c0, c1, col in _iter_runs_1d(g[r, :], 0, sep_col, bg=bg):
                length = c1 - c0 + 1
                dist = (sep_col - 1) - c1
                bars.append((dist, r, c0, length, col))
        else:
            for c0, c1, col in _iter_runs_1d(g[r, :], sep_col + 1, g.shape[1], bg=bg):
                length = c1 - c0 + 1
                dist = c0 - (sep_col + 1)
                bars.append((dist, r, c0, length, col))

    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    for _, r, _c0, length, col in bars:
        if length <= 0:
            continue
        if bars_on_left:
            for cc in range(sep_col + 1, g.shape[1], length):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col
        else:
            for cc in range(sep_col - 1, -1, -length):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col

    return out, sep_col, bars_on_left, bg


def main(grid: np.ndarray) -> np.ndarray:
    full_out, sep_col, bars_on_left, _bg = _solve_full(grid)
    if sep_col is None or bars_on_left is None:
        return full_out
    return _crop_ext_side(full_out, sep_col, bars_on_left)


# --------------- generator (D6 variant) ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 4, 14

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


def _sample_row_bars(
    rng: np.random.Generator,
    palette: list[int],
    primary: int,
    bar_side_len: int,
) -> tuple[list[tuple[int, int]], int]:
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap
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


def _clear_ext_side(inp: np.ndarray, sep_col: int, bars_on_left: bool) -> None:
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0


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
    ext_side_len = (w - 1 - sep_col) if bars_on_left else sep_col
    if not (EXT_SIDE_MIN <= ext_side_len <= EXT_SIDE_MAX):
        return None

    sep_color, palette = _sample_palette(rng)

    max_active = min(ACTIVE_MAX, (h + 1) // 2)
    active_count = int(rng.integers(ACTIVE_MIN, max_active + 1))
    active_rows = _choose_active_rows(rng, h, active_count)
    active_row_set = set(active_rows)
    inactive_rows = [r for r in range(h) if r not in active_row_set]

    full_out = np.zeros((h, w), dtype=int)
    full_out[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        full_out[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for idx, row in enumerate(active_rows):
        bars, sep_gap = _sample_row_bars(rng, palette, row_primary[idx], bar_side_len)
        row_bars_map[row] = bars
        _place_bars(full_out, row, sep_col, bars_on_left, bars, sep_gap)

    for row, bars in row_bars_map.items():
        _paint_extension(full_out, row, sep_col, bars_on_left, bars)

    expected = _crop_ext_side(full_out, sep_col, bars_on_left)
    if expected.shape != (h, ext_side_len):
        return None
    if np.count_nonzero(expected) < active_count:
        return None

    inp = full_out.copy()
    _clear_ext_side(inp, sep_col, bars_on_left)

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
