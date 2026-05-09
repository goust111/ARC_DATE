# Variant of 1ae2feb7
# Dimension: D7 — 多步处理顺序 (multi_step)
# Change: ordered_processing——按色条距分隔条的距离从近到远排序处理，先涂的不被后涂的覆盖（no-overwrite）→ 允许覆盖（后到先得）

from __future__ import annotations

"""
Design analysis for the D7 variant
==================================

a) Concrete behavior and trace
------------------------------
- The scene keeps the original structure: one mostly solid vertical separator column, bars
  on exactly one side, and an empty extension side on the other.
- Each active row still contains 1-2 horizontal monochrome bars arranged from nearest to
  farthest relative to the separator.
- The core change is only in the extension phase on the empty side:
  * Original rule: process bars from near to far, but only paint onto background cells.
    Once the nearest bar paints a dot, a farther bar cannot replace it.
  * Variant rule: process bars from near to far, but always write the color. Later bars
    overwrite earlier dots on collisions, so the farthest processed bar wins.
- Small trace on a left-bar row with separator at column 5 and two bars:
  * Near bar: length 2, color A -> paints columns 6, 8, 10, 12, ...
  * Far bar:  length 3, color B -> paints columns 6, 9, 12, ...
  * Original output: column 6 stays A, column 12 stays A because B cannot overwrite.
  * Variant output:  column 6 becomes B, column 12 becomes B because B is processed later.
- The first extension cell next to the separator is especially diagnostic: every bar starts
  there, so on any two-bar row the final color flips from the near bar to the far bar.

b) Complexity calibration
-------------------------
- A trivial generator could accidentally produce mostly single-bar rows, making the new
  overwrite semantics invisible. To avoid that, every generated sample is forced to include
  at least one two-bar row.
- The rule still has low conceptual complexity: only 1-2 bars per active row, same periodic
  stepping as the source task, and only one changed semantic bit (overwrite policy).
- Distinct colors are kept within a two-bar row so collisions are visually readable.

c) Old assumptions that break under the variant
-----------------------------------------------
- In the original main():
  * `bars.sort(...)` assumes "near first" ordering. This remains valid.
  * `if int(out[r, cc]) == bg: out[r, cc] = col` assumes earlier writes must be preserved.
    This is the exact old invariant that no longer holds.
  * The rest of the solver only identifies separator side, bar runs, bar lengths, and row
    positions; none of those depend on no-overwrite semantics.
- Therefore the only required solver change is to remove the background guard during painting.

d) Placement legality
---------------------
- Bars remain on one side of the separator, contiguous within a row, and fully inside the
  available bar-side width.
- A row forced to have two bars is only sampled when its effective capacity is at least 2;
  this is guaranteed because `bar_side_len >= 3` and the optional separator gap is at most 1.
- Active rows still keep one empty row between them, so extension patterns from different rows
  never interact; overwrite only happens between bars on the same row.

e) Visual observability
-----------------------
- The changed rule is human-visible in 2-3 examples because a two-bar row always has a clash
  at the first extension cell. Under the old rule that cell matches the near bar; under the
  new rule it matches the far bar.
- No hidden ordering by color id is used. Ordering is still purely geometric: distance from
  the separator.

f) Visual cleanliness
---------------------
- The grid is still sparse and row-separated. Overwrite only affects individual dots inside a
  single extension row, so the output remains legible rather than chaotic.
- No extra markers or decorations are introduced; the viewer only has to notice which bar's
  color survives at overlapping extension positions.

g) Color inferability
---------------------
- Colors are not remapped. Each extension dot always copies one of the bar colors already
  visible in the same row.
- Because two-bar rows use two distinct bar colors, the final surviving color at collision
  sites clearly indicates the "later overrides earlier" rule.

Post-change main() assumptions audit
------------------------------------
- `_find_sep_col` assumes there is one mostly solid monochrome separator column. Generator keeps
  that, with at most one gap on an inactive row.
- `bars_on_left = left_mass >= right_mass` assumes the non-separator objects live on one side.
  Generator clears the extension side in the input, so this remains unambiguous.
- Run extraction assumes every bar is a contiguous monochrome 1D segment. Generator preserves it.
- Overwrite semantics assume same-row collisions are allowed and meaningful. Generator guarantees
  at least one such row by forcing at least one two-bar row and rejecting samples where the new
  and old semantics would coincide.
"""

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color


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
    overwrite: bool,
) -> None:
    """Project bar periods onto the extension side using either legacy or variant semantics."""
    w = out.shape[1]
    for length, color in bars:
        if bars_on_left:
            positions = range(sep_col + 1, w, length)
        else:
            positions = range(sep_col - 1, -1, -length)
        for cc in positions:
            if overwrite or out[row, cc] == 0:
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

    # D7 variant: keep the original near->far processing order, but later bars overwrite
    # earlier dots on the extension side.
    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    for _, r, _c0, _c1, length, col in bars:
        if length <= 0:
            continue
        if bars_on_left:
            for cc in range(sep_col + 1, w, length):
                out[r, cc] = col
        else:
            for cc in range(sep_col - 1, -1, -length):
                out[r, cc] = col

    return out


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 2, 14

PALETTE_MIN, PALETTE_MAX = 5, 8

ACTIVE_MIN, ACTIVE_MAX = 2, 5

P_TWO_BARS = 0.6
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5

P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 300


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    """Pick `count` row indices in [0, h) with pairwise spacing >= 2."""
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
    *,
    force_two: bool,
) -> tuple[list[tuple[int, int]], int]:
    """Return (bars, sep_gap) where bars is a list of (length, color) ordered near->far."""
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap
    if capacity < 1:
        raise ValueError("bar capacity must stay positive")

    two_bars = force_two or (rng.random() < P_TWO_BARS)
    if not two_bars or capacity == 1:
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

    base = np.zeros((h, w), dtype=int)
    base[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        base[gap_row, sep_col] = 0

    forced_two_row = int(rng.choice(active_rows))
    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for idx, row in enumerate(active_rows):
        bars, sep_gap = _sample_row_bars(
            rng,
            palette,
            row_primary[idx],
            bar_side_len,
            force_two=(row == forced_two_row),
        )
        row_bars_map[row] = bars
        _place_bars(base, row, sep_col, bars_on_left, bars, sep_gap)

    out = base.copy()
    legacy_out = base.copy()
    for row, bars in row_bars_map.items():
        _paint_extension(out, row, sep_col, bars_on_left, bars, overwrite=True)
        _paint_extension(legacy_out, row, sep_col, bars_on_left, bars, overwrite=False)

    if np.array_equal(out, legacy_out):
        return None

    inp = base.copy()

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
