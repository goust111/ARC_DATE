# Variant of 1ae2feb7
# Dimension: D15 — no-overwrite规则的精确行为 (multi_step)
# Change: 不覆盖已有的非背景像素——先涂的点永远保留，后涂的射线遇到已涂位置直接跳过... → 冲突位置留白

from __future__ import annotations

import math

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for the D15 variant
# -----------------------------------
# spec:
#   preserved_structure:
#     - Grid size, separator geometry, active-row spacing, and bar placement stay the same as the base task.
#     - Each active row still contains 1 or 2 horizontal monochrome bars on exactly one side of a near-solid
#       vertical separator column.
#     - The opposite side is still blank in the input and is the only painted region in the output.
#   concretized_variant:
#     - Each bar still emits a periodic "ray" onto the extension side, starting at the cell immediately across
#       the separator and stepping by the bar length.
#     - Old rule: process nearer bars first; a later ray skips cells that were already painted.
#     - New rule: treat all rays in the same row as simultaneous claims. A cell is colored only if exactly one
#       bar claims it. If two bars claim the same cell, that cell stays background.
#   behavior_trace:
#     - Example row: separator at c=5, bars on the left, bars = [(2, red), (3, blue)] ordered near->far.
#     - Original output on the right side:
#         red claims c=6,8,10,12,...
#         blue claims c=6,9,12,...
#         nearest-first gives c=6 red, c=8 red, c=9 blue, c=10 red, c=12 red, ...
#     - Variant output:
#         c=6 is claimed by both -> stays 0
#         c=8 only red -> red
#         c=9 only blue -> blue
#         c=10 only red -> red
#         c=12 both -> stays 0
#     - The visible change is a regular set of holes, with the first extension cell always blank on 2-bar rows.
#   complexity_calibration:
#     - A row with only one bar would look identical to the original task, so the generator must force at least
#       one 2-bar row per sample.
#     - To make the holes inferable from a few examples, forced 2-bar rows prefer unequal lengths whose least
#       common multiple fits inside the extension width, producing repeated blank cells beyond the first one.
#     - We keep only 1 or 2 bars per row so the cancellation rule remains legible rather than chaotic.
#   implicit_assumption_review:
#     - `_find_sep_col` still works because the separator column shape is unchanged.
#     - `bars_on_left` mass inference still works because the input extension side remains blank.
#     - `_iter_runs_1d` still works because bars are still contiguous horizontal monochrome runs.
#     - The original solver's only semantically invalid fragment for D15 was:
#         `if out[r, cc] == bg: out[r, cc] = col`
#       That line encoded "first claimant wins". It is replaced by per-row claim counting.
#     - The new solver assumes extension-side cells begin as background in valid inputs; the generator preserves
#       that invariant by constructing output first and then zeroing the extension side to form the input.
#   placement_legality:
#     - 2-bar rows require total length <= bar-side capacity after any separator gap.
#     - Forced conflict pairs are chosen only from legal length pairs, so bars never overlap or cross the separator.
#     - Active rows remain separated by at least one empty row, keeping each row's pattern visually isolated.
#   visual_observability:
#     - The rule does not depend on color ids or hidden ordering by color. Only visible bar lengths and proximity
#       to the separator matter.
#     - Every generated sample contains at least one row where the variant differs from the base task.
#   visual_tidy_and_color_inference:
#     - Colors are not remapped; they are copied from the uniquely claiming bar, so there is no extra color logic
#       to infer beyond the hole-creation rule.
#     - Because rows are independent and spaced apart, overlapping explanations between rows are minimized.


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


def _extension_positions(w: int, sep_col: int, bars_on_left: bool, step: int):
    if bars_on_left:
        return range(sep_col + 1, w, step)
    return range(sep_col - 1, -1, -step)


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

    row_bars: dict[int, list[tuple[int, int, int, int]]] = {}
    for r in range(h):
        if bars_on_left:
            for c0, c1, col in _iter_runs_1d(g[r, :], 0, sep_col, bg=bg):
                length = c1 - c0 + 1
                dist = (sep_col - 1) - c1
                row_bars.setdefault(r, []).append((dist, c0, length, col))
        else:
            for c0, c1, col in _iter_runs_1d(g[r, :], sep_col + 1, w, bg=bg):
                length = c1 - c0 + 1
                dist = c0 - (sep_col + 1)
                row_bars.setdefault(r, []).append((dist, c0, length, col))

    for r, bars in row_bars.items():
        claims: dict[int, int] = {}
        claim_color: dict[int, int] = {}
        for _dist, c0, length, col in sorted(bars, key=lambda t: (t[0], t[1])):
            if length <= 0:
                continue
            for cc in _extension_positions(w, sep_col, bars_on_left, length):
                claims[cc] = claims.get(cc, 0) + 1
                claim_color.setdefault(cc, col)
        for cc, count in claims.items():
            if count == 1:
                out[r, cc] = claim_color[cc]
            else:
                out[r, cc] = bg

    return out


# --------------- generator for D15 variant ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 2, 14

PALETTE_MIN, PALETTE_MAX = 5, 8

ACTIVE_MIN, ACTIVE_MAX = 2, 5

P_TWO_BARS = 0.55
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5
P_SEP_COLUMN_GAP = 0.2

MIN_FORCED_CONFLICT_ROWS = 1
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


def _pick_second_color(
    rng: np.random.Generator,
    palette: list[int],
    primary: int,
) -> int:
    others = [c for c in palette if c != primary]
    return int(rng.choice(others))


def _conflict_length_pairs(capacity: int, ext_side_len: int) -> list[tuple[int, int]]:
    pairs = [
        (a, b)
        for a in range(1, capacity)
        for b in range(1, capacity - a + 1)
        if a != b
    ]
    rich_pairs = [pair for pair in pairs if math.lcm(pair[0], pair[1]) <= ext_side_len - 1]
    return rich_pairs if rich_pairs else pairs


def _sample_row_bars(
    rng: np.random.Generator,
    palette: list[int],
    primary: int,
    bar_side_len: int,
    ext_side_len: int,
    *,
    force_conflict: bool = False,
) -> tuple[list[tuple[int, int]], int]:
    """Return (bars, sep_gap) where bars is a list of (length, color) ordered near->far."""
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap
    if capacity <= 0:
        raise ValueError("bar-side capacity must stay positive")

    want_two_bars = force_conflict or (capacity >= 2 and rng.random() < P_TWO_BARS)
    if not want_two_bars or capacity < 2:
        length = int(rng.integers(1, capacity + 1))
        return [(length, primary)], sep_gap

    pair_choices = _conflict_length_pairs(capacity, ext_side_len)
    if not pair_choices:
        length = int(rng.integers(1, capacity + 1))
        return [(length, primary)], sep_gap

    near_len, far_len = pair_choices[int(rng.integers(0, len(pair_choices)))]
    if rng.random() < 0.5:
        near_len, far_len = far_len, near_len

    colors = [primary, _pick_second_color(rng, palette, primary)]
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


def _paint_extension(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
) -> None:
    """Simulate the D15 rule: unique claims paint, multi-claims leave holes."""
    claims: dict[int, int] = {}
    claim_color: dict[int, int] = {}
    w = out.shape[1]
    for length, color in bars:
        for cc in _extension_positions(w, sep_col, bars_on_left, length):
            claims[cc] = claims.get(cc, 0) + 1
            claim_color.setdefault(cc, color)
    for cc, count in claims.items():
        out[row, cc] = claim_color[cc] if count == 1 else 0


def _count_conflict_rows(bars_by_row: dict[int, list[tuple[int, int]]]) -> int:
    return sum(1 for bars in bars_by_row.values() if len(bars) >= 2)


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
    active_set = set(active_rows)
    inactive_rows = [r for r in range(h) if r not in active_set]

    out = np.zeros((h, w), dtype=int)
    out[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        out[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    forced_conflict_rows = set(
        active_rows[: min(MIN_FORCED_CONFLICT_ROWS, len(active_rows))]
    )
    if len(active_rows) > 1 and rng.random() < 0.5:
        extra = active_rows[int(rng.integers(1, len(active_rows)))]
        forced_conflict_rows.add(extra)

    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for idx, row in enumerate(active_rows):
        bars, sep_gap = _sample_row_bars(
            rng,
            palette,
            row_primary[idx],
            bar_side_len,
            ext_side_len,
            force_conflict=row in forced_conflict_rows,
        )
        row_bars_map[row] = bars
        _place_bars(out, row, sep_col, bars_on_left, bars, sep_gap)

    if _count_conflict_rows(row_bars_map) < MIN_FORCED_CONFLICT_ROWS:
        return None

    for row, bars in row_bars_map.items():
        _paint_extension(out, row, sep_col, bars_on_left, bars)

    inp = out.copy()
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0

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
