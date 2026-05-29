# Combo variant of 1ae2feb7 (index 26)
# combo_spec: D8/3+D15/1
# Name: GCD间隔冲突混色

from __future__ import annotations

from math import gcd
from pathlib import Path
import sys

import numpy as np

ARC_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS) not in sys.path:
    sys.path.append(str(ARC_UTILS))

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
# - The separator and bar-side geometry stay the same as the source task: one near-solid
#   vertical separator, 2-5 sparse active rows, and 1-2 horizontal monochrome bars per
#   active row on exactly one side.
# - D8/3 changes the period selection: a row no longer paints with each bar's own length.
#   Instead every bar on that row emits claims at the same spacing = gcd(all bar lengths).
# - D15/1 changes conflict handling: if different bars claim the same extension cell, the
#   output is not "first bar wins" and not "leave blank". The cell becomes a third color
#   determined by the unordered pair of claim colors.
# - Because all bars on one row now share the same start cell (the cell immediately across
#   the separator) and the same step, a 2-bar row produces full-row collisions on every
#   claimed extension position. That is the key synergy of the fusion.
# - Small trace, bars on the left, row bars near->far = [(length 4, color 1), (length 6, color 2)]:
#   original task:
#     near bar paints columns +0,+4,+8,... ; far bar paints +0,+6,+12,... and skips any
#     occupied cells, so the visible right side is mostly color 1 with a few color-2 points.
#   D8 only:
#     gcd(4,6)=2, both bars target +0,+2,+4,+6,... ; nearest-first means color 1 captures
#     every target, so the whole extension collapses to color 1 at period 2.
#   combo:
#     the same targets +0,+2,+4,+6,... are still produced, but every one is a 1-vs-2
#     collision, so every target becomes mix(1,2)=4. The output is a period-2 dotted line
#     in a third color not present in that row's input bars.
#
# b) Complexity calibration
# - If rows with two bars used pairs like (3,6), the gcd would equal one visible length and
#   the GCD rule would be less diagnostic. The generator therefore forces at least two rows
#   with non-trivial gcd: gcd > 1 and gcd < min(lengths).
# - To keep the color rule learnable, the generator uses a tiny 3-color algebra on
#   {1,2,4}: mixing any two different colors yields the third one. This gives only three
#   unordered pair cases and repeated support across rows/samples.
# - Non-forced rows can still be single-bar rows, so the task does not collapse into "all
#   outputs are third-color lines". Humans can compare unchanged single-bar behavior against
#   mixed 2-bar behavior.
#
# c) Hidden-assumption audit of the original main()
# - `_find_sep_col` assumed one near-full monochrome separator column. Still valid; the
#   generator preserves that exact object.
# - `bars_on_left` inferred the bar side by non-background mass. Still valid because the
#   extension side is blank in the input.
# - `_iter_runs_1d` assumed bars are contiguous runs. Still valid; the generator still
#   creates 1-2 contiguous runs per active row.
# - The original `step = length` is invalid after D8/3. Step becomes a row property:
#   `gcd(lengths_on_row)`.
# - The original `if out[r, cc] == bg: out[r, cc] = col` encoded "first claimant wins".
#   That is invalid after D15/1. The solver must aggregate all claim colors landing on a
#   cell before deciding the output color.
# - The fused solver introduces one new assumption: the intended collision logic is easiest
#   to infer when a row has at most two distinct claim colors. The generator guarantees this
#   by keeping the original 1-2 bar cap and using distinct bar colors on 2-bar rows.
#
# d) Placement legality
# - Bars still occupy only the bar side, are stacked contiguously from the separator
#   outward, and their total length fits within the side width after any optional 1-cell gap.
# - Forced mixed rows only use legal length pairs whose gcd is non-trivial and whose shared
#   step can be shown at least three times on the extension side.
# - The input is created output-first and then the extension side is zeroed, so the solver
#   always receives a clean blank target region.
#
# e) Visual observability
# - The GCD rule is visible because mixed rows use pairs like (4,6) or (6,9), producing a
#   denser cadence than either bar length alone.
# - The mix rule is visible because the extension color on a 2-bar row is neither bar color
#   but the third member of the visible triad {1,2,4}. Across 2-4 samples, humans can infer
#   the closed pairwise table: 1+2->4, 1+4->2, 2+4->1.
# - The generator rejects any sample where the combo output matches either the original task
#   or the "GCD without mixing" intermediate, preventing silent degeneration.
#
# f) Visual cleanliness
# - Rows remain separated by at least one empty row, so each periodic pattern is isolated.
# - Mixing happens only on the extension side and never stacks three different colors in one
#   cell under generator-produced inputs.
# - The separator color is chosen outside the 3-color mix triad, so control structure is not
#   visually conflated with bar/mix colors.
#
# g) Fusion-specific inferability
# - The color mapping is deliberately tiny and symmetric rather than a large arbitrary table.
# - At least two distinct mixed color pairs appear in each generated puzzle, and often all
#   three do. This gives repeated support for the "third color" rule instead of relying on a
#   single isolated coincidence.
# - `behavior_metrics()` below measures the core fusion signal: non-trivial gcd rows, mixed
#   rows, mixed cells, and differences vs both the original solver and the D8-only solver.


BAR_COLORS = [1, 2, 4]
SEPARATOR_COLORS = [3, 5, 6, 7, 8, 9]
MIX_TABLE = {
    frozenset((1, 2)): 4,
    frozenset((1, 4)): 2,
    frozenset((2, 4)): 1,
}
PAIR_TYPES = [(1, 2), (1, 4), (2, 4)]

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20
BAR_SIDE_MIN, BAR_SIDE_MAX = 6, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 5, 14
ACTIVE_MIN, ACTIVE_MAX = 3, 5
P_SEP_GAP = 0.15
P_EXTRA_TWO_BAR = 0.35
P_SWAP_COLORS = 0.5
P_SEP_COLUMN_GAP = 0.2
MAX_ATTEMPTS = 500


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


def _extension_positions(w: int, sep_col: int, bars_on_left: bool, step: int):
    if bars_on_left:
        return range(sep_col + 1, w, step)
    return range(sep_col - 1, -1, -step)


def _fallback_mix_pair(a: int, b: int) -> int:
    palette = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    idx = (palette.index(int(a)) + palette.index(int(b))) % len(palette)
    for _ in range(len(palette)):
        cand = int(palette[idx])
        if cand not in {int(a), int(b)}:
            return cand
        idx = (idx + 1) % len(palette)
    return int(a)


def _mix_two_colors(a: int, b: int) -> int:
    a = int(a)
    b = int(b)
    if a == b:
        return a
    return int(MIX_TABLE.get(frozenset((a, b)), _fallback_mix_pair(a, b)))


def _mix_claim_colors(claim_colors: list[int]) -> int:
    unique: list[int] = []
    for color in claim_colors:
        color = int(color)
        if color not in unique:
            unique.append(color)
    if not unique:
        return 0
    if len(unique) == 1:
        return int(unique[0])
    mixed = int(unique[0])
    for color in unique[1:]:
        mixed = _mix_two_colors(mixed, int(color))
    return mixed


def _collect_row_bars(
    g: np.ndarray, sep_col: int, bars_on_left: bool, bg: int
) -> dict[int, list[tuple[int, int, int, int, int]]]:
    h, w = g.shape
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
            row_bars[r] = sorted(runs, key=lambda t: (t[0], t[1]))
    return row_bars


def _paint_row(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int, int, int, int]],
    *,
    use_gcd_step: bool,
    mix_conflicts: bool,
    bg: int,
) -> None:
    lengths = [length for _dist, _c0, _c1, length, _col in bars]
    if not lengths:
        return
    shared_step = _gcd_lengths(lengths)
    if shared_step <= 0:
        return

    claims: dict[int, list[int]] = {}
    for _dist, _c0, _c1, length, col in bars:
        step = shared_step if use_gcd_step else length
        for cc in _extension_positions(out.shape[1], sep_col, bars_on_left, step):
            claims.setdefault(int(cc), []).append(int(col))

    for cc, claim_colors in claims.items():
        if not claim_colors:
            continue
        if mix_conflicts:
            out[row, cc] = _mix_claim_colors(claim_colors)
            continue
        first = int(claim_colors[0])
        if int(out[row, cc]) == bg:
            out[row, cc] = first


def _simulate(
    grid: np.ndarray, *, use_gcd_step: bool, mix_conflicts: bool
) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy(), {"sep_col": None, "bars_on_left": True, "row_bars": {}}

    out = g.copy()
    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < g.shape[1] else 0
    bars_on_left = left_mass >= right_mass
    row_bars = _collect_row_bars(g, sep_col, bars_on_left, bg)

    for row, bars in row_bars.items():
        _paint_row(
            out,
            row,
            sep_col,
            bars_on_left,
            bars,
            use_gcd_step=use_gcd_step,
            mix_conflicts=mix_conflicts,
            bg=bg,
        )
    return out, {"sep_col": sep_col, "bars_on_left": bars_on_left, "row_bars": row_bars}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overview: Extend each active row across the separator using a shared GCD spacing and
    mix conflicting color claims into a third color.

    Input structure:
    A near-solid vertical separator column splits the grid into a bar side and a blank
    extension side. On 3-5 spaced rows, the bar side contains 1 or 2 contiguous horizontal
    bars, stacked from the separator outward.

    Rule:
    For each active row, compute one step = gcd(all bar lengths on that row). Every bar
    claims extension cells starting at the cell immediately across the separator and then
    every `step` cells. If a cell receives one color, paint that color. If it receives two
    different colors, replace the conflict by the pair's fixed third-color mix.

    Key insight:
    On 2-bar rows the shared GCD step makes both bars land on the same cadence, so the new
    behavior is not about overwrite priority anymore. The row's visible output is a repeated
    mixed-color dotted line whose spacing comes from the bars' gcd.
    """
    out, _meta = _simulate(grid, use_gcd_step=True, mix_conflicts=True)
    return out


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _valid_nontrivial_pairs(capacity: int, ext_side_len: int) -> list[tuple[int, int]]:
    pairs = []
    for a in range(2, capacity):
        for b in range(2, capacity - a + 1):
            step = gcd(a, b)
            if step <= 1:
                continue
            if step >= min(a, b):
                continue
            if ext_side_len < 2 * step + 1:
                continue
            pairs.append((a, b))
    return pairs


def _sample_forced_mixed_bars(
    rng: np.random.Generator,
    pair: tuple[int, int],
    bar_side_len: int,
    ext_side_len: int,
) -> tuple[list[tuple[int, int]], int] | None:
    for sep_gap in [0, 1]:
        capacity = bar_side_len - sep_gap
        if capacity < 4:
            continue
        candidates = _valid_nontrivial_pairs(capacity, ext_side_len)
        if not candidates:
            continue
        near_len, far_len = candidates[int(rng.integers(0, len(candidates)))]
        colors = [int(pair[0]), int(pair[1])]
        if rng.random() < P_SWAP_COLORS:
            colors.reverse()
        return [(near_len, colors[0]), (far_len, colors[1])], sep_gap
    return None


def _sample_free_row_bars(
    rng: np.random.Generator,
    bar_side_len: int,
    ext_side_len: int,
    *,
    force_single: bool = False,
) -> tuple[list[tuple[int, int]], int]:
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap
    if capacity <= 0:
        raise ValueError("bar-side capacity must be positive")

    use_two = (not force_single) and capacity >= 4 and rng.random() < P_EXTRA_TWO_BAR
    if not use_two:
        length = int(rng.integers(1, capacity + 1))
        color = int(BAR_COLORS[int(rng.integers(0, len(BAR_COLORS)))])
        return [(length, color)], sep_gap

    candidates = _valid_nontrivial_pairs(capacity, ext_side_len)
    if not candidates:
        length = int(rng.integers(1, capacity + 1))
        color = int(BAR_COLORS[int(rng.integers(0, len(BAR_COLORS)))])
        return [(length, color)], sep_gap

    near_len, far_len = candidates[int(rng.integers(0, len(candidates)))]
    pair = PAIR_TYPES[int(rng.integers(0, len(PAIR_TYPES)))]
    colors = [int(pair[0]), int(pair[1])]
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
            out[row, cursor - length : cursor] = int(color)
            cursor -= length
    else:
        cursor = sep_col + 1 + sep_gap
        for length, color in bars:
            out[row, cursor : cursor + length] = int(color)
            cursor += length


def _clear_extension_side(grid: np.ndarray, sep_col: int, bars_on_left: bool) -> np.ndarray:
    inp = np.array(grid, dtype=int, copy=True)
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0
    return inp


def behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    variant, meta = _simulate(grid, use_gcd_step=True, mix_conflicts=True)
    original, _ = _simulate(grid, use_gcd_step=False, mix_conflicts=False)
    gcd_nomix, _ = _simulate(grid, use_gcd_step=True, mix_conflicts=False)

    sep_col = meta["sep_col"]
    row_bars = meta["row_bars"]
    mixed_rows = 0
    nontrivial_gcd_rows = 0
    mixed_cells = 0
    pair_kinds: set[tuple[int, int]] = set()

    for row, bars in row_bars.items():
        colors = sorted({int(col) for _dist, _c0, _c1, _length, col in bars})
        lengths = [int(length) for _dist, _c0, _c1, length, _col in bars]
        if len(colors) >= 2:
            mixed_rows += 1
            pair_kinds.add(tuple(colors[:2]))
        if len(lengths) >= 2:
            step = _gcd_lengths(lengths)
            if 1 < step < min(lengths):
                nontrivial_gcd_rows += 1
        if sep_col is None:
            continue
        claim_map: dict[int, list[int]] = {}
        step = _gcd_lengths(lengths)
        for _dist, _c0, _c1, _length, col in bars:
            for cc in _extension_positions(variant.shape[1], int(sep_col), bool(meta["bars_on_left"]), step):
                claim_map.setdefault(int(cc), []).append(int(col))
        for claim_colors in claim_map.values():
            if len({int(c) for c in claim_colors}) >= 2:
                mixed_cells += 1

    return {
        "mixed_rows": int(mixed_rows),
        "distinct_pair_kinds": int(len(pair_kinds)),
        "nontrivial_gcd_rows": int(nontrivial_gcd_rows),
        "mixed_cells": int(mixed_cells),
        "changed_vs_original": int(np.count_nonzero(variant != original)),
        "changed_vs_gcd_nomix": int(np.count_nonzero(variant != gcd_nomix)),
    }


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
    sep_color = int(SEPARATOR_COLORS[int(rng.integers(0, len(SEPARATOR_COLORS)))])

    max_active = min(ACTIVE_MAX, (h + 1) // 2)
    active_count = int(rng.integers(ACTIVE_MIN, max_active + 1))
    active_rows = _choose_active_rows(rng, h, active_count)
    active_set = set(active_rows)
    inactive_rows = [r for r in range(h) if r not in active_set]

    base = np.zeros((h, w), dtype=int)
    base[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        base[gap_row, sep_col] = 0

    forced_mixed_count = 2 if active_count == 3 else int(rng.integers(2, min(3, active_count) + 1))
    forced_rows = active_rows[:forced_mixed_count]
    pair_perm = list(map(int, rng.permutation(len(PAIR_TYPES))))
    forced_pairs = [PAIR_TYPES[pair_perm[i % len(PAIR_TYPES)]] for i in range(forced_mixed_count)]

    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for row, pair in zip(forced_rows, forced_pairs):
        sample = _sample_forced_mixed_bars(rng, pair, bar_side_len, ext_side_len)
        if sample is None:
            return None
        bars, sep_gap = sample
        row_bars_map[int(row)] = bars
        _place_bars(base, int(row), sep_col, bars_on_left, bars, sep_gap)

    free_rows = active_rows[forced_mixed_count:]
    for idx, row in enumerate(free_rows):
        # Keep one plain row as a visual control so the fusion rows are easier to infer.
        bars, sep_gap = _sample_free_row_bars(
            rng,
            bar_side_len,
            ext_side_len,
            force_single=(idx == 0),
        )
        row_bars_map[int(row)] = bars
        _place_bars(base, int(row), sep_col, bars_on_left, bars, sep_gap)

    inp = _clear_extension_side(base, sep_col, bars_on_left)
    variant = main(inp)
    metrics = behavior_metrics(inp)

    if metrics["mixed_rows"] < 2:
        return None
    if metrics["distinct_pair_kinds"] < 2:
        return None
    if metrics["nontrivial_gcd_rows"] < 2:
        return None
    if metrics["mixed_cells"] < 6:
        return None
    if metrics["changed_vs_original"] <= 0:
        return None
    if metrics["changed_vs_gcd_nomix"] <= 0:
        return None
    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if not np.array_equal(main(inp), variant):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
