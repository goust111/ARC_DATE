# Combo variant of 1ae2feb7 (index 25)
# combo_spec: D4/2+D5/0+D7/0
# Name: 远优先有限涂色后擦源

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    The scene still contains one nearly full-height monochrome vertical separator, visible source
#    bars on exactly one side, and an empty extension side in the input. The fused rule combines
#    three edits on the original periodic extension:
#    - D7/0: process bars from farthest-to-nearest relative to the separator.
#    - D4/2: each bar of length L fires only L periodic shots instead of extending to the edge.
#    - D5/0: after all shots land, erase the original source bars from the output.
#    Example with bars on the left, separator at column 5, extension width 7:
#      source row: [A A][B] | 0 0 0 0 0 0 0
#      near bar = B, length 1; far bar = A, length 2.
#      Original row: near-first, to-edge => B claims every ext cell before A can act.
#      Combo row: far-first, limited =>
#        A paints only columns 6 and 8 first; B then paints only column 6 but is blocked there.
#        Final ext side shows A at 6 and 8, source bars are erased, so output row becomes
#        0 0 0 0 0 | A 0 A 0 0 0 0.
#    The crucial visual difference is that the output contains only derived dots; the missing bars
#    must be inferred backward from spacing, count, and far-priority collisions.
#
# b) Complexity calibration
#    D5 alone would already make the output sparser, so the generator must also make D4 and D7
#    observable. To do that it enforces two witness roles:
#    - one single-bar showcase row with length 1 or 2, so limited-shot behavior visibly differs
#      from edge-to-edge extension on a wide extension side;
#    - one two-bar row with distinct colors, so far-first visibly differs from near-first on the
#      first extension cell and often on later overlap cells too.
#    All bar lengths are capped at 3, and the extension side is kept wide enough for every bar
#    to place all L shots on-grid. This keeps the limited-shot rule visible under the original
#    width bounds and avoids accidental equality with edge-stop behavior.
#
# c) Hidden-assumption audit
#    - `_find_sep_col` still only relies on a mostly filled monochrome separator column.
#    - side detection by non-background mass still works because the generator keeps the input
#      extension side empty; erasure happens only in the output.
#    - run extraction on the bar side still assumes each object is a horizontal monochrome run.
#    - the original ascending distance sort encoded near-first; the combo must reverse it.
#    - the original open-ended `range(..., step=length)` encoded edge-stop; the combo must limit
#      each bar to at most `length` target cells.
#    - erasing bars cannot happen before bar collection or during shooting, otherwise later shots
#      would lose access to their source length/color and side inference on the input would break.
#
# d) Placement legality
#    Bars remain contiguous and lie fully on one side of the separator. The generator keeps total
#    per-row occupied width within the sampled bar-side capacity, allows at most a one-cell gap to
#    the separator, and paints only onto the opposite side. Output-first generation constructs a
#    latent "bars + combo dots" board, derives the input by clearing the extension side, derives
#    the output by erasing the bars, and validates with `main(input) == output`.
#
# e) Visual observability
#    Human-solvable clues are all spatial and visible. Far-first is visible because the far bar
#    wins the first extension cell on a two-bar row. Limited-shot behavior is visible because a
#    short single bar produces only 1-2 dots rather than continuing to the edge. Bar erasure is
#    visible because the output keeps the separator but removes all source runs.
#
# f) Visual cleanliness
#    Active rows stay separated by at least one empty row. The output is cleaner than the input:
#    the bar side is blanked, while the extension side contains only one layer of dots due to the
#    no-overwrite rule. This keeps the fused logic legible even when two bars compete on a row.
#
# g) Color inference
#    No hidden mapping by color id is introduced. Colors are copied directly from the source bars,
#    and the inference burden comes from spacing and priority, not from arbitrary recoloring.
#
# Post-change solver assumptions guaranteed by the generator:
#    - The input extension side is entirely background, so side inference from mass is reliable.
#    - At least one witness row has two distinct-color bars, making far-first different from
#      near-first even after the source bars are erased.
#    - At least one separate showcase row has a single short bar and extension width >= 7, making
#      limited-shot behavior different from edge-stop behavior after erasure.


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


def _collect_bars(
    g: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
    bg: int,
) -> list[tuple[int, int, int, int, int, int]]:
    h, w = g.shape
    bars: list[tuple[int, int, int, int, int, int]] = []
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
    return bars


def _iter_targets(
    sep_col: int,
    length: int,
    bars_on_left: bool,
    w: int,
    *,
    limited: bool,
):
    start = sep_col + 1 if bars_on_left else sep_col - 1
    delta = length if bars_on_left else -length
    if limited:
        for shot in range(length):
            cc = start + shot * delta
            if 0 <= cc < w:
                yield cc
            else:
                break
        return
    if bars_on_left:
        yield from range(sep_col + 1, w, length)
    else:
        yield from range(sep_col - 1, -1, -length)


def _solve(
    grid: np.ndarray,
    *,
    far_first: bool,
    limited: bool,
    erase_source: bool,
) -> np.ndarray:
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
    bars.sort(key=(lambda t: (-t[0], t[1], t[2])) if far_first else (lambda t: (t[0], t[1], t[2])))

    for _dist, r, _c0, _c1, length, col in bars:
        if length <= 0:
            continue
        for cc in _iter_targets(sep_col, length, bars_on_left, w, limited=limited):
            if int(out[r, cc]) == bg:
                out[r, cc] = col

    if erase_source:
        for _dist, r, c0, c1, _length, _col in bars:
            out[r, c0 : c1 + 1] = bg

    return out


def _solve_combo(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, far_first=True, limited=True, erase_source=True)


def _solve_near_limited_erase(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, far_first=False, limited=True, erase_source=True)


def _solve_far_edge_erase(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, far_first=True, limited=False, erase_source=True)


def _solve_far_limited_keep(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, far_first=True, limited=True, erase_source=False)


def _solve_original(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, far_first=False, limited=False, erase_source=False)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build periodic dots on the empty side of a vertical separator, then erase the sources.

    Input structure:
    One nearly full-height monochrome separator column splits the grid into a source-bar side and
    an empty extension side. Active rows contain one or two horizontal monochrome bars on the same
    side of the separator; the other side is all background in the input.

    Rule:
    For each bar, use its length L as both the spacing and the shot count. Paint at most L target
    cells starting from the cell adjacent to the separator and stepping by L across the opposite
    side. When multiple bars compete on a row, process the farthest bar first and never overwrite
    a non-background cell. After all dots are painted, erase the original source bars.

    Key insight:
    The output hides the bars that generated the dots, so the solver must reconstruct them from
    the periodic spacing and from the fact that farther bars get priority on overlap cells.
    """
    return _solve_combo(grid)


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 7, 14

PALETTE_MIN, PALETTE_MAX = 5, 8
ACTIVE_MIN, ACTIVE_MAX = 2, 5

P_TWO_BARS = 0.45
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5
P_SEP_COLUMN_GAP = 0.2

MAX_BAR_LENGTH = 3
SHOWCASE_MAX_LENGTH = 2
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


def _sample_row_bars(
    rng: np.random.Generator,
    palette: list[int],
    primary: int,
    bar_side_len: int,
    *,
    force_showcase: bool = False,
    force_two_bars: bool = False,
) -> tuple[list[tuple[int, int]], int]:
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap
    if capacity <= 0:
        raise ValueError("bar-side capacity must stay positive")

    max_len = min(MAX_BAR_LENGTH, capacity)
    if force_showcase:
        show_len = int(rng.integers(1, min(SHOWCASE_MAX_LENGTH, max_len) + 1))
        return [(show_len, primary)], sep_gap

    two_bars = force_two_bars or (capacity >= 2 and rng.random() < P_TWO_BARS)
    if not two_bars:
        length = int(rng.integers(1, max_len + 1))
        return [(length, primary)], sep_gap

    max_near = min(MAX_BAR_LENGTH, capacity - 1)
    if max_near <= 0:
        if force_two_bars:
            raise ValueError("cannot force two bars with capacity < 2")
        length = int(rng.integers(1, max_len + 1))
        return [(length, primary)], sep_gap

    near_len = int(rng.integers(1, max_near + 1))
    max_far = min(MAX_BAR_LENGTH, capacity - near_len)
    if max_far <= 0:
        if force_two_bars:
            raise ValueError("cannot fit far bar after sampled near bar")
        return [(near_len, primary)], sep_gap

    far_len = int(rng.integers(1, max_far + 1))
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
    ordered = list(reversed(bars))
    w = out.shape[1]
    for length, color in ordered:
        for cc in _iter_targets(sep_col, length, bars_on_left, w, limited=True):
            if out[row, cc] == 0:
                out[row, cc] = color


def _erase_source_bars(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
    sep_gap: int,
) -> None:
    if bars_on_left:
        cursor = sep_col - sep_gap
        for length, _color in bars:
            out[row, cursor - length : cursor] = 0
            cursor -= length
    else:
        cursor = sep_col + 1 + sep_gap
        for length, _color in bars:
            out[row, cursor : cursor + length] = 0
            cursor += length


def _can_fit_limited_shots(ext_side_len: int, length: int) -> bool:
    """Need cells at offsets 0, L, 2L, ..., (L-1)L from the separator-adjacent cell."""
    return 1 + (length - 1) * length <= ext_side_len


def _has_two_bar_witness(row_bars_map: dict[int, tuple[list[tuple[int, int]], int]]) -> bool:
    for bars, _sep_gap in row_bars_map.values():
        if len(bars) >= 2 and bars[0][1] != bars[1][1]:
            return True
    return False


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

    showcase_row = int(rng.choice(active_rows))
    other_rows = [r for r in active_rows if r != showcase_row]
    if not other_rows:
        return None
    witness_row = int(rng.choice(other_rows))

    latent = np.zeros((h, w), dtype=int)
    latent[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        latent[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    row_bars_map: dict[int, tuple[list[tuple[int, int]], int]] = {}
    for idx, row in enumerate(active_rows):
        try:
            bars, sep_gap = _sample_row_bars(
                rng,
                palette,
                row_primary[idx],
                bar_side_len,
                force_showcase=(row == showcase_row),
                force_two_bars=(row == witness_row),
            )
        except ValueError:
            return None
        row_bars_map[row] = (bars, sep_gap)
        _place_bars(latent, row, sep_col, bars_on_left, bars, sep_gap)

    showcase_bars, _showcase_gap = row_bars_map[showcase_row]
    if len(showcase_bars) != 1 or showcase_bars[0][0] > SHOWCASE_MAX_LENGTH:
        return None
    if ext_side_len < EXT_SIDE_MIN:
        return None
    for bars, _sep_gap in row_bars_map.values():
        if any(not _can_fit_limited_shots(ext_side_len, length) for length, _color in bars):
            return None
    if not _has_two_bar_witness(row_bars_map):
        return None

    for row, (bars, _sep_gap) in row_bars_map.items():
        _paint_extension(latent, row, sep_col, bars_on_left, bars)

    inp = latent.copy()
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0

    out = latent.copy()
    for row, (bars, sep_gap) in row_bars_map.items():
        _erase_source_bars(out, row, sep_col, bars_on_left, bars, sep_gap)

    erased_pixels = int(np.sum((inp != 0) & (out == 0)))
    if erased_pixels <= 0:
        return None
    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None

    solved_combo = main(inp)
    if not np.array_equal(solved_combo, out):
        return None
    if np.array_equal(_solve_near_limited_erase(inp), solved_combo):
        return None
    if np.array_equal(_solve_far_edge_erase(inp), solved_combo):
        return None
    if np.array_equal(_solve_far_limited_keep(inp), solved_combo):
        return None
    if np.array_equal(_solve_original(inp), solved_combo):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
