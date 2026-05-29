# Variant of 1ae2feb7
# Dimension: D4 — 终止条件/目标 (dst)
# Change: null（网格边缘）——射线延伸直到到达网格的对侧边缘，没有其他终止条件... → 涂满指定数量后停止

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for the D4 variant (source_task: None, so no extra reference code):
#
# a) Concrete behavior and trace
#    The separator and bar layout stay the same as the original task: one vertical separator
#    column, bars on exactly one side, and the opposite side starts empty in the input.
#    The changed part is only how far each bar extends its periodic marks. A bar of length L
#    still uses period L, but it now only considers the first L periodic target cells past the
#    separator instead of scanning all the way to the far edge.
#    Example with bars on the left, separator at column 5:
#      near bar: length 2, color A  -> old targets 6,8,10,12,... ; new targets 6,8
#      far  bar: length 1, color B  -> old targets 6,7,8,9,...   ; new targets 6
#    Because processing order is still nearest-first and there is still no overwrite, the old
#    output keeps adding farther marks after the first few periods, while the new output stops.
#
# b) Complexity calibration
#    If long bars (length >= 4) are kept under the original width bounds, the new stop rule is
#    often visually identical to the old edge-stop rule because the grid edge is reached first.
#    To keep D4 observable, generation restricts bar lengths to 1..3 and forces each sample to
#    include one single-bar "showcase" row with length 1 or 2 on a sufficiently wide extension
#    side. That preserves the original object vocabulary while making the changed termination
#    condition visible in 2-3 examples.
#
# c) Old implicit assumptions audit
#    - `_find_sep_col`: still valid because separator geometry is unchanged.
#    - side detection by non-background mass: still valid because the input extension side stays
#      empty, so the bar side remains the denser side.
#    - bar extraction and sorting by distance-to-separator: still required; nearer bars must keep
#      priority when multiple bars on a row would hit the same extension cells.
#    - the original `range(..., step=length)` implicitly assumed "keep going until edge". This is
#      the only behavior that must change: the solver now enumerates at most `length` targets.
#
# d) Placement legality
#    Bars are still contiguous monochrome runs placed entirely on one side of the separator.
#    Generation keeps the original adjacency/gap rules, clamps total bar width to the available
#    side, and validates by re-running `main(input)`.
#
# e) Visual observability
#    No hidden ordering by color id is introduced. All ordering is spatial: nearest bar first,
#    then row/column tie-breakers exactly as before. The new clue is purely visible: the number
#    of periodic marks equals the bar length instead of extending to the edge.
#
# f) Visual cleanliness
#    Active rows stay separated by at least one empty row, and only the extension side is filled
#    in the output, so the scene remains easy to parse even when two bars compete on one row.
#
# g) Color inference
#    Colors are unchanged: each bar paints using its own visible color. No new color mapping is
#    added, so the human solver only has to infer the new stop condition.
#
# Post-change solver assumptions that the generator guarantees:
#    - At least one active row is a single short bar (length 1 or 2), so old/new outputs differ.
#    - Extension-side width is at least 6, which makes a length-2 showcase row visibly shorter
#      under the new rule than under the original edge-stop rule.


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
    bars: list[tuple[int, int, int, int, int, int]] = []
    h, w = g.shape
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
    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    return bars


def _iter_targets_to_edge(sep_col: int, length: int, bars_on_left: bool, w: int):
    if bars_on_left:
        yield from range(sep_col + 1, w, length)
    else:
        yield from range(sep_col - 1, -1, -length)


def _iter_targets_limited(sep_col: int, length: int, bars_on_left: bool, w: int):
    start = sep_col + 1 if bars_on_left else sep_col - 1
    delta = length if bars_on_left else -length
    for shot in range(length):
        cc = start + shot * delta
        if 0 <= cc < w:
            yield cc
        else:
            break


def _run_solver(grid: np.ndarray, *, limited: bool) -> np.ndarray:
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
    for _, r, _c0, _c1, length, col in bars:
        if length <= 0:
            continue
        targets = (
            _iter_targets_limited(sep_col, length, bars_on_left, w)
            if limited
            else _iter_targets_to_edge(sep_col, length, bars_on_left, w)
        )
        for cc in targets:
            if int(out[r, cc]) == bg:
                out[r, cc] = col

    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _run_solver(grid, limited=True)


def _main_to_edge(grid: np.ndarray) -> np.ndarray:
    """Original task behavior, kept only for generator-side observability checks."""
    return _run_solver(grid, limited=False)


# --------------- generator for the D4 variant ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 6, 14

PALETTE_MIN, PALETTE_MAX = 5, 8
ACTIVE_MIN, ACTIVE_MAX = 2, 5

P_TWO_BARS = 0.45
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5
P_SEP_COLUMN_GAP = 0.2

MAX_BAR_LENGTH = 3
SHOWCASE_MAX_LENGTH = 2
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
    *,
    force_showcase: bool,
) -> tuple[list[tuple[int, int]], int]:
    """Return (bars, sep_gap) where bars is a list of (length, color) ordered near->far."""
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap
    if capacity <= 0:
        raise ValueError("bar side too narrow for sampled separator gap")

    max_len = min(MAX_BAR_LENGTH, capacity)
    if force_showcase:
        show_len = int(rng.integers(1, min(SHOWCASE_MAX_LENGTH, max_len) + 1))
        return [(show_len, primary)], sep_gap

    can_two = capacity >= 2
    if not can_two or rng.random() >= P_TWO_BARS:
        length = int(rng.integers(1, max_len + 1))
        return [(length, primary)], sep_gap

    max_near = min(MAX_BAR_LENGTH, capacity - 1)
    near_len = int(rng.integers(1, max_near + 1))
    max_far = min(MAX_BAR_LENGTH, capacity - near_len)
    if max_far <= 0:
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


def _paint_extension_limited(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
) -> None:
    w = out.shape[1]
    for length, color in bars:
        for cc in _iter_targets_limited(sep_col, length, bars_on_left, w):
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
    showcase_row = int(rng.choice(active_rows))

    out = np.zeros((h, w), dtype=int)
    out[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        out[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for idx, row in enumerate(active_rows):
        bars, sep_gap = _sample_row_bars(
            rng,
            palette,
            row_primary[idx],
            bar_side_len,
            force_showcase=(row == showcase_row),
        )
        row_bars_map[row] = bars
        _place_bars(out, row, sep_col, bars_on_left, bars, sep_gap)

    for row, bars in row_bars_map.items():
        _paint_extension_limited(out, row, sep_col, bars_on_left, bars)

    inp = out.copy()
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if ext_side_len < EXT_SIDE_MIN:
        return None
    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_main_to_edge(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
