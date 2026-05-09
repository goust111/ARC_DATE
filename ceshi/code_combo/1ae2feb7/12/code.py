# Combo variant of 1ae2feb7 (index 12)
# combo_spec: D1/1+D10/0+D15/2
# Name: 双向密集射线冲突留白

from __future__ import annotations

from functools import lru_cache
from itertools import combinations, permutations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for the combo variant
# a) Concrete behavior and trace
#    The preserved scaffold is still the original task: a nearly solid vertical separator column
#    splits the grid into one bar side and one extension side, and active rows contain horizontal
#    monochrome bars only on the bar side. The combo changes three linked parts at once:
#    - D10/0: every active row now has 3 or 4 adjacent bars, all with distinct colors and lengths
#    - D1/1: each bar emits periodic rays to both sides of the separator, not just across it
#    - D15/2: rays do not use first-come wins; a background cell is colored only if exactly one
#      ray claims it, otherwise that target cell stays background
#    A concrete left-bar row with separator at c=8 and near->far bars [(2, A), (3, B), (4, C)]:
#    - original task: only the right side is painted, near-first, so offsets 0,2,4,6... from A
#      occupy many cells before B and C arrive
#    - D1 alone: the same no-overwrite rule also runs leftward; new dots appear in the outer
#      background on the bar side
#    - this combo: all three bars claim both directions simultaneously, but only on cells that are
#      background in the input. On the right side, the first extension cell is claimed by A/B/C so
#      it stays 0; later cells such as offset 2 may be unique to A and stay colored. On the left
#      side, inner positions fall on existing bars and are ignored, while farther outer background
#      cells can become colored if uniquely claimed or remain 0 if multiple bar periods coincide.
#    The visible difference is thus not merely "more bars" or "two-sided filling" but a dense,
#    hole-rich periodic texture on both sides with regular cancellation gaps.
# b) Complexity calibration
#    The generator forces 3 or 4 bars on every active row, keeps row spacing >= 2, and reserves
#    outer background on the bar side so the bidirectional rule is always visible. It also filters
#    row-length patterns so there are actual repeated collisions and at least one bar-side addition.
#    This avoids degenerate samples where the combo collapses to "many bars but visually same as
#    original" or where D1 becomes invisible because the bars fill the whole bar side.
# c) Hidden-assumption audit on the original main()
#    - `_find_sep_col` assumed one mostly solid monochrome separator column. Still valid.
#    - `bars_on_left` inferred the source side from non-background mass. Still valid because the
#      generator keeps the whole target area blank in the input.
#    - `_iter_runs_1d` assumed bars are contiguous monochrome runs. Still valid; D10 only raises
#      the run count per row to 3-4.
#    - The original bar loop encoded "one-sided" and "first paint sticks". Both assumptions break:
#      rays must be generated on both sides, and conflict cells can no longer inherit the first bar.
#    - A new subtle assumption is needed: source bars themselves are immutable anchors, so the
#      conflict-blank rule applies only to background cells that rays try to paint. Otherwise
#      bidirectional conflicts would erase the visible source objects and make the task unstable.
#      The solver enforces this by counting claims only on input-background cells.
# d) Placement legality
#    Each row's bars are still stacked contiguously from the separator outward and never overlap.
#    Length tuples are chosen with distinct positive lengths whose sum fits within the bar-side
#    capacity after reserving an outer margin. The optional separator gap remains at most 1 cell.
#    Active rows remain separated by at least one empty row, so row-local ray logic never clashes
#    vertically.
# e) Visual observability
#    A human can infer the combo from a few examples because the control variables are all visible:
#    the number of bars per row, each bar's length, the separator, the fact that both sides gain
#    periodic dots, and the regular holes where multiple periods coincide. No invisible ordering by
#    color id or hidden lookup table is used. The generator rejects samples unless bidirectional
#    additions and collision-created blanks both occur.
# f) Visual cleanliness
#    Although the ray field is denser than the source task, clutter is controlled by keeping rows
#    sparse, using only 3-4 bars, and restricting all interaction to within each row. The separator
#    still cleanly splits the layout, and the bar-side additions only occupy reserved outer
#    background rather than overwriting the source bars.
# g) Change inferability
#    Colors are copied directly from the uniquely claiming bar; there is no extra color remapping.
#    That keeps the inference burden on spatial-period reasoning instead of arbitrary color logic.
# 9) New assumptions introduced by this combo
#    The new solver assumes valid inputs leave every non-source target cell as background before
#    simulation and that every non-background run on the source side is a real bar. The generator
#    guarantees both properties by constructing only separator + contiguous bars in the input and
#    validating that the combo output differs from both the base task and the partial variants.


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


def _ray_scans(
    w: int,
    sep_col: int,
    bars_on_left: bool,
    step: int,
    *,
    bidirectional: bool,
) -> tuple[range, ...]:
    if bars_on_left:
        scans = [range(sep_col + 1, w, step)]
        if bidirectional:
            scans.append(range(sep_col - 1, -1, -step))
        return tuple(scans)
    scans = [range(sep_col - 1, -1, -step)]
    if bidirectional:
        scans.append(range(sep_col + 1, w, step))
    return tuple(scans)


def _collect_row_bars(
    g: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
    bg: int,
) -> dict[int, list[tuple[int, int, int, int]]]:
    h, w = g.shape
    row_bars: dict[int, list[tuple[int, int, int, int]]] = {}
    for r in range(h):
        runs = (
            _iter_runs_1d(g[r, :], 0, sep_col, bg=bg)
            if bars_on_left
            else _iter_runs_1d(g[r, :], sep_col + 1, w, bg=bg)
        )
        for c0, c1, col in runs:
            length = c1 - c0 + 1
            dist = (sep_col - 1) - c1 if bars_on_left else c0 - (sep_col + 1)
            row_bars.setdefault(r, []).append((dist, c0, length, col))
    for r in row_bars:
        row_bars[r].sort(key=lambda t: (t[0], t[1]))
    return row_bars


def _simulate(
    grid: np.ndarray,
    *,
    bidirectional: bool,
    conflict_blank: bool,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if h == 0 or w == 0:
        return g.copy()

    bg = _bg_color(g)
    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    out = g.copy()
    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < w else 0
    bars_on_left = left_mass >= right_mass

    row_bars = _collect_row_bars(g, sep_col, bars_on_left, bg)
    for r, bars in row_bars.items():
        if conflict_blank:
            claims: dict[int, int] = {}
            claim_color: dict[int, int] = {}
            for _dist, _c0, length, col in bars:
                if length <= 0:
                    continue
                for positions in _ray_scans(
                    w,
                    sep_col,
                    bars_on_left,
                    length,
                    bidirectional=bidirectional,
                ):
                    for cc in positions:
                        if int(g[r, cc]) != bg:
                            continue
                        claims[cc] = claims.get(cc, 0) + 1
                        claim_color.setdefault(cc, col)
            for cc, count in claims.items():
                out[r, cc] = claim_color[cc] if count == 1 else bg
            continue

        for _dist, _c0, length, col in bars:
            if length <= 0:
                continue
            for positions in _ray_scans(
                w,
                sep_col,
                bars_on_left,
                length,
                bidirectional=bidirectional,
            ):
                for cc in positions:
                    if int(out[r, cc]) == bg:
                        out[r, cc] = col
    return out


def _solve_base(grid: np.ndarray) -> np.ndarray:
    return _simulate(grid, bidirectional=False, conflict_blank=False)


def _solve_d1(grid: np.ndarray) -> np.ndarray:
    return _simulate(grid, bidirectional=True, conflict_blank=False)


def _solve_d15(grid: np.ndarray) -> np.ndarray:
    return _simulate(grid, bidirectional=False, conflict_blank=True)


def _solve_combo(grid: np.ndarray) -> np.ndarray:
    return _simulate(grid, bidirectional=True, conflict_blank=True)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overview:
        Extend every source bar as a periodic ray on both sides of the separator, then keep only
        the background cells that are claimed by exactly one bar.

    Input structure:
        A near-solid vertical separator column splits the grid into a source side and a blank
        target area. Active rows contain 3 or 4 adjacent monochrome bars on exactly one side of
        the separator; inactive rows contain only background except for the separator column.

    Rule:
        For each active row, every bar emits dots with period equal to its length on the opposite
        side of the separator and also back into the outer background of the source side. Existing
        source-bar cells are preserved. Among the background cells reached by these rays, a cell is
        colored only when exactly one bar claims it; if two or more rays land there, the cell stays
        background.

    Key insight:
        The output is determined row-by-row by combining all bar lengths into one bidirectional
        claim map. The visible pattern is a dense superposition of periodic rays with systematic
        holes at collision points.
    """
    return _solve_combo(grid)


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 16, 22

BAR_SIDE_MIN, BAR_SIDE_MAX = 8, 16
EXT_SIDE_MIN, EXT_SIDE_MAX = 5, 14

PALETTE_MIN, PALETTE_MAX = 6, 8

ACTIVE_MIN, ACTIVE_MAX = 2, 5

P_FOUR_BARS = 0.35
P_SEP_GAP = 0.1
P_SEP_COLUMN_GAP = 0.2

OUTER_MARGIN_MIN = 2
MIN_ROW_CONFLICTS = 3
MAX_ATTEMPTS = 500


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    bar_pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(bar_pool)) + 1))
    palette = list(map(int, rng.choice(bar_pool, size=size, replace=False)))
    return sep_color, palette


def _row_claim_summary(
    bar_side_len: int,
    ext_side_len: int,
    sep_gap: int,
    lengths: tuple[int, ...],
) -> dict[str, int]:
    w = bar_side_len + 1 + ext_side_len
    sep_col = bar_side_len
    row = np.zeros(w, dtype=int)
    row[sep_col] = 9

    cursor = sep_col - sep_gap
    for idx, length in enumerate(lengths, start=1):
        row[cursor - length : cursor] = idx
        cursor -= length

    claims: dict[int, list[int]] = {}
    for idx, length in enumerate(lengths):
        for positions in _ray_scans(w, sep_col, True, length, bidirectional=True):
            for cc in positions:
                if row[cc] != 0:
                    continue
                claims.setdefault(cc, []).append(idx)

    bar_unique = 0
    ext_unique = 0
    conflicts = 0
    visible_bars: set[int] = set()
    for cc, owners in claims.items():
        if len(owners) == 1:
            visible_bars.add(owners[0])
            if cc < sep_col:
                bar_unique += 1
            else:
                ext_unique += 1
        else:
            conflicts += 1

    return {
        "bar_unique": bar_unique,
        "ext_unique": ext_unique,
        "conflicts": conflicts,
        "visible_bars": len(visible_bars),
    }


@lru_cache(maxsize=None)
def _feasible_length_orders(
    bar_side_len: int,
    ext_side_len: int,
    sep_gap: int,
    bar_count: int,
) -> tuple[tuple[int, ...], ...]:
    max_total = bar_side_len - sep_gap - OUTER_MARGIN_MIN
    min_total = bar_count * (bar_count + 1) // 2
    if max_total < min_total:
        return ()

    orders: list[tuple[int, ...]] = []
    for comb in combinations(range(1, max_total + 1), bar_count):
        if sum(comb) > max_total:
            continue
        for perm in permutations(comb):
            summary = _row_claim_summary(bar_side_len, ext_side_len, sep_gap, perm)
            min_visible = 2
            if summary["bar_unique"] < 1:
                continue
            if summary["ext_unique"] < 2:
                continue
            if summary["conflicts"] < MIN_ROW_CONFLICTS:
                continue
            if summary["visible_bars"] < min_visible:
                continue
            orders.append(tuple(map(int, perm)))
    return tuple(orders)


@lru_cache(maxsize=None)
def _feasible_geometries(bar_count: int) -> tuple[tuple[int, int, int], ...]:
    geometries: list[tuple[int, int, int]] = []
    for w in range(W_MIN, W_MAX + 1):
        lo = max(BAR_SIDE_MIN, w - 1 - EXT_SIDE_MAX)
        hi = min(BAR_SIDE_MAX, w - 1 - EXT_SIDE_MIN)
        if lo > hi:
            continue
        for bar_side_len in range(lo, hi + 1):
            ext_side_len = w - 1 - bar_side_len
            if any(
                _feasible_length_orders(bar_side_len, ext_side_len, sep_gap, bar_count)
                for sep_gap in (0, 1)
            ):
                geometries.append((w, bar_side_len, ext_side_len))
    return tuple(geometries)


def _sample_row_bars(
    rng: np.random.Generator,
    palette: list[int],
    bar_side_len: int,
    ext_side_len: int,
    bar_count: int,
) -> tuple[list[tuple[int, int]], int]:
    sep_gap_order = [1, 0] if rng.random() < P_SEP_GAP else [0, 1]
    for sep_gap in sep_gap_order:
        candidates = _feasible_length_orders(bar_side_len, ext_side_len, sep_gap, bar_count)
        if not candidates:
            continue
        lengths = list(map(int, candidates[int(rng.integers(0, len(candidates)))]))
        colors = list(map(int, rng.choice(palette, size=len(lengths), replace=False)))
        return list(zip(lengths, colors)), sep_gap

    raise ValueError(f"no feasible dense multi-bar configuration for {bar_count} bars")


def _place_bars(
    inp: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
    sep_gap: int,
) -> None:
    if bars_on_left:
        cursor = sep_col - sep_gap
        for length, color in bars:
            inp[row, cursor - length : cursor] = color
            cursor -= length
    else:
        cursor = sep_col + 1 + sep_gap
        for length, color in bars:
            inp[row, cursor : cursor + length] = color
            cursor += length


def _collect_masks(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    sep_col, _sep_color = _find_sep_col(g, bg)
    bar_mask = np.zeros_like(g, dtype=bool)
    ext_mask = np.zeros_like(g, dtype=bool)
    if sep_col is None:
        return bar_mask, ext_mask

    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < g.shape[1] else 0
    bars_on_left = left_mass >= right_mass
    if bars_on_left:
        bar_mask[:, :sep_col] = True
        ext_mask[:, sep_col + 1 :] = True
    else:
        bar_mask[:, sep_col + 1 :] = True
        ext_mask[:, :sep_col] = True
    return bar_mask, ext_mask


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    combo = _solve_combo(grid)
    base = _solve_base(grid)
    d1 = _solve_d1(grid)
    d15 = _solve_d15(grid)
    bar_mask, ext_mask = _collect_masks(grid)
    bg_targets = grid == 0
    all_targets = bg_targets & (bar_mask | ext_mask)
    return {
        "bar_side_added": int(np.count_nonzero((combo != 0) & bg_targets & bar_mask)),
        "ext_side_added": int(np.count_nonzero((combo != 0) & bg_targets & ext_mask)),
        "blank_vs_d1": int(np.count_nonzero((combo == 0) & (d1 != 0) & all_targets)),
        "diff_vs_base": int(np.count_nonzero((combo != base) & all_targets)),
        "diff_vs_d15": int(np.count_nonzero((combo != d15) & all_targets)),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(H_MIN, H_MAX + 1))
    preferred_bar_count = 4 if rng.random() < P_FOUR_BARS else 3
    geometry_pool = _feasible_geometries(preferred_bar_count)
    if not geometry_pool:
        return None
    w, bar_side_len, ext_side_len = geometry_pool[int(rng.integers(0, len(geometry_pool)))]

    bars_on_left = bool(rng.integers(0, 2) == 0)
    sep_col = bar_side_len if bars_on_left else (w - 1 - bar_side_len)

    sep_color, palette = _sample_palette(rng)

    max_active = min(ACTIVE_MAX, (h + 1) // 2)
    active_count = int(rng.integers(ACTIVE_MIN, max_active + 1))
    active_rows = _choose_active_rows(rng, h, active_count)
    active_set = set(active_rows)
    inactive_rows = [r for r in range(h) if r not in active_set]

    inp = np.zeros((h, w), dtype=int)
    inp[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        inp[gap_row, sep_col] = 0

    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for row in active_rows:
        try:
            bars, sep_gap = _sample_row_bars(
                rng,
                palette,
                bar_side_len,
                ext_side_len,
                preferred_bar_count,
            )
        except ValueError:
            return None
        row_bars_map[row] = bars
        _place_bars(inp, row, sep_col, bars_on_left, bars, sep_gap)

    if any(len(bars) != preferred_bar_count for bars in row_bars_map.values()):
        return None

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None

    out = _solve_combo(inp)
    if not np.array_equal(main(inp), out):
        return None

    metrics = _variant_metrics(inp)
    if metrics["bar_side_added"] <= 0:
        return None
    if metrics["blank_vs_d1"] <= 0:
        return None
    if metrics["diff_vs_base"] <= metrics["bar_side_added"]:
        return None
    if metrics["diff_vs_d15"] <= 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
