# Variant of 1ae2feb7
# Dimension: D8 — 间隔计算规则 (meta)
# Change: interval = 色条像素数（bar_length）——直接以色条的水平长度作为周期性涂点的步长
#   → interval = 色条所在行的行号（这里实现为 0-based 行号 r 的 1-based 版本，即 step = r + 1）

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for the D8 variant
# ----------------------------------
# a) Concrete behavior:
#    The separator is still one nearly-full vertical monochrome column. Bars still live entirely
#    on one side, and the opposite side starts empty in the input. The original task paints each
#    active row outward from the separator using each bar's horizontal length as the stride.
#    Here the painted stride no longer depends on bar length. Instead every bar on row r uses
#    step = r + 1, so upper rows paint more densely than lower rows.
#    Small trace:
#      - Original: row r=5 with a 3-cell bar paints ext columns at offsets 0,3,6,9,...
#      - Variant:  the same row paints offsets 0,6,12,... because step = r+1 = 6.
#      - Difference: the bar's visible width no longer controls density; vertical placement does.
# b) Complexity calibration:
#    If a row contains two bars, the original rule lets both contribute because their lengths may
#    differ. Under the row-based stride, two bars on the same row would share the same phase and
#    stride, so the farther bar would usually be completely masked by the nearer bar. To keep the
#    D8 change observable instead of cluttered, the generator primarily uses one bar per active row.
#    The solver still supports multiple runs and preserves the original nearest-first ordering.
# c) Implicit assumptions checked against main():
#      - _find_sep_col only assumes one mostly solid monochrome separator column. Still valid.
#      - bars_on_left mass comparison only assumes one side contains the bars. Still valid.
#      - _iter_runs_1d still assumes bars are monochrome horizontal runs. Still valid.
#      - Sorting by distance to separator still matters because "nearest first, no overwrite"
#        remains part of the task, even though rows are typically single-bar in generation.
#      - The old line `step = length` is the exact D8 dependency that breaks the variant.
#        It must become row-derived instead of bar-derived.
# d) Placement legality:
#    Generation still keeps all bars on one side, inside bounds, optionally with a 1-cell gap to
#    the separator, and keeps the ext side empty in the input. Active rows keep spacing >= 2 so
#    horizontal patterns do not visually merge between adjacent rows.
# e) Visual observability:
#    The generator enforces row spread (at least one upper and one lower active row) and rejects
#    samples where the variant output matches the original length-based rule. This ensures the new
#    control variable (row index) actually changes the visible pattern.
# f) Visual cleanliness:
#    Only the ext side is painted in the output, one row at a time, with no overwrite. There is no
#    extra marker or mixed-object overlay beyond the separator and the source bars.
# g) Color inference:
#    Colors are unchanged by the variant. The only change is spatial rhythm, so no hidden color-id
#    mapping is introduced.
#
# Assumption audit after the change:
#   - main() now assumes the intended stride is 1-based row number. generate_input() guarantees
#     all active rows are within the grid and distributed enough that multiple stride scales appear.
#   - main() still allows multiple bars per row. generate_input() usually avoids them because they
#     become causally redundant under shared row-based stride/phase.
#   - main() still depends on background being the majority color and separator being the only
#     nearly-full monochrome column. generate_input() preserves both conditions.


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


def _row_step(r: int) -> int:
    return int(r) + 1


def _collect_bars(g: np.ndarray, sep_col: int, bars_on_left: bool, bg: int):
    h, w = g.shape
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
    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    return bars


def _solve_with_step_rule(grid: np.ndarray, *, use_row_step: bool) -> np.ndarray:
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
        step = _row_step(r) if use_row_step else length
        if step <= 0:
            continue
        if bars_on_left:
            for cc in range(sep_col + 1, w, step):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col
        else:
            for cc in range(sep_col - 1, -1, -step):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve_with_step_rule(grid, use_row_step=True)


# --------------- generator (D8 row-index stride variant) ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 7, 14

PALETTE_MIN, PALETTE_MAX = 5, 8

ACTIVE_MIN, ACTIVE_MAX = 3, 5

P_SEP_GAP = 0.15
P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 500


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    bar_pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(bar_pool)) + 1))
    palette = list(map(int, rng.choice(bar_pool, size=size, replace=False)))
    return sep_color, palette


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    top_band = list(range(min(3, h)))
    bottom_band = list(range(max(0, h - 4), h))

    for _ in range(200):
        top = int(rng.choice(top_band))
        bottom = int(rng.choice(bottom_band))
        if abs(top - bottom) < 2:
            continue
        selected = [top, bottom]
        while len(selected) < count:
            candidates = [r for r in range(h) if all(abs(r - s) >= 2 for s in selected)]
            if not candidates:
                break
            selected.append(int(rng.choice(candidates)))
        if len(selected) == count:
            return sorted(selected)

    raise RuntimeError("failed to choose informative active rows")


def _sample_row_bar(
    rng: np.random.Generator,
    primary: int,
    bar_side_len: int,
    row: int,
    ext_side_len: int,
) -> tuple[list[tuple[int, int]], int]:
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap
    if capacity <= 0:
        raise RuntimeError("invalid bar capacity")

    step = _row_step(row)
    informative = [
        length
        for length in range(1, capacity + 1)
        if length != step and (step <= ext_side_len - 1 or length <= ext_side_len - 1)
    ]
    if informative:
        length = int(rng.choice(informative))
    else:
        fallback = [length for length in range(1, capacity + 1) if length != step]
        choices = fallback if fallback else list(range(1, capacity + 1))
        length = int(rng.choice(choices))
    return [(length, primary)], sep_gap


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
    step = _row_step(row)
    for _length, color in bars:
        if bars_on_left:
            positions = range(sep_col + 1, w, step)
        else:
            positions = range(sep_col - 1, -1, -step)
        for cc in positions:
            if out[row, cc] == 0:
                out[row, cc] = color


def _ext_metrics(out: np.ndarray, sep_col: int, bars_on_left: bool, active_rows: list[int]) -> tuple[int, int]:
    if bars_on_left:
        ext = out[:, sep_col + 1 :]
    else:
        ext = out[:, :sep_col]
    painted = int(np.sum(ext != 0))
    rows_with_repeat = 0
    for r in active_rows:
        if int(np.sum(ext[r] != 0)) >= 2:
            rows_with_repeat += 1
    return painted, rows_with_repeat


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

    out = np.zeros((h, w), dtype=int)
    out[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        out[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for idx, row in enumerate(active_rows):
        bars, sep_gap = _sample_row_bar(rng, row_primary[idx], bar_side_len, row, ext_side_len)
        row_bars_map[row] = bars
        _place_bars(out, row, sep_col, bars_on_left, bars, sep_gap)

    for row, bars in row_bars_map.items():
        _paint_extension(out, row, sep_col, bars_on_left, bars)

    painted, rows_with_repeat = _ext_metrics(out, sep_col, bars_on_left, active_rows)
    if painted == 0 or rows_with_repeat < 2:
        return None

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
    if np.array_equal(_solve_with_step_rule(inp, use_row_step=False), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
