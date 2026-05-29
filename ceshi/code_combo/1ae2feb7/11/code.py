# Combo variant of 1ae2feb7 (index 11)
# combo_spec: D2/1+D7/2+D15/0
# Name: 长度排序下的颜色循环与优先覆盖

from __future__ import annotations

import math

import numpy as np

from arc_utils import colors_present, get_bg


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior trace
#    - Preserve the original scene: one mostly-full vertical separator, bars on
#      only one side, ext_side empty in the input.
#    - Original row rule:
#        near bar paints its own color every L cells, then the farther bar tries
#        the same with its own step, and filled cells are never overwritten.
#    - Pure D2 union-coloring would erase the meaning of processing order and of
#      overwrite competition, because every candidate position would be colored
#      only once after taking a union.
#    - Therefore the fused rule keeps per-bar periodic hits, but changes what
#      each hit emits and how conflicts resolve:
#        1. Sort the row's bars by length descending.
#        2. The sorted bar colors form one shared cycle for that row.
#        3. Every bar still hits positions 0, L, 2L, ... on ext_side, but the
#           k-th hit uses cycle[k mod m] instead of the bar's own fixed color.
#        4. If multiple hits target the same cell, compare the attempted color
#           priorities induced by the same length-sorted cycle: earlier cycle
#           colors are stronger and may overwrite weaker existing colors.
#    - Example on the left of the separator:
#        input row = `7777774442........`
#        far bar: color 7, len 6; near bar: color 4, len 3; separator color 2
#        length order = [(6,7), (3,4)], so cycle = [7,4] and priority 7 > 4
#        process len-6 first: offsets 0,6,12 -> colors 7,4,7
#        process len-3 next:  offsets 0,3,6,9,12 -> colors 7,4,7,4,7
#        at offset 6 the earlier color is 4, the later color is 7, so the
#        later hit overwrites because 7 has higher priority in this row.
#    - Difference from the source task:
#        distance no longer decides either the global color progression or the
#        winner at conflicts; both now derive from length order.
#
# b) Complexity calibration
#    - Restrict rows to at most 2 bars, as in the source task, so the puzzle is
#      still visually readable.
#    - Force both "near longer" and "far longer" rows in generated tasks. This
#      prevents the rule from collapsing into "always near first" or "always far
#      first" heuristics.
#    - Force at least one row where color-priority overwrite actually changes
#      the result relative to no-overwrite. Otherwise D15 would be invisible.
#    - Keep palette size moderate and reuse the same row-local logic across
#      multiple active rows, so humans get repeated support from 2-4 examples.
#
# c) Implicit assumption review for main()
#    - `_find_sep_col` assumes a mostly full monochrome separator column. The
#      generator keeps the separator at H or H-1 cells of one non-bg color.
#    - Side detection by non-bg mass assumes ext_side is empty in the input.
#      The generator clears ext_side after rendering the output.
#    - Run detection treats every horizontal monochrome run as a bar. The
#      generator keeps exactly 1-2 adjacent bars per active row.
#    - Length order introduces a tie-break that is not human-visible if equal
#      lengths are allowed. The generator rejects equal-length 2-bar rows.
#    - Priority overwrite assumes all ext-side colors on a row come from that
#      row's cycle. The solver only paints on ext_side, and the generator never
#      seeds ext_side with other non-bg colors.
#
# d) Placement legality
#    - Bars are placed contiguously from the separator outward, optionally with
#      one gap before the nearest bar, exactly as in the source generator.
#    - Active rows remain separated by at least one empty row.
#    - Total bar length never exceeds the bar-side capacity.
#
# e) Visual observability
#    - A far-longer row shows that the first emitted color can belong to the far
#      bar, revealing length order instead of distance order.
#    - Repeated alternating colors along one row show that bars do not emit only
#      their source color anymore; they share a row-level cycle.
#    - A later repaint at a shared periodic position shows that overwrite is no
#      longer "first claim wins" but depends on the attempted color.
#
# f) Visual cleanliness
#    - Each row only interacts with its own ext_side cells; rows do not overlap.
#    - Restricting to 1-2 bars and moderate ext-side length keeps the dotted
#      pattern legible instead of becoming a dense texture.
#
# g) Color-rule inferability
#    - Priority is not based on raw color ids. It is induced by a visible
#      structural cue: the length-sorted order of the bars in that row.
#    - The same visible order also defines the emitted color cycle, coupling the
#      three dimensions into one inferable mechanism instead of three unrelated
#      tweaks.
# ---------------------------------------------------------------------------


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


def _ext_positions_abs(sep_col: int, width: int, bars_on_left: bool, step: int) -> list[int]:
    if step <= 0:
        return []
    if bars_on_left:
        return list(range(sep_col + 1, width, step))
    return list(range(sep_col - 1, -1, -step))


def _ordered_bars(
    bars: list[tuple[int, int]],
    order: str,
) -> list[tuple[int, int]]:
    indexed = list(enumerate(bars))  # bars are stored near->far
    if order == "near_first":
        indexed.sort(key=lambda t: t[0])
    elif order == "length_desc":
        indexed.sort(key=lambda t: (-t[1][0], t[0]))
    else:
        raise ValueError(order)
    return [bar for _, bar in indexed]


def _collect_row_bars(
    g: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
    bg: int,
) -> dict[int, list[tuple[int, int]]]:
    h, w = g.shape
    row_bars: dict[int, list[tuple[int, int]]] = {}
    for r in range(h):
        found: list[tuple[int, int, int]] = []
        if bars_on_left:
            for c0, c1, col in _iter_runs_1d(g[r, :], 0, sep_col, bg=bg):
                length = c1 - c0 + 1
                dist = (sep_col - 1) - c1
                found.append((dist, length, col))
        else:
            for c0, c1, col in _iter_runs_1d(g[r, :], sep_col + 1, w, bg=bg):
                length = c1 - c0 + 1
                dist = c0 - (sep_col + 1)
                found.append((dist, length, col))
        if found:
            found.sort(key=lambda t: t[0])
            row_bars[r] = [(length, int(col)) for _, length, col in found]
    return row_bars


def _paint_grid_row(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
    bg: int,
    *,
    order: str,
    color_mode: str,
    overwrite_mode: str,
) -> None:
    ordered = _ordered_bars(bars, order=order)
    cycle = [int(color) for _, color in ordered]
    priority = {int(color): idx for idx, color in enumerate(cycle)}
    weak_rank = len(priority) + 1

    for length, source_color in ordered:
        positions = _ext_positions_abs(sep_col, out.shape[1], bars_on_left, length)
        for idx, cc in enumerate(positions):
            if color_mode == "cycle":
                new_color = cycle[idx % len(cycle)]
            elif color_mode == "source":
                new_color = int(source_color)
            else:
                raise ValueError(color_mode)

            cur = int(out[row, cc])
            if cur == bg:
                out[row, cc] = new_color
                continue
            if overwrite_mode == "no_overwrite":
                continue
            if overwrite_mode != "priority":
                raise ValueError(overwrite_mode)
            if priority.get(new_color, weak_rank) < priority.get(cur, weak_rank):
                out[row, cc] = new_color


def _paint_ext_line(
    ext: np.ndarray,
    bars: list[tuple[int, int]],
    *,
    order: str,
    color_mode: str,
    overwrite_mode: str,
) -> None:
    ordered = _ordered_bars(bars, order=order)
    cycle = [int(color) for _, color in ordered]
    priority = {int(color): idx for idx, color in enumerate(cycle)}
    weak_rank = len(priority) + 1

    for length, source_color in ordered:
        for idx, pos in enumerate(range(0, ext.shape[0], length)):
            if color_mode == "cycle":
                new_color = cycle[idx % len(cycle)]
            elif color_mode == "source":
                new_color = int(source_color)
            else:
                raise ValueError(color_mode)

            cur = int(ext[pos])
            if cur == 0:
                ext[pos] = new_color
                continue
            if overwrite_mode == "no_overwrite":
                continue
            if overwrite_mode != "priority":
                raise ValueError(overwrite_mode)
            if priority.get(new_color, weak_rank) < priority.get(cur, weak_rank):
                ext[pos] = new_color


def _row_signature(
    ext_side_len: int,
    bars: list[tuple[int, int]],
    *,
    order: str,
    color_mode: str,
    overwrite_mode: str,
) -> tuple[int, ...]:
    ext = np.zeros(ext_side_len, dtype=int)
    _paint_ext_line(ext, bars, order=order, color_mode=color_mode, overwrite_mode=overwrite_mode)
    return tuple(map(int, ext))


def _row_signal_profile(ext_side_len: int, bars: list[tuple[int, int]]) -> dict[str, bool]:
    combo = _row_signature(
        ext_side_len, bars, order="length_desc", color_mode="cycle", overwrite_mode="priority"
    )
    return {
        "order": combo
        != _row_signature(
            ext_side_len, bars, order="near_first", color_mode="cycle", overwrite_mode="priority"
        ),
        "cycle": combo
        != _row_signature(
            ext_side_len, bars, order="length_desc", color_mode="source", overwrite_mode="priority"
        ),
        "priority": combo
        != _row_signature(
            ext_side_len, bars, order="length_desc", color_mode="cycle", overwrite_mode="no_overwrite"
        ),
        "original": combo
        != _row_signature(
            ext_side_len, bars, order="near_first", color_mode="source", overwrite_mode="no_overwrite"
        ),
    }


def _has_priority_override(long_len: int, short_len: int, ext_side_len: int) -> bool:
    if short_len <= 0 or long_len <= 0:
        return False
    shared = math.lcm(long_len, short_len)
    for pos in range(shared, ext_side_len, shared):
        if (pos // long_len) % 2 == 1 and (pos // short_len) % 2 == 0:
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each active row across the empty side using a fused periodic painter.

    Input structure:
    - A mostly full vertical separator column splits the grid into a bar side and
      an empty extension side.
    - Active rows contain 1 or 2 horizontal monochrome bars on the same side of
      the separator; inactive rows are otherwise background.

    Rule:
    - For each active row, read its bars from near the separator to far away.
    - Reorder those bars by length descending.
    - Use the reordered bar colors as a repeating cycle for every bar's periodic
      hits on ext_side.
    - When multiple hits target the same ext-side cell, the attempted color with
      higher priority wins; priority follows the same length-desc color order.

    Key insight:
    - Distance no longer determines ownership. The visually inferable control
      variable is bar length: it sets the cycle order, the processing order, and
      the color-overwrite hierarchy.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    out = g.copy()
    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < g.shape[1] else 0
    bars_on_left = left_mass >= right_mass

    row_bars = _collect_row_bars(g, sep_col, bars_on_left, bg)
    for row, bars in row_bars.items():
        _paint_grid_row(
            out,
            row,
            sep_col,
            bars_on_left,
            bars,
            bg,
            order="length_desc",
            color_mode="cycle",
            overwrite_mode="priority",
        )
    return out


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 6, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 5, 14

PALETTE_MIN, PALETTE_MAX = 5, 8

ACTIVE_MIN, ACTIVE_MAX = 3, 5
P_RANDOM_MULTI = 0.65
P_SEP_GAP = 0.12
P_SEP_COLUMN_GAP = 0.2

FORCED_ROW_TYPES = ("near_longer", "far_longer_override")
MAX_ATTEMPTS = 700


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    bar_pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(bar_pool)) + 1))
    palette = list(map(int, rng.choice(bar_pool, size=size, replace=False)))
    return sep_color, palette


def _candidate_pairs(
    capacity: int,
    ext_side_len: int,
    *,
    relation: str | None,
    require_priority_override: bool,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for near_len in range(1, capacity):
        max_far = capacity - near_len
        for far_len in range(1, max_far + 1):
            if near_len == far_len:
                continue
            if relation == "near_longer" and near_len <= far_len:
                continue
            if relation == "far_longer" and far_len <= near_len:
                continue
            long_len = max(near_len, far_len)
            short_len = min(near_len, far_len)
            if require_priority_override and not _has_priority_override(long_len, short_len, ext_side_len):
                continue
            pairs.append((near_len, far_len))
    return pairs


def _sample_row_bars(
    rng: np.random.Generator,
    palette: list[int],
    primary: int,
    bar_side_len: int,
    ext_side_len: int,
    row_type: str,
) -> tuple[list[tuple[int, int]], int] | None:
    if row_type == "single":
        sep_gap = 1 if rng.random() < P_SEP_GAP else 0
        capacity = bar_side_len - sep_gap
        if capacity <= 0:
            return None
        length = int(rng.integers(1, capacity + 1))
        return [(length, primary)], sep_gap

    if row_type == "random":
        if rng.random() >= P_RANDOM_MULTI:
            return _sample_row_bars(rng, palette, primary, bar_side_len, ext_side_len, "single")
        sep_gap = 1 if rng.random() < P_SEP_GAP else 0
        relation = None
        require_priority = False
        require_cycle = True
    elif row_type == "near_longer":
        sep_gap = 0
        relation = "near_longer"
        require_priority = False
        require_cycle = True
    elif row_type == "far_longer_override":
        sep_gap = 0
        relation = "far_longer"
        require_priority = True
        require_cycle = True
    else:
        raise ValueError(row_type)

    capacity = bar_side_len - sep_gap
    if capacity < 2:
        return None

    pairs = _candidate_pairs(
        capacity,
        ext_side_len,
        relation=relation,
        require_priority_override=require_priority,
    )
    if not pairs:
        return None

    order = rng.permutation(len(pairs))
    others = [c for c in palette if c != primary]
    if not others:
        return None

    for idx in order[: min(len(order), 120)]:
        near_len, far_len = pairs[int(idx)]
        far_color = int(rng.choice(others))
        bars = [(near_len, primary), (far_len, far_color)]
        profile = _row_signal_profile(ext_side_len, bars)
        if require_cycle and not profile["cycle"]:
            continue
        if row_type == "far_longer_override" and not (profile["order"] and profile["priority"]):
            continue
        return bars, sep_gap
    return None


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


def _render_output(
    base: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
    row_bars_map: dict[int, list[tuple[int, int]]],
    *,
    order: str,
    color_mode: str,
    overwrite_mode: str,
) -> np.ndarray:
    out = base.copy()
    for row, bars in row_bars_map.items():
        _paint_grid_row(
            out,
            row,
            sep_col,
            bars_on_left,
            bars,
            0,
            order=order,
            color_mode=color_mode,
            overwrite_mode=overwrite_mode,
        )
    return out


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
    min_active = max(ACTIVE_MIN, len(FORCED_ROW_TYPES))
    if max_active < min_active:
        return None
    active_count = int(rng.integers(min_active, max_active + 1))
    active_rows = _choose_active_rows(rng, h, active_count)
    active_set = set(active_rows)
    inactive_rows = [r for r in range(h) if r not in active_set]

    base = np.zeros((h, w), dtype=int)
    base[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        base[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    forced_map = {active_rows[idx]: row_type for idx, row_type in enumerate(FORCED_ROW_TYPES)}

    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    signal_counts = {"order": 0, "cycle": 0, "priority": 0, "original": 0}
    near_longer_rows = 0
    far_longer_rows = 0

    for idx, row in enumerate(active_rows):
        row_type = forced_map.get(row, "random")
        sampled = _sample_row_bars(
            rng,
            palette,
            row_primary[idx],
            bar_side_len,
            ext_side_len,
            row_type=row_type,
        )
        if sampled is None:
            return None
        bars, sep_gap = sampled
        row_bars_map[row] = bars
        _place_bars(base, row, sep_col, bars_on_left, bars, sep_gap)

        if len(bars) == 2:
            near_len, far_len = bars[0][0], bars[1][0]
            if near_len > far_len:
                near_longer_rows += 1
            elif far_len > near_len:
                far_longer_rows += 1
            profile = _row_signal_profile(ext_side_len, bars)
            for key, value in profile.items():
                signal_counts[key] += int(value)

    if near_longer_rows < 1 or far_longer_rows < 1:
        return None
    if signal_counts["order"] < 1 or signal_counts["priority"] < 1:
        return None
    min_cycle_rows = 2 if active_count >= 4 else 1
    if signal_counts["cycle"] < min_cycle_rows:
        return None

    out_combo = _render_output(
        base,
        sep_col,
        bars_on_left,
        row_bars_map,
        order="length_desc",
        color_mode="cycle",
        overwrite_mode="priority",
    )
    out_order_baseline = _render_output(
        base,
        sep_col,
        bars_on_left,
        row_bars_map,
        order="near_first",
        color_mode="cycle",
        overwrite_mode="priority",
    )
    out_cycle_baseline = _render_output(
        base,
        sep_col,
        bars_on_left,
        row_bars_map,
        order="length_desc",
        color_mode="source",
        overwrite_mode="priority",
    )
    out_priority_baseline = _render_output(
        base,
        sep_col,
        bars_on_left,
        row_bars_map,
        order="length_desc",
        color_mode="cycle",
        overwrite_mode="no_overwrite",
    )
    out_original = _render_output(
        base,
        sep_col,
        bars_on_left,
        row_bars_map,
        order="near_first",
        color_mode="source",
        overwrite_mode="no_overwrite",
    )

    if np.array_equal(out_combo, out_order_baseline):
        return None
    if np.array_equal(out_combo, out_cycle_baseline):
        return None
    if np.array_equal(out_combo, out_priority_baseline):
        return None
    if np.array_equal(out_combo, out_original):
        return None

    inp = base.copy()
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if not np.array_equal(main(inp), out_combo):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
