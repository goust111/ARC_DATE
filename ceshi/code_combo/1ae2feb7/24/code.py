# Combo variant of 1ae2feb7 (index 24)
# combo_spec: D10/0+D11/2+D13/0
# Name: 多色条交错涂色颜色映射
#
# Design analysis
# a) Concrete form + behavior trace
#    - Geometry stays close to 1ae2feb7: one mostly solid vertical separator, all
#      source bars on one side, ext_side blank in the input.
#    - The fused change is row-local and has three parts at once:
#        1) every active row has 3 or 4 adjacent bars;
#        2) bars are processed near->far, but bar i starts at
#           floor(i * length / n_bars) instead of always starting at ext offset 0;
#        3) the painted color is COLOR_MAP[source_color], not the source color itself.
#    - Example row with bars on the left, separator at column 8, ext_side length 9:
#      near->far lengths [5, 3, 4], source colors [1, 2, 6], so mapped colors [2, 3, 1].
#      Original task would paint:
#        bar0: offsets 0,5      with color 1
#        bar1: offsets 0,3,6    with color 2, skipping filled cells
#        bar2: offsets 0,4,8    with color 6, skipping filled cells
#      This combo paints:
#        n=3, offsets are floor(i*L/3) => [0,1,2]
#        bar0: offsets 0,5      with mapped color 2
#        bar1: offsets 1,4,7    with mapped color 3
#        bar2: offsets 2,6      with mapped color 1
#      The difference is not "more bars" plus "different colors" independently;
#      the shifted starts make the 3-4 mapped periodic rays interleave into a denser,
#      more regular stripe field on ext_side.
# b) Complexity calibration
#    - A per-sample random color map would be too hard to infer, so the mapping is
#      fixed globally on a 5-color cycle:
#        1->2, 2->3, 3->4, 4->6, 6->1.
#    - Every sample exposes all five source colors somewhere on the bar side, and every
#      generated bar is required to contribute at least one visible mapped dot.
#    - Active rows are forced to have 3 or 4 bars only; this prevents regression to the
#      original 1-2 bar regime and makes the staggered starts visually obvious.
# c) Implicit-assumption audit of the original solver
#    - `_find_sep_col`: still valid because the separator remains a mostly solid single-color
#      column with at most one missing cell.
#    - `bars_on_left` from left/right non-bg mass: still valid because the input keeps the
#      extension side entirely background.
#    - `_iter_runs_1d`: still valid because each bar is a monochrome contiguous run.
#    - Original paint loop assumed every bar starts at ext offset 0 and paints with its own
#      color. Both assumptions break here; the solver must know each row's local bar count
#      and each bar's local index in near->far order.
#    - Because color mapping is partial, the solver now assumes source bars use only colors
#      inside `SOURCE_COLORS`; the generator guarantees that.
# d) Placement legality
#    - Bars remain contiguous and all lie on one side of the separator.
#    - Each row uses distinct positive lengths whose sum fits in the available bar-side
#      capacity after an optional 1-cell separator gap.
#    - Feasible length orders are precomputed and only accepted if all bars stay visible on
#      ext_side under the staggered rule, so "hidden bars" are rejected before placement.
# e) Visual observability
#    - A human can see three repeated cues at once: near->far bar order, different bar
#      lengths, and consistent source->target color conversion.
#    - The stagger formula depends only on visible structure within a row:
#      index among the row's bars and that bar's own length. It does not depend on raw color
#      ids, randomness, or hidden metadata.
# f) Visual cleanliness
#    - Overlap still resolves with the original no-overwrite rule, but only within a single
#      row's ext_side. Sparse row spacing keeps the denser stripe fields readable.
#    - The generator rejects row profiles where one early ray fills everything and hides the
#      rest, so the output stays structured instead of turning into solid noise.
# g) Change inferability
#    - The mapping is intentionally low-entropy and repeated in every sample.
#    - Because all five source colors appear and every bar leaves at least one mapped dot,
#      the train pairs provide repeated evidence for both the geometric staggering and the
#      color remapping.
# main() new assumptions review
#    - Runs on the bar side must correspond exactly to the intended 3-4 bars in that row;
#      the generator guarantees distinct adjacent colors so runs do not merge accidentally.
#    - The local bar index must reflect near->far order; the generator places bars contiguously
#      from the separator outward, so sorting by separator distance is sufficient.
#    - `_bar_offset(length, i, n)` assumes `n >= 1` and `length >= 1`; all sampled rows satisfy
#      that. The generator additionally enforces row profiles whose staggered output differs from
#      the zero-offset output, preventing silent collapse back to the original trajectory rule.

from __future__ import annotations

from functools import lru_cache
from itertools import combinations, permutations

import numpy as np

from arc_utils import colors_present, get_bg


SOURCE_COLORS = (1, 2, 3, 4, 6)
COLOR_MAP = {1: 2, 2: 3, 3: 4, 4: 6, 6: 1}
SEPARATOR_COLORS = tuple(c for c in range(1, 10) if c not in SOURCE_COLORS)


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


def _collect_row_bars(
    g: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
    bg: int,
) -> dict[int, list[tuple[int, int, int]]]:
    row_bars: dict[int, list[tuple[int, int, int]]] = {}
    h, w = g.shape
    for r in range(h):
        runs = (
            _iter_runs_1d(g[r, :], 0, sep_col, bg=bg)
            if bars_on_left
            else _iter_runs_1d(g[r, :], sep_col + 1, w, bg=bg)
        )
        for c0, c1, col in runs:
            length = c1 - c0 + 1
            dist = (sep_col - 1) - c1 if bars_on_left else c0 - (sep_col + 1)
            row_bars.setdefault(r, []).append((dist, length, col))
    for r in row_bars:
        row_bars[r].sort(key=lambda t: t[0])
    return row_bars


def _bar_offset(length: int, bar_index: int, bar_count: int) -> int:
    return (bar_index * length) // bar_count


def _paint_row_extension(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int, int]],
    bg: int,
) -> None:
    bar_count = len(bars)
    width = out.shape[1]
    for bar_index, (_dist, length, source_color) in enumerate(bars):
        if length <= 0 or source_color not in COLOR_MAP:
            continue
        paint_color = int(COLOR_MAP[source_color])
        offset = _bar_offset(length, bar_index, bar_count)
        if bars_on_left:
            positions = range(sep_col + 1 + offset, width, length)
        else:
            positions = range(sep_col - 1 - offset, -1, -length)
        for cc in positions:
            if int(out[row, cc]) == bg:
                out[row, cc] = paint_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each multi-bar row across the separator using staggered mapped rays.

    Input structure:
    - One mostly solid vertical separator column of a single non-background color.
    - All source bars lie on the same side of the separator; the opposite side is blank.
    - Each active row contains 3 or 4 adjacent monochrome bars, ordered by distance from
      the separator, with distinct colors per row drawn from a fixed source palette.

    Rule:
    - Detect the separator and which side contains the bars.
    - For each active row, sort its bars from nearest to farthest from the separator.
    - If the row has n bars, then the i-th bar (0-indexed in that near->far order) paints
      onto ext_side every `length` cells starting from ext offset floor(i*length/n).
    - Paint with `COLOR_MAP[source_color]`, not the source color, and never overwrite a
      non-background cell already painted by a nearer bar.

    Key insight:
    The output is still controlled by visible bar lengths and separator proximity, but the
    periodic dots are now both phase-shifted within each row and recolored by a fixed global
    mapping, so solving requires combining geometry with color-pair inference.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if h == 0 or w == 0:
        return g.copy()

    bg = int(get_bg(g))
    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    out = g.copy()

    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < w else 0
    bars_on_left = left_mass >= right_mass

    row_bars = _collect_row_bars(g, sep_col, bars_on_left, bg)
    for r, bars in row_bars.items():
        _paint_row_extension(out, r, sep_col, bars_on_left, bars, bg)

    return out


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 6, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 4, 14

ACTIVE_MIN, ACTIVE_MAX = 3, 5

P_FOUR_BARS = 0.4
P_SEP_GAP = 0.12
P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 500


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_separator_color(rng: np.random.Generator) -> int:
    return int(rng.choice(SEPARATOR_COLORS))


def _simulate_row_pattern(
    ext_side_len: int,
    lengths: tuple[int, ...],
    *,
    staggered: bool,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    cells = [0] * ext_side_len
    counts = [0] * len(lengths)
    bar_count = len(lengths)
    for bar_index, length in enumerate(lengths):
        offset = _bar_offset(length, bar_index, bar_count) if staggered else 0
        for pos in range(offset, ext_side_len, length):
            if cells[pos] == 0:
                cells[pos] = bar_index + 1
                counts[bar_index] += 1
    return tuple(cells), tuple(counts)


@lru_cache(maxsize=None)
def _feasible_length_orders(
    capacity: int,
    ext_side_len: int,
    bar_count: int,
) -> tuple[tuple[int, ...], ...]:
    orders: list[tuple[int, ...]] = []
    for comb in combinations(range(1, capacity + 1), bar_count):
        if sum(comb) > capacity:
            continue
        for perm in permutations(comb):
            staggered_pattern, visible_counts = _simulate_row_pattern(
                ext_side_len,
                tuple(map(int, perm)),
                staggered=True,
            )
            if any(count == 0 for count in visible_counts):
                continue
            plain_pattern, _ = _simulate_row_pattern(
                ext_side_len,
                tuple(map(int, perm)),
                staggered=False,
            )
            if staggered_pattern == plain_pattern:
                continue
            orders.append(tuple(map(int, perm)))
    return tuple(orders)


def _profile_choices(
    bar_side_len: int,
    ext_side_len: int,
) -> dict[int, tuple[tuple[tuple[int, ...], int], ...]]:
    choices: dict[int, list[tuple[tuple[int, ...], int]]] = {3: [], 4: []}
    for sep_gap in (0, 1):
        capacity = bar_side_len - sep_gap
        if capacity < 6:
            continue

        three_orders = _feasible_length_orders(capacity, ext_side_len, 3)
        choices[3].extend((order, sep_gap) for order in three_orders)
        four_orders = _feasible_length_orders(capacity, ext_side_len, 4)
        choices[4].extend((order, sep_gap) for order in four_orders)
    return {count: tuple(items) for count, items in choices.items()}


def _choose_bar_side_len(
    rng: np.random.Generator,
    w: int,
) -> tuple[int, int, dict[int, tuple[tuple[tuple[int, ...], int], ...]]] | None:
    lo = max(BAR_SIDE_MIN, w - 1 - EXT_SIDE_MAX)
    hi = min(BAR_SIDE_MAX, w - 1 - EXT_SIDE_MIN)
    if lo > hi:
        return None

    candidates: list[tuple[int, int, dict[int, tuple[tuple[tuple[int, ...], int], ...]]]] = []
    for bar_side_len in range(lo, hi + 1):
        ext_side_len = w - 1 - bar_side_len
        profile_choices = _profile_choices(bar_side_len, ext_side_len)
        if profile_choices[3] and profile_choices[4]:
            candidates.append((bar_side_len, ext_side_len, profile_choices))

    if not candidates:
        return None
    return candidates[int(rng.integers(0, len(candidates)))]


def _sample_row_profile(
    rng: np.random.Generator,
    profile_choices: dict[int, tuple[tuple[tuple[int, ...], int], ...]],
    bar_count: int,
) -> tuple[tuple[int, ...], int]:
    choices = profile_choices[bar_count]
    if not choices:
        raise ValueError(f"no feasible fused {bar_count}-bar row profile")
    zero_gap = [item for item in choices if item[1] == 0]
    one_gap = [item for item in choices if item[1] == 1]
    if one_gap and (not zero_gap or rng.random() < P_SEP_GAP):
        pool = one_gap
    else:
        pool = zero_gap if zero_gap else one_gap
    lengths, sep_gap = pool[int(rng.integers(0, len(pool)))]
    return tuple(map(int, lengths)), int(sep_gap)


def _assign_row_colors(
    rng: np.random.Generator,
    row_counts: list[int],
) -> list[list[int]]:
    total_slots = sum(row_counts)
    extras_needed = total_slots - len(SOURCE_COLORS)
    pool = list(SOURCE_COLORS)
    if extras_needed > 0:
        pool.extend(map(int, rng.choice(SOURCE_COLORS, size=extras_needed, replace=True)))

    for _ in range(400):
        perm = list(map(int, rng.permutation(pool)))
        rows: list[list[int]] = []
        idx = 0
        ok = True
        for count in row_counts:
            colors = perm[idx : idx + count]
            idx += count
            if len(set(colors)) != count:
                ok = False
                break
            rows.append(colors)
        if ok and set(c for row in rows for c in row) == set(SOURCE_COLORS):
            return rows

    raise RuntimeError("failed to assign row colors for fused variant")


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
) -> list[int]:
    visible_counts = [0] * len(bars)
    bar_count = len(bars)
    for bar_index, (length, source_color) in enumerate(bars):
        paint_color = int(COLOR_MAP[source_color])
        offset = _bar_offset(length, bar_index, bar_count)
        if bars_on_left:
            positions = range(sep_col + 1 + offset, out.shape[1], length)
        else:
            positions = range(sep_col - 1 - offset, -1, -length)
        for cc in positions:
            if out[row, cc] == 0:
                out[row, cc] = paint_color
                visible_counts[bar_index] += 1
    return visible_counts


def _paint_extension_plain(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
) -> None:
    for length, source_color in bars:
        paint_color = int(COLOR_MAP[source_color])
        if bars_on_left:
            positions = range(sep_col + 1, out.shape[1], length)
        else:
            positions = range(sep_col - 1, -1, -length)
        for cc in positions:
            if out[row, cc] == 0:
                out[row, cc] = paint_color


def _outputs_differ_from_plain(
    inp: np.ndarray,
    row_bars_map: dict[int, list[tuple[int, int]]],
    sep_col: int,
    bars_on_left: bool,
) -> bool:
    staggered = inp.copy()
    plain = inp.copy()
    for row, bars in row_bars_map.items():
        _paint_extension(staggered, row, sep_col, bars_on_left, bars)
        _paint_extension_plain(plain, row, sep_col, bars_on_left, bars)
    return not np.array_equal(staggered, plain)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(H_MIN, H_MAX + 1))
    w = int(rng.integers(W_MIN, W_MAX + 1))

    picked = _choose_bar_side_len(rng, w)
    if picked is None:
        return None
    bar_side_len, ext_side_len, profile_choices = picked

    bars_on_left = bool(rng.integers(0, 2) == 0)
    sep_col = bar_side_len if bars_on_left else (w - 1 - bar_side_len)
    sep_color = _sample_separator_color(rng)

    max_active = min(ACTIVE_MAX, (h + 1) // 2)
    if max_active < ACTIVE_MIN:
        return None
    active_count = int(rng.integers(ACTIVE_MIN, max_active + 1))
    active_rows = _choose_active_rows(rng, h, active_count)
    active_set = set(active_rows)
    inactive_rows = [r for r in range(h) if r not in active_set]

    row_bar_counts = [3] * active_count
    if profile_choices[4]:
        target_four = max(1, min(active_count - 1, int(round(active_count * P_FOUR_BARS))))
        for idx in rng.choice(active_count, size=target_four, replace=False):
            row_bar_counts[int(idx)] = 4

    row_profiles: list[tuple[tuple[int, ...], int]] = []
    try:
        for bar_count in row_bar_counts:
            row_profiles.append(_sample_row_profile(rng, profile_choices, bar_count))
    except (ValueError, RuntimeError):
        return None

    row_counts = [len(lengths) for lengths, _sep_gap in row_profiles]
    try:
        row_colors = _assign_row_colors(rng, row_counts)
    except RuntimeError:
        return None

    out = np.zeros((h, w), dtype=int)
    out[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        out[gap_row, sep_col] = 0

    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for row, (lengths, sep_gap), colors in zip(active_rows, row_profiles, row_colors):
        bars = list(zip(lengths, colors))
        row_bars_map[row] = bars
        _place_bars(out, row, sep_col, bars_on_left, bars, sep_gap)

    for row, bars in row_bars_map.items():
        visible_counts = _paint_extension(out, row, sep_col, bars_on_left, bars)
        if any(count == 0 for count in visible_counts):
            return None

    inp = out.copy()
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0

    if get_bg(inp) != 0:
        return None

    non_bg = colors_present(inp, exclude=[0])
    if sep_color not in non_bg:
        return None
    if set(c for c in non_bg if c != sep_color) != set(SOURCE_COLORS):
        return None
    if not any(len(bars) == 3 for bars in row_bars_map.values()):
        return None
    if not any(len(bars) == 4 for bars in row_bars_map.values()):
        return None
    if not _outputs_differ_from_plain(inp, row_bars_map, sep_col, bars_on_left):
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
