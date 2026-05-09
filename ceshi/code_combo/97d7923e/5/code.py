# Combo variant of 97d7923e (index 5)
# combo_spec: D2/0+D6/2
# Name: 升序排名+顶部行排序

from __future__ import annotations

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - The visible vocabulary stays close to the original task: a top-left legend of
#      one-column stacks, and several vertical cap/fill/cap bars grouped by cap color.
#    - The fused change is not "pick by length" anymore. Within one cap-color group,
#      bars are ordered by the row of the top cap: smaller row index means the cap sits
#      higher, so it comes earlier in the ranking.
#    - Concrete trace for one group:
#        bars = [
#          (col A, top row 2, fill length 1),
#          (col B, top row 4, fill length 4),
#          (col C, top row 6, fill length 2),
#        ], legend height k = 2
#      original rule:
#        sort by length descending -> [B, C, A] -> pick B? no, k=2 -> pick C
#      D2-only rule:
#        sort by length ascending  -> [A, C, B] -> pick C
#      fused rule:
#        sort by top row ascending -> [A, B, C] -> pick B
#      The chosen column changes because spatial order and length order are decoupled.
# b) Complexity calibration
#    - If bars remained bottom-aligned, top-row order would collapse back to length
#      order and the combo would be visually fake. The generator therefore uses floating
#      bars with varied bottoms.
#    - Each group still has only 2-3 bars and legend heights still range from 1 to the
#      group size, keeping the ordinal task small enough to infer from a few examples.
#    - Within each group, top rows are distinct and fill lengths are distinct. This
#      avoids ties for both the new top-row rule and the old length-based references.
# c) Implicit-assumption audit against the original main()
#    - `_vertical_sandwiches` only assumes a valid bar is one column with equal caps and
#      a solid non-background interior. Floating the bar does not break that assumption.
#    - `_legend_entries` only reads top-touching stacks and their heights. The meaning of
#      that height changes from a length rank to a top-row rank, but the parser is still valid.
#    - The original `segs.sort(key=lambda s: (-(...), s[0], s[2]))` assumed fill length
#      was the grouping order. That is the line that must change: the primary key is now
#      `top_r`, with column only as a stable fallback.
#    - The original generator assumed bottom alignment, so `top_r` was derived from
#      length. That assumption is invalid here and must be removed from construction and oracle.
# d) Placement legality
#    - Grid height stays 10 and width stays 10-20.
#    - Bars remain one column wide and columns remain distinct, so floating them only
#      changes vertical placement; it does not create overlap between different bars.
#    - The legend may still share a column with one bar per group, but the shared-column
#      bar starts at least one background row below the legend stack, so the solver sees
#      two separate structures rather than one merged component.
# e) Visual observability
#    - The critical clue is now spatial: within a same-cap group, "which bar starts
#      highest?" is directly visible without measuring lengths numerically.
#    - The generator rejects samples unless at least one group selects a different bar
#      under fused top-row ranking than under both length-desc and length-asc references.
#      That prevents hidden degeneration.
# f) Visual neatness
#    - The legend remains a compact left cluster and the bars remain sparse single
#      columns. Floating bars add vertical variation without introducing clutter or overlap.
# g) Change inferability
#    - Colors still serve only as group labels and repaint targets. The decisive cue is
#      the row of the top cap, which is a visible geometric property rather than a color id.
# 9) New assumptions introduced by the combo main()
#    - Every cap-color group must have unique top rows so the top-based ranking is unambiguous.
#    - Every legend height `k` must satisfy `1 <= k <= bars_in_group`.
#    - At least one group must produce a fused target column that differs from both
#      length-desc and length-asc references; the generator enforces this before accepting a sample.

from pathlib import Path
import sys
import itertools

import numpy as np

for _parent in Path(__file__).resolve().parents:
    if (_parent / "arc_utils.py").exists():
        if str(_parent) not in sys.path:
            sys.path.append(str(_parent))
        break

from arc_utils import get_bg, neighbors4, paste


def _vertical_sandwiches(g: np.ndarray, cap: int) -> list[tuple[int, int, int, int]]:
    """
    Find vertical bar candidates (top_r, bottom_r, col, fill_color) where:
      - g[top_r, col] == cap and g[bottom_r, col] == cap
      - (top_r, bottom_r) are consecutive occurrences of cap in that column
      - interior is a solid run of ONE color fill_color != cap and fill_color != background
      - length >= 1 (bottom_r >= top_r + 2)
    """
    h, w = g.shape
    bg = get_bg(g)
    segs: list[tuple[int, int, int, int]] = []
    for c in range(w):
        rows = [int(r) for r in np.flatnonzero(g[:, c] == cap)]
        if len(rows) < 2:
            continue
        for a, b in zip(rows, rows[1:]):
            if b <= a + 1:
                continue
            fill = int(g[a + 1, c])
            if fill == cap or fill == bg:
                continue
            if np.all(g[a + 1 : b, c] == fill):
                segs.append((a, b, c, fill))
    return segs


def _legend_entries(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    """
    Legend is in the top-left: a cluster of non-bg pixels on the top row near the left edge,
    allowing small gaps (typically 1 bg column) between entries.

    Returns entries as (x_pos, legend_color, block_count), sorted by x_pos.
    block_count is the vertical height of the 4-connected component of that color that touches
    the top row within the legend cluster (i.e., how many stacked blocks).
    """
    h, w = g.shape
    top_pos = [int(c) for c in np.flatnonzero(g[0, :] != bg)]
    assert top_pos, "legend row must be non-empty"
    top_pos.sort()

    cols = [top_pos[0]]
    for c in top_pos[1:]:
        if c - cols[-1] <= 2:
            cols.append(c)
        else:
            break
    legend_max_c = max(cols)

    entries: list[tuple[int, int, int]] = []
    seen = np.zeros((h, w), dtype=bool)
    for c in cols:
        col = int(g[0, c])
        if col == bg or seen[0, c]:
            continue
        st = [(0, c)]
        seen[0, c] = True
        pix = []
        while st:
            r, cc = st.pop()
            pix.append((r, cc))
            for rr, ccc in neighbors4(r, cc):
                if 0 <= rr < h and 0 <= ccc <= legend_max_c and (not seen[rr, ccc]) and int(g[rr, ccc]) == col:
                    seen[rr, ccc] = True
                    st.append((rr, ccc))
        x_pos = min(cc for _, cc in pix)
        rs = [r for r, _ in pix]
        block_count = max(rs) - min(rs) + 1
        entries.append((x_pos, col, int(block_count)))

    entries.sort(key=lambda t: t[0])
    return entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor vertical bars by legend-driven rank selection using top-cap row order.

    Input structure:
    - Background fills most cells. In the upper-left corner, each legend entry is a one-column
      stack of a cap color K touching row 0, and its height k is an ordinal cue.
    - The rest of the grid contains one-column cap-fill-cap bars. Bars sharing cap color K
      form a group, but unlike the original task they do not need to share a common bottom row.

    Rule:
    - For each legend entry (K, k), collect all bars whose top and bottom caps are color K.
    - Sort that group by the row of the top cap, from highest on the board to lowest
      (smaller row index = earlier rank).
    - Recolor the interior of the rank-k bar to K.

    Key insight: The legend height is still a rank index, but the rank now comes from visible
    vertical placement of the top caps rather than from bar length.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, "legend must be non-empty"

    for _x, cap, k in legend:
        segs = _vertical_sandwiches(g, cap)
        assert segs, f"legend cap {cap} must have matching vertical bars"
        segs.sort(key=lambda s: (s[0], s[2]))
        assert 1 <= k <= len(segs), f"legend rank {k} out of range for {len(segs)} bars"
        top_r, bot_r, col, _fill = segs[k - 1]
        out[top_r + 1 : bot_r, col] = cap

    return out


# --------------- generator (O2) ---------------

GRID_H = 10
WIDTH_MIN = 10
WIDTH_MAX = 20
MAX_GROUPS = 4
MAX_TOTAL_BARS = 10
BARS_PER_GROUP = (2, 3)
MAX_FILL_LEN = 7
COLOR_RANGE = (1, 10)
COL_SPACING = 2
MAX_RETRIES = 500

MIN_BAR_TOP = 1
MAX_BAR_TOP = GRID_H - 3

_FEASIBLE_COUNTS = {
    ng: [
        counts
        for counts in itertools.product(BARS_PER_GROUP, repeat=ng)
        if sum(counts) <= MAX_TOTAL_BARS and any(count == 3 for count in counts)
    ]
    for ng in range(1, MAX_GROUPS + 1)
}


def _build_bar(cap: int, fill: int, fill_len: int) -> np.ndarray:
    """Return a (fill_len+2, 1) column sprite: cap / fill*N / cap."""
    bar = np.full((fill_len + 2, 1), fill, dtype=int)
    bar[0, 0] = cap
    bar[-1, 0] = cap
    return bar


def _assign_columns(rng, n_groups: int, counts: list[int], width: int) -> list[list[int]]:
    """Distribute bar columns across groups with random inter-group spacing."""
    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    gap_slots = total_bars - n_groups + 1
    extra_width = width - min_width
    extra_parts = rng.multinomial(extra_width, np.full(gap_slots, 1.0 / gap_slots))

    inter_gap_bonus = np.zeros(total_bars - 1, dtype=int)
    if total_bars > n_groups:
        inter_gap_bonus[n_groups - 1 :] = extra_parts[:-1]

    occupied = [0]
    for idx in range(total_bars - 1):
        occupied.append(occupied[-1] + COL_SPACING + int(inter_gap_bonus[idx]))

    legend_cols = occupied[:n_groups]
    extra_cols = occupied[n_groups:]

    group_columns = [[legend_cols[i]] for i in range(n_groups)]
    owners = [i for i, count in enumerate(counts) for _ in range(count - 1)]
    rng.shuffle(owners)
    for col, owner in zip(extra_cols, owners):
        group_columns[owner].append(col)

    return group_columns


def _bar_len_limit(top_r: int) -> int:
    return min(MAX_FILL_LEN, GRID_H - top_r - 2)


def _sample_group_specs(rng, count: int, rank: int) -> list[tuple[int, int]] | None:
    """
    Sample one group's bars as (top_r, fill_len), aligned with that group's columns.
    Index 0 is the legend-sharing column and must sit below the legend with one bg gap row.
    """
    for _ in range(200):
        legend_top_choices = [r for r in range(rank + 1, MAX_BAR_TOP + 1) if _bar_len_limit(r) >= 1]
        if not legend_top_choices:
            return None
        legend_top = int(legend_top_choices[int(rng.integers(len(legend_top_choices)))])

        other_pool = [r for r in range(MIN_BAR_TOP, MAX_BAR_TOP + 1) if r != legend_top and _bar_len_limit(r) >= 1]
        if len(other_pool) < count - 1:
            return None
        rng.shuffle(other_pool)
        top_rows = [legend_top] + other_pool[: count - 1]

        used_lengths: set[int] = set()
        specs: list[tuple[int, int]] = []
        ok = True
        for top_r in top_rows:
            choices = [fill_len for fill_len in range(1, _bar_len_limit(top_r) + 1) if fill_len not in used_lengths]
            if not choices:
                ok = False
                break
            fill_len = int(choices[int(rng.integers(len(choices)))])
            used_lengths.add(fill_len)
            specs.append((int(top_r), fill_len))
        if ok:
            return specs
    return None


def _sorted_group(
    bars: list[tuple[int, int, int]],
    mode: str,
) -> list[tuple[int, int, int]]:
    """
    Bars are (col, top_r, fill_len).
    """
    if mode == "top":
        return sorted(bars, key=lambda item: (item[1], item[0]))
    if mode == "len_desc":
        return sorted(bars, key=lambda item: (-item[2], item[1], item[0]))
    if mode == "len_asc":
        return sorted(bars, key=lambda item: (item[2], item[1], item[0]))
    raise ValueError(f"unknown mode: {mode}")


def _paint_choice(out: np.ndarray, cap: int, bar: tuple[int, int, int]) -> None:
    col, top_r, fill_len = bar
    out[top_r + 1 : top_r + 1 + fill_len, col] = cap


def _expected_output(
    inp: np.ndarray,
    caps: list[int],
    groups: list[list[tuple[int, int, int]]],
    ranks: list[int],
    mode: str,
) -> np.ndarray:
    out = inp.copy()
    for cap, bars, rank in zip(caps, groups, ranks):
        ranked = _sorted_group(bars, mode)
        _paint_choice(out, cap, ranked[rank - 1])
    return out


def _signal_counts(groups: list[list[tuple[int, int, int]]], ranks: list[int]) -> tuple[int, int]:
    """
    Returns:
      strong_groups: fused target differs from both length-desc and length-asc targets
      original_diff_groups: fused target differs from the original length-desc target
    """
    strong_groups = 0
    original_diff_groups = 0
    for bars, rank in zip(groups, ranks):
        top_pick = _sorted_group(bars, "top")[rank - 1][0]
        desc_pick = _sorted_group(bars, "len_desc")[rank - 1][0]
        asc_pick = _sorted_group(bars, "len_asc")[rank - 1][0]
        if top_pick != desc_pick:
            original_diff_groups += 1
        if top_pick != desc_pick and top_pick != asc_pick:
            strong_groups += 1
    return strong_groups, original_diff_groups


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch or weak fused signal."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    if not feasible:
        return None
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, count + 1)) for count in counts]

    palette = rng.choice(np.arange(*COLOR_RANGE), size=2 * n_groups, replace=False)
    caps = [int(v) for v in palette[:n_groups]]
    fills = [int(v) for v in palette[n_groups:]]

    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    group_columns = _assign_columns(rng, n_groups, counts, width)

    grid = np.zeros((GRID_H, width), dtype=int)
    groups: list[list[tuple[int, int, int]]] = []
    for cap, fill, count, rank, cols in zip(caps, fills, counts, ranks, group_columns):
        paste(grid, np.full((rank, 1), cap, dtype=int), 0, cols[0])

        specs = _sample_group_specs(rng, count, rank)
        if specs is None:
            return None

        bars: list[tuple[int, int, int]] = []
        for col, (top_r, fill_len) in zip(cols, specs):
            paste(grid, _build_bar(cap, fill, fill_len), top_r, col)
            bars.append((int(col), int(top_r), int(fill_len)))
        groups.append(bars)

    strong_groups, original_diff_groups = _signal_counts(groups, ranks)
    if strong_groups < 1 or original_diff_groups < 1:
        return None

    expected = _expected_output(grid, caps, groups, ranks, "top")
    if not np.array_equal(main(grid), expected):
        return None

    if np.array_equal(expected, _expected_output(grid, caps, groups, ranks, "len_desc")):
        return None
    if np.array_equal(expected, _expected_output(grid, caps, groups, ranks, "len_asc")):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
