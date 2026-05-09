# Variant of 97d7923e
# Dimension: D14 — 条的底部对齐约束 (meta)
# Change: 所有条底部对齐在最后一行(row 9)，顶部位置由长度决定... → 无固定对齐（自由浮动）

import itertools

import numpy as np

from arc_utils import get_bg, neighbors4, paste


# Design analysis for the D14 variant
# ===================================
# a) Concrete behavior + trace
#    - The legend stays in the top-left corner exactly as in the original task:
#      each entry is a 1-cell-wide monochrome stack whose height k encodes rank k.
#    - The main objects are still vertical cap/fill/cap columns. The change is
#      purely geometric: their bottom caps no longer line up on row 9.
#    - A bar's length must now be read from the distance between its two cap cells,
#      not inferred from a shared baseline. Example for cap color K with rank k=2:
#      original generator: bars of fill lengths [5, 3, 1] all ended on row 9, so
#      the longest bar was visually the highest. Variant generator: the same three
#      lengths might occupy rows [1..7], [4..8], [2..4]. The solver must still pick
#      the second-longest by comparing the cap-to-cap span, then recolor only that
#      interior.
# b) Complexity calibration
#    - The free-floating placement is visible because every generated group contains
#      at least two different bottom rows, so the sample never collapses back to a
#      hidden "all bars still end together" layout.
#    - The task is not made noisy: bars remain single-column, groups keep one cap
#      color and one fill color, lengths remain distinct within a group, and only
#      1-4 legend entries exist.
# c) Implicit-assumption audit of main()
#    - get_bg(g): assumes the background is still the dominant color. The generator
#      preserves the original sparse layout and width budget, so 0 remains most common.
#    - _legend_entries(...): assumes row 0 only contains the legend cluster near the
#      left edge. The generator forbids bar top caps on row 0, so free-floating bars
#      never leak into legend parsing.
#    - _vertical_sandwiches(...): assumes a valid bar is represented by one column
#      containing two consecutive cap occurrences with a solid non-bg interior.
#      This still holds because each column hosts at most one bar, and when a legend
#      shares that column there is always at least one bg cell between legend and bar.
#    - segs.sort(...): the semantic key is interior length. Top-row / column tie-breaks
#      are retained only as deterministic fallbacks; lengths are generated distinct
#      inside every cap group, so the tie-break never determines the answer.
# d) Placement legality
#    - Columns remain unique and separated by at least one blank column.
#    - The first column of each group still doubles as the legend column; therefore
#      its bar must start below the legend with one bg gap row.
#    - Every bar is placed fully inside the 10xW grid, and group-local rejection
#      guarantees non-uniform bottoms.
# e) Visual observability
#    - Across 2-4 ARC examples a human can now see that the chosen bar is not the
#      "one reaching the common floor", because there is no common floor anymore.
#      The only stable cue is the cap-to-cap span, which is directly visible.
# f) Visual cleanliness
#    - No object overlaps another. Legend occupies a compact top-left strip; bars
#      stay in separate columns with background between them, so the grid remains
#      easy to parse despite the added vertical freedom.
# g) Color-change inferability
#    - Colors are unchanged from the original logic: legend color K selects among
#      bars capped by K, and the chosen bar's fill becomes K. No invisible ordering
#      by color id is introduced.


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
    assert top_pos, 'legend row must be non-empty'
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
    Recolor free-floating vertical bars using the top-left rank legend.

    Input structure:
    - bg is the background. In the top-left, each legend entry is a monochrome stack touching row 0; its color K names one bar group and its height k is that group's target rank.
    - Elsewhere, bars are single columns with matching cap color K at both ends and one solid interior color between them. Bars may start and end on different rows; there is no shared bottom alignment.

    Rule:
    - For each legend entry (K, k), collect every vertical cap/fill/cap sandwich using cap color K.
    - Measure each bar by the number of interior cells between the two caps, then sort from longest to shortest.
    - Recolor the interior of the rank-k bar to K.

    Key insight:
    - The legend height does not describe a literal bar size. It selects an order statistic among same-cap bars, and that ordering must now be recovered from cap-to-cap distance rather than from any common baseline.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, 'legend must be non-empty'

    for _x, cap, k in legend:
        segs = _vertical_sandwiches(g, cap)
        assert segs, f'legend cap {cap} must have matching vertical bars'
        segs.sort(key=lambda s: (-(s[1] - s[0] - 1), s[0], s[2]))
        assert 1 <= k <= len(segs), f'legend rank {k} out of range for {len(segs)} bars'
        top_r, bot_r, col, _fill = segs[k - 1]
        out[top_r + 1 : bot_r, col] = cap

    return out


# --------------- generator (O2) ---------------

# --- Configuration ---
GRID_H = 10                  # fixed grid height
WIDTH_MIN = 10               # minimum grid width
WIDTH_MAX = 20               # maximum grid width (inclusive)
MAX_GROUPS = 4               # max number of cap-color groups
MAX_TOTAL_BARS = 10          # max bars across all groups
BARS_PER_GROUP = (2, 3)      # allowed bars-per-group counts
MAX_FILL_LEN = 7             # max interior fill length of a bar
COLOR_RANGE = (1, 10)        # non-bg color ids (exclusive upper)
COL_SPACING = 2              # minimum column stride between bars
MAX_RETRIES = 300            # retry budget for generate_input
TOP_ROW_MIN = 1              # keep row 0 reserved for legend parsing
GROUP_TOP_TRIES = 80         # retry budget for one group's floating placement

_BOTTOM_ROW = GRID_H - 1

_FEASIBLE_COUNTS = {
    ng: [c for c in itertools.product(BARS_PER_GROUP, repeat=ng) if sum(c) <= MAX_TOTAL_BARS]
    for ng in range(1, MAX_GROUPS + 1)
}


def _build_bar(cap: int, fill: int, fill_len: int) -> np.ndarray:
    """Return a (fill_len+2, 1) column sprite: cap / fill*N / cap."""
    bar = np.full((fill_len + 2, 1), fill, dtype=int)
    bar[0, 0] = cap
    bar[-1, 0] = cap
    return bar


def _assign_columns(rng, n_groups: int, counts: list[int], width: int) -> list[list[int]]:
    """Distribute bar columns across groups with random inter-group spacing.

    Returns one list of column indices per group; the first column in each
    group doubles as that group's legend column.
    """
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


def _choose_fill_lengths(rng, count: int, rank: int) -> list[int]:
    """Pick distinct fill lengths for one group.

    The first bar shares a column with the legend, so it must leave one bg row
    below the legend before its top cap.
    """
    max_legend_len = min(MAX_FILL_LEN, GRID_H - rank - 3)
    shared_len = int(rng.integers(1, max_legend_len + 1))
    pool = [length for length in range(1, MAX_FILL_LEN + 1) if length != shared_len]
    rng.shuffle(pool)
    return [shared_len] + pool[: count - 1]


def _choose_top_rows(rng, lengths: list[int], rank: int) -> list[int] | None:
    """Choose floating top rows for one group's bars.

    Rejection rules enforce that the group does not accidentally collapse back
    to the original bottom-aligned layout.
    """
    for _ in range(GROUP_TOP_TRIES):
        tops: list[int] = []
        bottoms: list[int] = []
        for idx, fill_len in enumerate(lengths):
            min_top = rank + 1 if idx == 0 else TOP_ROW_MIN
            max_top = GRID_H - fill_len - 2
            if min_top > max_top:
                return None
            top_r = int(rng.integers(min_top, max_top + 1))
            tops.append(top_r)
            bottoms.append(top_r + fill_len + 1)

        if len(set(bottoms)) == 1:
            continue
        if all(bottom == _BOTTOM_ROW for bottom in bottoms):
            continue
        return tops
    return None


def _expected_output(
    inp: np.ndarray,
    caps: list[int],
    placements: list[list[tuple[int, int, int, int]]],
    ranks: list[int],
) -> np.ndarray:
    """Compute the expected output by recoloring the rank-k bar's fill to cap."""
    out = inp.copy()
    for cap, group, rank in zip(caps, placements, ranks):
        ranked = sorted(group, key=lambda item: (-item[3], item[0], item[2]))
        top_r, bot_r, col, _fill_len = ranked[rank - 1]
        out[top_r + 1 : bot_r, col] = cap
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid floating-bar grid. Returns None on oracle mismatch."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, c + 1)) for c in counts]

    palette = rng.choice(np.arange(*COLOR_RANGE), size=2 * n_groups, replace=False)
    caps = [int(v) for v in palette[:n_groups]]
    fills = [int(v) for v in palette[n_groups:]]

    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    group_columns = _assign_columns(rng, n_groups, counts, width)

    grid = np.zeros((GRID_H, width), dtype=int)
    placements: list[list[tuple[int, int, int, int]]] = []
    all_bottoms: list[int] = []

    for cap, fill, count, rank, cols in zip(caps, fills, counts, ranks, group_columns):
        paste(grid, np.full((rank, 1), cap, dtype=int), 0, cols[0])

        lengths = _choose_fill_lengths(rng, count, rank)
        tops = _choose_top_rows(rng, lengths, rank)
        if tops is None:
            return None

        group_places: list[tuple[int, int, int, int]] = []
        for col, fill_len, top_r in zip(cols, lengths, tops):
            paste(grid, _build_bar(cap, fill, fill_len), top_r, col)
            bot_r = top_r + fill_len + 1
            group_places.append((top_r, bot_r, col, fill_len))
            all_bottoms.append(bot_r)
        placements.append(group_places)

    if len(set(all_bottoms)) == 1:
        return None
    if all(bottom == _BOTTOM_ROW for bottom in all_bottoms):
        return None

    expected = _expected_output(grid, caps, placements, ranks)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f'failed to generate a valid input after {MAX_RETRIES} attempts')
