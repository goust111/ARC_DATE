# Variant of 97d7923e
# Dimension: D1 — 匹配条件：端点颜色分组 (match_cond)
# Change: s.endpoint_colors == d.color_group — 垂直条的顶部和底部帽色(cap color)与... → 填充颜色分组替代端点颜色

# Design analysis for the D1 variant
# ----------------------------------
# spec_variant:
#   core_change:
#     original: legend color names the bar group by cap color
#     variant: legend color names the bar group by fill color
#   visible_objects:
#     legend:
#       still sits at the top-left and still encodes rank by vertical height
#       but each legend block now uses the shared interior fill color of one group
#     bars:
#       remain one-column cap/fill/cap vertical bars, bottom-aligned
#       bars in the same group share the same fill color but may use different cap colors
#   output:
#     the selected rank-k bar still becomes monochrome, but it is repainted to its own cap color
#     this is the smallest compatibility adjustment that keeps the D1 change visible
#
# a) Concrete behavior trace
#   Example group:
#     legend block = color 5, height 2
#     bars with fill 5 have lengths 6, 4, 2 and caps [2, 3, 2]
#   Original logic would have interpreted legend color 5 as a cap-color query.
#   Variant logic interprets legend color 5 as a fill-color query, sorts the three fill-5 bars
#   by length, picks rank 2, and repaints that bar's interior from 5 to its own cap 3.
#   Difference:
#     old pipeline: legend color -> match cap -> repaint to legend color
#     new pipeline: legend color -> match fill -> repaint to matched bar's cap
#
# b) Complexity calibration
#   If every fill group also had a unique cap color, the partition by fill would look equivalent
#   to the partition by cap and the D1 change would be weak. To avoid that, the generator uses a
#   small shared cap palette across multiple fill groups and mixes at least two cap colors inside
#   each fill group. This keeps the bars simple while making the grouping cue visibly shift from
#   endpoints to interiors.
#
# c) Implicit-assumption review of the original main()
#   - `_vertical_sandwiches(g, cap)` assumed the legend color is directly a cap color.
#     This fails under the variant because legend colors are fill colors.
#   - `out[top_r + 1 : bot_r, col] = cap` assumed the legend color is also the repaint color.
#     Under the variant that would repaint to the wrong color or become a no-op if repainting to fill.
#   - Sorting by interior length is still valid and unchanged.
#   Therefore the solver now first extracts all cap/fill/cap bars, then groups them by fill color,
#   and repaints the chosen bar to the matched bar's cap color.
#
# d) Placement legality
#   - Legend and bars may still share a column only when a background gap row separates them.
#   - Bars keep distinct columns and remain bottom-aligned.
#   - The first bar in each group is length-limited so the legend column remains non-overlapping.
#   - Fill colors are distinct across groups and disjoint from all cap colors, preventing accidental
#     cap-based matches to legend colors.
#
# e) Visual observability
#   Humans can now identify a group only by scanning for repeated fill color inside bars.
#   Endpoint colors are deliberately unreliable because the same cap colors are reused across
#   different fill groups. The selected output bar becomes monochrome in its endpoint color, so the
#   change remains easy to spot after the ranking step.
#
# f) Visual cleanliness
#   The layout stays sparse: one-column legend blocks on the left/top, one-column bars with at
#   least one blank column between neighbors, and only one bar per column.
#
# g) Color-change inferability
#   Each legend color already appears as the shared interior color of several bars, so the legend
#   to group mapping is structurally signaled. The output color is the chosen bar's endpoint color,
#   which is locally visible on the same column.

import itertools
from pathlib import Path
import sys

import numpy as np

for _parent in Path(__file__).resolve().parents:
    if (_parent / "arc_utils.py").exists():
        if str(_parent) not in sys.path:
            sys.path.append(str(_parent))
        break

from arc_utils import get_bg, neighbors4, paste


def _all_vertical_sandwiches(g: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """
    Find all vertical bar candidates as (top_r, bottom_r, col, cap_color, fill_color) where:
      - g[top_r, col] == g[bottom_r, col] == cap_color
      - (top_r, bottom_r) are consecutive occurrences of that cap_color in the column
      - the interior is a solid run of one non-bg fill_color different from the cap
      - the bar spans at least 3 cells total
    """
    h, w = g.shape
    bg = get_bg(g)
    segs: list[tuple[int, int, int, int, int]] = []
    for c in range(w):
        col_vals = g[:, c]
        for cap in {int(v) for v in col_vals if int(v) != bg}:
            rows = [int(r) for r in np.flatnonzero(col_vals == cap)]
            if len(rows) < 2:
                continue
            for a, b in zip(rows, rows[1:]):
                if b <= a + 1:
                    continue
                fill = int(col_vals[a + 1])
                if fill == cap or fill == bg:
                    continue
                if np.all(col_vals[a + 1 : b] == fill):
                    segs.append((a, b, c, cap, fill))
    return segs


def _legend_entries(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    """
    Legend is in the top-left: a cluster of non-bg pixels on the top row near the left edge,
    allowing small gaps (typically 1 bg column) between entries.

    Returns entries as (x_pos, legend_color, block_count), sorted by x_pos.
    block_count is the vertical height of the 4-connected component of that color that touches
    the top row within the legend cluster.
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
    Recolor vertical bars by legend-driven rank selection.

    Input structure:
    - bg = background. Top-left: monochrome blocks touching row 0; each has color F and height k.
    - Main grid: vertical bars capped top and bottom by one color, with solid-color interiors.
    - Bars in one logical group share the same interior fill color F, while cap colors may repeat across groups.

    Rule:
    - For each legend entry (F, k), find vertical bars whose interior fill color is F.
    - Rank those bars by interior length (longest = rank 1).
    - Recolor the rank-k bar's interior to that bar's own cap color.

    Key insight: Legend height still encodes a rank index, but legend color now points to a fill-color group rather than an endpoint-color group.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, 'legend must be non-empty'
    bars = _all_vertical_sandwiches(g)

    for _x, fill, k in legend:
        segs = [s for s in bars if s[4] == fill]
        assert segs, f'legend fill {fill} must have matching vertical bars'
        segs.sort(key=lambda s: (-(s[1] - s[0] - 1), s[0], s[2]))
        assert 1 <= k <= len(segs), f'legend rank {k} out of range for {len(segs)} bars'
        top_r, bot_r, col, cap, _fill = segs[k - 1]
        out[top_r + 1 : bot_r, col] = cap

    return out


# --------------- generator (O2) ---------------

# --- Configuration ---
GRID_H = 10                  # fixed grid height
WIDTH_MIN = 10               # minimum grid width
WIDTH_MAX = 20               # maximum grid width (inclusive)
MIN_GROUPS = 2               # keep the fill-group distinction observable
MAX_GROUPS = 4               # max number of fill-color groups
MAX_TOTAL_BARS = 10          # max bars across all groups
BARS_PER_GROUP = (2, 3)      # allowed bars-per-group counts
MIN_CAP_COLORS = 2           # use a shared, smaller cap palette to force cross-group reuse
MAX_CAP_COLORS = 3
MAX_FILL_LEN = 7             # max interior fill length of a bar
COLOR_RANGE = (1, 10)        # non-bg color ids (exclusive upper)
COL_SPACING = 2              # minimum column stride between bars
MAX_RETRIES = 300            # retry budget for generate_input

_BOTTOM_ROW = GRID_H - 1    # row index where all bars end
_MAX_CAP_ROW = GRID_H - 2   # highest row a bar's top cap can occupy

_FEASIBLE_COUNTS = {
    ng: [c for c in itertools.product(BARS_PER_GROUP, repeat=ng) if sum(c) <= MAX_TOTAL_BARS]
    for ng in range(MIN_GROUPS, MAX_GROUPS + 1)
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
    """Pick `count` distinct fill lengths for one group.

    The first length (legend column bar) is constrained so the bar fits
    below the legend block with a bg gap row.
    """
    max_legend_len = MAX_FILL_LEN - rank
    shared_len = int(rng.integers(1, max_legend_len + 1))
    pool = [l for l in range(1, MAX_FILL_LEN + 1) if l != shared_len]
    rng.shuffle(pool)
    return [shared_len] + pool[: count - 1]


def _choose_bar_caps(rng, count: int, cap_pool: list[int]) -> list[int]:
    """Assign caps to bars in one fill group using at least two distinct shared cap colors."""
    first_two = [int(v) for v in rng.choice(np.array(cap_pool), size=2, replace=False)]
    caps = [first_two[i % 2] for i in range(count)]
    rng.shuffle(caps)
    return caps


def _expected_output(
    inp: np.ndarray,
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    group_caps: list[list[int]],
    ranks: list[int],
) -> np.ndarray:
    """Compute the expected output by recoloring the rank-k bar's fill to that bar's cap."""
    out = inp.copy()
    for cols, lengths, caps, rank in zip(group_columns, group_lengths, group_caps, ranks):
        ranked = sorted(zip(cols, lengths, caps), key=lambda item: -item[1])
        target_col, target_len, target_cap = ranked[rank - 1]
        top_r = _MAX_CAP_ROW - target_len
        out[top_r + 1 : _BOTTOM_ROW, target_col] = target_cap
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch."""
    rng = np.random.default_rng()

    # --- structure choices ---
    n_groups = int(rng.integers(MIN_GROUPS, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, c + 1)) for c in counts]

    # --- palette ---
    all_colors = list(range(*COLOR_RANGE))
    rng.shuffle(all_colors)
    fills = [int(v) for v in all_colors[:n_groups]]
    remaining = [int(v) for v in all_colors[n_groups:]]
    max_caps_here = min(MAX_CAP_COLORS, len(remaining))
    n_cap_colors = int(rng.integers(MIN_CAP_COLORS, max_caps_here + 1))
    cap_pool = remaining[:n_cap_colors]

    # --- layout ---
    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    group_columns = _assign_columns(rng, n_groups, counts, width)

    # --- render legend + bars ---
    grid = np.zeros((GRID_H, width), dtype=int)
    group_lengths: list[list[int]] = []
    group_caps: list[list[int]] = []
    for fill, count, rank, cols in zip(fills, counts, ranks, group_columns):
        paste(grid, np.full((rank, 1), fill, dtype=int), 0, cols[0])  # legend uses fill color

        lengths = _choose_fill_lengths(rng, count, rank)
        caps = _choose_bar_caps(rng, count, cap_pool)
        group_lengths.append(lengths)
        group_caps.append(caps)

        for col, fill_len, cap in zip(cols, lengths, caps):
            paste(grid, _build_bar(cap, fill, fill_len), _MAX_CAP_ROW - fill_len, col)

    # --- oracle check ---
    expected = _expected_output(grid, group_columns, group_lengths, group_caps, ranks)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f'failed to generate a valid input after {MAX_RETRIES} attempts')
