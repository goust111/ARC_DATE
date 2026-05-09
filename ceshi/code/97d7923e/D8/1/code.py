# Variant of 97d7923e
# Dimension: D8 — 图例与条的空间关系 (meta)
# Change: 图例占据左上角顶部若干行，条从更低的行开始向下延伸到最后一行；图例列和条列可共享同一列（图例在上方且有背景色行间隔）... → 图例嵌入在条之间

import itertools

import numpy as np

from arc_utils import get_bg, paste


# Design analysis
# - concrete_layout:
#   grid: 10 x 10..20, bg=0
#   groups: 2..4 cap-color groups, each with 2..3 vertical bars
#   bars: one column each, cap / solid fill / cap, all bottom-aligned
#   legend_variant: each group's first bar column also carries that group's legend:
#     a same-color vertical stack touching row 0, with height k = desired rank.
#     The stack is separated from the actual bar in that column by >= 1 bg row.
#   effect: the top row no longer contains one left-corner legend cluster; instead,
#     non-bg top-row cells are spread across the bar field and each is locally tied
#     to the bar directly below it.
# - behavior_trace:
#   old: collect one contiguous top-left cluster, read entry heights there, then rank
#     every cap-matching bar globally.
#   new: scan every non-bg top-row column, keep only stacks whose own column also has
#     a lower bar with the same cap, use that local stack as the legend entry for the
#     whole same-cap group, then perform the same rank-based recolor.
#   difference: legend detection changes from "global corner region" to "distributed
#     per-group anchors embedded in bar columns".
# - complexity_calibration:
#   keep cap colors unique and fill colors disjoint from caps so grouping is still
#   visually readable; require at least 2 groups so the scattered-legend layout is
#   usually observable and does not collapse into the original single-cluster cue.
# - hidden_assumptions_audit:
#   old _legend_entries depended on all legends appearing in one short leftmost run of
#   top-row pixels with gaps <= 2; that is false now and would ignore later groups.
#   _vertical_sandwiches itself remains valid because the extra legend pixels only add
#   cap-to-cap pairs whose interiors are bg, so they are rejected as non-bars.
#   main() must therefore stop using a bounded top-left flood-fill and instead inspect
#   each top-row colored column independently.
# - placement_legality:
#   the first bar in each group uses a constrained fill length so its top cap stays at
#   least one row below the legend stack. Columns are unique, groups are laid out left
#   to right with spacing 2, and extra width is injected as left padding / inter-group
#   gaps only, so nothing overlaps or leaves the grid.
# - visual_observability:
#   the only changed cue is spatial: legends are distributed above their own groups.
#   Bars keep repeated structure and repeated cap colors, so humans can still see that
#   each local top stack supplies the rank for nearby same-cap bars.
# - cleanliness_and_color_reasoning:
#   no color-id arithmetic is used; color changes are always "fill -> that bar group's
#   cap color". The legend remains one-column wide, preventing clutter at the top.


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


def _embedded_legend_entries(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    """
    Legends are distributed across the top row: each legend is a one-column stack
    touching row 0, and its own column also contains a lower bar with the same cap.

    Returns entries as (x_pos, legend_color, block_count), sorted by x_pos.
    """
    h, w = g.shape
    entries: list[tuple[int, int, int]] = []

    for c in range(w):
        cap = int(g[0, c])
        if cap == bg:
            continue

        k = 1
        while k < h and int(g[k, c]) == cap:
            k += 1
        if k >= h or int(g[k, c]) != bg:
            continue

        segs = [s for s in _vertical_sandwiches(g, cap) if s[2] == c and s[0] > k - 1]
        if segs:
            entries.append((c, cap, k))

    assert entries, "embedded legend must be non-empty"
    return entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor vertical bars by rank marks embedded inside the bar field.

    Input structure:
    - bg = background. Along the top row, each group has a one-column cap-colored stack.
      That stack sits in the same column as the group's first bar, with a bg gap below it.
    - The main objects are bottom-aligned vertical bars whose top and bottom cells share
      cap color K and whose interiors are solid fill color F.

    Rule:
    - Treat each top stack (K, k) as the legend for the lower same-cap bar group in that
      column.
    - Among all bars capped by K, sort by interior length descending (longest = rank 1).
    - Recolor the interior of the rank-k bar to K.

    Key insight: The legend is no longer a separate corner cluster; it is a local marker
    embedded above one bar column in each group, and its height still encodes the rank.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _embedded_legend_entries(g, bg)
    assert legend, "legend must be non-empty"

    for _x, cap, k in legend:
        segs = _vertical_sandwiches(g, cap)
        assert segs, f"legend cap {cap} must have matching vertical bars"
        segs.sort(key=lambda s: (-(s[1] - s[0] - 1), s[0], s[2]))
        assert 1 <= k <= len(segs), f"legend rank {k} out of range for {len(segs)} bars"
        top_r, bot_r, col, _fill = segs[k - 1]
        out[top_r + 1 : bot_r, col] = cap

    return out


# --------------- generator (O2) ---------------

# --- Configuration ---
GRID_H = 10
WIDTH_MIN = 10
WIDTH_MAX = 20
MIN_GROUPS = 2
MAX_GROUPS = 4
MAX_TOTAL_BARS = 10
BARS_PER_GROUP = (2, 3)
MAX_FILL_LEN = 7
COLOR_RANGE = (1, 10)
COL_SPACING = 2
MAX_RETRIES = 300

_BOTTOM_ROW = GRID_H - 1

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


def _assign_group_columns(rng, counts: list[int], width: int) -> list[list[int]]:
    """
    Lay groups out left-to-right as local bar blocks.

    Each group's first column is its legend column, so the top-row legend markers end up
    distributed across the bar field instead of forming one contiguous cluster.
    """
    n_groups = len(counts)
    total_bars = sum(counts)
    base_span = COL_SPACING * total_bars - 1
    extra_width = width - base_span

    gap_parts = rng.multinomial(extra_width, np.full(n_groups + 1, 1.0 / (n_groups + 1)))
    left_pad = int(gap_parts[0])
    inter_group_bonus = [int(v) for v in gap_parts[1:n_groups]]

    group_columns: list[list[int]] = []
    cur = left_pad
    for idx, count in enumerate(counts):
        cols = [cur + COL_SPACING * j for j in range(count)]
        group_columns.append(cols)
        if idx + 1 < n_groups:
            cur = cols[-1] + COL_SPACING + inter_group_bonus[idx]
    return group_columns


def _choose_fill_lengths(rng, count: int, rank: int) -> list[int]:
    """Pick distinct fill lengths; the first one must leave room for the embedded legend."""
    max_legend_len = MAX_FILL_LEN - rank
    shared_len = int(rng.integers(1, max_legend_len + 1))
    pool = [l for l in range(1, MAX_FILL_LEN + 1) if l != shared_len]
    rng.shuffle(pool)
    return [shared_len] + pool[: count - 1]


def _bar_top_row(fill_len: int) -> int:
    return _BOTTOM_ROW - fill_len - 1


def _expected_output(
    inp: np.ndarray,
    caps: list[int],
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
) -> np.ndarray:
    """Compute the expected output by recoloring the rank-k bar's fill to cap."""
    out = inp.copy()
    for cap, cols, lengths, rank in zip(caps, group_columns, group_lengths, ranks):
        ranked = sorted(zip(cols, lengths), key=lambda item: (-item[1], item[0]))
        target_col, target_len = ranked[rank - 1]
        top_r = _bar_top_row(target_len)
        out[top_r + 1 : _BOTTOM_ROW, target_col] = cap
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(MIN_GROUPS, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, c + 1)) for c in counts]

    palette = rng.choice(np.arange(*COLOR_RANGE), size=2 * n_groups, replace=False)
    caps = [int(v) for v in palette[:n_groups]]
    fills = [int(v) for v in palette[n_groups:]]

    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    group_columns = _assign_group_columns(rng, counts, width)

    grid = np.zeros((GRID_H, width), dtype=int)
    group_lengths: list[list[int]] = []
    for cap, fill, count, rank, cols in zip(caps, fills, counts, ranks, group_columns):
        paste(grid, np.full((rank, 1), cap, dtype=int), 0, cols[0])

        lengths = _choose_fill_lengths(rng, count, rank)
        group_lengths.append(lengths)

        for col, fill_len in zip(cols, lengths):
            paste(grid, _build_bar(cap, fill, fill_len), _bar_top_row(fill_len), col)

    expected = _expected_output(grid, caps, group_columns, group_lengths, ranks)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
