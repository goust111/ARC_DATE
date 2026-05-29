# Variant of 97d7923e
# Dimension: D8 — 图例与条的空间关系 (meta)
# Change: 图例占据左上角顶部若干行，条从更低的行开始向下延伸到最后一行；图例列和条列可共享同一列（图例在上方且有背景色行间隔）... → 图例在网格底部

import itertools

import numpy as np

from arc_utils import get_bg, neighbors4, paste

# Design analysis for the D8 variant
# a) Concrete behavior:
#    - The original task uses legend stacks touching row 0 and bars that all end at the last row.
#    - This variant vertically flips that spatial relationship: legend stacks now touch the last
#      row and rise upward, while every bar starts at row 0 and extends downward.
#    - A concrete trace for one color K with rank k=2:
#      old layout: legend cells at rows 0-1, candidate bars are cap/fill/cap columns whose bottom
#      cap is at row 9; recolor the second-longest bar for K.
#      new layout: legend cells at rows 8-9, candidate bars are cap/fill/cap columns whose top cap
#      is at row 0; recolor the second-longest bar for K.
#      The ranking rule is unchanged; only the spatial anchoring is inverted.
# b) Complexity calibration:
#    - Keep 1-4 legend groups and 2-3 bars per group so each sample still has repeated evidence.
#    - Keep per-group fill lengths distinct to avoid ambiguous rank ties.
#    - Keep legend columns near the left edge with small gaps, preserving the visible "legend zone".
# c) Solver assumptions that break if left unchanged:
#    - Reading legend seeds from row 0 no longer works; the legend must now be detected from row 9.
#    - The old prose in main() says bars are bottom-aligned; that must be rewritten because the
#      candidate bars now share a common top cap at row 0 instead.
#    - The bar finder itself can stay unchanged because it only looks for consecutive cap-color
#      occurrences in a column with a solid non-bg fill between them.
# d) Placement legality:
#    - Bars occupy one column each and are pasted from row 0.
#    - Shared legend/bar columns are still allowed, but the shared-column bar length is limited so
#      there is at least one background row between the bar's bottom cap and the bottom legend stack.
#    - MAX_FILL_LEN=7 keeps all bars off the last row, so the bottom row cleanly identifies legend.
# e) Visual observability:
#    - The bottom row always contains the legend and never any bar endpoint, making the inversion
#      obvious across few examples.
#    - The output change remains structured: one bar interior per legend group changes from fill F
#      to cap color K, with the selected bar determined by the visible legend height.
# f) Visual cleanliness:
#    - Legend stays in a compact bottom-left cluster; bars remain separated by at least one empty
#      column, so the two regions are easy to parse.
#    - Same-color connections between legend and bars are prevented by a guaranteed background gap.
# g) Color inference:
#    - No invisible property such as numeric color id affects ranking or recoloring.
#    - Color changes are still explained by visible pairings: legend color K matches the bar caps of
#      the same K, and legend height k chooses the k-th longest bar within that color group.
# New solver-side assumptions to guarantee in the generator:
#    - The first bottom-row non-bg cluster is the full legend region.
#    - Each legend color appears in exactly one bottom-touching vertical stack inside that region.
#    - For every legend cap color, at least k bars with that cap exist, with distinct interior lengths.


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
    Legend is in the bottom-left: a cluster of non-bg pixels on the last row near the left edge,
    allowing small gaps (typically 1 bg column) between entries.

    Returns entries as (x_pos, legend_color, block_count), sorted by x_pos.
    block_count is the vertical height of the 4-connected component of that color that touches
    the bottom row within the legend cluster.
    """
    h, w = g.shape
    bottom_r = h - 1
    bottom_pos = [int(c) for c in np.flatnonzero(g[bottom_r, :] != bg)]
    assert bottom_pos, 'legend row must be non-empty'
    bottom_pos.sort()

    cols = [bottom_pos[0]]
    for c in bottom_pos[1:]:
        if c - cols[-1] <= 2:
            cols.append(c)
        else:
            break
    legend_max_c = max(cols)

    entries: list[tuple[int, int, int]] = []
    seen = np.zeros((h, w), dtype=bool)
    for c in cols:
        col = int(g[bottom_r, c])
        if col == bg or seen[bottom_r, c]:
            continue
        st = [(bottom_r, c)]
        seen[bottom_r, c] = True
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
    Recolor vertical bars using rank values stored in the bottom legend.

    Input structure:
    - bg is the background color. Near the lower-left corner, monochrome legend blocks touch the
      last row; each block has color K and vertical height k.
    - Elsewhere, the main objects are vertical cap-fill-cap bars whose top cap is on row 0 and
      whose interiors are solid non-background colors.

    Rule:
    - For each legend entry (K, k), collect all vertical bars capped by K.
    - Order those bars by interior length from longest to shortest.
    - Recolor the interior of the rank-k bar so that its fill becomes K.

    Key insight:
    - The legend height is not copied geometrically; it serves as a rank index into the bars that
      share the same cap color, even though the legend now sits below the bars instead of above.
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

_TOP_CAP_ROW = 0
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
    """Pick `count` distinct fill lengths for one group.

    The first length (legend column bar) is constrained so the bar fits
    above the bottom legend block with a bg gap row.
    """
    max_shared_len = MAX_FILL_LEN - rank
    shared_len = int(rng.integers(1, max_shared_len + 1))
    pool = [l for l in range(1, MAX_FILL_LEN + 1) if l != shared_len]
    rng.shuffle(pool)
    return [shared_len] + pool[: count - 1]


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
        out[1 : target_len + 1, target_col] = cap
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, c + 1)) for c in counts]
    if all(rank == 1 for rank in ranks):
        return None

    palette = rng.choice(np.arange(*COLOR_RANGE), size=2 * n_groups, replace=False)
    caps = [int(v) for v in palette[:n_groups]]
    fills = [int(v) for v in palette[n_groups:]]

    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    group_columns = _assign_columns(rng, n_groups, counts, width)

    grid = np.zeros((GRID_H, width), dtype=int)
    group_lengths: list[list[int]] = []
    for cap, fill, count, rank, cols in zip(caps, fills, counts, ranks, group_columns):
        paste(grid, np.full((rank, 1), cap, dtype=int), GRID_H - rank, cols[0])

        lengths = _choose_fill_lengths(rng, count, rank)
        group_lengths.append(lengths)

        for col, fill_len in zip(cols, lengths):
            paste(grid, _build_bar(cap, fill, fill_len), _TOP_CAP_ROW, col)

    expected = _expected_output(grid, caps, group_columns, group_lengths, ranks)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f'failed to generate a valid input after {MAX_RETRIES} attempts')
