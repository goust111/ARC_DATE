# Combo variant of 97d7923e (index 14)
# combo_spec: D9/1+D13/0
# Name: 更多条+列位置消歧

import itertools

import numpy as np

from arc_utils import get_bg, neighbors4, paste


# Design analysis for the D9/1 + D13/0 combo
# a) Concrete behavior:
#    The scene still has a top-left legend of one-column color stacks and bottom-aligned
#    cap/fill/cap bars grouped by cap color. The combo changes the search space inside each
#    color family: every family now has 4-5 bars instead of 2-3, and equal fill lengths are
#    legal. Example: for one cap color K, suppose the family columns are [0, 8, 10, 14, 18],
#    the fill lengths are [2, 6, 6, 4, 1], and the legend height is 5. Sorting by
#    (-length, col) gives [(8,6), (10,6), (14,4), (0,2), (18,1)], so rank 5 repaints the
#    bar at col 18. If the legend height were 2, the tie between the two length-6 bars would
#    be resolved by column and col 10 would be repainted. In the original task, neither the
#    5-way search nor the equal-length tie was supposed to appear.
# b) Complexity calibration:
#    The visual change must be obvious, so every sample uses 4-5 bars per group and at least
#    one legend stack has height 4 or 5. To keep 10-row grids readable, the generator caps the
#    total bars at 16; that still allows up to four groups while making the widened layout
#    manageable. Non-forced groups are also sampled from a small length range, so repeated
#    lengths occur often enough to look natural rather than like a single isolated corner case.
# c) Hidden assumptions audit:
#    The solver can no longer rely on unique lengths, so ranking must be defined explicitly as
#    (-fill_length, col). The original code's top-row key was only a deterministic fallback;
#    here equal lengths are intentional and bottom alignment makes equal lengths share the same
#    top row anyway. The generator must also sort each group's columns so "left-to-right" is a
#    geometric property, not an artifact of construction order, and it must permit legend ranks
#    up to 5 because group size increased.
# d) Placement legality:
#    Bars remain one cell wide, occupy distinct columns, and keep at least one blank column
#    between neighbors via a stride of 2. The leftmost column of each group is still shared by
#    that group's legend stack, so the bar placed there is constrained to leave at least one
#    background row below the legend block. Width is expanded to fit the extra bars without
#    overlap or clipping.
# e) Visual observability:
#    Every generated grid contains one decisive family whose chosen rank lies inside a tie block
#    and whose legend height is 4 or 5. Reversing the column tiebreak would recolor a different
#    bar in that family, so the new rule is visible from pixels alone. The larger family size is
#    also directly observable because each legend color points to four or five candidate bars.
# f) Visual cleanliness:
#    The picture still uses the original sparse vocabulary: legend at the upper left, bar field
#    to the right and below, no new markers. The legend columns remain contiguous near the left
#    edge, and all bars stay bottom-aligned, so even wider samples read as clean grouped columns
#    instead of cluttered overlap.
# g) Inferability of changes:
#    Color continues to only identify families and repaint targets; no ranking decision depends
#    on raw color ids. The human-solvable cues are visible structure only: family membership by
#    shared cap color, rank by legend height, and tie resolution by left-to-right column order.


def _fill_len(seg: tuple[int, int, int, int]) -> int:
    return int(seg[1] - seg[0] - 1)


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
    Repaint one bar per legend entry inside larger bar families, using columns to break ties.

    Input structure:
    - The upper-left corner contains one-column monochrome legend stacks touching row 0.
      A stack's color K matches one family of vertical bars, and its height k is the rank.
    - The rest of the grid contains bottom-aligned cap/fill/cap bars. Each family has 4-5
      bars sharing cap color K, and equal fill lengths may appear inside the family.

    Rule:
    - For each legend entry (K, k), collect every vertical bar whose two caps are color K.
    - Sort that family by fill length from longest to shortest; when lengths tie, choose the
      left bar before the right bar.
    - Recolor the interior of the k-th bar in that order to K.

    Key insight:
    - Legend height is an ordinal selector over a larger candidate set, not a literal size.
      Because same-length bars are allowed, horizontal position becomes the visible tiebreak.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, 'legend must be non-empty'

    for _x, cap, k in legend:
        segs = _vertical_sandwiches(g, cap)
        assert segs, f'legend cap {cap} must have matching vertical bars'
        segs.sort(key=lambda s: (-_fill_len(s), s[2]))
        assert 1 <= k <= len(segs), f'legend rank {k} out of range for {len(segs)} bars'
        top_r, bot_r, col, _fill = segs[k - 1]
        out[top_r + 1 : bot_r, col] = cap

    return out


# --------------- generator (O2) ---------------

# --- Configuration ---
GRID_H = 10
WIDTH_MIN = 14
WIDTH_MAX = 36
MAX_GROUPS = 4
MAX_TOTAL_BARS = 16
BARS_PER_GROUP = (4, 5)
MAX_FILL_LEN = 7
COLOR_RANGE = (1, 10)
COL_SPACING = 2
MAX_RETRIES = 300

_BOTTOM_ROW = GRID_H - 1
_MAX_CAP_ROW = GRID_H - 2

_FEASIBLE_COUNTS = {
    ng: [counts for counts in itertools.product(BARS_PER_GROUP, repeat=ng) if sum(counts) <= MAX_TOTAL_BARS]
    for ng in range(1, MAX_GROUPS + 1)
}


def _build_bar(cap: int, fill: int, fill_len: int) -> np.ndarray:
    """Return a (fill_len+2, 1) column sprite: cap / fill*N / cap."""
    bar = np.full((fill_len + 2, 1), fill, dtype=int)
    bar[0, 0] = cap
    bar[-1, 0] = cap
    return bar


def _assign_columns(rng, n_groups: int, counts: list[int], width: int) -> list[list[int]]:
    """Distribute bar columns across groups while keeping legend columns clustered on the left."""
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

    return [sorted(cols) for cols in group_columns]


def _rank_pairs(cols: list[int], lengths: list[int]) -> list[tuple[int, int]]:
    return sorted(zip(cols, lengths), key=lambda item: (-item[1], item[0]))


def _selected_rank_is_tied(cols: list[int], lengths: list[int], rank: int) -> bool:
    ranked = _rank_pairs(cols, lengths)
    target_len = ranked[rank - 1][1]
    return sum(1 for _col, fill_len in ranked if fill_len == target_len) >= 2


def _tiebreak_changes_target(cols: list[int], lengths: list[int], rank: int) -> bool:
    left_ranked = _rank_pairs(cols, lengths)
    right_ranked = sorted(zip(cols, lengths), key=lambda item: (-item[1], -item[0]))
    return left_ranked[rank - 1][0] != right_ranked[rank - 1][0]


def _choose_deep_decisive_tie_profile(rng, count: int) -> tuple[int, list[int]]:
    """
    Build one family where:
      - the legend rank is 4 or 5,
      - the chosen rank lands inside a tie block,
      - reversing the left/right tiebreak picks a different bar.
    """
    rank = int(rng.integers(4, count + 1))
    low = int(rng.integers(1, MAX_FILL_LEN - rank + 1))
    high_count = count - 2
    high_pool = np.arange(low + 1, MAX_FILL_LEN + 1)
    highs = [int(v) for v in rng.choice(high_pool, size=high_count, replace=False)]
    rng.shuffle(highs)
    lengths = [low] + highs + [low]

    cols = list(range(count))
    assert lengths[0] <= MAX_FILL_LEN - rank
    assert _selected_rank_is_tied(cols, lengths, rank)
    assert _tiebreak_changes_target(cols, lengths, rank)
    return rank, lengths


def _choose_other_lengths(rng, count: int, rank: int) -> list[int]:
    """
    Sample lengths for a non-forced group.

    Duplicates are preferred so the wider families frequently show ties, but only one group
    must be decisively controlled by the tiebreak.
    """
    max_legend_len = MAX_FILL_LEN - rank
    prefer_duplicate = bool(rng.random() < 0.7)
    for _ in range(200):
        lengths = [int(rng.integers(1, max_legend_len + 1))]
        lengths.extend(int(v) for v in rng.integers(1, MAX_FILL_LEN + 1, size=count - 1))
        if prefer_duplicate and len(set(lengths)) == count:
            continue
        return lengths

    fallback = [1] + [1 + (idx % max(1, MAX_FILL_LEN - 1)) for idx in range(count - 1)]
    fallback[0] = min(fallback[0], max_legend_len)
    return fallback


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
        ranked = _rank_pairs(cols, lengths)
        target_col, target_len = ranked[rank - 1]
        top_r = _MAX_CAP_ROW - target_len
        out[top_r + 1 : _BOTTOM_ROW, target_col] = cap
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    tie_group = int(rng.integers(n_groups))

    palette = rng.choice(np.arange(*COLOR_RANGE), size=2 * n_groups, replace=False)
    caps = [int(v) for v in palette[:n_groups]]
    fills = [int(v) for v in palette[n_groups:]]

    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    group_columns = _assign_columns(rng, n_groups, counts, width)

    grid = np.zeros((GRID_H, width), dtype=int)
    group_lengths: list[list[int]] = []
    ranks: list[int] = []
    decisive_tie_verified = False

    for gi, (cap, fill, count, cols) in enumerate(zip(caps, fills, counts, group_columns)):
        if gi == tie_group:
            rank, lengths = _choose_deep_decisive_tie_profile(rng, count)
            decisive_tie_verified = True
        else:
            rank = int(rng.integers(1, count + 1))
            lengths = _choose_other_lengths(rng, count, rank)

        ranks.append(rank)
        group_lengths.append(lengths)

        paste(grid, np.full((rank, 1), cap, dtype=int), 0, cols[0])
        for col, fill_len in zip(cols, lengths):
            paste(grid, _build_bar(cap, fill, fill_len), _MAX_CAP_ROW - fill_len, col)

    assert decisive_tie_verified
    assert any(rank >= 4 for rank in ranks)

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
