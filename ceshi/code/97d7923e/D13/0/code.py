# Variant of 97d7923e
# Dimension: D13 — 消歧规则（Tiebreak） (meta)
# Change: 组内条的填充长度全部不同，不需要消歧；排序键为 (-(fill_length), top_row, col)... → 允许同长度条，用列位置消歧

import itertools

import numpy as np

from arc_utils import get_bg, neighbors4, paste


# Design analysis for the D13 variant
# a) Concrete behavior:
#    The picture still has a top-left legend made of single-color vertical stacks,
#    and each legend color still points to a family of bottom-aligned cap/fill/cap bars.
#    The change is localized inside one bar family: equal fill lengths are now allowed.
#    Example: if cap color 2 has bars at columns [2, 7, 10] with fill lengths [5, 5, 2]
#    and the legend height is 2, the ranked order is [(2,5), (7,5), (10,2)] because the
#    two length-5 bars tie and the left one comes first. The output repaints column 7.
#    In the original task, such an input was impossible because lengths had to be distinct.
# b) Complexity calibration:
#    Only one group is forced to contain a decisive tie; all other groups keep the old
#    distinct-length behavior. This makes the D13 change visible without flooding the grid
#    with ambiguous cases or changing the overall object vocabulary.
# c) Hidden assumptions audit:
#    The old generator relied on distinct lengths, so its expected-output sorter only used
#    length. That is no longer safe: both solver and oracle must rank by (-length, col).
#    Group column lists are sorted explicitly so "left-to-right" means spatial position,
#    not incidental construction order. The old tertiary key top_row is no longer the rule
#    explanation; bottom alignment makes equal lengths share a top row anyway.
# d) Placement legality:
#    Bars remain one column wide, never share columns, and keep at least one blank column
#    between neighbors. The first bar of each group still shares its column with the legend,
#    so its fill length is capped by MAX_FILL_LEN - rank to preserve a blank row between the
#    legend block and the bar's top cap.
# e) Visual observability:
#    Every generated grid contains one group where the selected rank lands inside a tie block,
#    and reversing the left/right tiebreak would pick a different bar. That guarantees at least
#    one output change depends on left-to-right disambiguation.
# f) Visual cleanliness:
#    The layout remains the original sparse legend-plus-bars arrangement. No extra markers are
#    added, and only one tie-heavy group is introduced, so the scene stays easy to parse.
# g) Color inference:
#    Colors still only identify bar groups and repaint targets. No color id participates in
#    ranking; the new secondary rule is purely geometric and visible in the columns.


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
    Repaint one bar per legend entry using rank selection with a left-to-right tiebreak.

    Input structure:
    - The upper-left corner contains one-column monochrome legend stacks touching row 0.
      A stack's color K matches a family of bars, and its height k is the requested rank.
    - The rest of the grid contains bottom-aligned vertical cap/fill/cap bars. Bars in the
      same family share cap color K and have a solid non-background interior.

    Rule:
    - For each legend entry (K, k), collect every vertical bar whose two caps are color K.
    - Sort that family by fill length from longest to shortest; if lengths tie, use column
      position from left to right as the secondary key.
    - Recolor the interior of the k-th bar in that order to K.

    Key insight:
    - Legend height encodes an ordinal choice inside each cap-color family, not a literal bar
      size. When two bars are equally long, the puzzle resolves the tie spatially by column.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, "legend must be non-empty"

    for _x, cap, k in legend:
        segs = _vertical_sandwiches(g, cap)
        assert segs, f"legend cap {cap} must have matching vertical bars"
        segs.sort(key=lambda s: (-_fill_len(s), s[2]))
        assert 1 <= k <= len(segs), f"legend rank {k} out of range for {len(segs)} bars"
        top_r, bot_r, col, _fill = segs[k - 1]
        out[top_r + 1 : bot_r, col] = cap

    return out


# --------------- generator (O2) ---------------

# --- Configuration ---
GRID_H = 10
WIDTH_MIN = 10
WIDTH_MAX = 20
MAX_GROUPS = 4
MAX_TOTAL_BARS = 10
BARS_PER_GROUP = (2, 3)
MAX_FILL_LEN = 7
COLOR_RANGE = (1, 10)
COL_SPACING = 2
MAX_RETRIES = 300

_BOTTOM_ROW = GRID_H - 1
_MAX_CAP_ROW = GRID_H - 2

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


def _choose_distinct_fill_lengths(rng, count: int, rank: int) -> list[int]:
    """Pick distinct fill lengths for a non-tie group."""
    max_legend_len = MAX_FILL_LEN - rank
    first_len = int(rng.integers(1, max_legend_len + 1))
    pool = [fill_len for fill_len in range(1, MAX_FILL_LEN + 1) if fill_len != first_len]
    rng.shuffle(pool)
    return [first_len] + pool[: count - 1]


def _choose_decisive_tie_profile(rng, count: int) -> tuple[int, list[int]]:
    """Pick (rank, lengths) so the chosen rank falls inside an equal-length tie block."""
    for _ in range(200):
        if count == 2:
            rank = int(rng.integers(1, 3))
            shared_len = int(rng.integers(1, MAX_FILL_LEN - rank + 1))
            lengths = [shared_len, shared_len]
            return rank, lengths

        case = int(rng.integers(3))
        if case == 0:
            rank = int(rng.integers(1, 3))
            tie_len = int(rng.integers(2, MAX_FILL_LEN + 1))
            other_len = int(rng.integers(1, tie_len))
            tie_positions = sorted(rng.choice(3, size=2, replace=False).tolist())
            lengths = [other_len] * 3
            for idx in tie_positions:
                lengths[idx] = tie_len
        elif case == 1:
            rank = int(rng.integers(2, 4))
            other_len = int(rng.integers(2, MAX_FILL_LEN + 1))
            tie_len = int(rng.integers(1, other_len))
            tie_positions = sorted(rng.choice(3, size=2, replace=False).tolist())
            lengths = [other_len] * 3
            for idx in tie_positions:
                lengths[idx] = tie_len
        else:
            rank = int(rng.choice([1, 3]))
            shared_len = int(rng.integers(1, MAX_FILL_LEN + 1))
            lengths = [shared_len, shared_len, shared_len]

        if lengths[0] <= MAX_FILL_LEN - rank and _tiebreak_changes_target([0, 1, 2], lengths, rank):
            return rank, lengths

    raise RuntimeError("failed to sample a decisive tie profile")


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
    tie_verified = False

    for gi, (cap, fill, count, cols) in enumerate(zip(caps, fills, counts, group_columns)):
        if gi == tie_group:
            rank, lengths = _choose_decisive_tie_profile(rng, count)
            assert _selected_rank_is_tied(cols, lengths, rank)
            assert _tiebreak_changes_target(cols, lengths, rank)
            tie_verified = True
        else:
            rank = int(rng.integers(1, count + 1))
            lengths = _choose_distinct_fill_lengths(rng, count, rank)

        ranks.append(rank)
        group_lengths.append(lengths)

        paste(grid, np.full((rank, 1), cap, dtype=int), 0, cols[0])
        for col, fill_len in zip(cols, lengths):
            paste(grid, _build_bar(cap, fill, fill_len), _MAX_CAP_ROW - fill_len, col)

    assert tie_verified

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
