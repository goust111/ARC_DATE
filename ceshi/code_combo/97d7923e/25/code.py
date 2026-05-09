# Combo variant of 97d7923e (index 25)
# combo_spec: D6/3+D9/1+D13/0
# Name: 多键排序+更多条+列位置消歧

import itertools

import numpy as np

from arc_utils import get_bg, neighbors4, paste


# Design analysis for the combo variant
# a) Concrete behavior:
#    The scene still has a legend in the upper-left and one vertical-bar family per legend
#    color. The fused change is that each family now has 4-5 bars, equal fill lengths are
#    allowed, and ties are resolved by spatial order. Example: if cap color 2 owns columns
#    [0, 8, 12, 18, 24] with fill lengths [2, 6, 6, 4, 1] and the legend stack has height 3,
#    the ranked order is [(8,6), (12,6), (18,4), (0,2), (24,1)] because length is primary and
#    the two length-6 bars are broken left-to-right. Rank 3 repaints column 18. In the seed
#    task, groups only had 2-3 bars and lengths were distinct, so neither deep ranking nor the
#    tie rule mattered.
# b) Complexity calibration:
#    Every group uses 4 or 5 bars so the larger search space is always visible. At least one
#    group per grid is forced to have a decisive equal-length tie at the selected rank, but not
#    every group is made tie-heavy; this keeps the new rule observable without turning the whole
#    picture into a wall of duplicates.
# c) Hidden assumptions audit:
#    The original solver sorted by length and then by geometric accident `(top_row, col)`. With
#    bottom-aligned bars, equal lengths share the same top row, so the actual intended tiebreak
#    must now be stated directly as column order. The generator can no longer rely on all lengths
#    being distinct, and its oracle must use the same `(-fill_length, col)` ranking as `main()`.
#    Group column lists are sorted explicitly so "left to right" refers to the rendered grid.
# d) Placement legality:
#    Bars remain one cell wide, occupy distinct columns, and keep one blank column between any two
#    occupied columns. The first bar of each group still shares the legend column, so its fill
#    length is capped by `MAX_FILL_LEN - rank` to preserve one blank row between the legend stack
#    and that bar's top cap. Wider grids are required because 4 groups x 5 bars needs up to 20
#    occupied columns.
# e) Visual observability:
#    A solver can see the new rule from examples because one family in every generated grid has a
#    selected rank that lands inside a tie block, and reversing the left/right order would pick a
#    different bar. The bars are still compared by visible geometric height only; no color id or
#    hidden metadata participates in ranking.
# f) Visual cleanliness:
#    The layout stays sparse: legend entries are confined to the top-left cluster, bars are vertical
#    single-column objects, and extra width is spent on horizontal spacing instead of adding more
#    marker types. That keeps bar families easy to parse even when each family has 5 members.
# g) Change inferability:
#    Recoloring is still structurally cued: a legend color always maps to the family with the same
#    cap color, and the output only changes one bar interior per family. The new information to
#    infer is purely ordinal: compare lengths first, then use left-to-right order when lengths tie.


def _fill_len(seg: tuple[int, int, int, int]) -> int:
    return int(seg[1] - seg[0] - 1)


def _rank_segs(segs: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    return sorted(segs, key=lambda seg: (-_fill_len(seg), seg[2]))


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
    the top row within the legend cluster.
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
        rows = [r for r, _ in pix]
        block_count = max(rows) - min(rows) + 1
        entries.append((x_pos, col, int(block_count)))

    entries.sort(key=lambda item: item[0])
    return entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint one bar per legend entry using deep rank selection with an explicit column tiebreak.

    Input structure:
    - The upper-left corner contains one-column monochrome legend stacks touching row 0. A stack
      has color K and height k, where k now ranges across larger 4-5 bar families.
    - The rest of the grid contains bottom-aligned vertical cap/fill/cap bars. Bars in the same
      family share cap color K and have solid interiors; equal interior lengths are allowed.

    Rule:
    - For each legend entry (K, k), gather every vertical bar capped by color K.
    - Rank that family by fill length from longest to shortest. If two bars have the same fill
      length, the one in the smaller column ranks earlier.
    - Recolor the interior of the k-th ranked bar to K.

    Key insight:
    - Legend height is an ordinal selector inside a cap-color family, not a literal bar size. In
      this variant, longer families and equal-length bars make the secondary left-to-right rule
      necessary for deciding which bar receives the repaint.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, "legend must be non-empty"

    for _x, cap, rank in legend:
        segs = _vertical_sandwiches(g, cap)
        assert segs, f"legend cap {cap} must have matching vertical bars"
        ranked = _rank_segs(segs)
        assert 1 <= rank <= len(ranked), f"legend rank {rank} out of range for {len(ranked)} bars"
        top_r, bot_r, col, _fill = ranked[rank - 1]
        out[top_r + 1 : bot_r, col] = cap

    return out


# --------------- generator (O2) ---------------

# --- Configuration ---
GRID_H = 10
WIDTH_MIN = 18
WIDTH_MAX = 44
MAX_GROUPS = 4
MAX_TOTAL_BARS = 20
BARS_PER_GROUP = (4, 5)
MAX_FILL_LEN = 7
COLOR_RANGE = (1, 10)
COL_SPACING = 2
MAX_RETRIES = 400

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
    """Distribute bar columns across groups while keeping the legend cluster at the far left."""
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


def _sample_regular_lengths(rng, count: int, rank: int) -> list[int]:
    """Pick 4-5 visible lengths, allowing ties but keeping at least two distinct heights."""
    max_first_len = MAX_FILL_LEN - rank
    for _ in range(200):
        first_len = int(rng.integers(1, max_first_len + 1))
        anchor_count = int(rng.integers(2, 5))
        anchors = [int(v) for v in rng.choice(np.arange(1, MAX_FILL_LEN + 1), size=anchor_count, replace=False)]
        lengths = [first_len]
        for _idx in range(count - 1):
            if rng.random() < 0.6:
                lengths.append(int(anchors[int(rng.integers(len(anchors)))]))
            else:
                lengths.append(int(rng.integers(1, MAX_FILL_LEN + 1)))
        if len(set(lengths)) >= 2:
            return lengths
    raise RuntimeError("failed to sample regular lengths")


def _sample_decisive_tie_group(rng, count: int, cols: list[int]) -> tuple[int, list[int]]:
    """Pick (rank, lengths) so the selected rank depends on left-to-right tiebreaking."""
    for _ in range(1000):
        tie_size = int(rng.integers(2, min(3, count) + 1))
        tie_start = int(rng.integers(1, count - tie_size + 2))
        rank = int(rng.integers(tie_start, tie_start + tie_size))
        max_first_len = MAX_FILL_LEN - rank

        above_count = tie_start - 1
        below_count = count - tie_size - above_count

        tie_min = 2 if below_count > 0 else 1
        tie_max = MAX_FILL_LEN - 1 if above_count > 0 else MAX_FILL_LEN
        if tie_min > tie_max:
            continue
        tie_len = int(rng.integers(tie_min, tie_max + 1))

        above = []
        below = []
        if above_count > 0:
            above = [int(rng.integers(tie_len + 1, MAX_FILL_LEN + 1)) for _ in range(above_count)]
        if below_count > 0:
            below = [int(rng.integers(1, tie_len)) for _ in range(below_count)]

        lengths = above + [tie_len] * tie_size + below
        rng.shuffle(lengths)
        if lengths[0] > max_first_len:
            continue
        if len(set(lengths)) < 2:
            continue
        if _selected_rank_is_tied(cols, lengths, rank) and _tiebreak_changes_target(cols, lengths, rank):
            return rank, lengths

    raise RuntimeError("failed to sample a decisive tie group")


def _expected_output(
    inp: np.ndarray,
    caps: list[int],
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
) -> np.ndarray:
    """Compute the expected output by recoloring the rank-selected bar's fill to cap."""
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
    decisive_group = int(rng.integers(n_groups))

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
    decisive_verified = False

    for gi, (cap, fill, count, cols) in enumerate(zip(caps, fills, counts, group_columns)):
        if gi == decisive_group:
            rank, lengths = _sample_decisive_tie_group(rng, count, cols)
            decisive_verified = True
        else:
            rank = int(rng.integers(1, count + 1))
            lengths = _sample_regular_lengths(rng, count, rank)

        ranks.append(rank)
        group_lengths.append(lengths)

        paste(grid, np.full((rank, 1), cap, dtype=int), 0, cols[0])
        for col, fill_len in zip(cols, lengths):
            paste(grid, _build_bar(cap, fill, fill_len), _MAX_CAP_ROW - fill_len, col)

    assert decisive_verified

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
