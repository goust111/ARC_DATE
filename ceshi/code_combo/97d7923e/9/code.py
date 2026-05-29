# Combo variant of 97d7923e (index 9)
# combo_spec: D4/1+D6/2
# Name: 浮动条+顶部行排序

import itertools

import numpy as np

from arc_utils import get_bg, neighbors4, paste

# Design analysis for the combo variant
# a) Concrete behavior:
#    The original task grouped bars by cap color K and used the legend stack
#    height k to select the k-th longest bar in that color group. Here the bars
#    still have the same cap / fill / cap structure, but they are no longer
#    bottom-aligned, and the selection key changes to the top-cap row. Example:
#    with three bars of the same cap color at rows top=[1, 4, 2] and lengths
#    len=[2, 6, 4], the old task would rank them by length as [6, 4, 2], while
#    the combo ranks them by how high they start as [1, 2, 4]. A legend height
#    of 2 therefore points to a different bar once the floating placement is
#    introduced.
# b) Complexity calibration:
#    The visual change must be obvious, so the generator enforces at least one
#    sample group whose selected target differs between top-row ranking and the
#    original length ranking. Bars still use distinct fill lengths within a
#    group so each column remains visually separable, but length is no longer
#    the deciding feature.
# c) Old implicit assumptions that no longer hold:
#    - Any formula derived from `bottom_row == GRID_H - 1` is invalid.
#    - Sorting by `-(bottom - top - 1)` is wrong for this combo, because length
#      is now only a distractor and the intended key is the top cap location.
#    - "Descending by row position" is geometrically ambiguous because larger row
#      indices are lower on the canvas. The D6 description clarifies the intended
#      semantics: the bar whose top cap is higher in the grid gets the better
#      rank, so the solver sorts `top_r` ascending.
#    - `_vertical_sandwiches` still assumes that valid bars are bracketed by two
#      consecutive cap pixels in the same column with a solid non-bg interior.
#      The generator must keep at least one blank row between a legend block and
#      any same-column bar top so the legend is not mistaken for a bar segment.
# d) Placement legality:
#    Each bar gets its own column, so floating vertically cannot cause overlap.
#    Non-legend columns keep bar tops at row >= 1 to protect the top row for the
#    legend cluster. The legend-sharing bar in each group starts at row >= rank+1
#    so the legend column contains a visible background gap. Every bar also obeys
#    `top + fill_len + 1 < GRID_H`.
# e) Visual observability:
#    A human can now see bottoms landing on different rows and, more importantly,
#    that the recolored target is the bar whose top cap sits at the requested
#    rank from the top rather than the longest bar. The generator explicitly
#    rejects samples where all bars still appear bottom-aligned or where the
#    selected bar would match the old length-based rule.
# f) Visual cleanliness:
#    The legend remains a compact cluster at the top-left, columns stay spaced by
#    at least one blank column, and each bar remains a one-cell-wide vertical
#    object. This keeps the grid readable even though bars float independently.
# g) Change inferability:
#    No hidden numeric property of the color ids is used. Cap color K still
#    links a legend entry to its candidate bars, while the deciding cue is the
#    visible top-row geometry. The legend height only encodes a rank index.
# h) Quantified anti-degeneracy target:
#    For each generated sample we can count how many groups choose a different
#    target under top-row ranking versus old length ranking. The generator forces
#    this "disagreement group" count to be at least 1, which proves the combo is
#    not collapsing back to the original task.
# i) New solver assumptions to guarantee:
#    The solver assumes one bar per candidate column, distinct top rows within a
#    group so top-order is unambiguous, and a non-empty legend parsed from the
#    leftmost top-row cluster. The generator constructs exactly that regime.


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


def _ranked_by_top(segs: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Higher bars rank earlier: smaller top-row index means better rank."""
    return sorted(segs, key=lambda s: (s[0], s[2]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor floating vertical bars using legend-specified top-order ranks.

    Input structure:
    - bg is the dominant background color.
    - At the top-left, each legend entry is a monochrome vertical stack touching row 0;
      its color K identifies one bar group and its height k encodes a rank.
    - Elsewhere, one-cell-wide cap-fill-cap bars of color K can float at different rows
      instead of sharing a common bottom edge.

    Rule:
    - For each legend entry (K, k), collect every vertical bar whose two caps are color K.
    - Order that group by the row of the top cap: the bar starting highest in the grid is
      rank 1, the next highest is rank 2, and so on.
    - Recolor the fill cells of the rank-k bar to K.

    Key insight:
    - The legend still encodes a rank, but the deciding feature is now vertical placement.
      Floating the bars breaks the old equivalence between "highest top" and "longest bar".
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, "legend must be non-empty"

    for _x, cap, k in legend:
        segs = _vertical_sandwiches(g, cap)
        assert segs, f"legend cap {cap} must have matching vertical bars"
        ranked = _ranked_by_top(segs)
        assert 1 <= k <= len(ranked), f"legend rank {k} out of range for {len(ranked)} bars"
        top_r, bot_r, col, _fill = ranked[k - 1]
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

    return group_columns


def _choose_fill_lengths(rng, count: int, rank: int) -> list[int]:
    """Pick distinct fill lengths; the legend-sharing bar must fit below its legend."""
    max_legend_len = GRID_H - rank - 3
    assert max_legend_len >= 1
    shared_len = int(rng.integers(1, min(MAX_FILL_LEN, max_legend_len) + 1))
    pool = [length for length in range(1, MAX_FILL_LEN + 1) if length != shared_len]
    rng.shuffle(pool)
    return [shared_len] + pool[: count - 1]


def _top_sort_order(top_rows: list[int]) -> tuple[int, ...]:
    return tuple(sorted(range(len(top_rows)), key=lambda i: (top_rows[i], i)))


def _length_sort_order(fill_lengths: list[int]) -> tuple[int, ...]:
    return tuple(sorted(range(len(fill_lengths)), key=lambda i: (-fill_lengths[i], i)))


def _target_index_by_top(top_rows: list[int], rank: int) -> int:
    return _top_sort_order(top_rows)[rank - 1]


def _target_index_by_length(fill_lengths: list[int], rank: int) -> int:
    return _length_sort_order(fill_lengths)[rank - 1]


def _enumerate_top_assignments(fill_lengths: list[int], rank: int) -> list[list[int]]:
    """Enumerate valid distinct top-row choices for one group."""
    count = len(fill_lengths)
    allowed = []
    for idx, fill_len in enumerate(fill_lengths):
        min_top = rank + 1 if idx == 0 else 1
        max_top = GRID_H - (fill_len + 2)
        if min_top > max_top:
            return []
        allowed.append(list(range(min_top, max_top + 1)))

    order = sorted(range(count), key=lambda i: (len(allowed[i]), -fill_lengths[i], i))
    assignments: list[list[int]] = []
    chosen = [-1] * count
    used: set[int] = set()

    def backtrack(pos: int) -> None:
        if pos == count:
            assignments.append(chosen.copy())
            return
        idx = order[pos]
        for top_r in allowed[idx]:
            if top_r in used:
                continue
            used.add(top_r)
            chosen[idx] = top_r
            backtrack(pos + 1)
            used.remove(top_r)
            chosen[idx] = -1

    backtrack(0)
    return assignments


def _choose_top_rows(rng, fill_lengths: list[int], rank: int, require_disagreement: bool) -> list[int] | None:
    """Pick a valid top-row assignment, optionally forcing disagreement with length ranking."""
    assignments = _enumerate_top_assignments(fill_lengths, rank)
    if not assignments:
        return None

    if require_disagreement:
        assignments = [
            tops
            for tops in assignments
            if _target_index_by_top(tops, rank) != _target_index_by_length(fill_lengths, rank)
        ]
        if not assignments:
            return None

    return assignments[int(rng.integers(len(assignments)))]


def _expected_output(
    inp: np.ndarray,
    caps: list[int],
    placements: list[list[tuple[int, int, int]]],
    ranks: list[int],
) -> np.ndarray:
    """Compute the expected output using top-row ranking within each cap-color group."""
    out = inp.copy()
    for cap, group_bars, rank in zip(caps, placements, ranks):
        ranked = sorted(group_bars, key=lambda item: (item[1], item[0]))
        target_col, target_top, target_len = ranked[rank - 1]
        out[target_top + 1 : target_top + 1 + target_len, target_col] = cap
    return out


def _has_visible_floating(placements: list[list[tuple[int, int, int]]]) -> bool:
    """Return True iff the sample visibly departs from full bottom alignment."""
    bottoms = [top + fill_len + 1 for group in placements for _col, top, fill_len in group]
    return len(set(bottoms)) >= 2 and any(bottom < _BOTTOM_ROW for bottom in bottoms)


def _disagreement_group_count(
    placements: list[list[tuple[int, int, int]]],
    ranks: list[int],
) -> int:
    """Count groups whose selected bar differs from the original length-based task."""
    count = 0
    for group_bars, rank in zip(placements, ranks):
        top_ranked = sorted(group_bars, key=lambda item: (item[1], item[0]))
        length_ranked = sorted(group_bars, key=lambda item: (-item[2], item[1], item[0]))
        if top_ranked[rank - 1][0] != length_ranked[rank - 1][0]:
            count += 1
    return count


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, count + 1)) for count in counts]

    palette = rng.choice(np.arange(*COLOR_RANGE), size=2 * n_groups, replace=False)
    caps = [int(v) for v in palette[:n_groups]]
    fills = [int(v) for v in palette[n_groups:]]

    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    group_columns = _assign_columns(rng, n_groups, counts, width)

    fill_lengths_per_group = [_choose_fill_lengths(rng, count, rank) for count, rank in zip(counts, ranks)]
    disagreeable = [
        idx
        for idx, (fill_lengths, rank) in enumerate(zip(fill_lengths_per_group, ranks))
        if _choose_top_rows(rng, fill_lengths, rank, require_disagreement=True) is not None
    ]
    if not disagreeable:
        return None
    focus_group = int(rng.choice(disagreeable))

    grid = np.zeros((GRID_H, width), dtype=int)
    placements: list[list[tuple[int, int, int]]] = []

    for group_idx, (cap, fill, rank, cols, fill_lengths) in enumerate(
        zip(caps, fills, ranks, group_columns, fill_lengths_per_group)
    ):
        paste(grid, np.full((rank, 1), cap, dtype=int), 0, cols[0])
        top_rows = _choose_top_rows(
            rng,
            fill_lengths,
            rank,
            require_disagreement=(group_idx == focus_group),
        )
        if top_rows is None:
            return None

        group_bars: list[tuple[int, int, int]] = []
        for col, fill_len, top_r in zip(cols, fill_lengths, top_rows):
            paste(grid, _build_bar(cap, fill, fill_len), top_r, col)
            group_bars.append((col, top_r, fill_len))
        placements.append(group_bars)

    if not _has_visible_floating(placements):
        return None
    if _disagreement_group_count(placements, ranks) < 1:
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
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
