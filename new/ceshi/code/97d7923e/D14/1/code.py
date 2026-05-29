# Variant of 97d7923e
# Dimension: D14 — 条的底部对齐约束 (meta)
# Change: 所有条底部对齐在最后一行(row 9)，顶部位置由长度决定... → 中心对齐

import itertools

import numpy as np

from arc_utils import get_bg, neighbors4, paste


# Design analysis
# variant_spec:
#   core_change:
#     old_geometry: all cap-fill-cap bars shared bottom row 9, so length only changed the top.
#     new_geometry: all bars share CENTER_ROW as their midpoint; longer bars extend farther
#       upward and downward, shorter bars stay closer to the middle.
#   concretization:
#     - Bars are still single-column cap/fill/cap objects grouped by cap color.
#     - The legend in the top-left still uses vertical stacks whose heights encode ranks 1-3.
#     - To make "same center row" exact on a discrete grid, fill lengths are restricted to
#       odd values {1, 3, 5, 7}. Then the whole bar height (fill+2) is also odd, so each
#       bar has a unique middle cell row instead of a center between rows.
#   behavior_trace:
#     - Old placement example: fill length 7 -> rows 1..9, fill length 3 -> rows 5..9.
#       Both end at row 9.
#     - New placement example with CENTER_ROW=5: fill length 7 -> rows 1..9, fill length 3
#       -> rows 3..7. They now share row 5 as the midpoint while their bottoms differ.
#     - Solver behavior is unchanged at the semantic level: the legend still picks the
#       length-ranked bar to recolor. Only the visible alignment cue changes.
#   complexity_calibration:
#     - Keeping 2-3 bars per group and 1-4 groups preserves repeated evidence for ranking.
#     - Four odd length choices are enough to give visibly different sizes without making the
#       layout chaotic or underconstrained.
#   hidden_assumptions_review:
#     - _vertical_sandwiches only assumes two consecutive cap cells in one column with a solid
#       non-background interior. It does not assume a fixed bottom row, so it remains valid.
#     - _legend_entries only assumes the legend is the top-left cluster touching row 0; this
#       also remains valid.
#     - The original generator assumed top_r = constant - fill_len and bottom_r = row 9.
#       That arithmetic must be replaced everywhere with center-based bounds.
#     - Sharing a legend column with a bar is still safe only if there is at least one
#       background row between the legend stack and the bar's top cap. The generator enforces
#       that through the allowed length for the legend column.
#   placement_legality:
#     - Every bar occupies its own column; columns stay at least one background column apart.
#     - CENTER_ROW=5 lets the longest allowed bar (fill 7, total height 9) fit inside rows
#       1..9, leaving row 0 free for legend cells.
#     - The legend column's bar length is chosen from a rank-dependent safe subset so the legend
#       and bar never touch.
#   visual_observability:
#     - Each generated instance has at least one group with multiple distinct lengths, so bottom
#       rows vary and the shared-center alignment is visible by eye.
#     - Color ids are never used for ranking; only visible bar lengths determine the target.
#   cleanliness:
#     - The grid keeps the original sparse left-to-right column layout, so legend and bars stay
#       easy to distinguish even after the alignment change.


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
        rs = [r for r, _ in pix]
        block_count = max(rs) - min(rs) + 1
        entries.append((x_pos, col, int(block_count)))

    entries.sort(key=lambda t: t[0])
    return entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint one centered vertical bar per legend entry.

    Input structure:
    - The background is 0. In the top-left, each legend mark is a vertical stack of one color K
      touching row 0, and the stack height k encodes a rank.
    - Elsewhere, bars are single-column cap/fill/cap objects. Bars in the same color group share
      cap color K and are aligned by a common center row rather than a common bottom edge.

    Rule:
    - For each legend entry (K, k), collect all vertical bars whose two endpoint caps are color K.
    - Sort those bars by interior length from longest to shortest.
    - Recolor the interior of the k-th bar in that ordering to K.

    Key insight:
    - The legend still supplies a rank index, but the visible spatial regularity to notice is
      center alignment: shorter and longer bars spread around the same middle row.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
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
MAX_GROUPS = 4
MAX_TOTAL_BARS = 10
BARS_PER_GROUP = (2, 3)
MAX_FILL_LEN = 7
COLOR_RANGE = (1, 10)
COL_SPACING = 2
MAX_RETRIES = 300

CENTER_ROW = 5
ALLOWED_FILL_LENS = (1, 3, 5, 7)

_FEASIBLE_COUNTS = {
    ng: [c for c in itertools.product(BARS_PER_GROUP, repeat=ng) if sum(c) <= MAX_TOTAL_BARS]
    for ng in range(1, MAX_GROUPS + 1)
}


def _bar_bounds(fill_len: int) -> tuple[int, int]:
    """Return inclusive (top_row, bottom_row) for a centered bar with odd fill length."""
    fill_len = int(fill_len)
    assert fill_len in ALLOWED_FILL_LENS
    half_span = (fill_len + 1) // 2
    top_r = CENTER_ROW - half_span
    bot_r = CENTER_ROW + half_span
    assert 0 <= top_r < bot_r < GRID_H
    return top_r, bot_r


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


def _safe_legend_lengths(rank: int) -> list[int]:
    """Lengths whose centered top cap stays at least one bg row below a rank-high legend."""
    safe = []
    for fill_len in ALLOWED_FILL_LENS:
        top_r, _bot_r = _bar_bounds(fill_len)
        if top_r >= rank + 1:
            safe.append(fill_len)
    assert safe, f"no centered bar length leaves a gap below legend rank {rank}"
    return safe


def _choose_fill_lengths(rng, count: int, rank: int) -> list[int]:
    """Pick distinct centered fill lengths; the legend column length must leave a gap."""
    shared_len = int(rng.choice(_safe_legend_lengths(rank)))
    pool = [l for l in ALLOWED_FILL_LENS if l != shared_len]
    rng.shuffle(pool)
    return [shared_len] + pool[: count - 1]


def _expected_output(
    inp: np.ndarray,
    caps: list[int],
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
) -> np.ndarray:
    """Compute the expected output by recoloring the rank-k centered bar's fill to its cap."""
    out = inp.copy()
    for cap, cols, lengths, rank in zip(caps, group_columns, group_lengths, ranks):
        ranked = sorted(zip(cols, lengths), key=lambda item: -item[1])
        target_col, target_len = ranked[rank - 1]
        top_r, bot_r = _bar_bounds(target_len)
        out[top_r + 1 : bot_r, target_col] = cap
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid centered-alignment grid. Returns None on oracle mismatch."""
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
    group_lengths: list[list[int]] = []
    for cap, fill, count, rank, cols in zip(caps, fills, counts, ranks, group_columns):
        paste(grid, np.full((rank, 1), cap, dtype=int), 0, cols[0])

        lengths = _choose_fill_lengths(rng, count, rank)
        group_lengths.append(lengths)

        for col, fill_len in zip(cols, lengths):
            top_r, _bot_r = _bar_bounds(fill_len)
            paste(grid, _build_bar(cap, fill, fill_len), top_r, col)

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
