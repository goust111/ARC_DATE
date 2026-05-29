# Variant of 97d7923e
# Dimension: D4 — 源对象：垂直条的结构 (src)
# Change: 垂直条结构为cap-fill-cap：顶部和底部各一个帽色像素，中间为单一填充色的连续区域，所有条底部对齐在最后一行... → 条无固定底部对齐（浮动条）

import itertools

import numpy as np

from arc_utils import get_bg, neighbors4, paste

# Design analysis for the D4 variant
# a) Concrete behavior:
#    The legend stays in the top-left and still uses cap-color K plus height k
#    to indicate "select the k-th longest bar among bars capped by K". The
#    changed object is the bar placement: bars are still cap / solid fill / cap,
#    but each bar now chooses its own top row and therefore its own bottom row.
#    Example: under the old construction, fill lengths [5, 3, 1] always ended at
#    row 9, so their top caps were forced to rows [3, 5, 7]. Under the variant,
#    the same lengths may sit at rows [(1..7), (4..8), (2..4)] as long as each
#    bar fits, so length must be measured from the distance between its two cap
#    cells rather than inferred from a shared bottom alignment.
# b) Complexity calibration:
#    The visual change is made explicit by requiring multiple bottom rows in the
#    generated sample, and by keeping distinct lengths within each color group so
#    the ranking task remains identifiable rather than ambiguous.
# c) Old implicit assumptions that no longer hold:
#    - The original generator assumed `bottom_row == GRID_H - 1` for every bar.
#      Any formula like `top = fixed_bottom - fill_len` is invalid now.
#    - The solver's `_vertical_sandwiches` scans consecutive cap occurrences in a
#      column. This still works for floating bars, but only if legend pixels do
#      not touch a same-color bar cap in the same column; otherwise the legend
#      block and bar cap would merge into one component or create spurious cap
#      pairs.
#    - The top row must remain reserved for the legend cluster. If a floating bar
#      started on row 0, `_legend_entries` could misread it as part of the legend.
# d) Placement legality:
#    Every bar occupies a unique column. For non-legend columns we keep the top
#    cap at row >= 1; for the legend-sharing column we keep the top cap at least
#    one background row below the legend block. Each chosen top row also respects
#    `top + fill_len + 1 < GRID_H`, so bars neither overlap nor go out of bounds.
# e) Visual observability:
#    The change is visible because bottoms are no longer aligned; multiple bars in
#    a sample terminate at different rows. The recoloring rule is still supported
#    by repeated groups and by the unchanged legend encoding.
# f) Visual cleanliness:
#    Columns remain separated by at least one blank column, bars stay vertical and
#    one-cell wide, and the legend is confined to the leftmost top cluster.
# g) Color inference:
#    Colors are not used for hidden ordering; only geometry is. Cap color K links
#    a legend entry to its candidate bars, and the recolored interior becomes K.
# h) Variant-specific validation:
#    The generator rejects samples where all bars share the same bottom row, and
#    an external check can count distinct bottom rows to confirm the D4 behavior.
# i) New solver assumptions to guarantee:
#    `_vertical_sandwiches` assumes any valid bar interior is a single non-bg
#    color between two consecutive cap pixels in one column. The generator keeps
#    that structure and avoids extra same-color cap pixels in the same column.


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
    Recolor floating vertical bars using legend-specified ranks.

    Input structure:
    - bg = background. Top-left: monochrome legend stacks touching row 0; each has color K and height k.
    - Elsewhere: one-column cap-fill-cap bars whose top and bottom caps match, but the bars can float at different rows.

    Rule:
    - For each legend entry (K, k), collect all bars capped by K.
    - Measure each bar by the number of fill cells between its two caps and sort longest first.
    - Recolor the fill of the rank-k bar to K.

    Key insight: the legend still encodes a rank, but length must be read from the cap-to-cap span because bars no longer share a common bottom row.
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
    """Pick `count` distinct fill lengths for one group."""
    max_legend_len = GRID_H - (rank + 1) - 2
    assert max_legend_len >= 1
    shared_len = int(rng.integers(1, max_legend_len + 1))
    pool = [length for length in range(1, MAX_FILL_LEN + 1) if length != shared_len]
    rng.shuffle(pool)
    return [shared_len] + pool[: count - 1]


def _pick_top_rows(rng, fill_lengths: list[int], rank: int) -> list[int]:
    """Choose floating top rows for one group's bars."""
    tops: list[int] = []
    for idx, fill_len in enumerate(fill_lengths):
        min_top = rank + 1 if idx == 0 else 1
        max_top = GRID_H - (fill_len + 2)
        assert min_top <= max_top
        tops.append(int(rng.integers(min_top, max_top + 1)))
    return tops


def _expected_output(
    inp: np.ndarray,
    caps: list[int],
    placements: list[list[tuple[int, int, int]]],
    ranks: list[int],
) -> np.ndarray:
    """Compute the expected output by recoloring the rank-k bar's fill to cap."""
    out = inp.copy()
    for cap, group_bars, rank in zip(caps, placements, ranks):
        ranked = sorted(group_bars, key=lambda item: (-item[2], item[1], item[0]))
        target_col, target_top, target_len = ranked[rank - 1]
        out[target_top + 1 : target_top + 1 + target_len, target_col] = cap
    return out


def _has_floating_bottoms(placements: list[list[tuple[int, int, int]]]) -> bool:
    """Return True iff the sample visibly departs from full bottom alignment."""
    bottoms = [top + fill_len + 1 for group in placements for _col, top, fill_len in group]
    return len(set(bottoms)) >= 2 and any(bottom < GRID_H - 1 for bottom in bottoms)


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

    grid = np.zeros((GRID_H, width), dtype=int)
    placements: list[list[tuple[int, int, int]]] = []
    for cap, fill, count, rank, cols in zip(caps, fills, counts, ranks, group_columns):
        paste(grid, np.full((rank, 1), cap, dtype=int), 0, cols[0])

        fill_lengths = _choose_fill_lengths(rng, count, rank)
        top_rows = _pick_top_rows(rng, fill_lengths, rank)

        group_bars: list[tuple[int, int, int]] = []
        for col, fill_len, top_r in zip(cols, fill_lengths, top_rows):
            paste(grid, _build_bar(cap, fill, fill_len), top_r, col)
            group_bars.append((col, top_r, fill_len))
        placements.append(group_bars)

    if not _has_floating_bottoms(placements):
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
