# Variant of 97d7923e
# Dimension: D8 — 图例与条的空间关系 (meta)
# Change: 图例占据左上角顶部若干行，条从更低的行开始向下延伸到最后一行；图例列和条列可共享同一列（图例在上方且有背景色行间隔）... → 图例与条完全空间分离

import itertools

import numpy as np

from arc_utils import get_bg, neighbors4, paste


# ---------------------------------------------------------------------------
# Design analysis for the D8 variant
#
# a) Concretization and behavior trace
# - Original layout: each cap-color group had one "shared" column. Its upper cells formed
#   the legend stack, and the same column continued below (after a bg gap row) as one of the
#   group's vertical bars.
# - Variant layout: the legend stays in the top-left, but it now occupies a dedicated left
#   column band. All bars live strictly to the right of that band, with 2-4 all-bg separator
#   columns in between. No legend column may also contain a bar.
# - Example trace:
#   original: cap 2 with rank 2 might use column 0 for both the legend block and one bar.
#   variant: the rank-2 legend block of color 2 still appears at column 0, but every cap-2
#   bar is moved to some column >= bar_start on the right. Ranking by bar length is unchanged;
#   only the spatial lookup changes from "scan whole grid" to "read legend on the left, then
#   scan candidate bars only in the right region".
#
# b) Complexity calibration
# - The rule itself is unchanged, so visual novelty must come from clearly separated regions.
# - The separator is forced to be multiple blank columns so the split is obvious in 2-3 examples.
# - Legend entries remain 1-4 and ranks remain 1-3, keeping the same small discrete reasoning
#   load as the source task.
#
# c) Audit of old implicit assumptions
# - Old `_legend_entries` assumed the leftmost top-row cluster was enough to describe the legend.
#   That still holds, but now it must also recover where the bar region begins after a full blank
#   separator.
# - Old `_vertical_sandwiches` scanned every column. That was harmless when legend/bar sharing was
#   allowed; in this variant it should honor the new split and scan only the right-side bar area.
# - Old generator relied on the first column of each group doubling as both legend and bar. That
#   assumption is invalid now and must be removed entirely.
#
# d) Placement legality
# - Legend columns are generated first, compactly near the left edge, with only 0-1 blank columns
#   inside the legend band.
# - A 2-4 column all-bg separator follows.
# - Bars are then assigned distinct right-side columns with at least one blank column between
#   neighboring bars, so nothing overlaps and every placed bar still fits in 10 rows.
#
# e) Visual observability
# - The variant is visible without reading color ids: the left legend band and right bar field are
#   spatially disjoint in every sample.
# - The output change still occurs on exactly one bar per legend entry, preserving the original
#   "height means rank" clue.
#
# f) Visual cleanliness
# - No region mixes legend blocks and bars anymore, so there is less stacking ambiguity than in the
#   source task.
# - Blank separator columns make the parse "legend area vs bar area" immediate.
#
# g) Color-change inferability
# - Colors keep their original structural roles: each legend color matches the cap color of one bar
#   group, and recoloring affects the selected bar interior only.
# - No invisible property such as color id ordering is used.
#
# Solver assumptions introduced by the variant and how the generator guarantees them
# - Row 0 contains legend pixels but no bar pixels. Guaranteed because max bar fill length is 7, so
#   every bar top cap starts at row >= 1.
# - The first non-bg pixels to the right of the legend band belong to bars, and all intervening
#   columns are all-bg. Guaranteed by an explicit separator width in generation.
# - Every legend color has 2-3 matching bars in the right region. Guaranteed by constructive bar
#   placement and oracle verification with `main(generate_input())`.
# ---------------------------------------------------------------------------


def _vertical_sandwiches(
    g: np.ndarray, cap: int, start_col: int
) -> list[tuple[int, int, int, int]]:
    """
    Find vertical bar candidates (top_r, bottom_r, col, fill_color) in the bar region where:
      - g[top_r, col] == cap and g[bottom_r, col] == cap
      - (top_r, bottom_r) are consecutive occurrences of cap in that column
      - interior is a solid run of ONE color fill_color != cap and fill_color != background
      - length >= 1 (bottom_r >= top_r + 2)
    """
    h, w = g.shape
    bg = get_bg(g)
    segs: list[tuple[int, int, int, int]] = []
    for c in range(start_col, w):
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


def _legend_entries(g: np.ndarray, bg: int) -> tuple[list[tuple[int, int, int]], int]:
    """
    Legend occupies the leftmost top-row cluster. Bars start only after a fully blank separator.

    Returns:
      - entries: (x_pos, legend_color, block_count), sorted by x_pos
      - bar_start: first column index that contains the bar field
    """
    h, w = g.shape
    top_pos = [int(c) for c in np.flatnonzero(g[0, :] != bg)]
    assert top_pos, "legend row must be non-empty"
    top_pos.sort()
    assert top_pos[0] == 0, "legend must begin at the left edge"

    cols = [top_pos[0]]
    for c in top_pos[1:]:
        if c - cols[-1] <= 2:
            cols.append(c)
        else:
            break
    legend_max_c = max(cols)

    right_activity = np.flatnonzero(np.any(g[:, legend_max_c + 1 :] != bg, axis=0))
    assert right_activity.size > 0, "bar region must exist to the right of the legend"
    bar_start = legend_max_c + 1 + int(right_activity[0])
    assert bar_start > legend_max_c + 1, "legend and bars must be separated by blank columns"
    assert np.all(g[:, legend_max_c + 1 : bar_start] == bg), "separator columns must be blank"

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
    return entries, bar_start


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor right-side vertical bars using rank markers stored in a separate left legend.

    Input structure:
    - bg = background. The top-left contains monochrome legend stacks touching row 0.
    - The legend uses its own left column band; after several fully blank separator columns,
      the bar field begins on the right with vertical cap-fill-cap bars.

    Rule:
    - For each legend entry (K, k), look only in the right bar field for bars capped by K.
    - Sort those bars by interior length, from longest to shortest.
    - Recolor the interior of the rank-k bar to K.

    Key insight: The legend height still encodes a rank, but the legend and the ranked bars are
    now separated into different column regions, so the solver must first split the layout.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend, bar_start = _legend_entries(g, bg)
    assert legend, "legend must be non-empty"

    for _x, cap, k in legend:
        segs = _vertical_sandwiches(g, cap, bar_start)
        assert segs, f"legend cap {cap} must have matching vertical bars"
        segs.sort(key=lambda s: (-(s[1] - s[0] - 1), s[0], s[2]))
        assert 1 <= k <= len(segs), f"legend rank {k} out of range for {len(segs)} bars"
        top_r, bot_r, col, _fill = segs[k - 1]
        out[top_r + 1 : bot_r, col] = cap

    return out


# --------------- generator (O2) ---------------

# --- Configuration ---
GRID_H = 10
WIDTH_MIN = 16
WIDTH_MAX = 30
MAX_GROUPS = 4
MAX_TOTAL_BARS = 10
BARS_PER_GROUP = (2, 3)
MAX_FILL_LEN = 7
COLOR_RANGE = (1, 10)
BAR_COL_STEP = 2
REGION_GAP_MIN = 2
REGION_GAP_MAX = 4
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


def _legend_columns(rng: np.random.Generator, n_groups: int) -> list[int]:
    """Place legend entries compactly in the left band with 0-1 blank columns between them."""
    cols = [0]
    for _ in range(1, n_groups):
        cols.append(cols[-1] + int(rng.integers(1, 3)))
    return cols


def _assign_bar_columns(
    rng: np.random.Generator, counts: list[int], start_col: int, width: int
) -> list[list[int]]:
    """Assign all bars to right-side columns with at least one blank column between neighbors."""
    n_groups = len(counts)
    total_bars = sum(counts)
    min_bar_span = 1 + BAR_COL_STEP * (total_bars - 1)
    extra_width = width - start_col - min_bar_span
    assert extra_width >= 0

    extra_parts = rng.multinomial(extra_width, np.full(total_bars, 1.0 / total_bars))
    bar_cols = [start_col]
    for idx in range(total_bars - 1):
        bar_cols.append(bar_cols[-1] + BAR_COL_STEP + int(extra_parts[idx]))

    owners = [gid for gid, count in enumerate(counts) for _ in range(count)]
    rng.shuffle(owners)
    group_columns = [[] for _ in range(n_groups)]
    for col, owner in zip(bar_cols, owners):
        group_columns[owner].append(int(col))

    for cols in group_columns:
        cols.sort()
    return group_columns


def _choose_fill_lengths(rng: np.random.Generator, count: int) -> list[int]:
    """Pick distinct fill lengths for one cap-color group."""
    vals = rng.choice(np.arange(1, MAX_FILL_LEN + 1), size=count, replace=False)
    return [int(v) for v in vals]


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
        ranked = sorted(zip(cols, lengths), key=lambda item: (-item[1], _MAX_CAP_ROW - item[1], item[0]))
        target_col, target_len = ranked[rank - 1]
        top_r = _MAX_CAP_ROW - target_len
        out[top_r + 1 : _BOTTOM_ROW, target_col] = cap
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid separated-layout grid. Returns None on oracle mismatch."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, c + 1)) for c in counts]

    palette = rng.choice(np.arange(*COLOR_RANGE), size=2 * n_groups, replace=False)
    caps = [int(v) for v in palette[:n_groups]]
    fills = [int(v) for v in palette[n_groups:]]

    legend_cols = _legend_columns(rng, n_groups)
    legend_width = legend_cols[-1] + 1
    region_gap = int(rng.integers(REGION_GAP_MIN, REGION_GAP_MAX + 1))

    total_bars = sum(counts)
    min_bar_span = 1 + BAR_COL_STEP * (total_bars - 1)
    min_width = max(WIDTH_MIN, legend_width + region_gap + min_bar_span)
    if min_width > WIDTH_MAX:
        return None
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    bar_start = legend_width + region_gap
    group_columns = _assign_bar_columns(rng, counts, bar_start, width)

    grid = np.zeros((GRID_H, width), dtype=int)
    group_lengths: list[list[int]] = []
    for cap, fill, rank, legend_col, cols, count in zip(caps, fills, ranks, legend_cols, group_columns, counts):
        paste(grid, np.full((rank, 1), cap, dtype=int), 0, legend_col)

        lengths = _choose_fill_lengths(rng, count)
        group_lengths.append(lengths)
        for col, fill_len in zip(cols, lengths):
            paste(grid, _build_bar(cap, fill, fill_len), _MAX_CAP_ROW - fill_len, col)

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
