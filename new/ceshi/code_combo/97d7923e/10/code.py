# Combo variant of 97d7923e (index 10)
# combo_spec: D5/1+D8/2
# Name: 水平图例+底部图例

import itertools

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# a) Concrete behavior + trace
#    - Original task: the legend is a set of vertical stacks in the upper-left, and every bar is
#      bottom-aligned. Stack height k means "pick the k-th longest bar among bars with this cap".
#    - Fused task: the legend moves to the bottom-left and each entry becomes a horizontal run on
#      the last row. The run width k now encodes the rank. Meanwhile the bars flip to top-aligned,
#      so every bar starts at row 0 and extends downward.
#    - Example trace for one cap color K with three bars of fill lengths 7, 4, 2:
#      original input cue = a vertical K-stack of height 2 near the top; fused input cue = a
#      horizontal run "KK" on the bottom row. Original solver reads height from above; fused solver
#      must first find the bottom legend, then read the run width left-to-right as rank 2. In both
#      cases the selected target is still the second-longest K-capped bar, whose fill is recolored.
# b) Complexity calibration
#    - Rank values stay in the small visible range 1-3 because each group still has only 2-3 bars.
#    - At least one sample entry is forced to have width > 1, otherwise the horizontal encoding
#      could visually collapse into isolated single cells and the D5 component would be underused.
#    - Cap colors remain distinct across groups, and each group keeps a single fill color with
#      distinct bar lengths, so the ranking target stays learnable and unambiguous.
# c) Implicit assumptions broken by the fusion
#    - Old legend parsing assumed row 0 and measured component height. That fails twice here:
#      the legend lives on the last row, and its code is width rather than height.
#    - Old generator assumed shared legend/bar columns were safe when bars ended at row 9. After
#      the D8 flip, any bar under a bottom-legend pixel must stop above row 8 to preserve a blank
#      separator row; otherwise the legend and bar would visually fuse.
#    - `_vertical_sandwiches` itself survives unchanged because it only requires consecutive cap
#      cells with a solid non-background interior; it does not care whether bars are top- or
#      bottom-anchored.
# d) Placement legality
#    - Legend runs are packed left-to-right on the bottom row with 1-2 background columns between
#      entries, mirroring the original compact legend cluster.
#    - The first column of each group is reused as both a legend anchor and a bar column so the
#      color grouping remains explicit. Any additional bar columns are kept at least one blank
#      column away from every other bar column.
#    - If a bar column lies under any bottom-row legend pixel, its fill length is capped at 6, so
#      the bar's lower cap ends on row 7 and row 8 stays blank.
# e) Visual observability
#    - The bottom row now clearly contains the code, and the code is horizontal rather than
#      vertical. A human comparing 2-4 examples can see both changes without relying on metadata.
#    - The output still changes one interior per legend group, giving repeated evidence that the
#      bottom run width is not copied as geometry but used as a rank into same-cap bars.
# f) Visual cleanliness
#    - Bars occupy sparse columns and the legend is confined to the last row, so there is no
#      overdraw inside the legend region.
#    - The forced blank row above shared legend columns prevents same-color touching between a bar
#      and the bottom legend, which keeps the two object types visually separable.
# g) Color-change inference
#    - No hidden ordering by numeric color id is introduced. The only selector is visible length:
#      the legend run width chooses among bars that visibly share the same cap color.
# h) Solver-side assumptions that the generator must guarantee
#    - The first non-background cluster on the last row is the whole legend region, with only
#      1-2-cell gaps between entries and no other bottom-row objects.
#    - Every legend color appears as a single horizontal run on the bottom row, and for that color
#      there are at least k top-aligned bars with distinct interior lengths.
#    - Any shared legend/bar column preserves at least one background row between the bar and the
#      legend, so `_legend_entries` never confuses bar pixels with legend pixels.


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
    Legend is a bottom-row cluster near the left edge.

    Each entry is a horizontal run of one non-background color on the last row.
    Entry width = target rank, and consecutive entries are separated by 1-2
    background cells. Returns (x_pos, legend_color, run_width).
    """
    h, w = g.shape
    bottom_r = h - 1
    c = 0
    while c < w and int(g[bottom_r, c]) == bg:
        c += 1
    assert c < w, 'legend row must be non-empty'

    entries: list[tuple[int, int, int]] = []
    while c < w:
        if int(g[bottom_r, c]) == bg:
            gap_start = c
            while c < w and int(g[bottom_r, c]) == bg:
                c += 1
            if c == w or c - gap_start > 2:
                break
            continue

        col = int(g[bottom_r, c])
        start = c
        while c < w and int(g[bottom_r, c]) == col:
            c += 1
        entries.append((start, col, c - start))

    return entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor vertical bars using rank values stored in a bottom horizontal legend.

    Input structure:
    - bg is the background color. Along the lower-left edge, the legend sits on the last row and
      each entry is a horizontal run of cap color K whose width k is the desired rank.
    - The main objects are vertical cap-fill-cap bars with solid interiors, and these bars are
      top-aligned so their upper cap lies on row 0.

    Rule:
    - For each legend entry (K, k), gather all vertical bars capped by K.
    - Sort those bars by interior length from longest to shortest.
    - Recolor the interior of the rank-k bar so that its fill becomes K.

    Key insight:
    - The fused variant changes both where the legend lives and how rank is encoded. The solver
      must read the bottom row, interpret run width as rank, and then apply the same length-based
      selection inside the matching cap-color group.
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
GRID_H = 10
WIDTH_MIN = 10
WIDTH_MAX = 20
MAX_GROUPS = 4
MAX_TOTAL_BARS = 10
BARS_PER_GROUP = (2, 3)
MAX_FILL_LEN = 7
COLOR_RANGE = (1, 10)
MAX_RETRIES = 300

_TOP_CAP_ROW = 0
_BOTTOM_ROW = GRID_H - 1
_MAX_FILL_LEN_UNDER_LEGEND = MAX_FILL_LEN - 1

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


def _legend_layout(rng, ranks: list[int]) -> tuple[list[int], set[int], int]:
    """Place bottom-row horizontal legend runs left-to-right with 1-2 bg columns between them."""
    starts = [0]
    for rank in ranks[:-1]:
        gap = 1 + int(rng.integers(0, 2))
        starts.append(starts[-1] + rank + gap)

    legend_cols: set[int] = set()
    for start, rank in zip(starts, ranks):
        legend_cols.update(range(start, start + rank))

    cluster_width = starts[-1] + ranks[-1]
    assert cluster_width <= WIDTH_MAX, 'legend cluster must fit'
    return starts, legend_cols, cluster_width


def _maximal_extra_columns(required: list[int], width: int) -> list[int]:
    """
    Greedily build a maximal set of extra bar columns that stay at least one
    blank column away from every required bar column and from each other.
    """
    required = sorted(int(c) for c in required)
    extras: list[int] = []
    c = 0
    while c < width:
        if any(abs(c - req) <= 1 for req in required):
            c += 1
            continue
        if extras and c - extras[-1] <= 1:
            c += 1
            continue
        extras.append(c)
        c += 2
    return extras


def _assign_columns(
    rng, counts: list[int], starts: list[int], legend_cols: set[int]
) -> list[list[int]] | None:
    """
    Build one bar-column list per group.

    Each group reuses the start column of its legend run as a shared anchor column.
    Remaining bar columns come from a 2-separated completion set; columns outside
    the legend footprint are preferred when there is enough width.
    """
    total_bars = sum(counts)
    extras_needed = total_bars - len(starts)
    extra_pool = _maximal_extra_columns(starts, WIDTH_MAX)
    if len(extra_pool) < extras_needed:
        return None

    clean = [c for c in extra_pool if c not in legend_cols]
    busy = [c for c in extra_pool if c in legend_cols]
    rng.shuffle(clean)
    rng.shuffle(busy)
    chosen_extras = (clean + busy)[:extras_needed]
    rng.shuffle(chosen_extras)

    group_columns = [[start] for start in starts]
    owners = [i for i, count in enumerate(counts) for _ in range(count - 1)]
    rng.shuffle(owners)
    for col, owner in zip(chosen_extras, owners):
        group_columns[owner].append(col)

    for idx, start in enumerate(starts):
        rest = sorted(c for c in group_columns[idx] if c != start)
        group_columns[idx] = [start] + rest
    return group_columns


def _choose_fill_lengths(rng, cols: list[int], legend_cols: set[int]) -> list[int]:
    """
    Pick distinct fill lengths for one group.

    Any bar under a bottom legend pixel is limited to length 6 so the bar ends on
    row 7 and row 8 remains blank above the legend row.
    """
    limits = [(_MAX_FILL_LEN_UNDER_LEGEND if c in legend_cols else MAX_FILL_LEN) for c in cols]
    order = sorted(range(len(cols)), key=lambda i: (limits[i], cols[i]))
    available = list(range(1, MAX_FILL_LEN + 1))
    lengths = [0] * len(cols)

    for idx in order:
        allowed = [length for length in available if length <= limits[idx]]
        if not allowed:
            raise ValueError('no valid distinct fill lengths for group')
        pick = int(allowed[int(rng.integers(len(allowed)))])
        lengths[idx] = pick
        available.remove(pick)

    return lengths


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
        bump = int(rng.integers(n_groups))
        ranks[bump] = int(rng.integers(2, counts[bump] + 1))

    palette = rng.choice(np.arange(*COLOR_RANGE), size=2 * n_groups, replace=False)
    caps = [int(v) for v in palette[:n_groups]]
    fills = [int(v) for v in palette[n_groups:]]

    starts, legend_cols, cluster_width = _legend_layout(rng, ranks)
    group_columns = _assign_columns(rng, counts, starts, legend_cols)
    if group_columns is None:
        return None

    max_used_col = max(max(cols) for cols in group_columns)
    min_width = max(WIDTH_MIN, cluster_width, max_used_col + 1)
    if min_width > WIDTH_MAX:
        return None
    width = int(rng.integers(min_width, WIDTH_MAX + 1))

    grid = np.zeros((GRID_H, width), dtype=int)
    group_lengths: list[list[int]] = []
    for cap, fill, rank, start, cols in zip(caps, fills, ranks, starts, group_columns):
        paste(grid, np.full((1, rank), cap, dtype=int), _BOTTOM_ROW, start)

        lengths = _choose_fill_lengths(rng, cols, legend_cols)
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
