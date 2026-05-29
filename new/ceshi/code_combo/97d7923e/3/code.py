# Combo variant of 97d7923e (index 3)
# combo_spec: D2/1+D5/1
# Name: 宽度编码+水平图例

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior + trace
#    - Original task: the legend lived in the top-left as thin vertical stacks. For each
#      cap color K, the stack height k selected the rank-k bar within that color group
#      after sorting bars by fill length descending.
#    - Combo task: the legend is still in the top-left, but each entry now occupies its
#      own row. On row i there is a single horizontal run of color K_i starting at col 0.
#      The run width w_i, not any vertical extent, is the rank index.
#    - Example trace:
#      * Suppose row 0 contains `2 2 2` at cols 0..2, so width = 3 for cap color 2.
#      * Suppose the three color-2 bars have interior lengths 6, 4, 1.
#      * Original logic would have looked for a vertical height and, on this row-style
#        legend, would only see height 1. That would incorrectly pick the longest bar.
#      * Combo logic reads width 3, ranks the color-2 bars as [6, 4, 1], and recolors the
#        third one, i.e. the shortest bar in that group.
#    - The visual difference is therefore not just "horizontal instead of vertical"; it is
#      specifically "one legend row per group, width = rank".
# b) Complexity calibration
#    - Group count remains 1-4 and bars per group remain 2-3, so ranks stay in the small
#      human-readable range 1-3.
#    - The generator forbids the degenerate all-rank-1 case. At least one legend row has
#      width 2 or 3, otherwise the width cue could collapse into an uninformative single
#      pixel and become visually too close to the original task.
#    - Cap colors still identify which bars belong to which legend row, while width alone
#      selects which bar within the group changes. This keeps the puzzle learnable.
# c) Hidden assumptions review
#    - `_legend_entries` in the original solver assumed that legend items touch row 0 and
#      encode rank by vertical height. That assumption is invalid here because every entry
#      has height 1. The parser must instead scan consecutive non-bg rows from row 0 and
#      measure the left-anchored run width in each row.
#    - The main recoloring loop still assumes that each legend color matches one bar group,
#      and that bars are rankable by distinct fill lengths. The generator preserves both.
#    - The new parser assumes that the legend ends when column 0 becomes background. The
#      generator therefore leaves a guaranteed blank row below the last legend row in col 0.
#    - The parser also assumes each legend row begins at col 0 with a contiguous run. The
#      generator builds exactly that and prevents same-color top caps from touching the run
#      on its immediate right edge.
# d) Placement legality
#    - Legend rows occupy rows 0..n_groups-1 and columns 0..rank-1 for their own row.
#    - Bar columns are unique and spaced by at least `COL_SPACING`, so bars do not merge.
#    - If a bar column lies underneath legend pixels, its fill length is capped so the top
#      cap starts at least one blank row below the lowest legend cell in that column.
#    - If a bar of a given group is placed at column `rank`, its fill length is also capped
#      so its top cap cannot land on that group's legend row and accidentally extend the
#      horizontal legend run by adjacency.
# e) Visual observability
#    - Across 2-4 examples, a human can see that the top-left code is a staircase of rows,
#      not a cluster of vertical stacks. The only varying quantity inside each row is width.
#    - Because at least one row has width > 1 in every sample, the width-reading rule is
#      exercised directly rather than being hidden in width-1 edge cases.
#    - The transformation remains local and easy to compare: within each color group,
#      exactly one bar interior changes to its cap color.
# f) Visual cleanliness
#    - The legend is confined to the left margin in consecutive rows, while bars remain as
#      separated vertical columns elsewhere in the grid.
#    - Overlap between legend rows and bars is allowed only when a blank row keeps them
#      visually distinct, so the input stays readable instead of becoming tangled.
# g) Change inferability
#    - No invisible property such as numeric color id participates in ranking.
#    - The only selector is visible run width, repeated across examples with the same bar
#      mechanism, which gives enough structure for humans to infer the mapping.
# h) New solver assumptions checked against the generator
#    - Assumption: legend rows are consecutive from row 0 and stop before any later bar cap
#      appears in col 0. Guaranteed by `_column_max_fill_len`.
#    - Assumption: each row's prefix run is monochrome and starts at col 0. Guaranteed by
#      explicit rendering of the legend before bars are pasted.
#    - Assumption: rank width is within the number of bars for that cap color. Guaranteed by
#      sampling `rank` from `1..bars_in_group`.
#    - Assumption: bars are valid cap/fill/cap sandwiches with solid non-bg fills. This is
#      unchanged from the original construction.

import itertools

import numpy as np

from arc_utils import get_bg, paste


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
    Legend is a stack of consecutive non-background rows starting at row 0.

    Each legend entry is a single horizontal run anchored at col 0. Returns
    (row_index, legend_color, run_width) for each legend row.
    """
    h, w = g.shape
    rows: list[int] = []
    for r in range(h):
        if int(g[r, 0]) == bg:
            break
        rows.append(r)
    assert rows, 'legend column must be non-empty'

    entries: list[tuple[int, int, int]] = []
    for r in rows:
        color = int(g[r, 0])
        c = 0
        while c < w and int(g[r, c]) == color:
            c += 1
        entries.append((r, color, c))

    return entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor vertical bars by legend-driven rank selection.

    Input structure:
    - `bg` is the background. The legend is a top-left stack of rows; each row begins at
      col 0 with a monochrome horizontal run of cap color K whose width w is the encoded
      rank for that color group.
    - The rest of the grid contains bottom-aligned vertical cap/fill/cap bars whose fill
      cells are solid and whose groups are identified by shared cap color.

    Rule:
    - For each legend row `(K, w)`, collect the bars capped by K.
    - Sort those bars by interior fill length from longest to shortest.
    - Recolor the interior of the rank-w bar to K.

    Key insight:
    - The legend rows are not miniature bars. Their horizontal width is the only quantity
      that matters, and it indexes into the length-sorted bar group of the same cap color.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, 'legend must be non-empty'

    for _row, cap, rank in legend:
        segs = _vertical_sandwiches(g, cap)
        assert segs, f'legend cap {cap} must have matching vertical bars'
        segs.sort(key=lambda s: (-(s[1] - s[0] - 1), s[0], s[2]))
        assert 1 <= rank <= len(segs), f'legend rank {rank} out of range for {len(segs)} bars'
        top_r, bot_r, col, _fill = segs[rank - 1]
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


def _assign_columns(rng, counts: list[int], width: int) -> list[list[int]]:
    """Distribute distinct bar columns with spacing >= `COL_SPACING`."""
    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    extra_width = width - min_width
    extra_parts = rng.multinomial(extra_width, np.full(total_bars + 1, 1.0 / (total_bars + 1)))

    left_pad = int(extra_parts[0])
    inter_gap_bonus = [int(v) for v in extra_parts[1:total_bars]]

    occupied = [left_pad]
    for bonus in inter_gap_bonus:
        occupied.append(occupied[-1] + COL_SPACING + bonus)

    owners = [i for i, count in enumerate(counts) for _ in range(count)]
    rng.shuffle(owners)

    group_columns = [[] for _ in counts]
    for col, owner in zip(occupied, owners):
        group_columns[owner].append(int(col))
    for cols in group_columns:
        cols.sort()

    return group_columns


def _column_max_fill_len(ranks: list[int], col: int) -> int:
    """
    Legend occupies (row r, cols 0..ranks[r]-1). A bar in a covered column must leave at
    least one blank row below the lowest legend cell in that column.
    """
    covered_rows = [r for r, width in enumerate(ranks) if col < width]
    if not covered_rows:
        return MAX_FILL_LEN
    lowest_legend_row = max(covered_rows)
    return min(MAX_FILL_LEN, 6 - lowest_legend_row)


def _choose_fill_lengths(
    rng,
    cols: list[int],
    ranks: list[int],
    row_idx: int,
    rank: int,
) -> list[int] | None:
    """
    Pick distinct fill lengths for one group while preserving legend readability.

    If a bar sits immediately to the right of its legend run (col == rank), its top cap is
    forbidden from landing on the legend row, because that would visually extend the run.
    """
    max_lens = [_column_max_fill_len(ranks, c) for c in cols]
    for i, col in enumerate(cols):
        if col == rank:
            max_lens[i] = min(max_lens[i], 7 - row_idx)

    count = len(cols)
    valid: list[tuple[int, ...]] = []
    for perm in itertools.permutations(range(1, MAX_FILL_LEN + 1), count):
        if all(length <= max_len for length, max_len in zip(perm, max_lens)):
            valid.append(tuple(int(v) for v in perm))

    if not valid:
        return None
    choice = valid[int(rng.integers(len(valid)))]
    return list(choice)


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
        top_r = _MAX_CAP_ROW - target_len
        out[top_r + 1 : _BOTTOM_ROW, target_col] = cap
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])

    while True:
        ranks = [int(rng.integers(1, c + 1)) for c in counts]
        if any(rank > 1 for rank in ranks):
            break

    palette = rng.choice(np.arange(*COLOR_RANGE), size=2 * n_groups, replace=False)
    caps = [int(v) for v in palette[:n_groups]]
    fills = [int(v) for v in palette[n_groups:]]

    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    group_columns = _assign_columns(rng, counts, width)

    group_lengths: list[list[int]] = []
    for row_idx, (cols, rank) in enumerate(zip(group_columns, ranks)):
        lengths = _choose_fill_lengths(rng, cols, ranks, row_idx, rank)
        if lengths is None:
            return None
        group_lengths.append(lengths)

    grid = np.zeros((GRID_H, width), dtype=int)
    for row_idx, cap, rank in zip(range(n_groups), caps, ranks):
        paste(grid, np.full((1, rank), cap, dtype=int), row_idx, 0)

    for cap, fill, cols, lengths in zip(caps, fills, group_columns, group_lengths):
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
    raise RuntimeError(f'failed to generate a valid input after {MAX_RETRIES} attempts')
