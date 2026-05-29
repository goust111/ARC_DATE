# Variant of 97d7923e
# Dimension: D5 — 目标结构：图例的编码方式 (symbol_instructions)
# Change: 图例在左上角：每个条目为一列，从第0行开始向下堆叠k个同色像素，k编码该帽色组内的目标排名；条目间间隔1-2列背景色... → 图例为水平色块行

import itertools

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the D5 variant
# a) Concrete behavior + trace
#    - Original legend: one vertical monochrome stack per cap color; stack height = rank.
#    - New legend: the legend lives only on row 0. Each entry is a horizontal run of one
#      cap color, and the run width = rank. Entries still appear left-to-right near the
#      top-left and are separated by 1-2 background cells.
#    - The bars below stay unchanged: each group is a set of single-column cap/fill/cap
#      vertical bars with equal cap color and distinct fill lengths.
#    - Example trace: suppose a cap color 2 has three bars with interior lengths 6, 4, 1.
#      In the original task the legend for rank 2 was a 2-cell vertical stack in one
#      column. In this variant it becomes a 2-cell horizontal run on row 0, e.g. "22".
#      Solver behavior changes only at legend reading time: width 2 must now be read as
#      rank 2; the later "sort bars by length and recolor the 2nd one" step is unchanged.
# b) Complexity calibration
#    - Ranks remain in a tiny, repeated range (1-3), so the new width code is learnable.
#    - We force at least one legend entry to have width > 1, otherwise some samples would
#      visually collapse back to the original task.
#    - Cap colors and fill colors remain distinct by group, so the ranking target is still
#      unambiguous even though the legend orientation changed.
# c) Old assumptions that break and the fix
#    - Old `_legend_entries` assumed a 4-connected component touching row 0, and measured
#      its height. That silently fails here because every legend entry is only one row tall.
#    - The main loop itself is still valid: it only needs `(cap, rank)` pairs, regardless
#      of how rank is encoded.
#    - `_vertical_sandwiches` still works, but now some bar columns may sit under row-0
#      legend pixels. To keep the legend visually separate, the generator ensures that any
#      bar placed under a legend cell starts at row >= 2, leaving a blank row in between.
# d) Placement legality
#    - Legend runs are placed from column 0 with 1-2 background columns between entries.
#    - Bar columns remain unique and at least one column apart.
#    - If a bar column lies under a row-0 legend pixel, its maximum fill length is reduced
#      from 7 to 6 so the bar's top cap cannot touch the legend.
#    - Generator checks the constructed grid with `main()` before returning it.
# e) Visual observability
#    - The new cue is plainly visible: the top-left code is now a row of colored dashes
#      instead of colored stacks.
#    - Because at least one run has width 2 or 3 in every sample, the direction change can
#      be inferred from 2-4 examples rather than being hidden in all-ones cases.
# f) Visual cleanliness
#    - The legend is confined to a single row, so it does not tangle with the vertical bars.
#    - Horizontal entries are kept contiguous and separated by small gaps, preserving a
#      compact legend cluster instead of scattering marks across the top row.
# g) Color-change inference
#    - The recoloring rule is still keyed by the shared cap color of each bar group.
#      No invisible ordering by color id is introduced; only visible run width encodes rank.
# h) New solver assumptions reviewed against the generator
#    - `_legend_entries` assumes the legend is the only non-background prefix cluster on
#      row 0 and that inter-entry gaps are at most 2. The generator constructs exactly that.
#    - The main loop assumes each legend rank is within the number of bars of that color.
#      The generator samples ranks from `1..bars_in_group`.
#    - `_vertical_sandwiches` assumes each valid bar has one solid non-background fill run
#      between consecutive cap cells. The generator keeps the original bar construction.


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
    Legend is a top-row cluster near the left edge.

    Each entry is a horizontal run of one non-background color on row 0.
    Entry width = target rank, and consecutive entries are separated by 1-2
    background cells. Returns (x_pos, legend_color, run_width).
    """
    _h, w = g.shape
    c = 0
    while c < w and int(g[0, c]) == bg:
        c += 1
    assert c < w, 'legend row must be non-empty'

    entries: list[tuple[int, int, int]] = []
    while c < w:
        if int(g[0, c]) == bg:
            gap_start = c
            while c < w and int(g[0, c]) == bg:
                c += 1
            if c == w or c - gap_start > 2:
                break
            continue

        col = int(g[0, c])
        start = c
        while c < w and int(g[0, c]) == col:
            c += 1
        entries.append((start, col, c - start))

    return entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor vertical bars by legend-driven rank selection.

    Input structure:
    - bg is the background color. The top-left legend sits only on row 0, where each entry
      is a horizontal run of cap color K and the run width k is the desired rank.
    - Elsewhere, the puzzle contains vertical cap/fill/cap bars whose interiors are solid.

    Rule:
    - For each legend entry (K, k), collect the vertical bars capped by K.
    - Sort those bars by interior length from longest to shortest.
    - Recolor the interior of the rank-k bar to K.

    Key insight:
    - The legend no longer encodes rank by height; the only thing that changed is the
      reading direction. The width of each top-row color run is the rank index.
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

_BOTTOM_ROW = GRID_H - 1
_MAX_CAP_ROW = GRID_H - 2
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
    """Place horizontal legend runs from left to right with 1-2 bg columns between them."""
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
    Greedily build a maximal set of additional columns that stay at least one
    column away from every required column and from each other.
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

    The first column of each group shares a column with the legend run start, so the
    relation between legend color and bar group remains explicit. Remaining columns are
    chosen from a maximal 2-separated completion set; cleaner columns outside the legend
    row are preferred when there is slack.
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

    Columns beneath row-0 legend cells are capped at length 6 so their top cap sits on
    row 2 or lower, leaving row 1 blank and visually separating the legend from the bars.
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
        ranked = sorted(zip(cols, lengths), key=lambda item: -item[1])
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
        paste(grid, np.full((1, rank), cap, dtype=int), 0, start)

        lengths = _choose_fill_lengths(rng, cols, legend_cols)
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
    raise RuntimeError(f'failed to generate a valid input after {MAX_RETRIES} attempts')
