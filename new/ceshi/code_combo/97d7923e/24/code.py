# Combo variant of 97d7923e (index 24)
# combo_spec: D2/1+D5/1+D8/2
# Name: 宽度编码+水平图例+底部图例

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior + trace
#    - Original task: the legend lived in the top-left as vertical color stacks; stack
#      height k selected the k-th longest bar in the matching cap-color group, and bars
#      were bottom-aligned.
#    - Fused task: the legend is moved to the bottom row and every legend entry becomes a
#      horizontal run of one cap color. The run width w is the rank index. Meanwhile all
#      bars are flipped to be top-aligned, so they start at row 0 and extend downward.
#    - Example trace: suppose cap color 2 has bars with fill lengths 6, 4, 1. In the
#      original task, a rank-2 cue would be a 2-cell-tall vertical stack near the top.
#      Here it is a 2-cell-wide horizontal run on the last row, such as cells (9,0..1).
#      The solver must read width=2 from the bottom legend, then recolor the second
#      longest 2-capped bar. Reading height would always give 1, and reading the top edge
#      would confuse bar caps with legend pixels, so both the orientation and position
#      changes matter simultaneously.
# b) Complexity calibration
#    - Keep the original grid size family (10 rows, width 10-20), 1-4 groups, and 2-3
#      bars per group. This preserves repeated evidence without exploding the layout.
#    - Rank values remain small (1-3), matching the allowed group sizes, so width-coded
#      legend entries are visually learnable from a few examples.
#    - The generator forbids the degenerate all-rank-1 case. Otherwise the bottom legend
#      would still be horizontal, but width would never influence the output and the D2/D5
#      changes would be under-exercised.
# c) Hidden assumptions review
#    - The original `_legend_entries` assumed legend pixels touched row 0 and rank came
#      from component height. That breaks twice here: legend is on the last row, and rank
#      is encoded by horizontal run length.
#    - The original prose in `main()` said bars were bottom-aligned. That is now false:
#      the generator constructs bars with their top cap on row 0.
#    - `_vertical_sandwiches` itself still survives because it only needs "cap / solid fill
#      / cap" in one column and does not care whether the bar is anchored at the top or
#      bottom of the grid.
#    - The new solver assumes the bottom-row legend forms one left-anchored cluster with
#      1-2 background columns between entries. The generator constructs exactly that.
# d) Placement legality
#    - Bottom legend entries are laid out left-to-right on row 9 with widths equal to the
#      chosen ranks and 1-2 background cells between neighboring entries.
#    - Bars occupy distinct columns with stride at least 2, so single-column objects stay
#      visually separate.
#    - Some bar columns may share a column with a bottom legend pixel. In those columns the
#      fill length is capped at 6, so the bar's bottom cap lands on row 7 and row 8 stays
#      blank, leaving a visible gap above the legend at row 9.
#    - Bars in columns outside the legend can still use fill length 7 and bottom out at
#      row 8, which keeps the "top-aligned bars vs bottom legend" inversion obvious.
# e) Visual observability
#    - The fused rule is readable from 2-4 examples because the legend sits exclusively on
#      the bottom row and its informative quantity is horizontal extent rather than height.
#    - The selected output change is still structured: for each legend color, exactly one
#      same-cap bar has its interior recolored from the group's fill color to the cap
#      color after sorting bars by visible fill length.
# f) Visual cleanliness
#    - Legend and bars occupy opposite vertical extremes of the grid, reducing clutter.
#    - Small gaps between legend entries keep the bottom row parseable as separate tokens,
#      while bar columns remain sparse enough that different groups do not visually merge.
# g) Change inferability
#    - No invisible property such as numeric color id or generation order participates in
#      ranking. The only signals are visible: legend color identifies the group, legend run
#      width identifies the rank, and bar fill length determines ordering.
#    - To verify the fusion is not degenerate, evaluation should check that generated
#      samples frequently contain at least one rank > 1; in those cases a mistaken
#      "bottom legend height = 1" solver would pick a different bar than this combo solver.
# h) New solver assumptions guaranteed by the generator
#    - The bottom row contains the full legend cluster and no bar endpoint.
#    - Each legend entry is one contiguous same-color run on the bottom row.
#    - For every legend color, there are at least `rank` bars with that cap color, and
#      their fill lengths are distinct within the group.

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
    Legend is a bottom-row cluster anchored at the left edge.

    Each entry is a horizontal run of one non-background color on the last row.
    Entry width = target rank, and neighboring entries are separated by 1-2
    background cells. Returns (x_pos, legend_color, run_width).
    """
    h, w = g.shape
    r = h - 1
    assert int(g[r, 0]) != bg, 'bottom legend must start at the left edge'

    entries: list[tuple[int, int, int]] = []
    c = 0
    while c < w:
        col = int(g[r, c])
        if col == bg:
            gap_start = c
            while c < w and int(g[r, c]) == bg:
                c += 1
            if c == w or c - gap_start > 2:
                break
            continue

        start = c
        while c < w and int(g[r, c]) == col:
            c += 1
        entries.append((start, col, c - start))

    return entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor top-aligned vertical bars using rank values stored in a bottom legend row.

    Input structure:
    - `bg` is the background color. The legend sits on the last row as a left-anchored
      sequence of horizontal monochrome runs; each run has cap color `K` and width `w`.
    - The main objects are vertical cap-fill-cap bars whose top cap is on row 0 and whose
      interior is a solid non-background fill color shared within each cap-color group.

    Rule:
    - For each bottom-row legend entry `(K, w)`, collect all vertical bars capped by `K`.
    - Rank those bars by interior fill length from longest to shortest.
    - Recolor the interior of the rank-`w` bar to `K`.

    Key insight:
    - The legend token is neither a miniature bar nor a height code anymore. Its horizontal
      width, read on the bottom row, is the rank index into the same-color bar group.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, 'legend must be non-empty'

    for _x, cap, rank in legend:
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
MAX_RETRIES = 300

_TOP_CAP_ROW = 0
_BOTTOM_LEGEND_ROW = GRID_H - 1
_MAX_FILL_LEN_SHARED = MAX_FILL_LEN - 1
COL_SPACING = 2

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
    """Place bottom-row horizontal legend runs from left to right with 1-2 bg gaps."""
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
        c += COL_SPACING
    return extras


def _assign_columns(
    rng,
    counts: list[int],
    starts: list[int],
    legend_cols: set[int],
) -> list[list[int]] | None:
    """
    Build one bar-column list per group.

    Each group reuses its legend run start as one bar column so the color-to-group link is
    explicit. Remaining columns come from a 2-separated completion set; columns outside
    the legend are preferred, but legend-covered columns are allowed when the layout is
    tight, as long as fill lengths later enforce a blank gap above the bottom legend.
    """
    total_bars = sum(counts)
    extras_needed = total_bars - len(starts)
    extra_pool = _maximal_extra_columns(starts, WIDTH_MAX)
    if len(extra_pool) < extras_needed:
        return None

    clean = [c for c in extra_pool if c not in legend_cols]
    shared = [c for c in extra_pool if c in legend_cols]
    rng.shuffle(clean)
    rng.shuffle(shared)
    chosen_extras = (clean + shared)[:extras_needed]
    rng.shuffle(chosen_extras)

    group_columns = [[start] for start in starts]
    owners = [i for i, count in enumerate(counts) for _ in range(count - 1)]
    rng.shuffle(owners)
    for col, owner in zip(chosen_extras, owners):
        group_columns[owner].append(int(col))

    for idx, start in enumerate(starts):
        rest = sorted(c for c in group_columns[idx] if c != start)
        group_columns[idx] = [start] + rest
    return group_columns


def _choose_fill_lengths(rng, cols: list[int], legend_cols: set[int]) -> list[int]:
    """
    Pick distinct fill lengths for one group.

    Bar columns that pass through bottom-row legend cells are limited to length 6 so their
    bottom cap lands on row 7, leaving row 8 blank and row 9 for the legend.
    """
    limits = [(_MAX_FILL_LEN_SHARED if c in legend_cols else MAX_FILL_LEN) for c in cols]
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

    for cap, rank, start in zip(caps, ranks, starts):
        paste(grid, np.full((1, rank), cap, dtype=int), _BOTTOM_LEGEND_ROW, start)

    for cap, fill, cols in zip(caps, fills, group_columns):
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
