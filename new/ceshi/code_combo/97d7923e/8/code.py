# Combo variant of 97d7923e (index 8)
# combo_spec: D4/0+D5/1
# Name: 水平条+水平图例

from pathlib import Path
import sys
import itertools

import numpy as np

for parent in Path(__file__).resolve().parents:
    solver_dir = parent / 'solvers'
    if solver_dir.exists():
        if str(solver_dir) not in sys.path:
            sys.path.append(str(solver_dir))
        break

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# a) Concrete behavior + trace
#    - Original task: the legend sits in the top-left as vertical stacks, each bar is a
#      bottom-aligned vertical cap/fill/cap column, and the stack height chooses the kth
#      longest bar inside the matching cap-color group.
#    - Combo task: the long objects rotate into right-aligned horizontal cap/fill/cap rows,
#      while the legend also changes into top-row horizontal runs whose widths encode rank.
#    - Example: suppose cap color 2 has three bars with fill lengths [6, 4, 1] on three
#      separate rows, all ending at the last column. If row 0 contains a run "22", that
#      width 2 means "pick the second-longest 2-capped row", so only the length-4 row has
#      its interior recolored to 2.
#    - The key difference from the source task is twofold at once: bars must be discovered
#      by scanning rows instead of columns, and legend rank must be read from horizontal
#      width instead of vertical height.
# b) Complexity calibration
#    - Ranks stay in the tiny range 1-3, repeated across samples, so width-as-rank remains
#      learnable rather than arbitrary.
#    - Fill lengths stay distinct inside each cap-color group, removing ranking ties.
#    - At least one legend run has width > 1 in every generated sample, otherwise the
#      horizontal legend could visually collapse into a trivial sequence of single pixels.
#    - Every bar ends on the right border, making the horizontal rotation obvious by sight.
# c) Implicit assumption audit
#    - The old legend parser assumed top-row components could be measured by height. That is
#      false now because each legend entry is only one row tall; the parser must read run
#      widths along row 0.
#    - The old bar detector assumed each candidate occupied one column. That breaks under
#      the combo and must become a row-wise cap/fill/cap detector.
#    - The old recolor slice was vertical. In the combo, recoloring must target a horizontal
#      interior slice or it will overwrite unrelated cells.
#    - The old generator could let legend and bars share a column because they were separated
#      vertically. Here the legend owns row 0, so the generator instead keeps all bars on
#      lower rows and relies only on color matching to connect legend entries to groups.
# d) Placement legality
#    - Grid width stays within 10-20 columns so both the top-row legend cluster and a
#      longest 9-cell bar (cap + 7 fill + cap) fit simultaneously.
#    - Grid height varies between 10 and 20 rows. Bars occupy distinct odd-numbered rows
#      starting at row 1, so even the maximum 10 bars fit without overlap.
#    - Legend runs are placed left-to-right from column 0 with 1-2 background cells between
#      entries, forming one compact top-row cluster.
#    - Bars on different rows never overlap, and each bar is fully determined by its row,
#      its fill length, and the shared right edge.
# e) Visual observability
#    - A solver can see two stable cues across examples: the legend is a row of colored
#      dashes at the top, and the bars are horizontal segments that all terminate on the
#      right boundary.
#    - No invisible property such as raw color id or generation order participates in the
#      ranking; only visible fill length and visible legend width matter.
# f) Visual cleanliness
#    - The legend is confined to row 0 and the bars live below it, so the two structures do
#      not tangle inside the same cells.
#    - Using every other row for bars keeps components visually separable even when there
#      are many groups and many bars.
# g) New solver assumptions reviewed against the generator
#    - `_legend_entries` assumes the legend is the only compact non-background cluster on
#      row 0 with inter-entry gaps at most 2. The generator constructs exactly that.
#    - `_horizontal_sandwiches` assumes each valid bar row contains consecutive cap pixels
#      with one solid non-background fill run between them. The generator draws every bar in
#      that exact cap/fill/cap form.
#    - `main()` assumes each legend rank k lies in `1..len(group)`. The generator samples
#      ranks directly from the corresponding bar counts.


def _horizontal_sandwiches(g: np.ndarray, cap: int) -> list[tuple[int, int, int, int]]:
    """
    Find horizontal bar candidates (row, left_c, right_c, fill_color) where:
      - g[row, left_c] == cap and g[row, right_c] == cap
      - (left_c, right_c) are consecutive occurrences of cap in that row
      - interior is a solid run of ONE color fill_color != cap and fill_color != background
      - length >= 1 (right_c >= left_c + 2)
    """
    h, _w = g.shape
    bg = get_bg(g)
    segs: list[tuple[int, int, int, int]] = []
    for r in range(h):
        cols = [int(c) for c in np.flatnonzero(g[r, :] == cap)]
        if len(cols) < 2:
            continue
        for a, b in zip(cols, cols[1:]):
            if b <= a + 1:
                continue
            fill = int(g[r, a + 1])
            if fill == cap or fill == bg:
                continue
            if np.all(g[r, a + 1 : b] == fill):
                segs.append((r, a, b, fill))
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
    Recolor right-aligned horizontal bars by legend-driven rank selection.

    Input structure:
    - `bg` is the background color. The legend lives on row 0 near the left edge, where
      each entry is a horizontal run of cap color `K` and the run width `k` is the rank.
    - Below the legend are horizontal cap/fill/cap bars. Every bar uses one row, has a
      solid interior color, and its right cap sits on the last column.

    Rule:
    - For each legend entry `(K, k)`, collect the horizontal bars capped by `K`.
    - Sort those bars by interior length from longest to shortest.
    - Recolor the interior of the rank-`k` bar to `K`.

    Key insight:
    - Two reading directions changed together: ranks are encoded by top-row run width, and
      candidate objects are horizontal rows rather than vertical columns.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, 'legend must be non-empty'

    for _x, cap, k in legend:
        segs = _horizontal_sandwiches(g, cap)
        assert segs, f'legend cap {cap} must have matching horizontal bars'
        segs.sort(key=lambda s: (-(s[2] - s[1] - 1), s[0], s[1]))
        assert 1 <= k <= len(segs), f'legend rank {k} out of range for {len(segs)} bars'
        row, left_c, right_c, _fill = segs[k - 1]
        out[row, left_c + 1 : right_c] = cap

    return out


# --------------- generator (O2) ---------------

# --- Configuration ---
HEIGHT_MIN = 10
HEIGHT_MAX = 20
WIDTH_MIN = 10
WIDTH_MAX = 20
MAX_GROUPS = 4
MAX_TOTAL_BARS = 10
BARS_PER_GROUP = (2, 3)
MAX_FILL_LEN = 7
COLOR_RANGE = (1, 10)
ROW_SPACING = 2
FIRST_BAR_ROW = 1
MAX_RETRIES = 300

FEASIBLE_COUNTS = {
    ng: [counts for counts in itertools.product(BARS_PER_GROUP, repeat=ng) if sum(counts) <= MAX_TOTAL_BARS]
    for ng in range(1, MAX_GROUPS + 1)
}


def _build_bar(cap: int, fill: int, fill_len: int) -> np.ndarray:
    """Return a (1, fill_len + 2) row sprite: cap / fill*N / cap."""
    bar = np.full((1, fill_len + 2), fill, dtype=int)
    bar[0, 0] = cap
    bar[0, -1] = cap
    return bar


def _legend_layout(rng, ranks: list[int]) -> tuple[list[int], int]:
    """Place top-row legend runs left-to-right with 1-2 background columns between them."""
    starts = [0]
    for rank in ranks[:-1]:
        gap = 1 + int(rng.integers(0, 2))
        starts.append(starts[-1] + rank + gap)
    cluster_width = starts[-1] + ranks[-1]
    return starts, cluster_width


def _min_height(total_bars: int) -> int:
    """Minimum height needed to place `total_bars` on distinct rows with stride >= 2."""
    return max(HEIGHT_MIN, FIRST_BAR_ROW + 1 + ROW_SPACING * (total_bars - 1))


def _assign_rows(rng, n_groups: int, counts: list[int], height: int) -> list[list[int]]:
    """
    Distribute bar rows across groups with random extra spacing.

    The first row in each group is one of the first `n_groups` occupied rows so the
    legend order and the earliest bar of each color remain loosely aligned.
    """
    total_bars = sum(counts)
    base_height = _min_height(total_bars)
    extra_height = height - base_height
    gap_slots = total_bars - 1
    if gap_slots > 0:
        extra_parts = rng.multinomial(extra_height, np.full(gap_slots, 1.0 / gap_slots))
    else:
        extra_parts = np.zeros(0, dtype=int)

    occupied = [FIRST_BAR_ROW]
    for idx in range(total_bars - 1):
        occupied.append(occupied[-1] + ROW_SPACING + int(extra_parts[idx]))

    anchor_rows = occupied[:n_groups]
    extra_rows = occupied[n_groups:]

    group_rows = [[anchor_rows[i]] for i in range(n_groups)]
    owners = [i for i, count in enumerate(counts) for _ in range(count - 1)]
    rng.shuffle(owners)
    for row, owner in zip(extra_rows, owners):
        group_rows[owner].append(row)

    for idx, anchor in enumerate(anchor_rows):
        rest = sorted(row for row in group_rows[idx] if row != anchor)
        group_rows[idx] = [anchor] + rest
    return group_rows


def _choose_fill_lengths(rng, count: int) -> list[int]:
    """Pick `count` distinct horizontal fill lengths for one cap-color group."""
    pool = list(range(1, MAX_FILL_LEN + 1))
    rng.shuffle(pool)
    return [int(length) for length in pool[:count]]


def _expected_output(
    inp: np.ndarray,
    caps: list[int],
    group_rows: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
) -> np.ndarray:
    """Compute the expected output by recoloring the rank-k bar's fill to cap."""
    out = inp.copy()
    width = inp.shape[1]
    right_c = width - 1
    for cap, rows, lengths, rank in zip(caps, group_rows, group_lengths, ranks):
        ranked = sorted(zip(rows, lengths), key=lambda item: (-item[1], item[0]))
        target_row, target_len = ranked[rank - 1]
        left_c = width - target_len - 2
        out[target_row, left_c + 1 : right_c] = cap
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, count + 1)) for count in counts]
    if all(rank == 1 for rank in ranks):
        bump = int(rng.integers(n_groups))
        ranks[bump] = int(rng.integers(2, counts[bump] + 1))

    palette = rng.choice(np.arange(*COLOR_RANGE), size=2 * n_groups, replace=False)
    caps = [int(v) for v in palette[:n_groups]]
    fills = [int(v) for v in palette[n_groups:]]

    starts, cluster_width = _legend_layout(rng, ranks)
    min_width = max(WIDTH_MIN, cluster_width, MAX_FILL_LEN + 2)
    if min_width > WIDTH_MAX:
        return None
    width = int(rng.integers(min_width, WIDTH_MAX + 1))

    total_bars = sum(counts)
    min_height = _min_height(total_bars)
    if min_height > HEIGHT_MAX:
        return None
    height = int(rng.integers(min_height, HEIGHT_MAX + 1))
    group_rows = _assign_rows(rng, n_groups, counts, height)

    grid = np.zeros((height, width), dtype=int)
    group_lengths: list[list[int]] = []
    for cap, fill, rank, start, rows, count in zip(caps, fills, ranks, starts, group_rows, counts):
        paste(grid, np.full((1, rank), cap, dtype=int), 0, start)

        lengths = _choose_fill_lengths(rng, count)
        group_lengths.append(lengths)

        for row, fill_len in zip(rows, lengths):
            left_c = width - fill_len - 2
            paste(grid, _build_bar(cap, fill, fill_len), row, left_c)

    expected = _expected_output(grid, caps, group_rows, group_lengths, ranks)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f'failed to generate a valid input after {MAX_RETRIES} attempts')
