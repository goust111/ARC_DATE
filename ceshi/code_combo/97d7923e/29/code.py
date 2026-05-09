# Combo variant of 97d7923e (index 29)
# combo_spec: D4/0+D5/1+D14/0
# Name: 水平条+水平图例+左端对齐

import itertools
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg, paste


"""
Design analysis for the combo variant.

a) Concrete behavior + trace
   - Original task:
     - read a top-row legend made of vertical stacks; stack height k is the target rank.
     - scan single-column cap/fill/cap bars that all share one bottom row.
     - rank each cap-color group by fill height and recolor the rank-k bar.
   - Combo task:
     - read a top-row legend made of horizontal runs; run width k is now the target rank.
     - scan single-row cap/fill/cap bars; every bar in the puzzle starts from the same left cap
       column, so longer bars extend farther to the right.
     - rank each cap-color group by horizontal fill length and recolor the rank-k bar.
   - Example trace:
     - suppose the legend contains a run "333" on row 0, so cap color 3 asks for rank 3.
     - bars capped by 3 might have fill lengths [7, 4, 1] on three different rows, all with
       their left cap at column 0.
     - the third longest bar is the length-1 row, so only that row's single fill cell changes
       from its fill color to 3.
   - The three edits cooperate rather than stack independently: the legend now reads left-to-right,
     the candidate objects read left-to-right, and the alignment cue also flips to a shared left edge.

b) Complexity calibration
   - Keep ranks in the small original range 1-3 by keeping 2-3 bars per group.
   - Keep 1-4 groups and distinct lengths within each group so ranking stays observable.
   - Force at least one legend run to have width > 1; otherwise the horizontal legend could
     visually collapse into isolated single pixels.
   - Keep bars one row tall with blank rows between occupied rows so the horizontal geometry is
     unmistakable and candidate extraction stays unambiguous.

c) Implicit assumption audit
   - The original legend parser assumed vertical connected components touching row 0 and measured
     their height. That breaks here; the parser must instead read contiguous same-color runs on row 0.
   - The original bar detector assumed candidates lived in columns. That also breaks; the detector
     must now search rows for consecutive cap occurrences with a solid fill interval between them.
   - The original recolor slice was vertical. In the combo it must be horizontal.
   - The combo introduces one new geometric assumption: within a cap-color group all bars share one
     left cap column. main() checks this explicitly, and generate_input() constructs it directly.

d) Placement legality
   - Width must fit both the longest legend cluster and the longest horizontal bar, so the generator
     uses variable width in 10..20 and enforces width >= cluster_width and width >= 9.
   - Height must fit up to 10 bars on separate rows with one blank row between them, so the generator
     uses variable height in 10..20 and places occupied rows at stride >= 2 starting from row 1.
   - Bars live strictly below the legend row, so legend pixels and bar pixels never overlap even when
     both use columns near the left edge.

e) Visual observability
   - A human sees three repeated cues at once: a compact top-row dash legend, long horizontal bars,
     and a common left edge instead of a common right edge.
   - Ranking depends only on visible bar length and visible legend width; no color-id ordering or
     hidden metadata is used.
   - Because groups contain multiple distinct lengths, the selected bar is inferable from a few I/O
     examples by comparing which bar interior changes for each legend width.

f) Visual cleanliness
   - The legend is confined to row 0 and bars are confined to later rows, so the two symbol systems
     stay separated.
   - Bars are separated by empty rows, and each row contains at most one candidate object.
   - Cap colors and fill colors are sampled without overlap, making the recolored target easy to spot.

g) Change inferability
   - Color change is keyed only by the shared cap color named in the legend and by the visible rank.
   - The generator preserves repeated support for each mechanism: there are multiple bars per group,
     multiple legend entries overall, and at least one nontrivial legend width in every sample.

h) New solver assumptions reviewed against the generator
   - _legend_entries() assumes row 0 contains one compact left-anchored cluster with 1-2 cell gaps.
     generate_input() builds exactly that.
   - _horizontal_sandwiches() assumes each valid row has one solid non-background fill segment between
     two cap pixels of the same color. generate_input() uses only that construction.
   - main() assumes every legend rank is within the number of bars of that cap color, and that every
     cap-color group shares one left cap column. generate_input() samples ranks from 1..count and
     places every bar with left cap at column 0.
"""


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
    Legend is a compact top-row cluster near the left edge.

    Each entry is a horizontal run of one non-background color on row 0.
    Entry width = target rank, and consecutive entries are separated by 1-2
    background cells. Returns (x_pos, legend_color, run_width).
    """
    _h, w = g.shape
    c = 0
    while c < w and int(g[0, c]) == bg:
        c += 1
    assert c < w, "legend row must be non-empty"

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
    Recolor left-aligned horizontal bars by legend-selected rank.

    Input structure:
    - bg is the background color. On row 0 near the left edge sits a compact legend of
      monochrome horizontal runs; each run uses cap color K and has width k.
    - Below the legend are horizontal cap/fill/cap bars. Within each cap-color group, all
      bars share the same left cap column and differ only by their fill length.

    Rule:
    - For each legend entry (K, k), collect the horizontal bars capped by K.
    - Rank those bars by interior length from longest to shortest.
    - Recolor the interior of the rank-k bar to K.

    Key insight:
    - The legend width is not a bar size to copy. It is a rank index into the length-sorted
      horizontal bars of the same cap color, and the shared left edge makes the new direction
      of comparison visually explicit.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, "legend must be non-empty"

    for _x, cap, k in legend:
        segs = _horizontal_sandwiches(g, cap)
        assert segs, f"legend cap {cap} must have matching horizontal bars"
        left_cols = {left_c for _row, left_c, _right_c, _fill in segs}
        assert len(left_cols) == 1, f"cap {cap} bars must share one left column in this variant"
        segs.sort(key=lambda s: (-(s[2] - s[1] - 1), s[0]))
        assert 1 <= k <= len(segs), f"legend rank {k} out of range for {len(segs)} bars"
        row, left_c, right_c, _fill = segs[k - 1]
        out[row, left_c + 1 : right_c] = cap

    return out


# --------------- generator (O2) ---------------

# --- Configuration ---
WIDTH_MIN = 10
WIDTH_MAX = 20
HEIGHT_MIN = 10
HEIGHT_MAX = 20
MAX_GROUPS = 4
MAX_TOTAL_BARS = 10
BARS_PER_GROUP = (2, 3)
MAX_FILL_LEN = 7
COLOR_RANGE = (1, 10)
ROW_SPACING = 2
MAX_RETRIES = 300

_LEFT_CAP_COL = 0

_FEASIBLE_COUNTS = {
    ng: [c for c in itertools.product(BARS_PER_GROUP, repeat=ng) if sum(c) <= MAX_TOTAL_BARS]
    for ng in range(1, MAX_GROUPS + 1)
}


def _build_bar(cap: int, fill: int, fill_len: int) -> np.ndarray:
    """Return a (1, fill_len+2) row sprite: cap / fill*N / cap."""
    bar = np.full((1, fill_len + 2), fill, dtype=int)
    bar[0, 0] = cap
    bar[0, -1] = cap
    return bar


def _legend_layout(rng, ranks: list[int]) -> tuple[list[int], int]:
    """Place horizontal legend runs from left to right with 1-2 bg columns between them."""
    starts = [0]
    for rank in ranks[:-1]:
        gap = 1 + int(rng.integers(0, 2))
        starts.append(starts[-1] + rank + gap)
    cluster_width = starts[-1] + ranks[-1]
    return starts, cluster_width


def _assign_rows(rng, n_groups: int, counts: list[int], height: int) -> list[list[int]]:
    """
    Distribute bar rows across groups with at least one empty row between occupied rows.

    The earliest `n_groups` occupied rows serve as each group's anchor row so every group
    appears near the top of the bar field. Remaining occupied rows are assigned randomly.
    """
    total_bars = sum(counts)
    min_height = max(HEIGHT_MIN, 2 * total_bars)
    assert height >= min_height

    extra_height = height - min_height
    inter_gap_bonus = np.zeros(total_bars - 1, dtype=int)
    if total_bars > 1 and extra_height > 0:
        inter_gap_bonus = rng.multinomial(extra_height, np.full(total_bars - 1, 1.0 / (total_bars - 1)))

    occupied = [1]
    for idx in range(total_bars - 1):
        occupied.append(occupied[-1] + ROW_SPACING + int(inter_gap_bonus[idx]))

    anchor_rows = occupied[:n_groups]
    extra_rows = occupied[n_groups:]

    group_rows = [[anchor_rows[i]] for i in range(n_groups)]
    owners = [i for i, count in enumerate(counts) for _ in range(count - 1)]
    rng.shuffle(owners)
    for row, owner in zip(extra_rows, owners):
        group_rows[owner].append(row)

    for rows in group_rows:
        rows.sort()
    return group_rows


def _choose_fill_lengths(rng, count: int) -> list[int]:
    """Pick `count` distinct fill lengths for one group."""
    pool = list(range(1, MAX_FILL_LEN + 1))
    rng.shuffle(pool)
    return pool[:count]


def _expected_output(
    inp: np.ndarray,
    caps: list[int],
    group_rows: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
) -> np.ndarray:
    """Compute the expected output by recoloring the rank-k bar's fill to cap."""
    out = inp.copy()
    for cap, rows, lengths, rank in zip(caps, group_rows, group_lengths, ranks):
        ranked = sorted(zip(rows, lengths), key=lambda item: (-item[1], item[0]))
        target_row, target_len = ranked[rank - 1]
        right_c = _LEFT_CAP_COL + target_len + 1
        out[target_row, _LEFT_CAP_COL + 1 : right_c] = cap
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

    total_bars = sum(counts)
    min_height = max(HEIGHT_MIN, 2 * total_bars)
    height = int(rng.integers(min_height, HEIGHT_MAX + 1))
    group_rows = _assign_rows(rng, n_groups, counts, height)

    starts, cluster_width = _legend_layout(rng, ranks)
    min_width = max(WIDTH_MIN, MAX_FILL_LEN + 2, cluster_width)
    if min_width > WIDTH_MAX:
        return None
    width = int(rng.integers(min_width, WIDTH_MAX + 1))

    grid = np.zeros((height, width), dtype=int)
    group_lengths: list[list[int]] = []
    for cap, fill, rank, start, count, rows in zip(caps, fills, ranks, starts, counts, group_rows):
        paste(grid, np.full((1, rank), cap, dtype=int), 0, start)

        lengths = _choose_fill_lengths(rng, count)
        group_lengths.append(lengths)

        for row, fill_len in zip(rows, lengths):
            paste(grid, _build_bar(cap, fill, fill_len), row, _LEFT_CAP_COL)

    expected = _expected_output(grid, caps, group_rows, group_lengths, ranks)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
