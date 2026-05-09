# Variant of 97d7923e
# Dimension: D4 — 源对象：垂直条的结构 (src)
# Change: 垂直条结构为cap-fill-cap：顶部和底部各一个帽色像素，中间为单一填充色的连续区域，所有条底部对齐在最后一行... → 水平条替代垂直条

"""
Design analysis for the D4 variant.

spec_variant:
  core_change:
    from: vertical cap-fill-cap bars, bottom-aligned, legend stacked by columns along the top row
    to: horizontal cap-fill-cap bars, right-aligned, legend laid out by rows along the left column
  grid:
    width: fixed 10
    height: 10-20
    bg: 0
  legend:
    anchor: top-left corner, touching column 0
    entries: 1-4
    layout: one horizontal monochrome block per entry, sorted by row
    width_meaning: width k is the rank index 1-3 for that color group
    spacing: consecutive legend rows are separated by 1-2 cells in the compact top-left cluster
  bars:
    shape: cap-fill-cap on a single row
    alignment: all bars end at the last column
    left_cap: at column 8-fill_len, so the right cap stays at column 9
    fill_len: 1-7, distinct within each cap-color group
    groups: 1-4 groups, 2-3 bars per group, same cap color and same fill color inside a group
    row_sharing: the first row in each group also hosts that group's legend entry
  transform:
    for each legend entry (K, k), collect horizontal bars capped by K
    sort by fill length descending
    recolor the interior of the rank-k bar to K

behavior_trace:
  original_local_step:
    - read legend from the top row by columns
    - for cap K, scan each column for cap / fill... / cap
    - choose the kth longest column and recolor its interior
  variant_local_step:
    - read legend from the left column by rows
    - for cap K, scan each row for cap fill... cap
    - choose the kth longest row and recolor its interior
  concrete_example:
    - legend row contains "222" at columns 0..2, so k=3 for cap color 2
    - rows capped by 2 might have fill lengths [6, 4, 1]
    - the third longest row is the length-1 bar, so only that row's middle cell changes to 2

complexity_calibration:
  - The orientation flip is visually obvious because the long objects now run horizontally and terminate on the right border.
  - Rank values remain small (1-3), preserving the original inference load.
  - Distinct fill lengths inside each group prevent ranking ambiguity.
  - Legend blocks remain compact in one corner, so the task still reads as "legend drives selection".

implicit_assumption_audit:
  - The old solver assumed top-row activity identifies the legend. That no longer holds, so the legend parser must pivot to the left column.
  - The old bar detector assumed each candidate lives in a column. In the variant, the detector must instead use consecutive cap occurrences in a row.
  - The old recolor slice was vertical. In the variant, it must be horizontal or it will overwrite unrelated cells.
  - Sorting still depends only on visible fill length, not on color ids or hidden metadata.

placement_legality:
  - Width is fixed at 10 so every horizontal bar fits exactly as left_cap / fill / right_cap with right_cap at column 9.
  - Height varies with the number of rows needed for 2-10 bars while preserving a blank row between occupied rows.
  - The first row of each group hosts both legend and a bar; legality is guaranteed by restricting that row's legend width k and bar fill length L to k + L <= 7, leaving at least one background gap.
  - Group colors are pairwise distinct across caps and fills, and cap colors never equal fill colors.

visual_observability:
  - A human can compare input/output and see that exactly one horizontal bar per legend color is recolored.
  - The legend width is repeated across examples as a small discrete index, making the selection rule inferable.
  - Because every bar reaches the right edge, the orientation change is not subtle and cannot be mistaken for the original vertical layout.

cleanliness:
  - Legend pixels live only in the compact upper-left strip.
  - Bars occupy separate rows with blank rows between them, so components remain easy to parse visually.
  - No sorting or recoloring uses invisible properties such as raw color id ordering.
"""

from pathlib import Path
import sys
import itertools

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / 'solvers'
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg, neighbors4, paste


def _horizontal_sandwiches(g: np.ndarray, cap: int) -> list[tuple[int, int, int, int]]:
    """
    Find horizontal bar candidates (row, left_c, right_c, fill_color) where:
      - g[row, left_c] == cap and g[row, right_c] == cap
      - (left_c, right_c) are consecutive occurrences of cap in that row
      - interior is a solid run of ONE color fill_color != cap and fill_color != background
      - length >= 1 (right_c >= left_c + 2)
    """
    h, w = g.shape
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
    Legend is in the top-left: a cluster of non-bg pixels on the left column near the top edge,
    allowing small gaps (typically 1 bg row) between entries.

    Returns entries as (y_pos, legend_color, block_count), sorted by y_pos.
    block_count is the horizontal width of the 4-connected component of that color that touches
    the left column within the legend cluster.
    """
    h, w = g.shape
    left_pos = [int(r) for r in np.flatnonzero(g[:, 0] != bg)]
    assert left_pos, 'legend column must be non-empty'
    left_pos.sort()

    rows = [left_pos[0]]
    for r in left_pos[1:]:
        if r - rows[-1] <= 2:
            rows.append(r)
        else:
            break
    legend_max_r = max(rows)

    entries: list[tuple[int, int, int]] = []
    seen = np.zeros((h, w), dtype=bool)
    for r in rows:
        col = int(g[r, 0])
        if col == bg or seen[r, 0]:
            continue
        st = [(r, 0)]
        seen[r, 0] = True
        pix = []
        while st:
            rr, cc = st.pop()
            pix.append((rr, cc))
            for rrr, ccc in neighbors4(rr, cc):
                if 0 <= rrr <= legend_max_r and 0 <= ccc < w and (not seen[rrr, ccc]) and int(g[rrr, ccc]) == col:
                    seen[rrr, ccc] = True
                    st.append((rrr, ccc))
        y_pos = min(rr for rr, _ in pix)
        cs = [cc for _, cc in pix]
        block_count = max(cs) - min(cs) + 1
        entries.append((y_pos, col, int(block_count)))

    entries.sort(key=lambda t: t[0])
    return entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor right-aligned horizontal bars by legend-driven rank selection.

    Input structure:
    - bg = background. Top-left: monochrome blocks touching column 0; each has color K and width k.
    - Main grid: horizontal bars capped left and right by a single color, with solid-color interiors.

    Rule:
    - For each legend entry (K, k), find horizontal bars capped by K.
    - Rank bars by interior length (longest = rank 1).
    - Recolor the rank-k bar's interior to K.

    Key insight: Legend width is a rank index into length-sorted horizontal bars, not a literal bar size.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, 'legend must be non-empty'

    for _y, cap, k in legend:
        segs = _horizontal_sandwiches(g, cap)
        assert segs, f'legend cap {cap} must have matching horizontal bars'
        segs.sort(key=lambda s: (-(s[2] - s[1] - 1), s[0], s[1]))
        assert 1 <= k <= len(segs), f'legend rank {k} out of range for {len(segs)} bars'
        row, left_c, right_c, _fill = segs[k - 1]
        out[row, left_c + 1 : right_c] = cap

    return out


# --------------- generator (O2) ---------------

# --- Configuration ---
GRID_W = 10                  # fixed grid width
HEIGHT_MIN = 10              # minimum grid height
HEIGHT_MAX = 20              # maximum grid height (inclusive)
MAX_GROUPS = 4               # max number of cap-color groups
MAX_TOTAL_BARS = 10          # max bars across all groups
BARS_PER_GROUP = (2, 3)      # allowed bars-per-group counts
MAX_FILL_LEN = 7             # max interior fill length of a bar
COLOR_RANGE = (1, 10)        # non-bg color ids (exclusive upper)
ROW_SPACING = 2              # minimum row stride between bars
MAX_RETRIES = 300            # retry budget for generate_input

_RIGHT_COL = GRID_W - 1      # column index where all bars end
_MAX_CAP_COL = GRID_W - 2    # furthest-right possible left cap before the fill

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


def _assign_rows(rng, n_groups: int, counts: list[int], height: int) -> list[list[int]]:
    """Distribute bar rows across groups with random inter-group spacing.

    Returns one list of row indices per group; the first row in each
    group doubles as that group's legend row.
    """
    total_bars = sum(counts)
    min_height = max(HEIGHT_MIN, ROW_SPACING * total_bars - 1)
    gap_slots = total_bars - n_groups + 1
    extra_height = height - min_height
    extra_parts = rng.multinomial(extra_height, np.full(gap_slots, 1.0 / gap_slots))

    inter_gap_bonus = np.zeros(total_bars - 1, dtype=int)
    if total_bars > n_groups:
        inter_gap_bonus[n_groups - 1 :] = extra_parts[:-1]

    occupied = [0]
    for idx in range(total_bars - 1):
        occupied.append(occupied[-1] + ROW_SPACING + int(inter_gap_bonus[idx]))

    legend_rows = occupied[:n_groups]
    extra_rows = occupied[n_groups:]

    group_rows = [[legend_rows[i]] for i in range(n_groups)]
    owners = [i for i, count in enumerate(counts) for _ in range(count - 1)]
    rng.shuffle(owners)
    for row, owner in zip(extra_rows, owners):
        group_rows[owner].append(row)

    return group_rows


def _choose_fill_lengths(rng, count: int, rank: int) -> list[int]:
    """Pick `count` distinct fill lengths for one group.

    The first length (legend row bar) is constrained so the bar fits
    to the right of the legend block with a bg gap column.
    """
    max_legend_len = MAX_FILL_LEN - rank  # legend occupies `rank` left columns
    shared_len = int(rng.integers(1, max_legend_len + 1))
    pool = [length for length in range(1, MAX_FILL_LEN + 1) if length != shared_len]
    rng.shuffle(pool)
    return [shared_len] + pool[: count - 1]


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
        left_c = _MAX_CAP_COL - target_len
        out[target_row, left_c + 1 : _RIGHT_COL] = cap
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch."""
    rng = np.random.default_rng()

    # --- structure choices ---
    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, c + 1)) for c in counts]

    # --- palette ---
    palette = rng.choice(np.arange(*COLOR_RANGE), size=2 * n_groups, replace=False)
    caps = [int(v) for v in palette[:n_groups]]
    fills = [int(v) for v in palette[n_groups:]]

    # --- layout ---
    total_bars = sum(counts)
    min_height = max(HEIGHT_MIN, ROW_SPACING * total_bars - 1)
    height = int(rng.integers(min_height, HEIGHT_MAX + 1))
    group_rows = _assign_rows(rng, n_groups, counts, height)

    # --- render legend + bars ---
    grid = np.zeros((height, GRID_W), dtype=int)
    group_lengths: list[list[int]] = []
    for cap, fill, count, rank, rows in zip(caps, fills, counts, ranks, group_rows):
        paste(grid, np.full((1, rank), cap, dtype=int), rows[0], 0)  # legend

        lengths = _choose_fill_lengths(rng, count, rank)
        group_lengths.append(lengths)

        for row, fill_len in zip(rows, lengths):
            left_c = _MAX_CAP_COL - fill_len
            paste(grid, _build_bar(cap, fill, fill_len), row, left_c)

    # --- oracle check ---
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
