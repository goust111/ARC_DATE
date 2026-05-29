# Combo variant of 97d7923e (index 26)
# combo_spec: D4/0+D14/0
# Name: 水平条+左端对齐

import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg, neighbors4, paste

# Design analysis for the combo variant.
#
# a) Concrete behavior and trace
# - Original task: read vertical legend stacks from the top row, then scan columns for
#   vertical cap-fill-cap bars whose bottoms all sit on the last row.
# - D4 rotates the long objects: bars become horizontal, legend entries become horizontal
#   runs touching the left edge, and the solver must scan rows instead of columns.
# - D14 flips the anchoring direction: instead of a shared far endpoint and variable starts,
#   all bars share the near endpoint and grow away from it.
# - Combined result: each bar is now a single-row cap/fill/cap segment with one shared left
#   cap column. The legend stays in the upper-left corner as horizontal runs on rows. Example:
#   if a legend row is "333" in columns 0..2, then k=3 for cap color 3. Suppose the bars
#   capped by 3 have fill lengths [4, 2, 1] and all start at column 4. Their right caps land
#   at columns [9, 7, 6]. The third longest bar is the length-1 one, so only that row's lone
#   fill cell is recolored to 3.
#
# b) Complexity calibration
# - Keep 1-4 groups and 2-3 bars per group so every legend rank has repeated support.
# - Keep fill lengths distinct within each cap-color group so ranking is never ambiguous.
# - Width stays fixed at 10 after the 90-degree rotation; height varies from 10 to 20 to hold
#   2-10 sparse horizontal bars. The shared left cap column depends on the widest legend entry,
#   which leaves 4-6 possible fill lengths. That is enough to support groups of size 3 while
#   making the left-alignment visibly different from the right-aligned D4 variant.
#
# c) Hidden-assumption audit
# - The original legend parser assumed the legend touches row 0. After rotation that is no
#   longer the stable cue; the new parser must instead anchor on column 0 and measure width.
# - The original bar detector assumed each candidate lived in one column. The combo needs a
#   row-based detector that finds consecutive cap occurrences with a solid non-background fill
#   interval between them.
# - The original recolor slice was vertical. Here it must repaint a horizontal open interval.
# - The original generator fixed one bottom row. The combo instead fixes one left cap column,
#   so placement logic must derive right endpoints from fill lengths.
# - The new solver now relies on a combo-specific geometric assumption: all matched bars share
#   one left cap column. The generator must construct that explicitly.
#
# d) Placement legality
# - Legend rows are the first occupied rows in a compact top-left cluster, separated by one
#   blank row at most. The first row of each group also hosts one bar.
# - The shared left cap column is `max(rank) + 1`, which guarantees at least one background
#   column between the widest legend block and any bar start.
# - With grid width 10, the usable fill range is `1 .. min(7, 10 - left_cap_col - 2)`. If that
#   range cannot provide enough distinct lengths for the largest group, the attempt is rejected.
# - Rows are assigned with stride 2, so horizontal bars never overlap and remain visually clean.
#
# e) Visual observability
# - Humans can see the fused change directly: long objects lie horizontally, they all start from
#   the same left position, and their right endpoints vary. This cannot be mistaken for the
#   original vertical bottom-aligned layout or for D4's right-aligned version.
# - Legend widths stay small (1-3) and repeat across samples, so they still read naturally as
#   discrete rank indices.
#
# f) Visual cleanliness
# - Legend pixels occupy only the left strip of the top cluster.
# - Bars occupy separate rows with background gaps, so the legend-bar overlap is only shared
#   row usage, not shared cells or touching components.
# - Cap colors and fill colors remain disjoint, making the recolored target easy to spot.
#
# g) Change inference
# - No invisible property such as raw color id, generation order, or hidden metadata influences
#   the ranking. Only visible legend width and visible bar length matter.
# - Because bars are left-aligned rather than right-aligned, geometry alone exposes the combo's
#   control variable even though the recoloring rule itself is unchanged.
#
# h) New solver assumptions to review
# - Every legend color must appear on at least one horizontal cap-fill-cap row.
# - All such rows must share one left cap column in this combo.
# - Each legend width k must be between 1 and the number of bars in that cap-color group.
# - generate_input() enforces all of those conditions directly.


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
    Legend is in the top-left: a compact cluster of non-bg pixels on the left column near the
    top edge, with small vertical gaps between entries.

    Returns entries as (y_pos, legend_color, block_count), sorted by y_pos.
    block_count is the horizontal width of the 4-connected component of that color that touches
    the left column within the legend cluster.
    """
    h, w = g.shape
    left_pos = [int(r) for r in np.flatnonzero(g[:, 0] != bg)]
    assert left_pos, "legend column must be non-empty"
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
        color = int(g[r, 0])
        if color == bg or seen[r, 0]:
            continue
        st = [(r, 0)]
        seen[r, 0] = True
        pix = []
        while st:
            rr, cc = st.pop()
            pix.append((rr, cc))
            for rrr, ccc in neighbors4(rr, cc):
                if 0 <= rrr <= legend_max_r and 0 <= ccc < w and (not seen[rrr, ccc]) and int(g[rrr, ccc]) == color:
                    seen[rrr, ccc] = True
                    st.append((rrr, ccc))
        y_pos = min(rr for rr, _ in pix)
        cs = [cc for _, cc in pix]
        block_count = max(cs) - min(cs) + 1
        entries.append((y_pos, color, int(block_count)))

    entries.sort(key=lambda t: t[0])
    return entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor left-anchored horizontal bars by legend-selected rank.

    Input structure:
    - The background is the dominant color. In the upper-left corner, each legend entry is a
      monochrome run touching column 0; its color is a cap color K and its width is k.
    - The main objects are single-row cap/fill/cap bars. All bars share one left cap column,
      while their right caps sit farther right when the visible fill is longer.

    Rule:
    - For each legend entry (K, k), gather the horizontal bars whose two end caps are color K.
    - Sort those bars by interior length from longest to shortest.
    - Repaint the interior of the rank-k bar with color K.

    Key insight: The legend width is an index into a length ordering; it does not describe where
    to draw a new bar. The only fused geometric change is that the ranked bars are horizontal and
    all start from the same left endpoint.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, "legend must be non-empty"

    all_left_caps: set[int] = set()
    cap_to_segs: dict[int, list[tuple[int, int, int, int]]] = {}
    for _y, cap, _k in legend:
        segs = _horizontal_sandwiches(g, cap)
        assert segs, f"legend cap {cap} must have matching horizontal bars"
        left_caps = {left_c for _row, left_c, _right_c, _fill in segs}
        assert len(left_caps) == 1, f"cap {cap} bars must share one left cap column in this variant"
        all_left_caps.update(left_caps)
        segs.sort(key=lambda s: (-(s[2] - s[1] - 1), s[0], s[1]))
        cap_to_segs[cap] = segs

    assert len(all_left_caps) == 1, "all bars must share one left cap column in this variant"

    for _y, cap, k in legend:
        segs = cap_to_segs[cap]
        assert 1 <= k <= len(segs), f"legend rank {k} out of range for {len(segs)} bars"
        row, left_c, right_c, _fill = segs[k - 1]
        out[row, left_c + 1 : right_c] = cap

    return out


# --------------- generator (O2) ---------------

# --- Configuration ---
GRID_W = 10                  # fixed grid width after the 90-degree rotation
HEIGHT_MIN = 10              # minimum grid height
HEIGHT_MAX = 20              # maximum grid height (inclusive)
MAX_GROUPS = 4               # max number of cap-color groups
MAX_TOTAL_BARS = 10          # max bars across all groups
BARS_PER_GROUP = (2, 3)      # allowed bars-per-group counts
MAX_FILL_LEN = 7             # nominal max interior fill length before layout clipping
COLOR_RANGE = (1, 10)        # non-bg color ids (exclusive upper)
ROW_SPACING = 2              # minimum row stride between bars
MAX_RETRIES = 300            # retry budget for generate_input

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

    Returns one list of row indices per group; the first row in each group doubles as that
    group's legend row.
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


def _bar_left_col(ranks: list[int]) -> int:
    """Shared left cap column: one blank column to the right of the widest legend block."""
    return max(ranks) + 1


def _max_fill_len(left_cap_col: int) -> int:
    """Largest fill length that still leaves room for the right cap within width 10."""
    return min(MAX_FILL_LEN, GRID_W - left_cap_col - 2)


def _choose_fill_lengths(rng, count: int, max_fill_len: int) -> list[int]:
    """Pick `count` distinct fill lengths that fit to the right of the shared left cap."""
    assert count <= max_fill_len, f"need {count} distinct lengths, only {max_fill_len} fit"
    pool = list(range(1, max_fill_len + 1))
    rng.shuffle(pool)
    return pool[:count]


def _expected_output(
    inp: np.ndarray,
    caps: list[int],
    group_rows: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
    left_cap_col: int,
) -> np.ndarray:
    """Compute the expected output by recoloring the rank-k bar's fill to cap."""
    out = inp.copy()
    for cap, rows, lengths, rank in zip(caps, group_rows, group_lengths, ranks):
        ranked = sorted(zip(rows, lengths), key=lambda item: (-item[1], item[0]))
        target_row, target_len = ranked[rank - 1]
        right_c = left_cap_col + target_len + 1
        out[target_row, left_cap_col + 1 : right_c] = cap
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch."""
    rng = np.random.default_rng()

    # --- structure choices ---
    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, c + 1)) for c in counts]
    left_cap_col = _bar_left_col(ranks)
    max_fill_len = _max_fill_len(left_cap_col)
    if max_fill_len < max(counts):
        return None

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

        lengths = _choose_fill_lengths(rng, count, max_fill_len)
        group_lengths.append(lengths)

        for row, fill_len in zip(rows, lengths):
            paste(grid, _build_bar(cap, fill, fill_len), row, left_cap_col)

    # --- oracle check ---
    expected = _expected_output(grid, caps, group_rows, group_lengths, ranks, left_cap_col)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
