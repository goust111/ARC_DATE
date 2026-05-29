# Variant of 97d7923e
# Dimension: D14 — 条的底部对齐约束 (meta)
# Change: 所有条底部对齐在最后一行(row 9)，顶部位置由长度决定... → 顶部对齐替代底部对齐

import itertools
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg, neighbors4, paste

# Design analysis
# source_task: none, so there is no external reference solver to inspect.
#
# a) Concrete behavior and trace
# - The legend stays in the top-left: each entry is a 1-cell-wide monochrome stack of cap
#   color K, and its height k still means "pick the k-th longest bar in color-group K".
# - The visual change is only in bar placement. In the original task every bar ended at row 9,
#   so long bars grew upward. In this variant every bar starts from the same top-cap row and
#   grows downward; the bottom cap is now length-dependent.
# - Example trace for one cap-color group with fill lengths [4, 2, 1] and legend height k=2:
#   original placement -> top caps at rows [4, 6, 7], all bottom caps at row 9.
#   variant placement -> all top caps at one shared row, say row 3; bottom caps land at
#   rows [8, 6, 5]. Sorting by fill length is unchanged, so the bar of fill length 2 is still
#   the target, but the observable alignment flips from "same bottom" to "same top".
#
# b) Complexity calibration
# - Keep 1-4 groups and 2-3 bars per group so each legend rank has repeated support.
# - Keep distinct fill lengths inside each cap-color group to avoid rank ambiguity.
# - Because the bars now start lower (below the legend cluster plus a gap row), the maximum
#   usable fill length becomes dynamic. This reduces length range but keeps enough variety:
#   even in the tightest case there are 4 available lengths, which still supports groups of 3.
#
# c) Hidden assumptions in the original main()
# - _vertical_sandwiches() does not depend on bottom alignment; it only assumes cap/fill/cap
#   columns with a solid non-background interior. That still holds.
# - _legend_entries() only reads the top-left legend cluster. Bar alignment does not affect it.
# - main() sorts by interior length and recolors the open interval between the two caps.
#   That remains valid because the target is still "rank by length, then recolor the fill".
# - The original generator assumed bottom cap row was constant and derived top rows from
#   length. That assumption is the part that must change. The new generator instead fixes one
#   shared top-cap row and derives bottom rows from length.
#
# d) Placement legality
# - A shared top-cap row is chosen as one row below the tallest legend block, plus one blank
#   separator row. This guarantees that a legend and its bar can share a column without
#   overlap and without touching.
# - Fill lengths are chosen from 1..max_fill_len where max_fill_len is constrained by the
#   shared top-cap row and grid height, so every bottom cap remains in-bounds.
# - Column assignment still keeps at least one empty column between adjacent bars, so groups
#   remain visually separable and sandwich detection stays unambiguous.
#
# e) Visual observability
# - The variant is visible from raw geometry alone: bars in a sample now have one common top
#   edge and multiple bottom endpoints, the exact opposite of the original layout.
# - Since each group has multiple distinct lengths, a sample cannot accidentally look bottom-
#   aligned as well; the bottom rows necessarily differ.
#
# f) Visual cleanliness
# - The legend remains compact in the top-left. The bar field remains sparse because bars are
#   one column wide with at least one blank column between neighbors.
# - Using one global top row for all bars keeps the grid easy to parse: legend above, bars
#   below, no mixed anchoring rules in the same canvas.
#
# g) Color-change inference
# - No ranking uses color ids. Only visible bar lengths and legend heights matter.
# - Cap colors and fill colors remain disjoint, so the recolored bar is easy to spot.
#
# h) New solver assumptions to review after the change
# - main() now explicitly checks that the bars within each cap-color group share one top row.
#   generate_input() enforces this by constructing every bar from the same shared start row.
# - main() still assumes each legend cap color has at least one matching bar group and that the
#   legend height k is within the number of bars of that color. generate_input() constructs both
#   conditions directly.


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
    Recolor one top-anchored vertical bar per legend-selected rank.

    Input structure:
    - bg is the dominant background. In the top-left sits a legend of monochrome stacks that
      touch row 0; each stack uses cap color K and has height k.
    - Elsewhere are single-column cap/fill/cap bars. In this variant the bars for a given cap
      color all begin on the same top-cap row, while their bottom caps drop lower as the fill
      gets longer.

    Rule:
    - For each legend entry (K, k), collect the vertical bars capped by K.
    - Order those bars by interior length from longest to shortest.
    - Repaint the interior of the rank-k bar with color K, leaving all other cells unchanged.

    Key insight: The legend height still encodes a rank index; the only geometric change is that
    bar length is now read from a shared top anchor instead of a shared bottom anchor.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, "legend must be non-empty"

    for _x, cap, k in legend:
        segs = _vertical_sandwiches(g, cap)
        assert segs, f"legend cap {cap} must have matching vertical bars"
        top_rows = {top_r for top_r, _bot_r, _col, _fill in segs}
        assert len(top_rows) == 1, f"cap {cap} bars must share one top row in this variant"
        segs.sort(key=lambda s: (-(s[1] - s[0] - 1), s[2]))
        assert 1 <= k <= len(segs), f"legend rank {k} out of range for {len(segs)} bars"
        top_r, bot_r, col, _fill = segs[k - 1]
        out[top_r + 1 : bot_r, col] = cap

    return out


# --------------- generator (O2) ---------------

# --- Configuration ---
GRID_H = 10                  # fixed grid height
WIDTH_MIN = 10               # minimum grid width
WIDTH_MAX = 20               # maximum grid width (inclusive)
MAX_GROUPS = 4               # max number of cap-color groups
MAX_TOTAL_BARS = 10          # max bars across all groups
BARS_PER_GROUP = (2, 3)      # allowed bars-per-group counts
MAX_FILL_LEN = 7             # nominal max interior fill length before layout clipping
COLOR_RANGE = (1, 10)        # non-bg color ids (exclusive upper)
COL_SPACING = 2              # minimum column stride between bars
MAX_RETRIES = 300            # retry budget for generate_input

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
    """Distribute bar columns across groups with random inter-group spacing.

    Returns one list of column indices per group; the first column in each
    group doubles as that group's legend column.
    """
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


def _bar_top_row(ranks: list[int]) -> int:
    """Shared top-cap row: one blank row below the tallest legend stack."""
    return max(ranks) + 1


def _max_fill_len(bar_top_row: int) -> int:
    """Largest fill length that still leaves room for the bottom cap."""
    return min(MAX_FILL_LEN, GRID_H - bar_top_row - 2)


def _choose_fill_lengths(rng, count: int, max_fill_len: int) -> list[int]:
    """Pick `count` distinct fill lengths that fit below the shared top row."""
    assert count <= max_fill_len, f"need {count} distinct lengths, only {max_fill_len} fit"
    pool = list(range(1, max_fill_len + 1))
    rng.shuffle(pool)
    return pool[:count]


def _expected_output(
    inp: np.ndarray,
    caps: list[int],
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
    bar_top_row: int,
) -> np.ndarray:
    """Compute the expected output by recoloring the rank-k bar's fill to cap."""
    out = inp.copy()
    for cap, cols, lengths, rank in zip(caps, group_columns, group_lengths, ranks):
        ranked = sorted(zip(cols, lengths), key=lambda item: (-item[1], item[0]))
        target_col, target_len = ranked[rank - 1]
        bottom_r = bar_top_row + target_len + 1
        out[bar_top_row + 1 : bottom_r, target_col] = cap
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch."""
    rng = np.random.default_rng()

    # --- structure choices ---
    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, c + 1)) for c in counts]
    bar_top_row = _bar_top_row(ranks)
    max_fill_len = _max_fill_len(bar_top_row)
    if max_fill_len < max(counts):
        return None

    # --- palette ---
    palette = rng.choice(np.arange(*COLOR_RANGE), size=2 * n_groups, replace=False)
    caps = [int(v) for v in palette[:n_groups]]
    fills = [int(v) for v in palette[n_groups:]]

    # --- layout ---
    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    group_columns = _assign_columns(rng, n_groups, counts, width)

    # --- render legend + bars ---
    grid = np.zeros((GRID_H, width), dtype=int)
    group_lengths: list[list[int]] = []
    for cap, fill, count, rank, cols in zip(caps, fills, counts, ranks, group_columns):
        paste(grid, np.full((rank, 1), cap, dtype=int), 0, cols[0])  # legend

        lengths = _choose_fill_lengths(rng, count, max_fill_len)
        group_lengths.append(lengths)

        for col, fill_len in zip(cols, lengths):
            paste(grid, _build_bar(cap, fill, fill_len), bar_top_row, col)

    # --- oracle check ---
    expected = _expected_output(grid, caps, group_columns, group_lengths, ranks, bar_top_row)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
