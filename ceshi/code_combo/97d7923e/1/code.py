# Combo variant of 97d7923e (index 1)
# combo_spec: D1/0+D2/0
# Name: 填充色分组+升序排名
#
# Design analysis for the combo variant
# -------------------------------------
# spec_variant:
#   core_change:
#     original: legend color points to a cap-color group, and height k selects the k-th longest bar.
#     combo: legend color points to a fill-color group, and height k selects the k-th shortest bar.
#   preserved_scene:
#     - grid is still 10 rows tall with sparse one-column vertical cap/fill/cap bars
#     - legend still sits at the upper-left as one-column stacks touching row 0
#     - chosen bars still become monochrome by repainting their interiors
#
# a) Concrete behavior trace
#   Example:
#     legend block = color 6, height 1
#     fill-6 bars = [(cap 2, len 5), (cap 3, len 2), (cap 2, len 7)]
#   Original task:
#     would interpret color 6 as a cap query and rank by longest-first.
#     On this combo input that interpretation is invalid because 6 is intentionally a fill color,
#     not a cap color.
#   D1-only view:
#     color 6 correctly identifies the three fill-6 bars, but longest-first would choose len 7.
#   Combo view:
#     color 6 identifies the same fill-6 bars, then shortest-first chooses len 2 and repaints its
#     interior to that bar's own cap color 3.
#   Difference:
#     the grouping cue moved from endpoints to interior fill, and the ordering cue flipped from
#     longest-first to shortest-first. Both changes alter which column is recolored.
#
# b) Complexity calibration
#   - Keep 2-4 groups and 2-3 bars per group so the task remains visually parseable.
#   - Within every fill group, lengths are distinct to make the rank unique.
#   - Every fill group uses at least two cap colors, and the global cap palette is smaller than the
#     number of groups. This prevents "group by cap" from looking equivalent to "group by fill".
#   - At least one group must have an informative rank (not the median of a size-3 group), so
#     shortest-first and longest-first choose different bars.
#
# c) Implicit-assumption review of the original main()
#   - `_vertical_sandwiches(g, cap)` assumed the legend color is itself a cap color. That fails
#     here, so the solver must extract all bars first and only then filter by fill color.
#   - The original `segs.sort(... negative length ...)` encoded longest-first ranking. The combo
#     flips this to ascending length while keeping stable tie-breakers by (top row, column).
#   - `out[top_r + 1 : bot_r, col] = cap` is still the right repaint action, but now `cap` must
#     come from the chosen bar rather than from the legend entry.
#   - `_legend_entries` still works unchanged because it only reads a top-left color stack and its
#     height; it does not care what semantic role the color plays.
#
# d) Placement legality
#   - Bars remain single-column and bottom-aligned at row 9.
#   - The legend column of each group can also host one bar, but its fill length is capped so there
#     is always at least one background row between the legend stack and the bar's top cap.
#   - Columns are spaced by at least one blank column, so bars never overlap.
#   - Fill colors are distinct across groups and disjoint from every cap color, preventing the
#     solver from confusing legend colors with endpoint colors.
#
# e) Visual observability
#   - A human can identify a group by scanning repeated interior fill colors.
#   - Endpoint colors are deliberately unreliable because the same cap colors reappear across
#     different fill groups.
#   - The shortest-first change is visible because at least one legend height selects a different
#     column than the original longest-first rule would have selected.
#
# f) Visual cleanliness
#   - The legend stays confined to the top-left strip.
#   - Bars occupy isolated columns with regular spacing and no overlapping structures.
#   - Repainting affects only one interior run per group, so outputs remain sparse and readable.
#
# g) Color-change inferability
#   - Legend colors already appear as bar fills, so the color-to-group mapping is directly visible.
#   - The repaint color is locally signaled by the chosen bar's two endpoint caps.
#   - No rule depends on numeric color ids or any hidden ordering; only visible fill matching and
#     visible bar length matter.
#
# new_main_assumptions:
#   - every legend color matches at least one bar fill group
#   - each fill group has 2-3 bars with distinct lengths
#   - at least one cap color is reused across multiple fill groups
#   - at least one group has a rank where shortest-first and longest-first disagree
#   - generate_input explicitly enforces these assumptions and oracle-checks the result

import itertools
from pathlib import Path
import sys

import numpy as np

for _parent in Path(__file__).resolve().parents:
    if (_parent / "arc_utils.py").exists():
        if str(_parent) not in sys.path:
            sys.path.append(str(_parent))
        break

from arc_utils import get_bg, neighbors4, paste


def _all_vertical_sandwiches(g: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """
    Find all vertical bar candidates as (top_r, bottom_r, col, cap_color, fill_color) where:
      - g[top_r, col] == g[bottom_r, col] == cap_color
      - (top_r, bottom_r) are consecutive occurrences of that cap_color in the column
      - interior is a solid non-bg run of fill_color different from the cap
      - total height is at least 3
    """
    h, w = g.shape
    bg = get_bg(g)
    segs: list[tuple[int, int, int, int, int]] = []
    for c in range(w):
        col_vals = g[:, c]
        for cap in {int(v) for v in col_vals if int(v) != bg}:
            rows = [int(r) for r in np.flatnonzero(col_vals == cap)]
            if len(rows) < 2:
                continue
            for a, b in zip(rows, rows[1:]):
                if b <= a + 1:
                    continue
                fill = int(col_vals[a + 1])
                if fill == cap or fill == bg:
                    continue
                if np.all(col_vals[a + 1 : b] == fill):
                    segs.append((a, b, c, cap, fill))
    return segs


def _legend_entries(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    """
    Legend is in the top-left: a cluster of non-bg pixels on the top row near the left edge,
    allowing small gaps between entries.

    Returns entries as (x_pos, legend_color, block_count), sorted by x_pos.
    block_count is the vertical height of the 4-connected component of that color that touches
    row 0 within the legend cluster.
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
    Recolor vertical bars by legend-driven rank selection under fill-group / shortest-first rules.

    Input structure:
    - Background cells dominate the grid; the top-left legend contains one-column stacks whose
      colors are group labels and whose heights are rank indices.
    - Elsewhere, each object is a single-column cap-fill-cap bar with a solid interior.
    - Bars in the same logical group share a fill color, while cap colors may repeat across groups.

    Rule:
    - For each legend entry (fill color F, stack height k), collect all bars whose interior fill is F.
    - Rank those bars by interior length in ascending order (shortest = rank 1).
    - Recolor the interior of the rank-k bar to that bar's own cap color.

    Key insight: The legend no longer names endpoint colors, and its height is not a size to copy;
    it is the index of the k-th shortest bar within the matching fill-color family.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, "legend must be non-empty"
    bars = _all_vertical_sandwiches(g)

    for _x, fill, k in legend:
        segs = [s for s in bars if s[4] == fill]
        assert segs, f"legend fill {fill} must have matching vertical bars"
        segs.sort(key=lambda s: (s[1] - s[0] - 1, s[0], s[2]))
        assert 1 <= k <= len(segs), f"legend rank {k} out of range for {len(segs)} bars"
        top_r, bot_r, col, cap, _fill = segs[k - 1]
        out[top_r + 1 : bot_r, col] = cap

    return out


# --------------- generator (O2) ---------------

# --- Configuration ---
GRID_H = 10
WIDTH_MIN = 10
WIDTH_MAX = 20
MIN_GROUPS = 2
MAX_GROUPS = 4
MAX_TOTAL_BARS = 10
BARS_PER_GROUP = (2, 3)
MIN_CAP_COLORS = 2
MAX_CAP_COLORS = 3
MAX_FILL_LEN = 7
COLOR_RANGE = (1, 10)
COL_SPACING = 2
MAX_RETRIES = 300

_BOTTOM_ROW = GRID_H - 1
_MAX_CAP_ROW = GRID_H - 2

_FEASIBLE_COUNTS = {
    ng: [c for c in itertools.product(BARS_PER_GROUP, repeat=ng) if sum(c) <= MAX_TOTAL_BARS]
    for ng in range(MIN_GROUPS, MAX_GROUPS + 1)
}


def _build_bar(cap: int, fill: int, fill_len: int) -> np.ndarray:
    """Return a (fill_len+2, 1) column sprite: cap / fill*N / cap."""
    bar = np.full((fill_len + 2, 1), fill, dtype=int)
    bar[0, 0] = cap
    bar[-1, 0] = cap
    return bar


def _assign_columns(rng, n_groups: int, counts: list[int], width: int) -> list[list[int]]:
    """Distribute bar columns across groups with random inter-group spacing."""
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


def _choose_fill_lengths(rng, count: int, rank: int) -> list[int]:
    """Pick distinct fill lengths; the legend-column bar must fit below the legend stack."""
    max_legend_len = MAX_FILL_LEN - rank
    shared_len = int(rng.integers(1, max_legend_len + 1))
    pool = [length for length in range(1, MAX_FILL_LEN + 1) if length != shared_len]
    rng.shuffle(pool)
    return [shared_len] + pool[: count - 1]


def _choose_bar_caps(rng, count: int, cap_pool: list[int]) -> list[int]:
    """Assign caps using at least two distinct colors from the shared cap pool."""
    base = [int(v) for v in rng.choice(np.array(cap_pool), size=2, replace=False)]
    caps = [base[i % 2] for i in range(count)]
    rng.shuffle(caps)
    return caps


def _rank_is_informative(count: int, rank: int) -> bool:
    """Return True iff shortest-first and longest-first choose different bars."""
    return 2 * rank != count + 1


def _has_ascending_signal(counts: list[int], ranks: list[int]) -> bool:
    """At least one group must visibly distinguish shortest-first from longest-first."""
    return any(_rank_is_informative(count, rank) for count, rank in zip(counts, ranks))


def _has_fill_group_signal(group_caps: list[list[int]]) -> bool:
    """
    Require both:
      - each fill group mixes at least two cap colors
      - at least one cap color is reused across multiple fill groups
    """
    if not all(len(set(caps)) >= 2 for caps in group_caps):
        return False
    cap_to_groups: dict[int, set[int]] = {}
    for group_idx, caps in enumerate(group_caps):
        for cap in set(caps):
            cap_to_groups.setdefault(int(cap), set()).add(group_idx)
    return any(len(owners) >= 2 for owners in cap_to_groups.values())


def _expected_output(
    inp: np.ndarray,
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    group_caps: list[list[int]],
    ranks: list[int],
) -> np.ndarray:
    """Compute the combo output: fill-group match + shortest-first rank + repaint to chosen cap."""
    out = inp.copy()
    for cols, lengths, caps, rank in zip(group_columns, group_lengths, group_caps, ranks):
        ranked = sorted(zip(cols, lengths, caps), key=lambda item: (item[1], item[0]))
        target_col, target_len, target_cap = ranked[rank - 1]
        top_r = _MAX_CAP_ROW - target_len
        out[top_r + 1 : _BOTTOM_ROW, target_col] = target_cap
    return out


def _descending_fill_reference_output(
    inp: np.ndarray,
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    group_caps: list[list[int]],
    ranks: list[int],
) -> np.ndarray:
    """Reference for the fill-group / longest-first single variant, used as a degeneration check."""
    out = inp.copy()
    for cols, lengths, caps, rank in zip(group_columns, group_lengths, group_caps, ranks):
        ranked = sorted(zip(cols, lengths, caps), key=lambda item: (-item[1], item[0]))
        target_col, target_len, target_cap = ranked[rank - 1]
        top_r = _MAX_CAP_ROW - target_len
        out[top_r + 1 : _BOTTOM_ROW, target_col] = target_cap
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid, visibly fused instance."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(MIN_GROUPS, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, count + 1)) for count in counts]
    if not _has_ascending_signal(counts, ranks):
        return None

    all_colors = list(range(*COLOR_RANGE))
    rng.shuffle(all_colors)
    fills = [int(v) for v in all_colors[:n_groups]]
    remaining = [int(v) for v in all_colors[n_groups:]]
    max_caps_here = min(MAX_CAP_COLORS, len(remaining))
    n_cap_colors = int(rng.integers(MIN_CAP_COLORS, max_caps_here + 1))
    cap_pool = remaining[:n_cap_colors]

    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    group_columns = _assign_columns(rng, n_groups, counts, width)

    grid = np.zeros((GRID_H, width), dtype=int)
    group_lengths: list[list[int]] = []
    group_caps: list[list[int]] = []
    for fill, count, rank, cols in zip(fills, counts, ranks, group_columns):
        paste(grid, np.full((rank, 1), fill, dtype=int), 0, cols[0])

        lengths = _choose_fill_lengths(rng, count, rank)
        caps = _choose_bar_caps(rng, count, cap_pool)
        group_lengths.append(lengths)
        group_caps.append(caps)

        for col, fill_len, cap in zip(cols, lengths, caps):
            paste(grid, _build_bar(cap, fill, fill_len), _MAX_CAP_ROW - fill_len, col)

    if not _has_fill_group_signal(group_caps):
        return None

    expected = _expected_output(grid, group_columns, group_lengths, group_caps, ranks)
    if not np.array_equal(main(grid), expected):
        return None

    if np.array_equal(expected, _descending_fill_reference_output(grid, group_columns, group_lengths, group_caps, ranks)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
