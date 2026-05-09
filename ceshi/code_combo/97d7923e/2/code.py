# Combo variant of 97d7923e (index 2)
# combo_spec: D1/0+D2/3
# Name: 填充色分组+绝对长度匹配

from __future__ import annotations

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


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior trace
#    - Visible input still has two parts:
#      1) top-left legend columns, each a solid stack of one color
#      2) bottom-aligned one-column bars of the form cap / fill*L / cap
#    - Fusion change 1 (D1/0): legend color no longer names a cap-color group.
#      It names a fill-color group instead.
#    - Fusion change 2 (D2/3): legend height k no longer means "rank k after sorting".
#      It literally means "choose the bar whose fill length is exactly k".
#    - Example group:
#      legend = color 5, height 4
#      bars with fill 5 are [(cap 2, len 6), (cap 3, len 4), (cap 2, len 1)]
#      original task: legend color 5 would try to match cap color 5, then use rank 4
#      fused task: legend color 5 matches the fill-5 group, then length 4 picks the
#      middle bar, whose interior repaints from 5 to its own cap color 3.
#    - The full pipeline is therefore:
#      old: legend color -> cap group -> sort by length -> choose rank k -> repaint to legend color
#      new: legend color -> fill group -> find exact length k -> repaint to matched bar's cap color
#
# b) Complexity calibration
#    - If each fill group had a unique cap color, grouping by fill could still look like
#      grouping by cap. The generator avoids that by reusing a small shared cap palette
#      across multiple fill groups and by mixing at least two cap colors inside each group.
#    - If target length k accidentally equaled the selected bar's rank, the D2 half of the
#      fusion would be visually weak. The generator therefore chooses each group's target
#      length so that literal-length selection disagrees with descending-rank selection.
#    - Lengths remain distinct inside each fill group, so exact-length matching is unique
#      and visually easy to verify.
#
# c) Implicit-assumption review of the original main()
#    - `_vertical_sandwiches(g, cap)` assumed the legend color can be queried as a cap.
#      That is false now because legend colors are fill colors, so the solver must first
#      extract all bars and then filter them by fill color.
#    - `segs.sort(...); segs[k - 1]` assumed k is a 1-based rank and `k <= len(segs)`.
#      That is false now because k is a literal fill length in 1..7, not an index.
#    - `out[...] = cap` is still conceptually valid only if `cap` refers to the matched
#      bar's own endpoint color, not the legend color. Under D1/0 those two are different.
#    - Combining both changes introduces one new requirement: every legend height must
#      match at least one bar length inside its fill group. The generator constructs that.
#
# d) Placement legality
#    - Legend columns are separated from bar columns. This removes overlap risk when
#      legend height can be as large as 7 and legend color equals a bar fill color.
#    - Bars stay one cell wide, bottom-aligned, and occupy distinct columns with at least
#      one blank column between neighbors.
#    - Width feasibility is enforced up front: with width <= 20, separate legend columns
#      and stride-2 bar placement allow at most 8 total bars across 2-4 groups.
#
# e) Visual observability
#    - A human can identify a legend entry's group only by scanning for repeated interior
#      fill color, because endpoint colors are intentionally unreliable and reused.
#    - The winning bar is the one whose interior height literally matches the legend stack.
#      It is not "1st/2nd/3rd longest". This becomes obvious across examples because the
#      generator forces those two choices to disagree.
#
# f) Visual cleanliness
#    - The left legend block and the right bar field are spatially separated, so there are
#      no merged same-color components between the cue area and the bars.
#    - The grid stays sparse and columnar, preserving the original task's clean geometry.
#
# g) Color-change inferability
#    - Each legend color already appears as the interior color of multiple bars, so the
#      fill-group mapping is directly visible without hidden arithmetic.
#    - The output color is locally justified by the selected bar itself: once chosen, its
#      interior simply adopts the endpoint color already sitting at its top and bottom.
#
# New solver assumptions enforced by the generator
# ------------------------------------------------
# - Legend colors are exactly the fill colors and are disjoint from all cap colors.
# - Every fill group has 2-3 bars with distinct fill lengths.
# - Every fill group uses at least two cap colors drawn from a shared global cap pool.
# - For every legend entry (fill, k), there is at least one same-fill bar of length k.
# - For every legend entry, exact-length selection differs from same-group rank selection.


def _all_vertical_sandwiches(g: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """
    Find all vertical bar candidates as (top_r, bottom_r, col, cap_color, fill_color) where:
      - g[top_r, col] == g[bottom_r, col] == cap_color
      - (top_r, bottom_r) are consecutive occurrences of that cap_color in the column
      - the interior is a solid run of one non-bg fill_color different from the cap
      - the bar spans at least 3 cells total
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
    allowing small gaps (typically 1 bg column) between entries.

    Returns entries as (x_pos, legend_color, block_count), sorted by x_pos.
    block_count is the vertical height of the 4-connected component of that color that touches
    the top row within the legend cluster.
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
    Recolor vertical bars by legend-driven fill-group exact-length selection.

    Input structure:
    - bg = background. Top-left: monochrome legend blocks touching row 0; each has color F and height k.
    - Main grid: bottom-aligned one-column bars with a cap / solid fill / cap structure.
    - Bars in one logical group share the same interior fill color F, while cap colors may repeat across groups.

    Rule:
    - For each legend entry (F, k), find vertical bars whose interior fill color is F.
    - Measure those bars by interior fill length and match the bar(s) whose length is exactly k.
    - Recolor each matched bar interior to that matched bar's own cap color.

    Key insight: The legend now fuses two direct cues: color chooses the fill-color group, and height is the literal target length, so ranking disappears entirely.
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
        matches = [s for s in segs if (s[1] - s[0] - 1) == k]
        assert matches, f"legend length {k} must match at least one bar for fill {fill}"
        for top_r, bot_r, _col, cap, _fill in matches:
            out[top_r + 1 : bot_r, _col] = cap

    return out


# --------------- generator (O2) ---------------

# --- Configuration ---
GRID_H = 10
WIDTH_MIN = 10
WIDTH_MAX = 20
MIN_GROUPS = 2
MAX_GROUPS = 4
MAX_TOTAL_BARS = 8
BARS_PER_GROUP = (2, 3)
MIN_CAP_COLORS = 2
MAX_CAP_COLORS = 3
MAX_FILL_LEN = 7
COLOR_RANGE = (1, 10)
COL_SPACING = 2
LEGEND_TO_BAR_GAP = 1
MAX_RETRIES = 300

_BOTTOM_ROW = GRID_H - 1
_MAX_CAP_ROW = GRID_H - 2


def _bar_start_col(n_groups: int) -> int:
    """First bar column after the compact legend cluster and one blank separator."""
    return int(n_groups + LEGEND_TO_BAR_GAP)


def _min_width_for(n_groups: int, total_bars: int) -> int:
    """Minimum width needed for a separate legend area plus stride-2 bar columns."""
    start = _bar_start_col(n_groups)
    return max(WIDTH_MIN, start + COL_SPACING * (total_bars - 1) + 1)


_FEASIBLE_COUNTS = {
    ng: [
        counts
        for counts in itertools.product(BARS_PER_GROUP, repeat=ng)
        if sum(counts) <= MAX_TOTAL_BARS and _min_width_for(ng, sum(counts)) <= WIDTH_MAX
    ]
    for ng in range(MIN_GROUPS, MAX_GROUPS + 1)
}


def _build_bar(cap: int, fill: int, fill_len: int) -> np.ndarray:
    """Return a (fill_len + 2, 1) sprite: cap / fill*N / cap."""
    bar = np.full((fill_len + 2, 1), fill, dtype=int)
    bar[0, 0] = cap
    bar[-1, 0] = cap
    return bar


def _assign_columns(rng: np.random.Generator, n_groups: int, counts: list[int], width: int) -> list[list[int]]:
    """Distribute bar columns on the right while keeping legend columns separate on the left."""
    total_bars = sum(counts)
    start_col = _bar_start_col(n_groups)
    min_width = _min_width_for(n_groups, total_bars)
    extra_width = width - min_width

    if total_bars == 1:
        occupied = [start_col]
    else:
        bonuses = rng.multinomial(extra_width, np.full(total_bars - 1, 1.0 / (total_bars - 1)))
        occupied = [start_col]
        for idx in range(total_bars - 1):
            occupied.append(occupied[-1] + COL_SPACING + int(bonuses[idx]))

    group_columns = [[] for _ in range(n_groups)]
    owners = [i for i, count in enumerate(counts) for _ in range(count)]
    rng.shuffle(owners)
    for col, owner in zip(occupied, owners):
        group_columns[owner].append(int(col))
    for cols in group_columns:
        cols.sort()
    return group_columns


def _choose_fill_lengths(rng: np.random.Generator, count: int) -> list[int]:
    """Pick distinct fill lengths for one fill-color group."""
    lengths = rng.choice(np.arange(1, MAX_FILL_LEN + 1), size=count, replace=False)
    return [int(v) for v in lengths]


def _pick_target_length(rng: np.random.Generator, lengths: list[int]) -> int:
    """Choose a target length that disagrees with descending-rank selection."""
    ranked = sorted(lengths, reverse=True)
    rank_of = {length: idx + 1 for idx, length in enumerate(ranked)}
    preferred = [length for length in lengths if rank_of[length] != length]
    assert preferred, "need at least one non-degenerate absolute-length target"
    return int(preferred[int(rng.integers(len(preferred)))])


def _choose_bar_caps(rng: np.random.Generator, count: int, cap_pool: list[int]) -> list[int]:
    """Assign caps using at least two colors from a shared global cap pool."""
    chosen = [int(v) for v in rng.choice(np.array(cap_pool), size=2, replace=False)]
    caps = [chosen[idx % 2] for idx in range(count)]
    rng.shuffle(caps)
    return caps


def _expected_output(
    inp: np.ndarray,
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    group_caps: list[list[int]],
    target_lengths: list[int],
) -> np.ndarray:
    """Compute the expected output under fill-group exact-length matching."""
    out = inp.copy()
    for cols, lengths, caps, target_len in zip(group_columns, group_lengths, group_caps, target_lengths):
        for col, fill_len, cap in zip(cols, lengths, caps):
            if fill_len != target_len:
                continue
            top_r = _MAX_CAP_ROW - fill_len
            out[top_r + 1 : _BOTTOM_ROW, col] = cap
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid non-degenerate combo instance."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(MIN_GROUPS, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])

    all_colors = list(range(*COLOR_RANGE))
    rng.shuffle(all_colors)
    fills = [int(v) for v in all_colors[:n_groups]]
    remaining = [int(v) for v in all_colors[n_groups:]]
    max_caps_here = min(MAX_CAP_COLORS, len(remaining))
    n_cap_colors = int(rng.integers(MIN_CAP_COLORS, max_caps_here + 1))
    cap_pool = [int(v) for v in remaining[:n_cap_colors]]

    total_bars = sum(counts)
    min_width = _min_width_for(n_groups, total_bars)
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    group_columns = _assign_columns(rng, n_groups, counts, width)

    grid = np.zeros((GRID_H, width), dtype=int)
    group_lengths: list[list[int]] = []
    group_caps: list[list[int]] = []
    target_lengths: list[int] = []

    for legend_col, fill, count, cols in zip(range(n_groups), fills, counts, group_columns):
        lengths = _choose_fill_lengths(rng, count)
        target_len = _pick_target_length(rng, lengths)
        caps = _choose_bar_caps(rng, count, cap_pool)

        paste(grid, np.full((target_len, 1), fill, dtype=int), 0, legend_col)
        group_lengths.append(lengths)
        group_caps.append(caps)
        target_lengths.append(target_len)

        for col, fill_len, cap in zip(cols, lengths, caps):
            paste(grid, _build_bar(cap, fill, fill_len), _MAX_CAP_ROW - fill_len, col)

    expected = _expected_output(grid, group_columns, group_lengths, group_caps, target_lengths)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
