# Combo variant of 97d7923e (index 12)
# combo_spec: D2/3+D11/0
# Name: 绝对长度匹配+未选中条擦除

import itertools

import numpy as np

from arc_utils import get_bg, neighbors4, paste


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - Visible input pieces remain the same family as the source task:
#      1) top-left legend columns, each a solid color K stack of height k
#      2) bottom-aligned vertical bars of the form cap K / fill F / cap K
#    - Original behavior for one group with fill lengths [6, 4, 2] and legend
#      height 4:
#        * sort by length descending -> ranks are 1, 2, 3
#        * rank-4 is invalid, so the source rule cannot even use this k
#    - D2 alone changes the meaning of k:
#        * k is a literal bar fill length, so the length-4 bar is selected
#    - D11 alone changes the scope of the transform:
#        * once a target is selected, non-target sibling bars are erased
#    - In the fused rule:
#        * legend (K, 4) selects the K-capped bar whose interior length is
#          exactly 4
#        * that selected bar becomes a solid K column
#        * the length-6 and length-2 bars are erased to bg
#    - Difference from the source task:
#        * selection is by exact visible size, not by rank in an ordering
#        * output keeps only the exact-length match instead of leaving the
#          other bars untouched
#
# b) Complexity calibration
#    - Distinct fill lengths inside each group preserve unique matching and keep
#      the "which bar matches k?" question visually clean.
#    - Bars per group stay at 2-3 so every group necessarily shows at least one
#      erased bar; the D11 effect is always visible.
#    - Generator target lengths are chosen to disagree with rank-based selection
#      for every group. Preferably k > group_size, otherwise choose a length
#      whose descending rank differs from its own value. This prevents collapse
#      into either the original task or the single D11 rank-and-erase variant.
#    - Legend heights still come from a small, human-countable range 1..7, so
#      the direct length match remains easy to infer from 2-4 examples.
#
# c) Implicit-assumption audit against the source main()
#    - `_legend_entries(...)` still assumes the legend is the leftmost top-row
#      cluster. The combo preserves that exactly.
#    - `_vertical_sandwiches(...)` still assumes each candidate bar is a clean
#      cap/fill/cap column with a solid non-bg interior. The generator keeps
#      that invariant.
#    - The source `segs.sort(...); segs[k - 1]` assumed k was a valid 1-based
#      rank and that k <= number of bars. D2 breaks that assumption because k is
#      now a literal fill length, potentially larger than the group size. The
#      fused solver therefore matches by `(bot_r - top_r - 1) == k` instead.
#    - The source `out[top_r + 1:bot_r, col] = cap` assumed non-selected bars
#      must survive. D11 breaks that assumption. The fused solver must rewrite
#      the whole detected bar span: selected bars become solid cap-colored
#      columns, all non-selected bars become bg.
#    - Erasing whole spans is only safe because the combo generator separates
#      legend columns from bar columns. That removes any risk of blanking legend
#      cells that share a column with a bar.
#
# d) Placement legality
#    - Legend columns are packed into cols 0..n_groups-1.
#    - A mandatory blank separator column follows the legend cluster.
#    - All bar columns are placed to the right of that separator, with stride 2
#      or more, so bars never overlap and always have at least one bg column
#      between neighbors.
#    - Bottom alignment remains fixed at row 9, and fill lengths stay in 1..7,
#      so bars fit within the 10-row frame.
#
# e) Visual observability
#    - The kept bar is identifiable by a simple visible equation: legend height
#      equals bar interior height. No hidden color ordering or palette math is
#      involved.
#    - Erasure makes the selection effect unmissable: each group shrinks from
#      several bars to one surviving solid column.
#    - Because generator targets disagree with rank selection, a human comparing
#      input and output can rule out the original "k-th longest" hypothesis.
#
# f) Visual cleanliness
#    - Separating legend and bars avoids merged same-color structures in the
#      upper-left corner, which would be especially messy when legend heights can
#      be as tall as 7.
#    - Output becomes cleaner than input: one solid survivor per group, all
#      other bars removed, and legend left intact.
#
# g) Change inferability
#    - Colors only encode group identity: legend color K points to K-capped bars,
#      and the surviving matched bar is recolored to the same K.
#    - The solver never relies on numeric color ids, only on visible grouping by
#      shared cap color and visible bar length.
#
# New solver assumptions to be enforced by generator
# --------------------------------------------------
# - Each legend height equals exactly one bar fill length in its own cap-color
#   group; distinct group lengths guarantee uniqueness.
# - For every group, the chosen legend height disagrees with rank-based
#   interpretation, so the combo does real D2 work rather than degenerating.
# - Legend columns are disjoint from all bar columns, making whole-span erasure
#   safe for D11 behavior.
# - Top row contains only legend cells, ensuring `_legend_entries(...)` sees one
#   contiguous leftmost legend cluster and nothing else.


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
    Keep only exact-length legend matches and erase all sibling bars.

    Input structure:
    - `bg` is the background color.
    - The top-left legend consists of 1-column monochrome stacks touching row 0; each stack has color K and height k.
    - To the right of the legend, each color group contains bottom-aligned vertical cap/fill/cap bars whose caps are K and whose interiors are solid non-bg fills.

    Rule:
    - For each legend entry (K, k), collect all bars capped by K and measure their interior fill lengths.
    - Keep the bar(s) whose fill length is exactly k, repainting the whole bar span to K so the survivor becomes a solid K column.
    - Erase every non-matching bar from that same cap-color group to `bg`, while leaving the legend unchanged.

    Key insight:
    - The legend height is a literal bar length to match, not a rank, and the output is exclusive: only exact-length matches survive.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, "legend must be non-empty"

    for _x, cap, k in legend:
        segs = _vertical_sandwiches(g, cap)
        assert segs, f"legend cap {cap} must have matching vertical bars"

        matches = [seg for seg in segs if (seg[1] - seg[0] - 1) == k]
        assert matches, f"legend length {k} must match at least one bar for cap {cap}"

        for top_r, bot_r, col, _fill in segs:
            if (bot_r - top_r - 1) == k:
                out[top_r : bot_r + 1, col] = cap
            else:
                out[top_r : bot_r + 1, col] = bg

    return out


# --------------- generator (O2) ---------------

# --- Configuration ---
GRID_H = 10
WIDTH_MIN = 10
WIDTH_MAX = 20
MAX_GROUPS = 4
MAX_TOTAL_BARS = 8           # separated legend columns must still fit within width 20
BARS_PER_GROUP = (2, 3)
MAX_FILL_LEN = 7
COLOR_RANGE = (1, 10)
COL_SPACING = 2              # at least one blank column between neighboring bars
LEGEND_TO_BAR_GAP = 1
MAX_RETRIES = 300

_BOTTOM_ROW = GRID_H - 1
_MAX_CAP_ROW = GRID_H - 2


def _bar_start_col(n_groups: int) -> int:
    """First bar column after the compact legend cluster and one blank separator."""
    return int(n_groups + LEGEND_TO_BAR_GAP)


def _min_width_for(n_groups: int, total_bars: int) -> int:
    """Minimum width needed for the separated legend-plus-bars layout."""
    start = _bar_start_col(n_groups)
    return max(WIDTH_MIN, start + COL_SPACING * (total_bars - 1) + 1)


_FEASIBLE_COUNTS = {
    ng: [
        counts
        for counts in itertools.product(BARS_PER_GROUP, repeat=ng)
        if sum(counts) <= MAX_TOTAL_BARS and _min_width_for(ng, sum(counts)) <= WIDTH_MAX
    ]
    for ng in range(1, MAX_GROUPS + 1)
}


def _build_bar(cap: int, fill: int, fill_len: int) -> np.ndarray:
    """Return a (fill_len+2, 1) column sprite: cap / fill*N / cap."""
    bar = np.full((fill_len + 2, 1), fill, dtype=int)
    bar[0, 0] = cap
    bar[-1, 0] = cap
    return bar


def _assign_columns(rng, n_groups: int, counts: list[int], width: int) -> list[list[int]]:
    """Distribute bar columns across groups while keeping legend columns separate on the left."""
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


def _choose_fill_lengths(rng, count: int) -> list[int]:
    """Pick `count` distinct fill lengths for one group."""
    lengths = rng.choice(np.arange(1, MAX_FILL_LEN + 1), size=count, replace=False)
    return [int(v) for v in lengths]


def _pick_target_length(rng, lengths: list[int]) -> int:
    """
    Choose a legend height that forces exact-length logic to differ from rank logic.

    Preference order:
    1) any length greater than the group's size, which rank selection cannot use
    2) otherwise, any length whose descending rank differs from its literal value
    """
    count = len(lengths)
    ranked = sorted(lengths, reverse=True)
    rank_of = {length: idx + 1 for idx, length in enumerate(ranked)}

    preferred = [length for length in lengths if length > count]
    if not preferred:
        preferred = [length for length in lengths if rank_of[length] != length]
    assert preferred, "each group must have a non-degenerate exact-length target"
    return int(preferred[int(rng.integers(len(preferred)))])


def _expected_output(
    inp: np.ndarray,
    caps: list[int],
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    target_lengths: list[int],
) -> np.ndarray:
    """Compute the expected output for the keep-match and erase-the-rest combo."""
    out = inp.copy()
    bg = get_bg(inp)
    for cap, cols, lengths, target_len in zip(caps, group_columns, group_lengths, target_lengths):
        for col, fill_len in zip(cols, lengths):
            top_r = _MAX_CAP_ROW - fill_len
            if fill_len == target_len:
                out[top_r : _BOTTOM_ROW + 1, col] = cap
            else:
                out[top_r : _BOTTOM_ROW + 1, col] = bg
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])

    palette = rng.choice(np.arange(*COLOR_RANGE), size=2 * n_groups, replace=False)
    caps = [int(v) for v in palette[:n_groups]]
    fills = [int(v) for v in palette[n_groups:]]

    total_bars = sum(counts)
    min_width = _min_width_for(n_groups, total_bars)
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    group_columns = _assign_columns(rng, n_groups, counts, width)

    grid = np.zeros((GRID_H, width), dtype=int)
    group_lengths: list[list[int]] = []
    target_lengths: list[int] = []

    for legend_col, cap, fill, count, cols in zip(range(n_groups), caps, fills, counts, group_columns):
        lengths = _choose_fill_lengths(rng, count)
        target_len = _pick_target_length(rng, lengths)

        paste(grid, np.full((target_len, 1), cap, dtype=int), 0, legend_col)
        group_lengths.append(lengths)
        target_lengths.append(target_len)

        for col, fill_len in zip(cols, lengths):
            paste(grid, _build_bar(cap, fill, fill_len), _MAX_CAP_ROW - fill_len, col)

    expected = _expected_output(grid, caps, group_columns, group_lengths, target_lengths)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
