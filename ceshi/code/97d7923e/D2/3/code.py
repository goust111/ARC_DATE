# Variant of 97d7923e
# Dimension: D2 — 匹配条件：排名与图例计数的对应 (match_cond)
# Change: s.length_rank_in_group == d.block_count — 图例条目的色块堆叠数k对应组内按长度... → 图例色块数对应条的绝对长度而非排名

import itertools

import numpy as np

from arc_utils import get_bg, neighbors4, paste


# Design analysis for the D2 variant
# ==================================
# a) Concrete behavior and trace
#    - Input still has two visible parts:
#      1) top-left legend columns: each is a solid vertical block of color K with height k
#      2) bottom-aligned vertical bars elsewhere: cap K / fill F / cap K
#    - Original behavior: for legend (K, k), gather all K-capped bars, sort them by
#      fill length descending, then recolor the rank-k bar.
#    - Variant behavior: for legend (K, k), gather all K-capped bars and recolor the
#      bar(s) whose fill length is exactly k.
#    - Example: one K-group has fill lengths [6, 3, 1].
#      Original with legend height 3: rank 3 -> the length-1 bar changes.
#      Variant with legend height 3: the length-3 bar changes.
#      This is the core D2 difference, and it is visible directly from the bar lengths.
#
# b) Complexity calibration
#    - The variant must not collapse back into the original rank rule.
#    - If legend heights stayed in 1..3 and always matched rank accidentally, the
#      change would be hard to observe. To prevent that, generator target lengths are
#      chosen so that "absolute length == k" and "descending rank == k" disagree for
#      every group. Many groups also use k > bar_count, which the original rule could
#      not even interpret.
#    - Bars within a cap-color group keep distinct lengths, preserving a unique visual
#      target and avoiding ambiguous direct-length matches.
#
# c) Implicit-assumption audit against the original main()
#    - _legend_entries(...): still valid. Legend is still a top-left cluster touching
#      row 0, and its component height still carries the numeric signal.
#    - _vertical_sandwiches(...): still valid. Bars are still cap/fill/cap columns with
#      solid interiors.
#    - The old `segs.sort(...)` + `segs[k - 1]` logic depended on "k is a 1-based rank"
#      and on `k <= len(segs)`. That assumption is false in the variant because k is now
#      an absolute fill length in 1..7, so sorting-by-rank must be removed.
#    - Recoloring all bars whose fill length equals k is safer than assuming uniqueness.
#      The generator still guarantees uniqueness by distinct group lengths.
#
# d) Placement legality
#    - Original generator reused a legend column as one of the bar columns. That worked
#      when legend height was a small rank (1..3), but it becomes fragile once legend
#      height can be any fill length up to 7.
#    - This variant therefore separates legend columns from bar columns. Legend stays in
#      columns 0..n_groups-1, bars start after one blank separator column, and bars are
#      bottom-aligned exactly as before.
#    - Width limits are preserved by capping total bars at 8, which is enough to keep
#      1-4 groups with 2-3 bars each while fitting inside width 20.
#
# e) Visual observability
#    - Human-visible clue: the legend height now literally equals the winning bar's
#      interior height. Across examples, the selected bar is not "1st/2nd/3rd longest"
#      but "the one whose height matches the stacked legend count".
#    - Choosing groups where absolute-length matching disagrees with rank matching makes
#      the variant inferable from a few examples.
#
# f) Visual cleanliness
#    - Legend and bars use disjoint columns, so no top-left overlap or confusing merged
#      same-color structures appear.
#    - Bars remain single-column, bottom-aligned, and separated by at least one blank
#      column, preserving the original clean layout.
#
# g) Color-change inferability
#    - Colors are still structured exactly as before: legend color K identifies the bar
#      group, and recoloring always changes fill F to cap K. No color-id arithmetic or
#      hidden palette ordering is used.
#
# New solver assumptions to be enforced by generator
# --------------------------------------------------
# - Each legend height equals at least one bar fill length in its same-cap group.
# - Top-left legend columns are the only non-background cells on row 0.
# - Bars keep solid interiors and distinct lengths within each cap-color group.


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
    Recolor vertical bars by legend-driven exact-length selection.

    Input structure:
    - bg = background. Top-left: monochrome legend blocks touching row 0; each has color K and height k.
    - Main grid: vertical bars capped top and bottom by a single color, with solid-color interiors.

    Rule:
    - For each legend entry (K, k), find vertical bars capped by K.
    - Measure each such bar by its interior fill length and match bars whose length is exactly k.
    - Recolor the matched bar interiors to K.

    Key insight: Legend height is a literal length value to match, not a rank inside a length ordering.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, "legend must be non-empty"

    for _x, cap, k in legend:
        segs = _vertical_sandwiches(g, cap)
        assert segs, f"legend cap {cap} must have matching vertical bars"
        matches = [s for s in segs if (s[1] - s[0] - 1) == k]
        assert matches, f"legend length {k} must match at least one bar for cap {cap}"
        for top_r, bot_r, col, _fill in matches:
            out[top_r + 1 : bot_r, col] = cap

    return out


# --------------- generator (O2) ---------------

# --- Configuration ---
GRID_H = 10                  # fixed grid height
WIDTH_MIN = 10               # minimum grid width
WIDTH_MAX = 20               # maximum grid width (inclusive)
MAX_GROUPS = 4               # max number of cap-color groups
MAX_TOTAL_BARS = 8           # reduced to keep separate legend columns within width 20
BARS_PER_GROUP = (2, 3)      # allowed bars-per-group counts
MAX_FILL_LEN = 7             # max interior fill length of a bar
COLOR_RANGE = (1, 10)        # non-bg color ids (exclusive upper)
COL_SPACING = 2              # stride 2 => at least one blank column between bars
LEGEND_TO_BAR_GAP = 1        # one blank column between legend cluster and first bar column
MAX_RETRIES = 300            # retry budget for generate_input

_BOTTOM_ROW = GRID_H - 1     # row index where all bars end
_MAX_CAP_ROW = GRID_H - 2    # highest row a bar's top cap can occupy


def _bar_start_col(n_groups: int) -> int:
    """First bar column after the compact legend cluster and one blank separator."""
    return int(n_groups + LEGEND_TO_BAR_GAP)


def _min_width_for(n_groups: int, total_bars: int) -> int:
    """Minimum width needed for the separated legend+bars layout."""
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
    """Distribute bar columns across groups, keeping legend columns separate on the left."""
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
    Choose the legend height for one group.

    Preference order:
    1) a length greater than the group's bar count (original rank logic cannot use it)
    2) otherwise, any length whose descending rank differs from its literal value
    """
    count = len(lengths)
    ranked = sorted(lengths, reverse=True)
    rank_of = {length: idx + 1 for idx, length in enumerate(ranked)}

    preferred = [length for length in lengths if length > count]
    if not preferred:
        preferred = [length for length in lengths if rank_of[length] != length]
    assert preferred, "each group must have a non-degenerate absolute-length target"
    return int(preferred[int(rng.integers(len(preferred)))])


def _expected_output(
    inp: np.ndarray,
    caps: list[int],
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    target_lengths: list[int],
) -> np.ndarray:
    """Compute the expected output by recoloring bars whose fill length equals the legend height."""
    out = inp.copy()
    for cap, cols, lengths, target_len in zip(caps, group_columns, group_lengths, target_lengths):
        for col, fill_len in zip(cols, lengths):
            if fill_len != target_len:
                continue
            top_r = _MAX_CAP_ROW - fill_len
            out[top_r + 1 : _BOTTOM_ROW, col] = cap
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch."""
    rng = np.random.default_rng()

    # --- structure choices ---
    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])

    # --- palette ---
    palette = rng.choice(np.arange(*COLOR_RANGE), size=2 * n_groups, replace=False)
    caps = [int(v) for v in palette[:n_groups]]
    fills = [int(v) for v in palette[n_groups:]]

    # --- layout ---
    total_bars = sum(counts)
    min_width = _min_width_for(n_groups, total_bars)
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    group_columns = _assign_columns(rng, n_groups, counts, width)

    # --- render legend + bars ---
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

    # --- oracle check ---
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
