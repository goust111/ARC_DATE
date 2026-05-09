# Combo variant of 97d7923e (index 22)
# combo_spec: D1/0+D2/0+D6/0
# Name: 填充色分组+升序排名+列位置排序

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace
#    - A legend entry now uses fill color F, not cap color K. Suppose one legend block is color 5
#      and height 2, and the fill-5 bars sit at columns [1, 7, 11] with caps [2, 3, 2].
#    - Original task: color 5 would have meant "look for cap-5 bars", then rank those by length
#      descending and repaint the chosen interior to 5.
#    - Combo task: color 5 means "look for fill-5 bars", then order them by column left-to-right,
#      take rank 2 (the middle column here), and repaint that bar's interior to its own cap 3.
#    - The three modified links in the reasoning chain are therefore coupled:
#      legend color -> fill-group match, rank -> left-to-right ascending index, repaint -> matched cap.
#
# b) Complexity calibration
#    - If each fill group had a unique cap, fill-grouping would look equivalent to cap-grouping.
#      To avoid that, the generator uses a small shared cap pool across multiple fill groups.
#    - If left-to-right rank always picked the same column as length-based rank, D6 would vanish.
#      The generator rejects such samples and keeps length differences as visible distractors.
#    - Counts stay small (2-3 bars per group, rank 1-3) so humans can inspect examples directly.
#
# c) Implicit-assumption audit of the original solver
#    - `_vertical_sandwiches(g, cap)` assumed legend colors were cap colors; that breaks under D1.
#      The combo solver therefore extracts all bars first, then filters by fill color.
#    - The original sort key assumed length was the only rank key. D6 replaces that with column.
#    - The original repaint step wrote the legend color into the bar. Under D1 that would repaint
#      fill to itself, so the combo solver must instead use the selected bar's own cap color.
#    - `_legend_entries()` still only needs to read (x, color, height), so it remains valid.
#
# d) Placement legality
#    - Grid shape stays 10 x (10..20); every bar is still a single bottom-aligned cap/fill/cap column.
#    - The leftmost bar of each fill group shares its column with the legend and is length-limited so
#      the legend and bar remain separated by at least one background row.
#    - Columns are unique and spaced by at least one background column, so bars never overlap.
#
# e) Visual observability
#    - Fill colors are the only reliable group cue because cap colors are intentionally reused across
#      groups. Humans must notice repeated interiors rather than repeated endpoints.
#    - Rank is determined by visible x-position, not by hidden ids. Leftmost = rank 1 is readable.
#    - Length remains varied but irrelevant, creating a clean distractor that distinguishes this combo
#      from the original length-based task.
#
# f) Visual cleanliness
#    - The layout stays sparse: thin legend stacks in the upper-left and thin vertical bars elsewhere.
#    - No extra markers or overlays are added; each object keeps the same simple one-column geometry.
#
# g) Color-change inferability
#    - Every legend color already appears inside multiple bars, so the legend-to-group mapping is
#      structurally signaled. The output color comes from the chosen bar's visible caps on that column.
#
# h) Anti-degeneration policy
#    - The generator enforces two signals: some cap colors must be shared across fill groups, and at
#      least one group's left-to-right target must differ from its length-descending target.
#    - This prevents accidental collapse back to cap-grouping or length-ranking behavior.
#
# i) New solver assumptions and generator guarantees
#    - Each legend height k satisfies 1 <= k <= number of bars in that fill group.
#    - Each group's columns are unique, so left-to-right order is strict.
#    - Fill colors are distinct across groups and disjoint from cap colors, so legend parsing and bar
#      matching stay unambiguous.

from pathlib import Path
import itertools
import sys

import numpy as np

for _parent in Path(__file__).resolve().parents:
    if (_parent / "arc_utils.py").exists():
        _parent_str = str(_parent)
        if _parent_str not in sys.path:
            sys.path.append(_parent_str)
        break

from arc_utils import get_bg, neighbors4, paste


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
    Recolor one vertical bar per fill-color group using a legend-defined spatial rank.

    Input structure:
    - The top-left legend is a set of one-column monochrome stacks touching row 0; each stack has
      fill color F and height k.
    - Elsewhere, the grid contains bottom-aligned vertical cap/fill/cap bars. Bars in one logical
      group share the same interior fill color F, while cap colors may repeat across groups.

    Rule:
    - For each legend entry (F, k), collect all bars whose interior fill color is F.
    - Order those bars by column position from left to right, with the leftmost bar as rank 1.
    - Recolor the interior of the rank-k bar to that selected bar's own cap color.

    Key insight:
    - The legend color names a fill-color group, not a cap-color group, and the legend height is an
      ordinal into left-to-right order rather than any kind of length comparison.
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
        segs.sort(key=lambda s: (s[2], s[0], s[1]))
        assert 1 <= k <= len(segs), f"legend rank {k} out of range for {len(segs)} bars"
        top_r, bot_r, col, cap, _fill = segs[k - 1]
        out[top_r + 1 : bot_r, col] = cap

    return out


# --------------- generator (combo variant) ---------------

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
        group_columns[owner].append(int(col))

    return [sorted(int(c) for c in cols) for cols in group_columns]


def _choose_fill_lengths(rng, count: int, rank: int) -> list[int]:
    """Pick distinct fill lengths; the leftmost bar must fit below its legend."""
    max_leftmost_len = MAX_FILL_LEN - rank
    leftmost_len = int(rng.integers(1, max_leftmost_len + 1))
    pool = [length for length in range(1, MAX_FILL_LEN + 1) if length != leftmost_len]
    rng.shuffle(pool)
    return [leftmost_len] + pool[: count - 1]


def _choose_bar_caps(rng, count: int, cap_pool: list[int]) -> list[int]:
    """Assign at least two distinct shared cap colors within one fill group."""
    chosen = [int(v) for v in rng.choice(np.array(cap_pool), size=2, replace=False)]
    caps = [chosen[i % 2] for i in range(count)]
    if count >= 3 and len(cap_pool) >= 3 and rng.random() < 0.5:
        caps[-1] = int(rng.choice(np.array(cap_pool)))
        if len(set(caps)) == 1:
            caps[-1] = chosen[1]
    rng.shuffle(caps)
    return caps


def _position_target(cols: list[int], rank: int) -> int:
    """Column chosen by the combo rule: k-th leftmost within a fill group."""
    return sorted(int(c) for c in cols)[rank - 1]


def _length_desc_target(cols: list[int], lengths: list[int], rank: int) -> int:
    """Reference column under the original length-descending ranking inside the same fill group."""
    ranked = sorted(zip(cols, lengths), key=lambda item: (-int(item[1]), int(item[0])))
    return int(ranked[rank - 1][0])


def _fill_grouping_is_visible(group_caps: list[list[int]]) -> bool:
    """True iff every fill group uses caps that also appear in another fill group."""
    cap_to_groups: dict[int, set[int]] = {}
    for gi, caps in enumerate(group_caps):
        for cap in set(int(v) for v in caps):
            cap_to_groups.setdefault(cap, set()).add(gi)
    return all(
        len(set(int(v) for v in caps)) >= 2
        and any(len(cap_to_groups[int(cap)]) >= 2 for cap in set(int(v) for v in caps))
        for caps in group_caps
    )


def _position_signal_count(group_columns: list[list[int]], group_lengths: list[list[int]], ranks: list[int]) -> int:
    """How many groups choose a different target under position-order vs length-order."""
    return sum(
        _position_target(cols, rank) != _length_desc_target(cols, lengths, rank)
        for cols, lengths, rank in zip(group_columns, group_lengths, ranks)
    )


def _expected_output(
    inp: np.ndarray,
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    group_caps: list[list[int]],
    ranks: list[int],
) -> np.ndarray:
    """Compute the expected output by repainting the k-th leftmost bar to its own cap."""
    out = inp.copy()
    for cols, lengths, caps, rank in zip(group_columns, group_lengths, group_caps, ranks):
        target_col = _position_target(cols, rank)
        target_idx = cols.index(target_col)
        target_len = int(lengths[target_idx])
        target_cap = int(caps[target_idx])
        top_r = _MAX_CAP_ROW - target_len
        out[top_r + 1 : _BOTTOM_ROW, target_col] = target_cap
    return out


def _has_fusion_signal(group_columns: list[list[int]], group_lengths: list[list[int]], group_caps: list[list[int]], ranks: list[int]) -> bool:
    """Require both fill-group ambiguity and visible position-based ranking."""
    return _fill_grouping_is_visible(group_caps) and _position_signal_count(group_columns, group_lengths, ranks) >= 1


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid combo sample. Returns None on mismatch or degeneration."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(MIN_GROUPS, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, count + 1)) for count in counts]

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

    if not _has_fusion_signal(group_columns, group_lengths, group_caps, ranks):
        return None

    expected = _expected_output(grid, group_columns, group_lengths, group_caps, ranks)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
