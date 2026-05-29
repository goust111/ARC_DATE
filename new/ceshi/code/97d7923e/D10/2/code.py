# Variant of 97d7923e
# Dimension: D10 — 颜色约束与调色板 (meta)
# Change: 帽色和填充色全部互不相同，帽色 != 填充色 != 背景色；同组内所有条共享相同的填充色... → 图例颜色与帽色不同，通过位置对应关系映射

import itertools

import numpy as np

from arc_utils import colors_present, get_bg, neighbors4, paste


# Design analysis for the D10 variant
# ----------------------------------
# a) Concrete behavior:
#    Original task: each legend block used the same color as one cap-color group, so
#    solver logic could jump directly from legend color K to bars capped by K.
#    Variant: the legend still sits in the top-left and each entry height still gives
#    the rank index, but the legend color L is decoupled from the bar cap color C.
#    The mapping is positional: the 1st legend entry from the left controls the 1st
#    cap-color group encountered from the left, the 2nd controls the 2nd group, etc.
#    Example trace:
#    - Original: legend entries [(x=0, color=6, k=2), (x=2, color=3, k=1)] directly
#      target cap colors 6 then 3.
#    - Variant: legend entries [(0, color=8, k=2), (2, color=4, k=1)] and cap groups
#      ordered by leftmost bar column [(0, cap=6), (2, cap=3)] mean 8 -> group cap 6,
#      4 -> group cap 3. The recolored bars use 6 and 3, not 8 and 4.
#    The changed step is the group lookup stage: color equality is replaced by
#    left-to-right order matching.
# b) Complexity calibration:
#    To keep the positional mapping visible, every group keeps a dedicated leftmost
#    column that also hosts its legend block above the bar with a guaranteed bg gap.
#    This produces repeated evidence inside one sample: each legend entry is vertically
#    aligned with its own bar group, but the legend color itself never appears on bars.
#    Because ARC has only 9 non-bg colors, we reduce MAX_GROUPS from 4 to 3 so legend,
#    cap, and fill palettes can remain fully disjoint without collisions.
# c) Implicit assumption review of original main():
#    - `_vertical_sandwiches(g, cap)` is still valid: bars are unchanged.
#    - `_legend_entries` still valid for finding top-left blocks, but the returned color
#      can no longer be passed into `_vertical_sandwiches`.
#    - `for _x, cap, k in legend:` was relying on legend_color == cap_color; this is
#      false now and must be replaced with zipping legend entries against ordered cap
#      groups.
#    - `out[...] = cap` remains safe because the recolor target is still the selected
#      group's cap color.
# d) Placement legality:
#    The generator keeps the legend above the group's leftmost bar column, with the
#    first bar length constrained so the top cap stays at least one bg row below the
#    legend block. Bar columns are unique and separated by at least one bg column, so
#    groups never overlap and their left-to-right order is unambiguous.
# e) Visual observability:
#    The solver-relevant novelty is visible because no legend color is reused as a cap.
#    A human must notice that the leftmost legend entry affects the leftmost cap group
#    despite the colors not matching. With 1-3 groups and 2-3 bars per group, examples
#    remain readable and still provide enough repeated rank-selection evidence.
# f) Visual neatness:
#    The scene stays in the same two-zone layout as the seed task: legend in the top
#    left, bars in the lower area. Using one legend column per group avoids adding new
#    cluttering markers elsewhere in the grid.
# g) Color inference:
#    Colors are partitioned into three disjoint roles per sample: legend-only, cap-only,
#    fill-only. This prevents accidental direct color clues and makes the positional
#    mapping the only consistent explanation for the recoloring.
# New main() assumptions and generator guarantees:
#    - Number of legend entries equals number of cap groups: generator constructs both
#      from the same `n_groups`.
#    - Left-to-right cap-group order is unique: each group has a unique leftmost column.
#    - Only true cap colors yield vertical sandwiches: fills are never used as caps and
#      legend colors appear only in top-left blocks.


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


def _ordered_cap_groups(g: np.ndarray, bg: int) -> list[tuple[int, int, list[tuple[int, int, int, int]]]]:
    """Return cap-color groups as (leftmost_col, cap_color, segments), ordered left to right."""
    groups: list[tuple[int, int, list[tuple[int, int, int, int]]]] = []
    for color in colors_present(g, exclude=[bg]):
        segs = _vertical_sandwiches(g, color)
        if not segs:
            continue
        leftmost = min(seg[2] for seg in segs)
        groups.append((leftmost, int(color), segs))
    groups.sort(key=lambda t: (t[0], t[1]))
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor vertical bars by legend-order rank selection.

    Input structure:
    - bg is the background. Top-left legend entries are monochrome vertical blocks; their
      heights encode rank indices, but their colors do not match any bar-cap color.
    - The main area contains vertical cap-fill-cap bars. Bars form groups by shared cap
      color, and each group's bars share one fill color.

    Rule:
    - Read legend entries from left to right, and independently order bar groups by the
      leftmost column where each cap-color group appears.
    - Pair the i-th legend entry with the i-th cap-color group, then rank that group's bars
      by interior length (longest = rank 1).
    - Recolor the rank-k bar's interior to the paired group's cap color.

    Key insight: The legend color is only a positional marker. Height still gives the rank,
    but the controlled bar group is determined by left-to-right correspondence, not by color.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    groups = _ordered_cap_groups(g, bg)
    assert legend, "legend must be non-empty"
    assert len(legend) == len(groups), "legend/group count mismatch"

    for (_legend_x, _legend_color, k), (_group_x, cap, segs) in zip(legend, groups):
        segs = sorted(segs, key=lambda s: (-(s[1] - s[0] - 1), s[0], s[2]))
        assert 1 <= k <= len(segs), f"legend rank {k} out of range for {len(segs)} bars"
        top_r, bot_r, col, _fill = segs[k - 1]
        out[top_r + 1 : bot_r, col] = cap

    return out


# --------------- generator (O2) ---------------

# --- Configuration ---
GRID_H = 10                  # fixed grid height
WIDTH_MIN = 10               # minimum grid width
WIDTH_MAX = 20               # maximum grid width (inclusive)
MAX_GROUPS = 3               # reduced from 4 so legend/cap/fill colors stay disjoint
MAX_TOTAL_BARS = 9           # still compatible with 2-3 bars per group
BARS_PER_GROUP = (2, 3)      # allowed bars-per-group counts
MAX_FILL_LEN = 7             # max interior fill length of a bar
COLOR_RANGE = (1, 10)        # non-bg color ids (exclusive upper)
COL_SPACING = 2              # minimum column stride between bars
MAX_RETRIES = 300            # retry budget for generate_input

_BOTTOM_ROW = GRID_H - 1    # row index where all bars end
_MAX_CAP_ROW = GRID_H - 2   # highest row a bar's top cap can occupy

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
    group doubles as that group's legend column, fixing the positional mapping.
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


def _choose_fill_lengths(rng, count: int, rank: int) -> list[int]:
    """Pick `count` distinct fill lengths for one group.

    The first length (legend column bar) is constrained so the bar fits
    below the legend block with a bg gap row.
    """
    max_legend_len = MAX_FILL_LEN - rank
    shared_len = int(rng.integers(1, max_legend_len + 1))
    pool = [l for l in range(1, MAX_FILL_LEN + 1) if l != shared_len]
    rng.shuffle(pool)
    return [shared_len] + pool[: count - 1]


def _expected_output(
    inp: np.ndarray,
    caps: list[int],
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
) -> np.ndarray:
    """Compute the expected output by recoloring the rank-k bar's fill to cap."""
    out = inp.copy()
    for cap, cols, lengths, rank in zip(caps, group_columns, group_lengths, ranks):
        ranked = sorted(zip(cols, lengths), key=lambda item: (-item[1], item[0]))
        target_col, target_len = ranked[rank - 1]
        top_r = _MAX_CAP_ROW - target_len
        out[top_r + 1 : _BOTTOM_ROW, target_col] = cap
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, c + 1)) for c in counts]

    palette = rng.choice(np.arange(*COLOR_RANGE), size=3 * n_groups, replace=False)
    caps = [int(v) for v in palette[:n_groups]]
    fills = [int(v) for v in palette[n_groups : 2 * n_groups]]
    legend_colors = [int(v) for v in palette[2 * n_groups :]]

    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    group_columns = _assign_columns(rng, n_groups, counts, width)

    grid = np.zeros((GRID_H, width), dtype=int)
    group_lengths: list[list[int]] = []
    for legend_color, cap, fill, count, rank, cols in zip(legend_colors, caps, fills, counts, ranks, group_columns):
        paste(grid, np.full((rank, 1), legend_color, dtype=int), 0, cols[0])

        lengths = _choose_fill_lengths(rng, count, rank)
        group_lengths.append(lengths)

        for col, fill_len in zip(cols, lengths):
            paste(grid, _build_bar(cap, fill, fill_len), _MAX_CAP_ROW - fill_len, col)

    expected = _expected_output(grid, caps, group_columns, group_lengths, ranks)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
