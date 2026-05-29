# Variant of 97d7923e
# Dimension: D7 — 网格形状与尺寸 (grid_shape)
# Change: 输出与输入网格尺寸相同(10行 x 10-20列)，仅修改匹配条的填充色... → 紧凑裁剪输出仅保留条区域

import itertools
import numpy as np

from arc_utils import crop_bbox, get_bg, neighbors4, paste

# Design analysis for the D7 variant
# a) Concrete behavior:
# - Input stays structurally identical to the seed task: a top-left legend made of vertical color
#   stacks, plus bottom-aligned cap-fill-cap bars scattered across the width.
# - Output changes in exactly one place: after the usual recoloring, we discard the full canvas and
#   return only the union bounding box of the bars. The legend is used for reasoning but disappears
#   from the answer.
# - Example trace: if the bars occupy columns 0, 4, 7 and their top caps begin on rows 3, 1, 2,
#   the original solver returned the whole 10xW board after recoloring one chosen bar per legend
#   entry. This variant still performs the same recoloring, but then returns rows 1:9 and cols 0:7,
#   so the empty top band and any right-side blank columns vanish.
# b) Complexity calibration:
# - Only D7 changes. The legend, ranking rule, tie-break order, and bar geometry are untouched.
# - The crop is a single rectangle around the whole bar field, not a per-column compaction. This
#   preserves inter-bar spacing and keeps the visual reasoning close to the seed task.
# c) Hidden assumptions audited from the original main():
# - Bar detection must continue to inspect the original input grid `g`, not the recolored `out`.
#   After recoloring, a selected bar no longer has a non-cap interior, so recomputing candidates on
#   `out` would silently delete the chosen bar from the candidate set.
# - The crop cannot be computed from all non-background pixels, because legend pixels sit above some
#   bar columns and would wrongly keep the top legend rows in the output.
# - Sorting by (-length, top_row, col) is preserved even though the generator keeps group lengths
#   distinct; the solver should not rely on that convenience.
# d) Placement legality:
# - The generator still guarantees that each legend stack sits above its shared-column bar with at
#   least one background row in between, so the bar bbox starts strictly below the legend.
# - Bars remain one column wide, non-overlapping, in-bounds, and bottom-aligned to the last row.
# e) Visual observability:
# - Every generated sample loses at least one top row in the output because the highest bar cap is
#   always below row 0. This makes the D7 change visible even when the width happens not to shrink.
# - Many samples also lose trailing blank columns, reinforcing that the output is a compact crop.
# f) Visual cleanliness:
# - Returning the bars' outer rectangle keeps the spacing pattern readable and avoids introducing a
#   second transformation like horizontal compression.
# g) Color inference:
# - No new color rule is introduced. The legend still encodes a rank, and the selected bar still
#   changes to the legend/cap color before cropping.


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


def _bars_bbox(segs: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    """Bounding box of all detected bars: (top_row, left_col, bottom_row, right_col)."""
    assert segs, "at least one bar is required for cropping"
    r0 = min(top_r for top_r, _bot_r, _col, _fill in segs)
    r1 = max(bot_r for _top_r, bot_r, _col, _fill in segs)
    c0 = min(col for _top_r, _bot_r, col, _fill in segs)
    c1 = max(col for _top_r, _bot_r, col, _fill in segs)
    return int(r0), int(c0), int(r1), int(c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor legend-selected vertical bars, then return only the compact bar field.

    Input structure:
    - bg = background. Top-left: monochrome blocks touching row 0; each block uses cap color K
      and its height k encodes a rank.
    - Elsewhere: one-column vertical bars with color K at both ends and a solid non-bg fill color
      in between; bars sharing K form a group.

    Rule:
    - For each legend entry (K, k), find all bars capped by K.
    - Rank those bars by interior length (longest = rank 1) and recolor the rank-k bar's interior
      to K.
    - After all recoloring, crop the output to the minimal bounding box that contains the bars only,
      excluding the legend area and any outer blank margin.

    Key insight: The legend still specifies which bars to recolor, but the final answer is not the
    full scene anymore; it is the tight rectangular crop of the transformed bar region.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, "legend must be non-empty"

    all_bar_segs: list[tuple[int, int, int, int]] = []
    segs_by_cap: dict[int, list[tuple[int, int, int, int]]] = {}

    for _x, cap, k in legend:
        segs = segs_by_cap.get(cap)
        if segs is None:
            segs = _vertical_sandwiches(g, cap)
            assert segs, f"legend cap {cap} must have matching vertical bars"
            segs_by_cap[cap] = segs
            all_bar_segs.extend(segs)
        ranked = sorted(segs, key=lambda s: (-(s[1] - s[0] - 1), s[0], s[2]))
        assert 1 <= k <= len(ranked), f"legend rank {k} out of range for {len(ranked)} bars"
        top_r, bot_r, col, _fill = ranked[k - 1]
        out[top_r + 1 : bot_r, col] = cap

    return crop_bbox(out, _bars_bbox(all_bar_segs))


# --------------- generator (O2) ---------------

# --- Configuration ---
GRID_H = 10                  # fixed grid height
WIDTH_MIN = 10               # minimum grid width
WIDTH_MAX = 20               # maximum grid width (inclusive)
MAX_GROUPS = 4               # max number of cap-color groups
MAX_TOTAL_BARS = 10          # max bars across all groups
BARS_PER_GROUP = (2, 3)      # allowed bars-per-group counts
MAX_FILL_LEN = 7             # max interior fill length of a bar
COLOR_RANGE = (1, 10)        # non-bg color ids (exclusive upper)
COL_SPACING = 2              # minimum column stride between bars
MAX_RETRIES = 300            # retry budget for generate_input

_BOTTOM_ROW = GRID_H - 1     # row index where all bars end
_MAX_CAP_ROW = GRID_H - 2    # highest row a bar's top cap can occupy

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

    # Extra spacing goes only to inter-group gaps (after first n_groups columns).
    inter_gap_bonus = np.zeros(total_bars - 1, dtype=int)
    if total_bars > n_groups:
        inter_gap_bonus[n_groups - 1 :] = extra_parts[:-1]

    occupied = [0]
    for idx in range(total_bars - 1):
        occupied.append(occupied[-1] + COL_SPACING + int(inter_gap_bonus[idx]))

    legend_cols = occupied[:n_groups]
    extra_cols = occupied[n_groups:]

    # Assign non-legend columns to groups randomly.
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
    max_legend_len = MAX_FILL_LEN - rank  # legend occupies `rank` top rows
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
    """Compute the expected cropped output for the variant."""
    out = inp.copy()
    for cap, cols, lengths, rank in zip(caps, group_columns, group_lengths, ranks):
        ranked = sorted(zip(cols, lengths), key=lambda item: (-item[1], item[0]))
        target_col, target_len = ranked[rank - 1]
        top_r = _MAX_CAP_ROW - target_len
        out[top_r + 1 : _BOTTOM_ROW, target_col] = cap

    all_cols = [col for cols in group_columns for col in cols]
    all_top_rows = [_MAX_CAP_ROW - fill_len for lengths in group_lengths for fill_len in lengths]
    bbox = (min(all_top_rows), min(all_cols), _BOTTOM_ROW, max(all_cols))
    return crop_bbox(out, bbox)


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
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    group_columns = _assign_columns(rng, n_groups, counts, width)

    # --- render legend + bars ---
    grid = np.zeros((GRID_H, width), dtype=int)
    group_lengths: list[list[int]] = []
    for cap, fill, count, rank, cols in zip(caps, fills, counts, ranks, group_columns):
        paste(grid, np.full((rank, 1), cap, dtype=int), 0, cols[0])  # legend

        lengths = _choose_fill_lengths(rng, count, rank)
        group_lengths.append(lengths)

        for col, fill_len in zip(cols, lengths):
            paste(grid, _build_bar(cap, fill, fill_len), _MAX_CAP_ROW - fill_len, col)

    # --- oracle check ---
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
