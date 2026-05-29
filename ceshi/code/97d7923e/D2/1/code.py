# Variant of 97d7923e
# Dimension: D2 — 匹配条件：排名与图例计数的对应 (match_cond)
# Change: s.length_rank_in_group == d.block_count — 图例条目的色块堆叠数k对应组内按长度... → 图例宽度编码替代高度编码

# Design analysis for the D2 variant
# ----------------------------------
# spec_variant:
#   core_change:
#     original: each legend entry was a vertical stack in the top-left; its height k
#       selected the rank-k bar within the cap-color group.
#     variant: each legend entry is now a one-row horizontal strip anchored at col 0;
#       its width k selects the rank-k bar within the cap-color group.
#   concrete_layout:
#     - Grid stays 10 x 10-20 with bg=0.
#     - Legend rows occupy the first n_groups rows. Row i contains a single horizontal
#       strip of color K_i starting at col 0 and extending for k_i cells.
#     - Bars remain vertical cap/fill/cap columns, bottom-aligned at row 9, grouped by
#       cap color and ranked by interior fill length descending.
#   behavior_trace:
#     - Original flow at the legend step: read a color component touching row 0, measure
#       its vertical extent, interpret that height as the rank index.
#     - Variant flow: read a color strip touching col 0, measure its horizontal extent,
#       interpret that width as the rank index.
#     - Example: if a red legend marker spans cells (row 1, cols 0..2), the solver must
#       choose the 3rd longest red-capped bar. Under the old logic its height would be 1,
#       so it would incorrectly choose the longest red bar instead.
#   complexity_calibration:
#     - Rank values stay small (1-3) because groups still have 2-3 bars, preserving the
#       original problem's search space.
#     - Generator forbids the degenerate all-rank-1 case so the width-based cue is always
#       visible in each sample and cannot collapse back to the original behavior.
#   hidden_assumptions_review:
#     - Old `_legend_entries` assumed legend components touched row 0 and encoded rank by
#       height. That fails here because every legend row is only one cell tall.
#     - The main bar-ranking code does not depend on legend orientation and remains valid.
#     - Any solver that still measures legend height would always read k=1, so width must
#       be parsed explicitly.
#   placement_legality:
#     - Legend strips occupy only columns `< rank_i` on row i.
#     - For any bar placed in a column touched by the legend, generator caps its maximum
#       fill length so the bar starts at least one blank row below the lowest legend cell
#       in that column. This prevents overlap and keeps legend/bar separation visible.
#     - Bars keep distinct columns and minimum stride `COL_SPACING`.
#   visual_observability:
#     - The top-left marker now changes along the horizontal axis only; all legend entries
#       have height 1, so the new rule is visually different from the original.
#     - Colors still identify groups, while width alone determines which bar is recolored.
#   cleanliness_and_color_inference:
#     - Legend uses one strip per row with no extra ornaments, so the left margin stays
#       readable.
#     - Fill colors remain group-consistent and distinct from cap colors, preserving the
#       same structural cue for which bars belong to which legend color.
#   main_new_assumptions:
#     - The legend is a top-left stack of consecutive non-bg rows starting at row 0.
#     - Each legend entry is a single horizontal run beginning at col 0.
#     - Generator guarantees those assumptions and also guarantees at least one row has
#       width > 1 so the variant behavior is exercised.

import itertools

import numpy as np

from arc_utils import get_bg, paste


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
    Legend is in the top-left: consecutive non-bg rows starting at row 0, each entry being
    a horizontal strip anchored at col 0.

    Returns entries as (y_pos, legend_color, block_count), sorted by y_pos.
    block_count is the horizontal width of that row's monochrome strip.
    """
    h, w = g.shape
    rows: list[int] = []
    for r in range(h):
        if int(g[r, 0]) == bg:
            break
        rows.append(r)
    assert rows, 'legend column must be non-empty'

    entries: list[tuple[int, int, int]] = []
    for r in rows:
        col = int(g[r, 0])
        c = 0
        while c < w and int(g[r, c]) == col:
            c += 1
        block_count = c
        entries.append((r, col, int(block_count)))

    return entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor vertical bars by legend-driven rank selection.

    Input structure:
    - bg = background. Top-left: one-row monochrome strips touching col 0; each has color K
      and width k.
    - Main grid: vertical bars capped top and bottom by a single color, with solid-color
      interiors.

    Rule:
    - For each legend entry (K, k), find vertical bars capped by K.
    - Rank bars by interior length (longest = rank 1).
    - Recolor the rank-k bar's interior to K.

    Key insight: Legend width is a rank index into length-sorted bars; the strip is not a
    literal bar preview and its height carries no information.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, 'legend must be non-empty'

    for _y, cap, k in legend:
        segs = _vertical_sandwiches(g, cap)
        assert segs, f'legend cap {cap} must have matching vertical bars'
        segs.sort(key=lambda s: (-(s[1] - s[0] - 1), s[0], s[2]))
        assert 1 <= k <= len(segs), f'legend rank {k} out of range for {len(segs)} bars'
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


def _assign_columns(rng, counts: list[int], width: int) -> list[list[int]]:
    """Distribute bar columns with spacing >= `COL_SPACING`, then assign them to groups."""
    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    extra_width = width - min_width
    extra_parts = rng.multinomial(extra_width, np.full(total_bars + 1, 1.0 / (total_bars + 1)))

    left_pad = int(extra_parts[0])
    inter_gap_bonus = [int(v) for v in extra_parts[1:total_bars]]

    occupied = [left_pad]
    for bonus in inter_gap_bonus:
        occupied.append(occupied[-1] + COL_SPACING + bonus)

    owners = [i for i, count in enumerate(counts) for _ in range(count)]
    rng.shuffle(owners)

    group_columns = [[] for _ in counts]
    for col, owner in zip(occupied, owners):
        group_columns[owner].append(int(col))
    for cols in group_columns:
        cols.sort()

    return group_columns


def _column_max_fill_len(ranks: list[int], col: int) -> int:
    """
    Legend occupies (row r, cols 0..ranks[r]-1). Bars using a covered column must start at
    least one blank row below the lowest legend cell in that column.
    """
    covered_rows = [r for r, width in enumerate(ranks) if col < width]
    if not covered_rows:
        return MAX_FILL_LEN
    lowest_legend_row = max(covered_rows)
    return min(MAX_FILL_LEN, 6 - lowest_legend_row)


def _choose_fill_lengths(
    rng,
    cols: list[int],
    ranks: list[int],
    row_idx: int,
    rank: int,
) -> list[int] | None:
    """Pick distinct fill lengths for one group that respect the legend clearance limits."""
    max_lens = [_column_max_fill_len(ranks, c) for c in cols]
    for i, col in enumerate(cols):
        if col == rank:
            # Avoid extending the legend strip by placing a same-color top cap flush against it.
            max_lens[i] = min(max_lens[i], 7 - row_idx)
    count = len(cols)

    valid: list[tuple[int, ...]] = []
    for perm in itertools.permutations(range(1, MAX_FILL_LEN + 1), count):
        if len(set(perm)) != count:
            continue
        if all(length <= max_len for length, max_len in zip(perm, max_lens)):
            valid.append(tuple(int(v) for v in perm))

    if not valid:
        return None
    choice = valid[int(rng.integers(len(valid)))]
    return list(choice)


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

    # --- structure choices ---
    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])

    while True:
        ranks = [int(rng.integers(1, c + 1)) for c in counts]
        if any(rank > 1 for rank in ranks):
            break

    # --- palette ---
    palette = rng.choice(np.arange(*COLOR_RANGE), size=2 * n_groups, replace=False)
    caps = [int(v) for v in palette[:n_groups]]
    fills = [int(v) for v in palette[n_groups:]]

    # --- layout ---
    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    group_columns = _assign_columns(rng, counts, width)

    # --- choose lengths under legend-clearance constraints ---
    group_lengths: list[list[int]] = []
    for row_idx, (cols, rank) in enumerate(zip(group_columns, ranks)):
        lengths = _choose_fill_lengths(rng, cols, ranks, row_idx, rank)
        if lengths is None:
            return None
        group_lengths.append(lengths)

    # --- render legend + bars ---
    grid = np.zeros((GRID_H, width), dtype=int)
    for row_idx, cap, rank in zip(range(n_groups), caps, ranks):
        paste(grid, np.full((1, rank), cap, dtype=int), row_idx, 0)

    for cap, fill, cols, lengths in zip(caps, fills, group_columns, group_lengths):
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
    raise RuntimeError(f'failed to generate a valid input after {MAX_RETRIES} attempts')
