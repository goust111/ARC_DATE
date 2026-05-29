# Variant of 97d7923e
# Dimension: D11 — 轨迹/变换范围 (trajectory)
# Change: 默认无轨迹——仅修改被选中条的填充色，其余所有内容保持不变... → 图例在输出中被擦除

# Design analysis for the D11 variant
# a) Concrete behavior and trace:
#    - The input keeps the original structure: a top-left legend made of 1-column vertical
#      stacks, plus bottom-aligned cap/fill/cap bars elsewhere.
#    - The original solver only repaints one ranked bar per legend entry and leaves the
#      legend visible. The variant keeps the same bar selection, then clears the legend
#      pixels to background. Example: if a legend entry is color 2 with height 2, and its
#      rank-2 bar is recolored from fill 5 to cap 2, the variant output also turns the
#      two legend cells at rows 0-1 of that legend column into 0. The difference is not in
#      which bar gets selected, only in the transformation scope after the instruction is used.
# b) Complexity calibration:
#    - The variant adds exactly one new visible effect: instruction pixels disappear.
#      This is visually obvious because every sample still contains 1-4 legend stacks, so
#      the top-left area changes in a repeated, structured way rather than via isolated noise.
#    - No new object types, ranking schemes, or color remappings are introduced, so the task
#      does not become ambiguous or overloaded.
# c) Hidden assumptions in the original main():
#    - `_legend_entries(...)` assumes the legend is the leftmost top-row non-bg cluster.
#      This remains true; the generator still places legend columns first.
#    - `_vertical_sandwiches(...)` assumes bars are separated from legend pixels by at least
#      one background row when they share a column; otherwise the flood-filled legend could
#      merge into a bar cap. The generator must continue guaranteeing that gap.
#    - `out = g.copy()` was previously harmless because untouched legend pixels were meant to
#      survive. In this variant that assumption is false, so `main()` must explicitly erase
#      the detected legend pixels after recoloring.
# d) Placement legality:
#    - Legend columns are still a subset of the bar columns, and the legend-column bar length
#      is constrained so its top cap starts strictly below the legend with one bg row in between.
#    - Erasure only targets exact legend pixels, not the whole top-left bounding box, so shared
#      columns remain valid and the bar region is preserved.
# e) Visual observability:
#    - Across examples, the same left-top instruction area disappears while ranked bars change
#      color. A solver can infer that the legend acts as a consumed command rather than an
#      output object.
# f) Visual cleanliness:
#    - The output removes clutter instead of adding overlays. Because only exact legend pixels
#      are cleared, no extra collisions or hard-to-parse composites are introduced.
# g) Color-change inferability:
#    - The recoloring rule remains keyed by visible cap colors and visible bar lengths only.
#      The new disappearance behavior is position- and structure-based, not tied to hidden
#      color ids or arbitrary case splits.
# Additional review of new assumptions:
#    - `main()` now depends on recovering a precise legend mask. The generator guarantees each
#      legend entry is a contiguous monochrome component touching row 0 within the initial
#      leftmost legend cluster, so the mask is well-defined.
#    - `main()` erases the legend after recoloring, so processing order does not affect bar
#      selection: all ranking is still computed from the original input grid `g`.

import itertools

import numpy as np

from arc_utils import get_bg, neighbors4, paste


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


def _legend_entries_and_mask(g: np.ndarray, bg: int) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    """
    Legend is the leftmost top-row cluster of non-background pixels.

    Returns:
      - entries as (x_pos, legend_color, block_count), sorted by x_pos
      - a boolean mask marking the exact legend pixels to erase in the output
    """
    h, w = g.shape
    top_pos = [int(c) for c in np.flatnonzero(g[0, :] != bg)]
    assert top_pos, 'legend row must be non-empty'
    top_pos.sort()

    cols = [top_pos[0]]
    for c in top_pos[1:]:
        if c - cols[-1] <= 2:
            cols.append(c)
        else:
            break
    legend_max_c = max(cols)

    entries: list[tuple[int, int, int]] = []
    mask = np.zeros((h, w), dtype=bool)
    seen = np.zeros((h, w), dtype=bool)
    for c in cols:
        col = int(g[0, c])
        if col == bg or seen[0, c]:
            continue
        st = [(0, c)]
        seen[0, c] = True
        pix: list[tuple[int, int]] = []
        while st:
            r, cc = st.pop()
            pix.append((r, cc))
            for rr, ccc in neighbors4(r, cc):
                if 0 <= rr < h and 0 <= ccc <= legend_max_c and (not seen[rr, ccc]) and int(g[rr, ccc]) == col:
                    seen[rr, ccc] = True
                    st.append((rr, ccc))
        x_pos = min(cc for _, cc in pix)
        rs = [r for r, _ in pix]
        for r, cc in pix:
            mask[r, cc] = True
        block_count = max(rs) - min(rs) + 1
        entries.append((x_pos, col, int(block_count)))

    entries.sort(key=lambda t: t[0])
    return entries, mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor ranked vertical bars, then remove the legend that encoded the ranks.

    Input structure:
    - The background surrounds two regions: a top-left legend of single-color vertical stacks,
    - and a field of bottom-aligned vertical bars with matching cap colors and solid interiors.

    Rule:
    - Each legend stack has color K and height k. Among bars capped by K, sort by interior
    - length descending and recolor the rank-k bar's interior to K. After all such selections,
    - erase the legend pixels to background.

    Key insight: The top-left stacks are consumable instructions: their heights select bars by
    rank, and the stacks themselves should disappear once that instruction has been applied.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend, legend_mask = _legend_entries_and_mask(g, bg)
    assert legend, 'legend must be non-empty'

    for _x, cap, k in legend:
        segs = _vertical_sandwiches(g, cap)
        assert segs, f'legend cap {cap} must have matching vertical bars'
        segs.sort(key=lambda s: (-(s[1] - s[0] - 1), s[0], s[2]))
        assert 1 <= k <= len(segs), f'legend rank {k} out of range for {len(segs)} bars'
        top_r, bot_r, col, _fill = segs[k - 1]
        out[top_r + 1 : bot_r, col] = cap

    out[legend_mask] = bg
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
    """Compute the expected output by recoloring rank-k bars and erasing the legend."""
    out = inp.copy()
    bg = get_bg(inp)
    for cap, cols, lengths, rank in zip(caps, group_columns, group_lengths, ranks):
        ranked = sorted(zip(cols, lengths), key=lambda item: -item[1])
        target_col, target_len = ranked[rank - 1]
        top_r = _MAX_CAP_ROW - target_len
        out[top_r + 1 : _BOTTOM_ROW, target_col] = cap
        out[:rank, cols[0]] = bg
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, c + 1)) for c in counts]

    palette = rng.choice(np.arange(*COLOR_RANGE), size=2 * n_groups, replace=False)
    caps = [int(v) for v in palette[:n_groups]]
    fills = [int(v) for v in palette[n_groups:]]

    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    group_columns = _assign_columns(rng, n_groups, counts, width)

    grid = np.zeros((GRID_H, width), dtype=int)
    group_lengths: list[list[int]] = []
    for cap, fill, count, rank, cols in zip(caps, fills, counts, ranks, group_columns):
        paste(grid, np.full((rank, 1), cap, dtype=int), 0, cols[0])

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
    raise RuntimeError(f'failed to generate a valid input after {MAX_RETRIES} attempts')
