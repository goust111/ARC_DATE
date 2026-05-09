# Variant of 97d7923e
# Dimension: D3 — 变换操作：重着色规则 (transform)
# Change: 将匹配条的整个填充区域重着色为帽色(cap color)，即图例颜色覆盖填充色... → 仅重着色填充区域的部分像素

# Design analysis for this D3 variant
# ----------------------------------
# concrete_behavior:
#   keep the original scene layout: top-left legend columns encode one cap color K
#   and a visible height k, while the main field contains vertical cap/fill/cap bars.
#   The original task used k only as a rank selector: choose the k-th longest bar in
#   cap-color group K and recolor its whole fill.
#   The variant keeps the same selection step, but changes the paint extent:
#   recolor only the first k fill pixels of the chosen bar, counted from the top.
#   Example with cap K and lengths [6, 4, 2]:
#   - if the legend height is k=2, the original output repaints all 4 fill cells of
#     the 2nd-longest bar.
#   - the variant repaints only the top 2 fill cells of that bar, leaving the lower
#     2 fill cells unchanged.
# complexity_calibration:
#   k still takes only a few visible values (1..3), matching the original legend
#   heights and bar counts, so the rule is learnable from 2-4 examples.
#   To keep the D3 change observable, the generator forces the selected bar to have
#   fill_len > k, so every target bar shows both painted and unpainted fill pixels.
# hidden_assumptions_from_original_main:
#   _vertical_sandwiches still works unchanged because inputs still contain solid
#   fill runs; only outputs gain mixed cap/fill columns.
#   Ranking must still be computed from the untouched input grid g, not from out,
#   otherwise earlier recolors could create extra cap occurrences and corrupt later
#   matching.
#   The original full-slice assignment out[top+1:bot] = cap is no longer valid;
#   it must become a bounded prefix assignment of exactly k pixels.
# placement_legality:
#   Legend and bars keep the same non-overlap logic: the legend occupies the top rows
#   of one column, and the bar in that column starts low enough to leave at least one
#   background cell between them.
#   Distinct bar lengths are preserved within each cap group, so ranking is unambiguous.
# observability_and_cleanliness:
#   Using a fixed direction ("from the top") avoids adding a second hidden variable.
#   Each chosen bar visibly looks like the top cap has grown downward by k cells.
#   The untouched lower tail keeps the output different from the original task and
#   makes the legend-height-to-paint-count relation readable.
# new_solver_assumptions:
#   Every legend rank k has at least k matching bars, as in the original task.
#   The selected bar has fill_len >= k; the generator strengthens this to fill_len > k
#   so the recolor is always partial rather than sometimes degenerating to full fill.

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
    Partially recolor selected vertical bars using legend-guided ranks.

    Input structure:
    - Background cells dominate the grid. In the top-left, each legend entry is a vertical
      monochrome stack of cap color K whose height k is visibly encoded by the number of cells.
    - Elsewhere, each relevant object is a one-column vertical bar with cap color K at both
      ends and a solid fill color between them; bars sharing K form a length-ranked group.

    Rule:
    - For each legend entry (K, k), gather all bars capped by K and sort them by fill length
      from longest to shortest.
    - Select the rank-k bar in that group.
    - Recolor only the first k fill pixels of that chosen bar, starting just below the top cap.

    Key insight:
    - The legend height does double duty: it chooses which bar is targeted within a cap-color
      group and also how many fill cells of that bar are repainted.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, 'legend must be non-empty'

    for _x, cap, k in legend:
        segs = _vertical_sandwiches(g, cap)
        assert segs, f'legend cap {cap} must have matching vertical bars'
        segs.sort(key=lambda s: (-(s[1] - s[0] - 1), s[0], s[2]))
        assert 1 <= k <= len(segs), f'legend rank {k} out of range for {len(segs)} bars'
        top_r, bot_r, col, _fill = segs[k - 1]
        fill_len = bot_r - top_r - 1
        assert k <= fill_len, f'legend height {k} exceeds target fill length {fill_len}'
        out[top_r + 1 : top_r + 1 + k, col] = cap

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
    """Pick distinct fill lengths for one group.

    Constraints:
    - The legend-column bar must fit below the legend with a blank row between them.
    - The rank-k target bar must have fill_len > k, so the repaint is visibly partial.
    """
    max_legend_len = MAX_FILL_LEN - rank
    attempts = 0
    while attempts < 500:
        attempts += 1
        legend_len = int(rng.integers(1, max_legend_len + 1))
        pool = [length for length in range(1, MAX_FILL_LEN + 1) if length != legend_len]
        rng.shuffle(pool)
        lengths = [legend_len] + pool[: count - 1]
        ranked = sorted(lengths, reverse=True)
        if ranked[rank - 1] > rank:
            return lengths
    raise RuntimeError(f'failed to choose visible partial lengths for count={count}, rank={rank}')


def _expected_output(
    inp: np.ndarray,
    caps: list[int],
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
) -> np.ndarray:
    """Compute the expected output by repainting only the top k fill cells."""
    out = inp.copy()
    for cap, cols, lengths, rank in zip(caps, group_columns, group_lengths, ranks):
        ranked = sorted(zip(cols, lengths), key=lambda item: -item[1])
        target_col, target_len = ranked[rank - 1]
        top_r = _MAX_CAP_ROW - target_len
        out[top_r + 1 : top_r + 1 + rank, target_col] = cap
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
