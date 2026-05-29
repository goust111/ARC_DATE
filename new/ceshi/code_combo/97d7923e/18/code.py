# Combo variant of 97d7923e (index 18)
# combo_spec: D2/1+D3/0
# Name: 宽度编码排名+部分重着色
#
# Design analysis for the fused variant
# ------------------------------------
# concrete_layout_and_trace:
#   - Keep the original bar field: each group is a set of bottom-aligned vertical
#     cap/fill/cap columns sharing a cap color K and a fill color F, with distinct
#     fill lengths inside the group.
#   - Change the legend from vertical stacks to horizontal strips. The legend now
#     occupies consecutive rows starting at row 0; row i contains a monochrome strip
#     of color K_i anchored at col 0 and extending rightward for width w_i.
#   - Fused behavior for one group:
#       original: measure legend height k -> choose the k-th longest K-capped bar ->
#       recolor that whole fill.
#       combo: measure legend width w -> choose the w-th longest K-capped bar ->
#       recolor only the first w fill cells below that bar's top cap.
#   - Example: if a cyan legend row covers cols 0..2, then w=3. Among cyan-capped
#     bars sorted by fill length descending, select rank 3; if its fill length is 5,
#     repaint only the top 3 fill pixels, leaving the bottom 2 unchanged.
#   - The same visible number w therefore controls both selection and paint extent,
#     which is the intended cross-dimension coupling rather than two unrelated edits.
# complexity_calibration:
#   - Group sizes remain 2-3, so the encoded widths also stay in 1..3. That keeps
#     the hypothesis space small enough for ARC while still making width differences
#     visually obvious.
#   - The generator rejects the degenerate all-width-1 case so the width cue cannot
#     collapse back to the original "all legend rows are one cell tall and one cell wide"
#     reading.
#   - The generator also enforces target_fill_len > w, ensuring the output always shows
#     a partially repainted bar rather than occasionally matching the original full-fill
#     transform by accident.
# hidden_assumptions_review:
#   - Original `_legend_entries` depended on components touching row 0 and encoded rank
#     by vertical height. That is false here because each legend entry is only one row tall.
#   - The bar parser still assumes the input bars have solid fill runs and that ranking is
#     computed from the untouched input grid. Both remain valid and must stay tied to `g`,
#     not `out`, because partial recolors introduce extra cap-colored cells in the output.
#   - The old full-slice repaint `out[top+1:bot, col] = cap` is invalid. The fused rule
#     requires a bounded prefix slice of exactly w cells from the top of the chosen fill.
# placement_legality:
#   - Legend rows are consecutive from row 0 and each strip starts at col 0.
#   - Bars keep distinct columns with spacing >= 2.
#   - If a bar uses a column covered by one or more legend strips, its top cap is forced
#     to start at least one blank row below the lowest covering legend cell.
#   - If a bar sits immediately to the right of its own legend strip (`col == w`), its
#     top cap is additionally kept below that legend row so the strip does not visually
#     extend sideways into the bar cap.
# visual_observability_and_cleanliness:
#   - Width is the only varying legend dimension; height no longer carries rank data.
#   - The selected bar visibly grows downward from its top cap by exactly w cells, while
#     the remaining lower fill tail stays unchanged. That makes the second use of w legible.
#   - Colors still identify groups structurally, so there is no hidden dependence on color
#     ids or any non-visual ordering.
# generator_guarantees_for_new_main:
#   - Each legend row corresponds to exactly one cap color group.
#   - Each width w satisfies 1 <= w <= number of bars in that group.
#   - The rank-w target bar has fill length strictly greater than w.
#   - At least one legend width exceeds 1, so the fused width-reading behavior is exercised.

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
        entries.append((r, col, c))
    return entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Partially recolor ranked vertical bars using width-coded legend entries.

    Input structure:
    - Background cells fill most of the grid. In the top-left, consecutive rows form the
      legend; each row is a horizontal monochrome strip of cap color K whose width w is
      the only encoded quantity.
    - Elsewhere, each relevant object is a one-column vertical bar with cap color K at
      both ends and a solid fill color between them; bars sharing K form one length-ranked
      group.

    Rule:
    - For each legend entry (K, w), collect all K-capped bars and sort them by fill length
      from longest to shortest.
    - Select the rank-w bar in that group.
    - Recolor only the first w fill pixels of that chosen bar, starting just below its
      top cap.

    Key insight:
    - Legend width does double duty: the strip length chooses which bar within the group is
      targeted, and that same visible width also sets how many fill cells are repainted.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, 'legend must be non-empty'

    for _y, cap, width in legend:
        segs = _vertical_sandwiches(g, cap)
        assert segs, f'legend cap {cap} must have matching vertical bars'
        segs.sort(key=lambda s: (-(s[1] - s[0] - 1), s[0], s[2]))
        assert 1 <= width <= len(segs), f'legend width {width} out of range for {len(segs)} bars'
        top_r, bot_r, col, _fill = segs[width - 1]
        fill_len = bot_r - top_r - 1
        assert width <= fill_len, f'legend width {width} exceeds target fill length {fill_len}'
        out[top_r + 1 : top_r + 1 + width, col] = cap

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


def _column_max_fill_len(widths: list[int], col: int) -> int:
    """
    Legend occupies (row r, cols 0..widths[r]-1). Bars using a covered column must start
    at least one blank row below the lowest legend cell in that column.
    """
    covered_rows = [r for r, width in enumerate(widths) if col < width]
    if not covered_rows:
        return MAX_FILL_LEN
    lowest_legend_row = max(covered_rows)
    return min(MAX_FILL_LEN, 6 - lowest_legend_row)


def _choose_fill_lengths(
    rng,
    cols: list[int],
    widths: list[int],
    row_idx: int,
    width: int,
) -> list[int] | None:
    """
    Pick distinct fill lengths for one group that satisfy:
      - legend/bar separation for every occupied column
      - no visual extension of the row's own legend strip
      - the rank-width target bar has fill_len > width, so the repaint is partial
    """
    max_lens = [_column_max_fill_len(widths, c) for c in cols]
    for i, col in enumerate(cols):
        if col == width:
            max_lens[i] = min(max_lens[i], 7 - row_idx)

    count = len(cols)
    valid: list[tuple[int, ...]] = []
    for perm in itertools.permutations(range(1, MAX_FILL_LEN + 1), count):
        if not all(length <= max_len for length, max_len in zip(perm, max_lens)):
            continue
        ranked = sorted(perm, reverse=True)
        if ranked[width - 1] <= width:
            continue
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
    widths: list[int],
) -> np.ndarray:
    """Compute the expected output by repainting only the top `width` fill cells."""
    out = inp.copy()
    for cap, cols, lengths, width in zip(caps, group_columns, group_lengths, widths):
        ranked = sorted(zip(cols, lengths), key=lambda item: (-item[1], item[0]))
        target_col, target_len = ranked[width - 1]
        top_r = _MAX_CAP_ROW - target_len
        out[top_r + 1 : top_r + 1 + width, target_col] = cap
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])

    while True:
        widths = [int(rng.integers(1, count + 1)) for count in counts]
        if any(width > 1 for width in widths):
            break

    palette = rng.choice(np.arange(*COLOR_RANGE), size=2 * n_groups, replace=False)
    caps = [int(v) for v in palette[:n_groups]]
    fills = [int(v) for v in palette[n_groups:]]

    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    group_columns = _assign_columns(rng, counts, width)

    group_lengths: list[list[int]] = []
    for row_idx, (cols, legend_width) in enumerate(zip(group_columns, widths)):
        lengths = _choose_fill_lengths(rng, cols, widths, row_idx, legend_width)
        if lengths is None:
            return None
        group_lengths.append(lengths)

    grid = np.zeros((GRID_H, width), dtype=int)
    for row_idx, cap, legend_width in zip(range(n_groups), caps, widths):
        paste(grid, np.full((1, legend_width), cap, dtype=int), row_idx, 0)

    for cap, fill, cols, lengths in zip(caps, fills, group_columns, group_lengths):
        for col, fill_len in zip(cols, lengths):
            paste(grid, _build_bar(cap, fill, fill_len), _MAX_CAP_ROW - fill_len, col)

    expected = _expected_output(grid, caps, group_columns, group_lengths, widths)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f'failed to generate a valid input after {MAX_RETRIES} attempts')
