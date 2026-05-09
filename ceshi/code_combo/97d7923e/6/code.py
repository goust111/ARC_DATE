# Combo variant of 97d7923e (index 6)
# combo_spec: D3/0+D11/0
# Name: 部分重着色+未选中条擦除

# Design analysis for this combo variant
# -------------------------------------
# concrete_behavior:
#   The input layout stays close to the source task: the top-left legend contains
#   one-column stacks of cap color K, and the visible stack height k is still the
#   group-specific rank selector. Each color group in the main field contains 2-3
#   bottom-aligned cap/fill/cap bars with distinct fill lengths.
#   The source task recolored the entire fill of the rank-k bar and left the rest
#   untouched. This combo changes two linked steps at once:
#   1. only the first k fill cells of the selected bar are recolored, counted from
#      the top cap downward;
#   2. every non-selected sibling bar in the same cap-color group is erased to bg.
#   Example: one group has fill lengths [6, 4, 2] and legend height k=2.
#   - source output: the length-4 bar becomes fully cap-colored, while lengths 6
#     and 2 stay visible.
#   - combo output: only the top 2 fill cells of the length-4 bar switch to cap;
#     its lower 2 fill cells stay in the original fill color; the length-6 and
#     length-2 bars disappear entirely.
# complexity_calibration:
#   k still comes from a tiny visible range tied to bars-per-group, so the rule is
#   learnable from a few examples. To keep the fusion visible, the generator forces
#   the selected bar to satisfy fill_len > k, guaranteeing a mixed kept bar with a
#   recolored prefix and an untouched tail. Because each group has at least 2 bars,
#   each sample also shows at least one erased bar.
# hidden_assumption_audit:
#   `_legend_entries` still relies on the legend being the leftmost top-row cluster;
#   the generator preserves that structure unchanged.
#   `_vertical_sandwiches` still reads the input grid only, where every bar remains
#   a clean cap/fill/cap column with a solid interior. This is important because
#   the output now contains both partial recolors and erased bars, which would break
#   bar detection if ranking were recomputed from `out`.
#   The source assignment `out[top+1:bot] = cap` is no longer valid: it would lose
#   the untouched tail required by D3. Likewise, modifying only the selected bar is
#   no longer sufficient: D11 requires blanking the non-selected spans, but only
#   within each detected bar span so a shared legend column is not harmed.
# placement_legality:
#   The constructor still places legend stacks above bars in the same column with at
#   least one background cell between them. Bars occupy distinct columns, stay
#   bottom-aligned inside the 10x10-20 frame, and use distinct fill lengths within
#   each group so ranking is unambiguous.
# observability_and_cleanliness:
#   Every surviving group contributes exactly one visible bar, which keeps the output
#   sparse and readable. That surviving bar is visually distinctive because its top
#   segment matches the cap color while its lower tail keeps the original fill. The
#   deleted siblings make the "keep one, erase the rest" rule obvious, and the top-
#   anchored recolor makes the legend height readable as a paint extent rather than
#   an arbitrary number.
# color_change_inference:
#   No hidden arithmetic on color ids is used. Colors only encode grouping, while
#   the legend height supplies both the rank choice and the recolor count.
# new_solver_assumptions_guaranteed_by_generator:
#   For every legend entry, there are at least k bars in that cap group, and the
#   selected bar has fill_len > k. The generator enforces both so the output always
#   contains the fusion's two signature effects: partial recolor and erasure.

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
    Keep only legend-selected bars, but recolor only the top part of the survivor.

    Input structure:
    - The background fills most cells. In the top-left, each legend entry is a vertical
      monochrome stack of cap color K whose visible height k encodes an index.
    - Elsewhere, each relevant object is a one-column cap/fill/cap bar: the top and
      bottom cells are color K, and the interior is a solid non-background fill color.
      Bars sharing cap color K form one group and have distinct fill lengths.

    Rule:
    - For each legend entry (K, k), collect all bars capped by K and sort them by fill
      length from longest to shortest.
    - Keep only the rank-k bar from that group and erase every other bar in the group to
      the background color.
    - On the kept bar, recolor only the first k fill cells just below the top cap to K;
      any remaining lower fill cells stay in their original fill color.

    Key insight:
    - The legend height controls two coupled decisions at once: it selects which ranked
      bar survives within its color group, and it also determines how deep the survivor's
      recolored prefix extends.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, "legend must be non-empty"

    for _x, cap, k in legend:
        segs = _vertical_sandwiches(g, cap)
        assert segs, f"legend cap {cap} must have matching vertical bars"
        segs.sort(key=lambda s: (-(s[1] - s[0] - 1), s[0], s[2]))
        assert 1 <= k <= len(segs), f"legend rank {k} out of range for {len(segs)} bars"

        for idx, (top_r, bot_r, col, _fill) in enumerate(segs):
            fill_len = bot_r - top_r - 1
            if idx == k - 1:
                assert k < fill_len, f"legend height {k} must be smaller than target fill length {fill_len}"
                out[top_r + 1 : top_r + 1 + k, col] = cap
            else:
                out[top_r : bot_r + 1, col] = bg

    return out


# --------------- generator (O2) ---------------

# --- Configuration ---
GRID_H = 10
WIDTH_MIN = 10
WIDTH_MAX = 20
MAX_GROUPS = 4
MAX_TOTAL_BARS = 10
BARS_PER_GROUP = (2, 3)
MAX_FILL_LEN = 7
COLOR_RANGE = (1, 10)
COL_SPACING = 2
MAX_RETRIES = 300

_BOTTOM_ROW = GRID_H - 1
_MAX_CAP_ROW = GRID_H - 2

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
        group_columns[owner].append(col)

    return group_columns


def _choose_fill_lengths(rng, count: int, rank: int) -> list[int]:
    """Pick distinct fill lengths with a visibly partial selected target."""
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
    raise RuntimeError(f"failed to choose visible partial lengths for count={count}, rank={rank}")


def _expected_output(
    inp: np.ndarray,
    caps: list[int],
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
) -> np.ndarray:
    """Compute the expected output for partial-keep plus erase-the-rest."""
    out = inp.copy()
    bg = get_bg(inp)
    for cap, cols, lengths, rank in zip(caps, group_columns, group_lengths, ranks):
        ranked = sorted(zip(cols, lengths), key=lambda item: (-item[1], item[0]))
        target_col, target_len = ranked[rank - 1]
        target_top = _MAX_CAP_ROW - target_len
        out[target_top + 1 : target_top + 1 + rank, target_col] = cap
        for col, fill_len in zip(cols, lengths):
            if col == target_col and fill_len == target_len:
                continue
            top_r = _MAX_CAP_ROW - fill_len
            out[top_r : _BOTTOM_ROW + 1, col] = bg
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, count + 1)) for count in counts]

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
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
