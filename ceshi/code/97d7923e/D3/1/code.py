# Variant of 97d7923e
# Dimension: D3 — 变换操作：重着色规则 (transform)
# Change: 将匹配条的整个填充区域重着色为帽色(cap color)，即图例颜色覆盖填充色... → 擦除匹配条替代重着色

"""
Design analysis for the D3 variant
==================================

a) Concrete behavior and trace
- The input still contains a top-left legend made of 1-cell-wide vertical stacks. Each stack uses
  the same color as one bar group, and its height k still means "pick rank k after sorting bars in
  that color group by fill length from longest to shortest".
- The main field still contains 1-column vertical bars of the form cap / fill... / cap, bottom
  aligned to the last row. Bars in the same group share cap color and fill color.
- Original action: after selecting the rank-k bar, repaint its entire interior run to the cap
  color, producing one solid monochrome cap-colored column between the two endpoints.
- Variant action: after selecting the same rank-k bar, erase its interior run to background 0,
  leaving only the top and bottom cap pixels in place.
- Small trace: if a bar column is [2,5,5,5,2] and legend says color 2 with height 1, the original
  output becomes [2,2,2,2,2], while this variant becomes [2,0,0,0,2]. The selection stage is
  unchanged; only the final write target differs.

b) Complexity calibration
- The task keeps the same ranking mechanism, so the solver still has to parse the legend, group by
  cap color, and compare lengths within each group.
- Visual support remains strong because every group has 2-3 bars with distinct lengths, so the rank
  index matters and is backed by repeated structure.
- The erase effect is visually stronger than recolor: selected bars turn into separated cap pixels,
  which is hard to confuse with untouched bars.

c) Review of old assumptions against the variant
- `_vertical_sandwiches()` assumes input interiors are solid, non-cap, non-bg runs. That is still
  true in the input; only the output changes, so the helper remains valid.
- `_legend_entries()` depends only on the top-left legend geometry. D3 does not change legend shape.
- In `main()`, the original write `out[top_r + 1 : bot_r, col] = cap` relied on "selected bars are
  marked by cap recoloring". Under the variant this exact line would silently implement the wrong
  task, so it must become background erasure.
- The solver must keep reading candidates from the original input `g`, not the partially modified
  `out`, because erased interiors would no longer satisfy the non-bg interior assumption if later
  logic re-read them. The original code already uses `g`, and that remains the correct choice.

d) Placement legality
- Legend columns still double as one bar column per group. The generator must preserve at least one
  background cell between the legend block bottom and the bar's top cap; the existing constraint
  `shared_len <= MAX_FILL_LEN - rank` still guarantees this.
- Bars stay one column wide, mutually non-overlapping, and spaced by at least one blank column.
- All selected bars remain valid after erasure because only the interior cells are zeroed; the cap
  endpoints remain inside bounds and unchanged.

e) Visual observability
- Across 2-4 examples, a human can see that legend-marked targets are not recolored to their cap
  color anymore; instead their middles disappear while non-selected bars remain intact.
- No hidden numeric property is introduced. Ranking is still by visible length only.

f) Visual cleanliness
- The grid stays sparse and columnar. Erasing interior cells reduces clutter rather than increasing
  it, and the surviving cap endpoints remain easy to pair with the original bar positions.
- Because fill colors are distinct from cap colors and from background in the input, the before/after
  contrast remains clean.

g) Color-change inference
- The new transform is structurally signaled: every changed pixel becomes background, and only inside
  the selected bar. This avoids arbitrary color remapping.
- Color count remains controlled: each group uses one cap color and one fill color, with no cap/fill
  overlap, so the solver's cue is spatial selection rather than a large palette lookup.

Post-change solver assumptions checked against the generator
------------------------------------------------------------
- `legend` is non-empty: generator always places 1-4 legend stacks touching row 0.
- Every legend color has at least one matching bar group: generator creates one group per legend
  entry, using the same cap color.
- `1 <= k <= len(segs)`: generator samples rank from the group's bar count.
- Candidate sorting is unambiguous enough: lengths are distinct within each group, and tie-breakers
  by row/column remain harmless.
- Selected interiors are non-empty: fill lengths are always 1-7, so erasing always changes at least
  one visible cell.
"""

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
    Erase selected vertical bar interiors using legend-driven rank selection.

    Input structure:
    - bg = background. In the top-left corner, each legend entry is a single-color vertical stack
      touching row 0; its color K names one bar group and its height k is a rank index.
    - The rest of the grid contains one-column vertical bars whose top and bottom endpoints are cap
      color K and whose interiors are solid non-background fill colors.

    Rule:
    - For each legend entry (K, k), collect all bars capped by K.
    - Sort those bars by interior length from longest to shortest.
    - Take the rank-k bar and replace its entire interior run with background, keeping both cap
      endpoints unchanged.

    Key insight: The legend does not describe a bar size to paint; it chooses which bar to remove
    internally, so the visible effect is a punched-out middle rather than a recolored bar.
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
        top_r, bot_r, col, _fill = segs[k - 1]
        out[top_r + 1 : bot_r, col] = bg

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
    pool = [length for length in range(1, MAX_FILL_LEN + 1) if length != shared_len]
    rng.shuffle(pool)
    return [shared_len] + pool[: count - 1]


def _expected_output(
    inp: np.ndarray,
    caps: list[int],
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
) -> np.ndarray:
    """Compute the expected output by erasing the rank-k bar's fill to background."""
    out = inp.copy()
    bg = get_bg(inp)
    for _cap, cols, lengths, rank in zip(caps, group_columns, group_lengths, ranks):
        ranked = sorted(zip(cols, lengths), key=lambda item: -item[1])
        target_col, target_len = ranked[rank - 1]
        top_r = _MAX_CAP_ROW - target_len
        out[top_r + 1 : _BOTTOM_ROW, target_col] = bg
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
