# Variant of 97d7923e
# Dimension: D11 — 轨迹/变换范围 (trajectory)
# Change: 默认无轨迹——仅修改被选中条的填充色，其余所有内容保持不变
#   → 被选中条的帽色也被重着色；这里实现为把整根被选中条改成其原填充色

from pathlib import Path
import sys
import itertools

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[5]
for _helper_dir in (_REPO_ROOT / "arc2_solver_select", _REPO_ROOT):
    _helper_dir_str = str(_helper_dir)
    if _helper_dir.is_dir() and _helper_dir_str not in sys.path:
        sys.path.append(_helper_dir_str)

from arc_utils import get_bg, neighbors4, paste


# Design analysis for the D11 variant
# a) Concrete behavior:
#    Input stays visually close to the source task: the top-left legend still gives
#    one column per cap color K, and its height k still selects rank k among the bars
#    whose endpoints are K. Each bar is still a 1-cell-wide vertical sandwich:
#    top cap K, solid fill F, bottom cap K. The changed behavior is in the output:
#    once the rank-k bar is selected, the transformation now spans the whole bar
#    rather than just its interior. To make this D11 expansion visible, the entire
#    selected bar is recolored to its original fill color F. Small trace:
#    original rule on K/F/K with fill length 4 -> only the 4 middle pixels become K;
#    variant rule -> all 6 pixels become F, so the two endpoint caps visibly change.
# b) Complexity calibration:
#    The task still has 1-4 groups and 2-3 bars per group, so every legend entry has
#    repeated support. Distinct lengths within a group keep the ranking unambiguous,
#    and the selected bar always has exactly two changed cap pixels, which makes the
#    variant easy to notice without adding extra symbols or hidden tie-breakers.
# c) Implicit assumptions from the original main():
#    - _legend_entries assumes the legend is the leftmost top-row cluster with gaps <= 1
#      background column; the generator preserves that.
#    - _vertical_sandwiches assumes bars are detected from consecutive cap-color
#      occurrences in a column and that the interior is one non-bg, non-cap color;
#      the generator keeps solid interiors and a bg gap below any legend block.
#    - The original write `out[top_r + 1 : bot_r, col] = cap` relied on caps being
#      intentionally immutable. That is the exact dependency that no longer holds.
#      The variant must instead use the selected bar's stored fill color and write to
#      `top_r : bot_r + 1` so the two endpoints are included.
# d) Placement legality:
#    Bars remain bottom-aligned, one column wide, and spaced at least one blank column
#    apart. The first bar column of each group can still share the legend column because
#    the generator constrains the bar to start at least one bg row below the legend.
#    That keeps the solver's sandwich parsing valid and prevents overlap.
# e) Visual observability:
#    The changed region is now exactly two additional pixels per selected bar, and those
#    pixels flip from cap color K to fill color F. This is directly visible in 2-4 ARC
#    examples because the selected bar loses its contrasting endcaps and becomes a solid
#    stripe, while all unselected bars keep the cap/fill/cap structure.
# f) Visual cleanliness:
#    No new objects are introduced. Legend, bars, spacing, and palette pressure stay the
#    same, so the grid remains sparse and readable.
# g) Color-change inferability:
#    Only colors already attached to the selected bar are used: grouping still comes from
#    cap color K, while the output color is the bar's own interior color F. There is no
#    hidden dependence on color ids, global counts, or background ordering.


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
    Select legend-ranked vertical bars and repaint the whole chosen bar.

    Input structure:
    - bg is the dominant background. At the top-left, each legend column is a solid stack
      of one color K, and its height k is the rank index for that color group.
    - The rest of the grid contains vertical 1-column bars with shape cap / fill* / cap:
      both endpoints are color K and the interior is a solid color F.

    Rule:
    - For each legend entry (K, k), gather every bar whose two endpoint pixels are K.
    - Sort those bars by interior height from longest to shortest.
    - Repaint the rank-k bar completely, including both endpoint caps, to that bar's
      original interior color F.

    Key insight:
    - The legend still identifies which bar to transform; the new twist is the transform
      range. Instead of only replacing the interior, the selected bar absorbs its own caps
      and turns into a single-color stripe.
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
        top_r, bot_r, col, fill = segs[k - 1]
        out[top_r : bot_r + 1, col] = fill

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

    # Extra spacing goes only to inter-group gaps (after first n_groups columns)
    inter_gap_bonus = np.zeros(total_bars - 1, dtype=int)
    if total_bars > n_groups:
        inter_gap_bonus[n_groups - 1 :] = extra_parts[:-1]

    occupied = [0]
    for idx in range(total_bars - 1):
        occupied.append(occupied[-1] + COL_SPACING + int(inter_gap_bonus[idx]))

    legend_cols = occupied[:n_groups]
    extra_cols = occupied[n_groups:]

    # Assign non-legend columns to groups randomly
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
    pool = [length for length in range(1, MAX_FILL_LEN + 1) if length != shared_len]
    rng.shuffle(pool)
    return [shared_len] + pool[: count - 1]


def _expected_output(
    inp: np.ndarray,
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
    fills: list[int],
) -> np.ndarray:
    """Compute the expected output by making the selected bar fully fill-colored."""
    out = inp.copy()
    for cols, lengths, rank, fill in zip(group_columns, group_lengths, ranks, fills):
        ranked = sorted(zip(cols, lengths), key=lambda item: -item[1])
        target_col, target_len = ranked[rank - 1]
        top_r = _MAX_CAP_ROW - target_len
        out[top_r : _BOTTOM_ROW + 1, target_col] = fill
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch."""
    rng = np.random.default_rng()

    # --- structure choices ---
    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, count + 1)) for count in counts]

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
    expected = _expected_output(grid, group_columns, group_lengths, ranks, fills)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
