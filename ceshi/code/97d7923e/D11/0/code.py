# Variant of 97d7923e
# Dimension: D11 — 轨迹/变换范围 (trajectory)
# Change: 默认无轨迹——仅修改被选中条的填充色，其余所有内容保持不变... → 未被选中的条被擦除

import itertools

import numpy as np

from arc_utils import get_bg, neighbors4, paste


# Design analysis for the D11 variant
# a) Concrete behavior:
#    Input objects stay the same as the source task: the top-left legend contains
#    1-column color stacks whose heights encode ranks, and each matching cap color
#    owns 2-3 bottom-aligned vertical bars of distinct lengths. In the original
#    task, the chosen rank-k bar is only "marked" by recoloring its interior to the
#    cap color while every other bar remains visible. In this variant, ranking is
#    unchanged, but the transform acts on the whole group: erase every non-selected
#    bar to bg and keep only the selected bar, with its interior recolored to the
#    cap color. Example within one color group:
#    lengths [6, 4, 2], legend height k=2. Original output keeps all 3 bars and
#    recolors the length-4 bar interior. Variant output leaves only the length-4
#    bar as one solid cap-colored column; the length-6 and length-2 bars become bg.
# b) Complexity calibration:
#    Counts remain 2-3 bars per group, so every group has at least one erased bar.
#    Distinct lengths inside each group keep ranking unambiguous. Multiple groups
#    remain allowed so the solver must repeat the same exclusion logic per color.
# c) Implicit assumptions from the source solver:
#    - `_legend_entries` assumes the legend is still the leftmost top-row cluster.
#      D11 does not change this, so the generator preserves it.
#    - `_vertical_sandwiches` assumes each bar is a cap/fill/cap column with solid
#      non-bg interior and no extra cap pixels inside. The variant still needs the
#      same bar detector, so construction is unchanged.
#    - The original `out[top_r+1:bot_r, col] = cap` was harmless because other bars
#      were meant to survive. Under D11 that is insufficient; the solver must also
#      blank all non-target segments, but only within each detected bar span so a
#      legend sharing the same column is not erased.
# d) Placement legality:
#    Legend stacks still sit above the corresponding legend-column bar with at least
#    one bg row between them. Bars occupy distinct columns with spacing >= 1 bg
#    column, stay inside the 10x10-20 frame, and remain bottom-aligned.
# e) Visual observability:
#    The new behavior is visible in every valid sample because each group has at
#    least 2 bars; therefore at least one full bar disappears. The output is not
#    explainable by simple recoloring alone, forcing the solver to infer exclusion.
# f) Clean visualization:
#    Only one retained bar per legend color avoids clutter in the output. Legend and
#    remaining bars stay spatially separated, so the "keep one / delete others"
#    effect reads clearly.
# g) Color-change inference:
#    No hidden numeric property of the color ids is used. Colors still only signal
#    group membership, while the visible height of the legend stack signals which
#    ranked bar survives.


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
    Keep only legend-selected bars and erase the rest of each color group.

    Input structure:
    - bg is the background color.
    - The top-left legend is made of 1-column monochrome stacks; color K identifies a bar group and stack height k encodes a rank.
    - Elsewhere, each group contains bottom-aligned vertical cap/fill/cap bars whose caps are color K and whose interiors are solid non-bg colors.

    Rule:
    - For each legend entry (K, k), collect all bars capped by K and sort them by interior length, longest first.
    - Keep only the rank-k bar from that sorted list; repaint its whole span to K by filling its interior with the cap color.
    - Erase every other bar from that same color group to bg, while leaving the legend untouched.

    Key insight:
    - The legend height still selects by rank, but now it decides which bar survives; all sibling bars are deleted rather than merely left unmodified.
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
            if idx == k - 1:
                out[top_r : bot_r + 1, col] = cap
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
    """Pick distinct fill lengths for one group."""
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
    """Compute the expected output for the keep-one-erase-the-rest variant."""
    out = inp.copy()
    bg = get_bg(inp)
    for cap, cols, lengths, rank in zip(caps, group_columns, group_lengths, ranks):
        ranked = sorted(zip(cols, lengths), key=lambda item: (-item[1], item[0]))
        target_col, target_len = ranked[rank - 1]
        for col, fill_len in zip(cols, lengths):
            top_r = _MAX_CAP_ROW - fill_len
            if col == target_col and fill_len == target_len:
                out[top_r : _BOTTOM_ROW + 1, col] = cap
            else:
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
