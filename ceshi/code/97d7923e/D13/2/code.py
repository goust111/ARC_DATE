# Variant of 97d7923e
# Dimension: D13 — 消歧规则（Tiebreak） (meta)
# Change: 组内条的填充长度全部不同，不需要消歧；排序键为 (-(fill_length), top_row, col)... → 多条同排名时全部重着色
#
# Design analysis
# - Concrete behavior:
#   The top-left legend still gives one cap color per group, and each legend height k still
#   points into that group's bars after sorting by fill length descending. The difference is
#   that the kth slot now acts as a threshold locator: find the fill length sitting at sorted
#   position k, then recolor every bar in that group whose fill length equals that threshold.
#   Example with cap color 2 and lengths [6, 4, 4]: old logic with k=2 recolored only one of
#   the length-4 bars after tie-break by column; new logic recolors both length-4 bars.
# - Complexity calibration:
#   The visual language is unchanged: single-column legend blocks and single-column bars.
#   To make the D13 change visible, generation guarantees at least one group per grid where
#   the selected threshold length appears on multiple bars. Other groups may stay distinct,
#   giving side-by-side examples of single-hit and multi-hit behavior.
# - Assumption audit:
#   Old main() relied on "segs[k-1] is the unique answer", which was harmless only because the
#   generator made all within-group lengths distinct. That assumption is removed: sorting is
#   retained only to locate the kth position's length, and recoloring is then length-based.
#   Because bars remain bottom-aligned, top_row is still determined by length; the old
#   secondary keys stay valid for locating the threshold slot but no longer decide output.
# - Placement legality:
#   The legend column still doubles as one bar column, so its bar length is constrained to fit
#   below the legend with at least one background row. Columns remain unique and spaced apart;
#   bars stay within bounds because fill lengths remain in 1..7 on a 10-row grid.
# - Visual observability:
#   Ties are expressed only through equal visible bar heights, never through color ids or any
#   hidden metadata. The output difference is directly visible as multiple bars of the same
#   height flipping to the cap color when the legend threshold lands on their shared length.
# - Visual cleanliness:
#   No new object types are added. The grid still separates legend and bars cleanly, with
#   single-color interiors and one-column bars, so tied selections remain easy to read.
# - Generator guarantee:
#   generate_input() enforces at least one "target multiplicity >= 2" group per sample and
#   validates the constructed input by comparing main(input) against an explicit oracle.

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


def _fill_length(seg: tuple[int, int, int, int]) -> int:
    return int(seg[1] - seg[0] - 1)


def _threshold_length_from_segs(segs: list[tuple[int, int, int, int]], rank: int) -> int:
    ranked = sorted(segs, key=lambda s: (-_fill_length(s), s[0], s[2]))
    return _fill_length(ranked[rank - 1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor vertical bars by legend-driven threshold selection.

    Input structure:
    - The background stays blank, while the top-left legend contains one-color vertical blocks.
    - Elsewhere, each relevant object is a single-column bar with matching cap cells and a
      solid interior, and bars sharing a cap color belong to the same group.

    Rule:
    - For each legend entry (cap color K, height k), collect every bar capped by K.
    - Sort those bars by interior length from longest to shortest to identify the kth slot.
    - Read the fill length at that slot, then recolor every bar in the group having that same
      fill length so its interior becomes K.

    Key insight:
    - The legend height no longer names one unique bar; it selects a length threshold position,
      and every tied bar at that selected length changes together.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, "legend must be non-empty"

    for _x, cap, k in legend:
        segs = _vertical_sandwiches(g, cap)
        assert segs, f"legend cap {cap} must have matching vertical bars"
        assert 1 <= k <= len(segs), f"legend rank {k} out of range for {len(segs)} bars"
        target_len = _threshold_length_from_segs(segs, k)
        for top_r, bot_r, col, _fill in segs:
            if _fill_length((top_r, bot_r, col, _fill)) == target_len:
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


def _threshold_multiplicity(lengths: list[int], rank: int) -> int:
    target_len = sorted(lengths, reverse=True)[rank - 1]
    return sum(length == target_len for length in lengths)


def _choose_group_lengths(rng, count: int, rank: int, force_tied_target: bool) -> list[int]:
    """Pick fill lengths for one group, optionally forcing the threshold rank to be tied."""
    max_legend_len = MAX_FILL_LEN - rank
    for _ in range(200):
        legend_len = int(rng.integers(1, max_legend_len + 1))
        if force_tied_target:
            others = [int(rng.integers(1, MAX_FILL_LEN + 1)) for _ in range(count - 1)]
            lengths = [legend_len] + others
            if _threshold_multiplicity(lengths, rank) >= 2:
                return lengths
        else:
            pool = [l for l in range(1, MAX_FILL_LEN + 1) if l != legend_len]
            rng.shuffle(pool)
            lengths = [legend_len] + pool[: count - 1]
            if len(set(lengths)) == count:
                return lengths
    raise RuntimeError("failed to choose group lengths")


def _expected_output(
    inp: np.ndarray,
    caps: list[int],
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
) -> np.ndarray:
    """Compute the expected output by recoloring all bars tied at the threshold rank."""
    out = inp.copy()
    for cap, cols, lengths, rank in zip(caps, group_columns, group_lengths, ranks):
        target_len = sorted(lengths, reverse=True)[rank - 1]
        for col, fill_len in zip(cols, lengths):
            if fill_len == target_len:
                top_r = _MAX_CAP_ROW - fill_len
                out[top_r + 1 : _BOTTOM_ROW, col] = cap
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, c + 1)) for c in counts]

    tie_groups = np.zeros(n_groups, dtype=bool)
    tie_groups[int(rng.integers(n_groups))] = True
    if n_groups > 1:
        extra_ties = rng.random(n_groups) < 0.35
        tie_groups |= extra_ties

    palette = rng.choice(np.arange(*COLOR_RANGE), size=2 * n_groups, replace=False)
    caps = [int(v) for v in palette[:n_groups]]
    fills = [int(v) for v in palette[n_groups:]]

    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    group_columns = _assign_columns(rng, n_groups, counts, width)

    grid = np.zeros((GRID_H, width), dtype=int)
    group_lengths: list[list[int]] = []
    for cap, fill, count, rank, cols, force_tie in zip(caps, fills, counts, ranks, group_columns, tie_groups):
        paste(grid, np.full((rank, 1), cap, dtype=int), 0, cols[0])

        lengths = _choose_group_lengths(rng, count, rank, bool(force_tie))
        group_lengths.append(lengths)

        for col, fill_len in zip(cols, lengths):
            paste(grid, _build_bar(cap, fill, fill_len), _MAX_CAP_ROW - fill_len, col)

    if not any(_threshold_multiplicity(lengths, rank) >= 2 for lengths, rank in zip(group_lengths, ranks)):
        return None

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
