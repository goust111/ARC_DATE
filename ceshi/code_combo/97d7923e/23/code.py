# Combo variant of 97d7923e (index 23)
# combo_spec: D3/0+D7/0+D11/0
# Name: 部分重着色+裁剪输出+未选中擦除
#
# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior and trace:
# - The input keeps the seed layout: the top-left legend is a row-adjacent cluster of
#   1-column color stacks, and the main field contains bottom-aligned cap/fill/cap bars.
# - For one legend entry (cap K, height k), the seed task ranked K-bars by fill length and
#   recolored the whole fill of the rank-k bar.
# - The fused rule changes three stages together:
#   1. selection stays the same: choose the rank-k bar inside cap-color group K;
#   2. painting becomes partial: recolor only the first k fill cells under the top cap;
#   3. exclusion becomes aggressive: erase every non-selected sibling bar to bg.
# - After all groups are processed, discard the full canvas and return only the bars'
#   union bounding box. Example with one group of lengths [6, 4, 2] and k=2:
#   seed output -> the length-4 bar becomes fully cap-colored and the other two bars stay.
#   fused output -> the length-4 bar keeps its lower 2 fill cells, only its top 2 fill
#   cells become K, the length-6 and length-2 bars vanish to bg, and the legend/top blank
#   band disappear because of the crop.
# b) Complexity calibration:
# - k still lives in a tiny visible range (1..3), inherited from bar counts, so humans can
#   infer both the rank choice and the repaint extent from a few examples.
# - The generator forces the selected bar to satisfy fill_len > k, preventing accidental
#   collapse into the D11-style "whole kept bar turns solid" behavior.
# - Each group keeps 2-3 bars, so every sample has erased bars and the D11 component stays
#   visually active.
# c) Hidden assumptions audited from the original main():
# - Candidate detection must still read from the untouched input grid `g`; once partial
#   repaint or erasure happens in `out`, recomputing candidates there would miss bars.
# - The original slice assignment `out[top+1:bot] = cap` no longer matches the rule; it
#   must become a bounded prefix repaint of exactly k cells.
# - Erasing must touch only the detected bar span `[top:bot]` so that a legend sharing the
#   same column remains untouched before the final crop removes it.
# - The crop bbox cannot be inferred from non-bg output pixels because erased bars would
#   shrink the box and partially repainting one kept bar would not recover the original bar
#   field. The bbox must be computed from all input bar candidates.
# d) Placement legality:
# - The generator preserves the seed constraints: bars occupy distinct columns, stay inside
#   the 10x10-20 frame, and the legend-column bar begins low enough to leave at least one bg
#   cell below the legend.
# - Distinct fill lengths are maintained within each cap group, so ranking stays unambiguous.
# e) Visual observability:
# - Every valid sample shows all three fused effects at once: some bars disappear, the kept
#   bars show a cap-colored prefix plus an untouched fill tail, and the legend is absent from
#   the cropped output.
# - No invisible property such as numeric color id participates in ranking or cropping.
# f) Visual cleanliness:
# - Cropping uses a single outer rectangle over the original bar field, preserving spacing
#   between surviving columns instead of horizontally compressing them.
# - Only one retained bar per cap color limits clutter, while the untouched lower fill tail
#   keeps the kept bars easy to distinguish from fully solid columns.
# g) New solver assumptions and generator guarantees:
# - For every legend entry, there are at least k matching bars.
# - The selected rank-k bar has fill_len >= k for solver safety, and fill_len > k in the
#   generator so the D3 component is always visible.
# - Legend colors stay unique, so processing one legend entry cannot collide with another.

import itertools

import numpy as np

from arc_utils import crop_bbox, get_bg, neighbors4, paste


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
    """Bounding box of all input bars: (top_row, left_col, bottom_row, right_col)."""
    assert segs, "at least one bar is required for cropping"
    r0 = min(top_r for top_r, _bot_r, _col, _fill in segs)
    r1 = max(bot_r for _top_r, bot_r, _col, _fill in segs)
    c0 = min(col for _top_r, _bot_r, col, _fill in segs)
    c1 = max(col for _top_r, _bot_r, col, _fill in segs)
    return int(r0), int(c0), int(r1), int(c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only legend-selected bars, repaint just their leading fill prefix, and return the compact bar field.

    Input structure:
    - Background cells dominate the board. At the top-left, each legend entry is a one-column stack
      of cap color K whose visible height k encodes a rank.
    - Elsewhere, bars are one-column cap/fill/cap objects: the top and bottom cells are color K and
      the interior is a solid non-background fill. Bars with the same cap color belong to one group.

    Rule:
    - For each legend entry (K, k), detect all K-capped bars and sort them by fill length from longest
      to shortest.
    - Keep only the rank-k bar from that group; recolor exactly its first k fill cells below the top cap
      to K, leaving the rest of that bar unchanged.
    - Erase every other bar in the same group to background, then crop the final grid to the minimal box
      containing the original bar field, which removes the legend and outer blank margins.

    Key insight:
    - The legend height now controls two coupled decisions at once: it chooses which bar survives and how
      deep the cap-colored repaint extends into that surviving bar, while the output only shows the cropped
      bar area instead of the full scene.
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

        for idx, (top_r, bot_r, col, _fill) in enumerate(ranked):
            if idx == k - 1:
                fill_len = bot_r - top_r - 1
                assert k <= fill_len, f"legend height {k} exceeds target fill length {fill_len}"
                out[top_r + 1 : top_r + 1 + k, col] = cap
            else:
                out[top_r : bot_r + 1, col] = bg

    return crop_bbox(out, _bars_bbox(all_bar_segs))


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
    """Pick distinct fill lengths and force the selected bar to show a partial repaint."""
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
    """Compute the fused-variant output from construction metadata."""
    out = inp.copy()
    bg = get_bg(inp)

    for cap, cols, lengths, rank in zip(caps, group_columns, group_lengths, ranks):
        ranked = sorted(zip(cols, lengths), key=lambda item: (-item[1], _MAX_CAP_ROW - item[1], item[0]))
        target_col, target_len = ranked[rank - 1]
        target_top = _MAX_CAP_ROW - target_len

        for col, fill_len in zip(cols, lengths):
            top_r = _MAX_CAP_ROW - fill_len
            if col == target_col:
                out[top_r : _BOTTOM_ROW + 1, col] = inp[top_r : _BOTTOM_ROW + 1, col]
                out[target_top + 1 : target_top + 1 + rank, target_col] = cap
            else:
                out[top_r : _BOTTOM_ROW + 1, col] = bg

    all_cols = [col for cols in group_columns for col in cols]
    all_top_rows = [_MAX_CAP_ROW - fill_len for lengths in group_lengths for fill_len in lengths]
    bbox = (min(all_top_rows), min(all_cols), _BOTTOM_ROW, max(all_cols))
    return crop_bbox(out, bbox)


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
