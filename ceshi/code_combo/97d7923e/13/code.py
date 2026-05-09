# Combo variant of 97d7923e (index 13)
# combo_spec: D7/0+D11/1
# Name: 裁剪输出+图例擦除

# Design analysis for the combo variant
# a) Concrete behavior and trace:
# - The input remains the seed layout: a left-top legend made of 1-cell-wide vertical stacks,
#   and several bottom-aligned cap/fill/cap bars. Each legend stack shares its color with one bar
#   group and its height is still a rank index.
# - The seed task recolors one ranked bar per legend entry and returns the full 10xW canvas.
# - This combo changes the post-processing in two coupled stages. First, after using the legend to
#   choose bars, the exact legend pixels are cleared to background because they were only
#   instructions. Second, the full board is cropped to the minimal bounding box containing all bar
#   columns and their caps, so the answer is only the compact bar field.
# - Example trace: suppose a color-2 legend stack of height 2 sits in column 0 above a shared bar,
#   and the rank-2 color-2 bar is repainted. The intermediate full-canvas combo state differs from
#   the seed by turning the legend cells at rows 0-1, col 0 back to background. The final combo
#   output then discards every row above the highest bar cap and every blank outer column, so only
#   the transformed bar rectangle remains.
# b) Complexity calibration:
# - The fusion still uses one global rectangular crop, not per-column compression, so bar spacing
#   stays readable and the visual language remains close to the seed task.
# - The legend keeps only one job: selecting ranked bars. It never becomes an extra output object.
# - The bar ranking rule, color grouping, and tie-break order remain unchanged, which prevents the
#   combo from becoming a multi-rule puzzle with weak support across examples.
# c) Hidden assumptions audited from the original main():
# - Candidate bars must still be detected from the original input grid `g`, not from a recolored or
#   legend-erased intermediate, because recolored targets would stop looking like cap/fill/cap bars.
# - Legend parsing still assumes the leftmost non-background cluster on row 0 is the whole legend.
#   The generator therefore continues to place legend columns first and uses only small gaps.
# - The crop now intentionally depends on the fused trajectory: it is computed from the remaining
#   non-background pixels after legend erasure. Without erasing the legend first, the crop would
#   wrongly keep the legend rows, so the two dimensions genuinely interact in one pipeline.
# - Because the legend can share columns with bars, erasure must target the exact legend component
#   mask rather than a coarse left-top rectangle.
# d) Placement legality:
# - Each legend column is also a bar column, but the generator constrains that column's fill length
#   so the top cap starts at least one background row below the legend stack.
# - Bars are one cell wide, bottom-aligned, non-overlapping in columns, and generated within the
#   10x10..20 canvas. Distinct lengths inside each cap group keep the rank selection unambiguous.
# e) Visual observability:
# - Relative to the seed, every sample now returns a smaller grid whose top edge starts at the
#   highest bar cap rather than at row 0, so the missing legend is obvious from shape alone.
# - The full-scene intermediate also loses the exact legend stacks, which supports the reading that
#   the top-left objects are consumed instructions rather than output objects.
# f) Visual cleanliness:
# - The output contains only the transformed bar field. No overlays, duplicated objects, or
#   competing annotations remain, so the result is visually cleaner than the seed.
# g) Change inferability:
# - No new arbitrary color mapping is introduced. Visible structure still explains the recoloring:
#   legend height chooses a rank among same-cap bars, and the chosen bar adopts that cap color.
#   The only additional behavior is that instruction pixels are omitted from the final compact view.
# Additional review of new assumptions:
# - `_legend_entries_and_mask(...)` assumes each legend entry is a 4-connected monochrome component
#   touching row 0 inside the initial legend cluster. The generator constructs exactly that.
# - The final crop assumes that after erasing the legend, every remaining non-background pixel
#   belongs to the bar field. The generator keeps the seed task's object inventory, so that holds.
# - The combo's visible result can still match the pure D7 crop on many samples, so verification
#   must also measure erased legend pixels and pre-vs-post-erasure crop shrinkage.

import itertools

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, neighbors4, paste


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
      - a boolean mask marking the exact legend pixels
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


def _recolor_and_erase(
    g: np.ndarray,
    legend: list[tuple[int, int, int]],
    legend_mask: np.ndarray,
    bg: int,
) -> np.ndarray:
    """Apply the legend-driven recoloring on `g`, then erase the legend pixels."""
    out = g.copy()
    segs_by_cap: dict[int, list[tuple[int, int, int, int]]] = {}

    for _x, cap, k in legend:
        segs = segs_by_cap.get(cap)
        if segs is None:
            segs = _vertical_sandwiches(g, cap)
            assert segs, f"legend cap {cap} must have matching vertical bars"
            segs_by_cap[cap] = segs
        ranked = sorted(segs, key=lambda s: (-(s[1] - s[0] - 1), s[0], s[2]))
        assert 1 <= k <= len(ranked), f"legend rank {k} out of range for {len(ranked)} bars"
        top_r, bot_r, col, _fill = ranked[k - 1]
        out[top_r + 1 : bot_r, col] = cap

    out[legend_mask] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor legend-selected bars, consume the legend, and keep only the compact bar field.

    Input structure:
    - Background surrounds a left-top legend of one-column monochrome stacks; each stack uses a
      bar-cap color K and its height k is a rank index.
    - The rest of the scene contains one-column vertical cap/fill/cap bars that end on the last
      row; bars sharing cap color K form a group and have solid non-background interiors.

    Rule:
    - For each legend entry (K, k), gather all bars capped by K, sort them by interior length
      descending, and repaint the rank-k bar's interior to K.
    - After the rank-guided recoloring is complete, erase the exact legend pixels to background.
    - Finally crop the board to the minimal bounding box that contains the bar region only, which
      removes the consumed legend area together with every outer blank margin.

    Key insight: The legend is an instruction source rather than part of the answer. Its heights
    choose bars by rank, then the instruction pixels disappear and the output becomes the tight
    rectangular view of the transformed bars alone.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    legend, legend_mask = _legend_entries_and_mask(g, bg)
    assert legend, "legend must be non-empty"

    transformed = _recolor_and_erase(g, legend, legend_mask, bg)
    cropped_bbox = bbox(transformed != bg)
    assert cropped_bbox is not None, "bar field must remain non-empty after legend erasure"
    return crop_bbox(transformed, cropped_bbox)


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
    """Pick `count` distinct fill lengths for one group."""
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
    """Compute the expected output by recoloring, erasing the legend, and cropping to the bars."""
    out = inp.copy()
    bg = get_bg(inp)

    for cap, cols, lengths, rank in zip(caps, group_columns, group_lengths, ranks):
        ranked = sorted(zip(cols, lengths), key=lambda item: (-item[1], item[0]))
        target_col, target_len = ranked[rank - 1]
        top_r = _MAX_CAP_ROW - target_len
        out[top_r + 1 : _BOTTOM_ROW, target_col] = cap
        out[:rank, cols[0]] = bg

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
