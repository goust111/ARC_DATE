# Variant of 97d7923e
# Dimension: D4 — 源对象：垂直条的结构 (src)
# Change: 垂直条结构为cap-fill-cap：顶部和底部各一个帽色像素，中间为单一填充色的连续区域，所有条底部对齐在最后一行... → 双层帽色条

import itertools

import numpy as np

from arc_utils import get_bg, neighbors4, paste


# Design analysis
# - source_task: None, so there is no extra reference solver to inspect.
# - a) Concretization and behavior trace:
#   The legend stays as one-pixel-wide stacks in the top-left corner. The changed
#   object is each vertical bar: instead of a one-column [cap][fill...][cap]
#   stick, each bar is now a two-column object whose top and bottom are cap-color
#   rectangles of thickness t in {2, 3}. The fill length is only the region
#   strictly between the two cap blocks. Example: for cap=4, fill=7, t=2,
#   fill_len=3, one bar is two identical columns shaped 4,4,7,7,7,4,4. Original
#   logic scans each column independently, so it would misread one two-column bar
#   as two separate bars. The new logic must first detect a matched pair of
#   adjacent columns, then parse the thick cap runs inside that object.
# - b) Complexity calibration:
#   Thickness is limited to 2 or 3 so the variation is visible but still easy to
#   infer from a few samples. Thickness is shared within each cap-color group,
#   which gives repeated evidence inside one sample and avoids noisy one-off bars.
#   Fill lengths remain distinct within a group so ranking is still unambiguous.
# - c) Hidden assumption audit against the original main():
#   The original solver assumed one detected column == one detected bar. That is
#   no longer valid once every bar is rendered as two equal neighboring columns.
#   Its recolor step also targeted one column only; the new solver must recolor
#   the whole two-column fill rectangle. Legend parsing does not depend on bar
#   width or cap thickness, so it can remain unchanged.
# - d) Placement legality:
#   Bars are now 2 columns wide, still bottom-aligned, and separated from one
#   another by at least one background column. Legend cells use fixed even
#   columns near the left edge; bars begin only after a full blank column to the
#   right of the legend cluster, so they cannot merge into legend components.
#   Bars are constrained to start at row >= 1 so they never add extra top-row
#   pixels that would corrupt legend detection.
# - e) Visual observability:
#   The thick-cap change is directly visible because every bar has 2-column cap
#   blocks at both ends. Output only recolors the middle rectangle, so the puzzle
#   still asks the solver to ignore cap thickness when ranking. The doubled width
#   makes the structural change operationally necessary rather than cosmetic.
# - f) Visual cleanliness:
#   Legend stays compact in the corner and bars stay as separated 2-column
#   blocks with one-column gaps, so there is no layered clutter.
# - g) Color-change inferability:
#   Colors keep the original role structure: legend color == cap color, fill
#   color is uniform per group and distinct from every cap color. The output
#   change is still "selected bar interior becomes cap color", now on thicker-end
#   bars, so the mapping remains strongly signposted by shape and position.
# - Step 9 / new assumptions guaranteed by generate_input():
#   main() now assumes each real bar object is encoded by two identical adjacent
#   columns containing exactly two cap runs of equal thickness around one solid
#   fill run, with no top-row bar pixels and with 1 <= fill_len <= 9 - 2*t. The
#   generator enforces these conditions directly.


def _cap_runs(col_vals: np.ndarray, cap: int) -> list[tuple[int, int]]:
    """Return contiguous [start, end) runs of `cap` in one column."""
    runs: list[tuple[int, int]] = []
    r = 0
    h = int(col_vals.shape[0])
    while r < h:
        if int(col_vals[r]) != cap:
            r += 1
            continue
        start = r
        while r < h and int(col_vals[r]) == cap:
            r += 1
        runs.append((start, r))
    return runs


def _vertical_sandwiches(g: np.ndarray, cap: int) -> list[tuple[int, int, int, int, int]]:
    """
    Find vertical bar candidates as (fill_top, fill_bottom, left_col, fill_color, cap_thickness)
    where each bar is:
      - two identical adjacent columns
      - one contiguous cap run of thickness t >= 2
      - one solid fill run of a single non-bg color
      - one lower contiguous cap run of the same thickness t
    """
    _h, w = g.shape
    bg = get_bg(g)
    segs: list[tuple[int, int, int, int, int]] = []
    for c in range(w - 1):
        if not np.array_equal(g[:, c], g[:, c + 1]):
            continue
        runs = _cap_runs(g[:, c], cap)
        if len(runs) < 2:
            continue
        for (top0, top1), (bot0, bot1) in zip(runs, runs[1:]):
            thickness = top1 - top0
            if thickness < 2 or bot1 - bot0 != thickness:
                continue
            if bot0 <= top1:
                continue
            fill = int(g[top1, c])
            if fill == cap or fill == bg:
                continue
            if np.all(g[top1:bot0, c] == fill):
                segs.append((top1, bot0, c, fill, thickness))
                break
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
    Recolor thick-capped vertical bars by legend-driven rank selection.

    Input structure:
    - bg is the background. In the top-left corner, one-column legend stacks touch row 0;
      each stack has color K and height k.
    - Elsewhere, each candidate bar is two columns wide and bottom-aligned, with cap color K
      repeated for the same thickness at the top and bottom and a solid fill rectangle between them.

    Rule:
    - For each legend entry (K, k), collect the bars whose top and bottom cap runs both use K.
    - Rank those bars by the length of the fill segment between the thick caps (longest = rank 1).
    - Recolor only the rank-k bar's fill rectangle to K.

    Key insight:
    - The legend still encodes a rank index, but bar size must now ignore the thicker end caps and
      measure only the middle fill span.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, "legend must be non-empty"

    for _x, cap, k in legend:
        segs = _vertical_sandwiches(g, cap)
        assert segs, f"legend cap {cap} must have matching vertical bars"
        segs.sort(key=lambda s: (-(s[1] - s[0]), s[0], s[2]))
        assert 1 <= k <= len(segs), f"legend rank {k} out of range for {len(segs)} bars"
        fill_top, fill_bottom, col, _fill, _thickness = segs[k - 1]
        out[fill_top:fill_bottom, col : col + 2] = cap

    return out


# --------------- generator (O2) ---------------

# --- Configuration ---
GRID_H = 10                  # fixed grid height
WIDTH_MIN = 10               # minimum grid width
WIDTH_MAX = 32               # maximum grid width (inclusive)
MAX_GROUPS = 4               # max number of cap-color groups
MAX_TOTAL_BARS = 8           # max bars across all groups
BARS_PER_GROUP = (2, 3)      # allowed bars-per-group counts
CAP_THICKNESS_CHOICES = (2, 3)
COLOR_RANGE = (1, 10)        # non-bg color ids (exclusive upper)
MAX_RETRIES = 300            # retry budget for generate_input

_BOTTOM_ROW = GRID_H - 1
_TOP_MARGIN = 1
_LEGEND_COLUMNS = [0, 2, 4, 6]

_FEASIBLE_COUNTS = {
    ng: [c for c in itertools.product(BARS_PER_GROUP, repeat=ng) if sum(c) <= MAX_TOTAL_BARS]
    for ng in range(1, MAX_GROUPS + 1)
}


def _build_bar(cap: int, fill: int, fill_len: int, cap_thickness: int) -> np.ndarray:
    """Return a thick-cap vertical sprite with shape (fill_len + 2*cap_thickness, 2)."""
    total_h = fill_len + 2 * cap_thickness
    bar = np.full((total_h, 2), fill, dtype=int)
    bar[:cap_thickness, :] = cap
    bar[-cap_thickness:, :] = cap
    return bar


def _assign_columns(rng, n_groups: int, counts: list[int], width: int) -> list[list[int]]:
    """Pick left columns for 2-wide bars with one bg column gap between bars."""
    total_bars = sum(counts)
    legend_right = _LEGEND_COLUMNS[n_groups - 1]
    start = legend_right + 2
    candidates = [c for c in range(start, width - 1, 3)]
    assert len(candidates) >= total_bars
    chosen = sorted(int(c) for c in rng.choice(candidates, size=total_bars, replace=False))

    group_columns: list[list[int]] = [[] for _ in range(n_groups)]
    owners = [i for i, count in enumerate(counts) for _ in range(count)]
    rng.shuffle(owners)
    for col, owner in zip(chosen, owners):
        group_columns[owner].append(col)
    for cols in group_columns:
        cols.sort()
    return group_columns


def _choose_fill_lengths(rng, count: int, cap_thickness: int) -> list[int]:
    """Pick distinct fill lengths that keep the thick-capped bar below row 0."""
    max_fill_len = GRID_H - 2 * cap_thickness - _TOP_MARGIN
    assert count <= max_fill_len
    pool = np.arange(1, max_fill_len + 1)
    rng.shuffle(pool)
    return [int(v) for v in pool[:count]]


def _expected_output(
    inp: np.ndarray,
    caps: list[int],
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
    cap_thicknesses: list[int],
) -> np.ndarray:
    """Compute the expected output by recoloring the chosen fill rectangle to the cap color."""
    out = inp.copy()
    for cap, cols, lengths, rank, thickness in zip(caps, group_columns, group_lengths, ranks, cap_thicknesses):
        ranked = sorted(zip(cols, lengths), key=lambda item: (-item[1], item[0]))
        target_col, target_len = ranked[rank - 1]
        top_r = GRID_H - (target_len + 2 * thickness)
        fill_top = top_r + thickness
        fill_bottom = fill_top + target_len
        out[fill_top:fill_bottom, target_col : target_col + 2] = cap
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid thick-cap grid. Returns None on oracle mismatch."""
    rng = np.random.default_rng()

    # --- structure choices ---
    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, c + 1)) for c in counts]
    cap_thicknesses = [int(rng.choice(CAP_THICKNESS_CHOICES)) for _ in range(n_groups)]

    # --- palette ---
    palette = rng.choice(np.arange(*COLOR_RANGE), size=2 * n_groups, replace=False)
    caps = [int(v) for v in palette[:n_groups]]
    fills = [int(v) for v in palette[n_groups:]]

    # --- layout ---
    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, 3 * total_bars + 2 * n_groups)
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    group_columns = _assign_columns(rng, n_groups, counts, width)

    # --- render legend + bars ---
    grid = np.zeros((GRID_H, width), dtype=int)
    group_lengths: list[list[int]] = []
    for idx, (cap, fill, count, rank, cols, thickness) in enumerate(
        zip(caps, fills, counts, ranks, group_columns, cap_thicknesses)
    ):
        paste(grid, np.full((rank, 1), cap, dtype=int), 0, _LEGEND_COLUMNS[idx])

        lengths = _choose_fill_lengths(rng, count, thickness)
        group_lengths.append(lengths)

        for col, fill_len in zip(cols, lengths):
            bar = _build_bar(cap, fill, fill_len, thickness)
            top_r = GRID_H - bar.shape[0]
            paste(grid, bar, top_r, col)

    # --- oracle check ---
    expected = _expected_output(grid, caps, group_columns, group_lengths, ranks, cap_thicknesses)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
