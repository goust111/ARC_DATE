# Variant of 97d7923e
# Dimension: D1 — 匹配条件：端点颜色分组 (match_cond)
# Change: s.endpoint_colors == d.color_group — 垂直条的顶部和底部帽色(cap color)与... → 列位置区域分组

import itertools
from pathlib import Path
import sys

import numpy as np

for _parent in Path(__file__).resolve().parents:
    if (_parent / "arc_utils.py").exists():
        if str(_parent) not in sys.path:
            sys.path.append(str(_parent))
        break

from arc_utils import get_bg, neighbors4, paste


# Design analysis
# ---------------
# variant_spec:
#   core_change:
#     original_match: legend color K selects the bar group whose cap color is also K.
#     variant_match: legend x-position selects a horizontal region; bars are grouped by
#       which legend marker they are horizontally closest to.
#   concrete_objects:
#     legend:
#       - 2-3 one-column blocks touch row 0.
#       - Their heights still encode rank k.
#       - Their colors are paint colors only, and are chosen to be absent from all bars.
#       - Their left-to-right placement defines the left/middle/right regions.
#     bars:
#       - Same cap-fill-cap vertical bars as the original task.
#       - Bars remain bottom-aligned and are ranked by interior length within a region.
#       - Cap/fill colors are reused across different regions so color can no longer be
#         the grouping cue.
#   behavior_trace:
#     original_flow:
#       - Legend entry (color 6, height 2) means "among bars capped by 6, take the
#         2nd-longest and repaint its fill to 6".
#     variant_flow:
#       - Legend entry at the right side (color 6, height 2) means "among bars in the
#         right-side region, take the 2nd-longest and repaint its fill to 6".
#     visible_difference:
#       - The selected bar can have cap colors unrelated to the legend color.
#       - The same cap colors can appear in multiple regions, so only spatial grouping works.
#   complexity_calibration:
#     - Use only 2-3 regions so the partition is learnable from a few examples.
#     - Keep 2-3 bars per region so the ranking action is repeated and visible.
#     - Reuse a tiny pool of bar cap/fill colors across regions, preventing hidden
#       color-based subgrouping while avoiding a visually noisy palette.
#   implicit_assumptions_audit:
#     - Original _vertical_sandwiches(g, cap) assumed the legend already told us which
#       cap color to search for. That assumption no longer holds, so the solver must
#       enumerate all bar candidates first.
#     - Original _legend_entries only looked at a top-left cluster. That breaks once
#       legend position itself carries meaning, so the variant scans every top-touching
#       legend block across the full top row.
#     - Original repaint used the legend color both as matcher and output color. In the
#       variant it is output color only; matching is fully spatial.
#   placement_legality:
#     - Legend columns are separate from bar columns, so legend height no longer limits
#       bar length and no overlap checks are needed between those object types.
#     - Regions are generated first, legend markers sit near each region center, and
#       every chosen bar column is verified to belong to that legend under the solver's
#       nearest-legend rule.
#   observability_and_cleanliness:
#     - Region markers stay on row 0 and bars stay below, so the scene remains layered
#       and readable.
#     - Because legend colors never appear on bar caps, the solver cannot silently
#       collapse back to the original "same color" behavior.
#   new_solver_assumptions:
#     - Each region has at least one legend marker and at least k bars.
#     - Every bar column has a unique nearest legend marker (the generator avoids ties).
#     - Only legend components touch row 0 with colors reserved for legends.


def _vertical_sandwiches(g: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """
    Find all vertical bar candidates (top_r, bottom_r, col, cap_color, fill_color) where:
      - g[top_r, col] == g[bottom_r, col] == cap_color
      - bottom_r is the next occurrence of cap_color below top_r in that column
      - interior is a solid run of one non-bg color fill_color != cap_color
      - length >= 1 (bottom_r >= top_r + 2)
    """
    h, w = g.shape
    bg = get_bg(g)
    segs: list[tuple[int, int, int, int, int]] = []
    for c in range(w):
        col_vals = g[:, c]
        for top_r in range(h - 2):
            cap = int(col_vals[top_r])
            if cap == bg:
                continue
            rel = np.flatnonzero(col_vals[top_r + 1 :] == cap)
            if len(rel) == 0:
                continue
            bot_r = int(top_r + 1 + rel[0])
            if bot_r <= top_r + 1:
                continue
            fill = int(col_vals[top_r + 1])
            if fill == bg or fill == cap:
                continue
            if np.all(col_vals[top_r + 1 : bot_r] == fill):
                segs.append((top_r, bot_r, c, cap, fill))
    return segs


def _legend_entries(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    """
    Legend entries are the non-bg components that touch row 0.

    Returns entries as (x_pos, legend_color, block_count), sorted by x_pos.
    block_count is the vertical height of the 4-connected monochrome component.
    """
    h, w = g.shape
    top_pos = [int(c) for c in np.flatnonzero(g[0, :] != bg)]
    assert top_pos, "legend row must be non-empty"

    entries: list[tuple[int, int, int]] = []
    seen = np.zeros((h, w), dtype=bool)
    for c in top_pos:
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
                if 0 <= rr < h and 0 <= ccc < w and (not seen[rr, ccc]) and int(g[rr, ccc]) == col:
                    seen[rr, ccc] = True
                    st.append((rr, ccc))
        x_pos = min(cc for _, cc in pix)
        rs = [r for r, _ in pix]
        block_count = max(rs) - min(rs) + 1
        entries.append((x_pos, col, int(block_count)))

    entries.sort(key=lambda t: t[0])
    return entries


def _legend_owner(col: int, legend_xs: list[int]) -> int:
    """Index of the legend entry whose x-position is closest to `col` (left-biased on ties)."""
    return min(range(len(legend_xs)), key=lambda idx: (abs(int(col) - int(legend_xs[idx])), int(legend_xs[idx])))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor vertical bars by legend-position-driven rank selection.

    Input structure:
    - bg = background. Row 0 contains several vertical legend blocks; each block has color P and height k.
    - Main field: vertical cap-fill-cap bars with solid-color interiors, spread across left/middle/right regions.

    Rule:
    - Use each legend block's horizontal position to define which nearby columns belong to its region.
    - Within that region, rank all vertical bars by interior length (longest = rank 1).
    - Recolor the rank-k bar's interior to the legend color P.

    Key insight: The legend color is no longer the grouping key; the x-location of the legend picks the bar set, and the legend height still supplies the rank.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, "legend must be non-empty"
    legend_xs = [int(x) for x, _paint, _k in legend]

    segs = _vertical_sandwiches(g)
    assert segs, "grid must contain vertical bars"

    for idx, (_x, paint, k) in enumerate(legend):
        region = [s for s in segs if _legend_owner(int(s[2]), legend_xs) == idx]
        assert region, f"legend region {idx} must have matching vertical bars"
        region.sort(key=lambda s: (-(s[1] - s[0] - 1), s[0], s[2]))
        assert 1 <= k <= len(region), f"legend rank {k} out of range for {len(region)} bars"
        top_r, bot_r, col, _cap, _fill = region[k - 1]
        out[top_r + 1 : bot_r, col] = int(paint)

    return out


# --------------- generator (D1 variant) ---------------

# --- Configuration ---
GRID_H = 10
WIDTH_MIN = 12
WIDTH_MAX = 20
MIN_GROUPS = 2
MAX_GROUPS = 3
MAX_TOTAL_BARS = 8
BARS_PER_GROUP = (2, 3)
MAX_FILL_LEN = 7
COLOR_RANGE = (1, 10)
COL_SPACING = 2
ZONE_GAP = 2
MAX_RETRIES = 400

_BOTTOM_ROW = GRID_H - 1
_MAX_CAP_ROW = GRID_H - 2

_FEASIBLE_COUNTS = {
    ng: [c for c in itertools.product(BARS_PER_GROUP, repeat=ng) if sum(c) <= MAX_TOTAL_BARS]
    for ng in range(MIN_GROUPS, MAX_GROUPS + 1)
}


def _build_bar(cap: int, fill: int, fill_len: int) -> np.ndarray:
    """Return a (fill_len+2, 1) column sprite: cap / fill*N / cap."""
    bar = np.full((fill_len + 2, 1), fill, dtype=int)
    bar[0, 0] = cap
    bar[-1, 0] = cap
    return bar


def _layout_regions(rng, counts: list[int], width: int) -> list[tuple[int, int]]:
    """Create disjoint horizontal intervals, one per legend region."""
    n_groups = len(counts)
    zone_mins = [2 * count for count in counts]
    min_width = sum(zone_mins) + ZONE_GAP * (n_groups - 1)
    assert width >= min_width

    slack = width - min_width
    parts = rng.multinomial(slack, np.full(2 * n_groups + 1, 1.0 / (2 * n_groups + 1)))
    left_margin = int(parts[0])
    zone_extras = [int(v) for v in parts[1 : 1 + n_groups]]
    gap_extras = [int(v) for v in parts[1 + n_groups : 1 + n_groups + (n_groups - 1)]]

    regions: list[tuple[int, int]] = []
    cur = left_margin
    for idx, base in enumerate(zone_mins):
        zone_w = int(base + zone_extras[idx])
        regions.append((int(cur), int(cur + zone_w - 1)))
        cur += zone_w
        if idx < n_groups - 1:
            cur += int(ZONE_GAP + gap_extras[idx])
    return regions


def _choose_group_columns(rng, region: tuple[int, int], count: int, legend_col: int) -> list[int] | None:
    """Pick `count` bar columns inside `region`, excluding `legend_col`, spaced by at least 2."""
    start, end = map(int, region)
    options = []
    for parity in (0, 1):
        cols = [c for c in range(start + parity, end + 1, COL_SPACING) if c != legend_col]
        if len(cols) >= count:
            options.append(cols)
    if not options:
        return None

    base = options[int(rng.integers(len(options)))]
    slack = len(base) - count
    offset = int(rng.integers(slack + 1)) if slack > 0 else 0
    chosen = [int(c) for c in base[offset : offset + count]]
    return chosen


def _choose_fill_lengths(rng, count: int) -> list[int]:
    """Pick `count` distinct fill lengths for one region."""
    pool = list(range(1, MAX_FILL_LEN + 1))
    rng.shuffle(pool)
    return [int(v) for v in pool[:count]]


def _bar_colors_for_group(rng, count: int, cap_pool: list[int], fill_pool: list[int]) -> list[tuple[int, int]]:
    """Reuse a small global palette so color is visibly decoupled from region ownership."""
    offset = int(rng.integers(len(cap_pool)))
    cols = []
    for i in range(count):
        cap = int(cap_pool[(offset + i) % len(cap_pool)])
        fill = int(fill_pool[(offset + i + 1) % len(fill_pool)])
        cols.append((cap, fill))
    return cols


def _expected_output(
    inp: np.ndarray,
    legend_colors: list[int],
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
) -> np.ndarray:
    """Compute the expected output by recoloring the rank-k bar in each region."""
    out = inp.copy()
    for paint, cols, lengths, rank in zip(legend_colors, group_columns, group_lengths, ranks):
        ranked = sorted(zip(cols, lengths), key=lambda item: (-item[1], item[0]))
        target_col, target_len = ranked[rank - 1]
        top_r = _MAX_CAP_ROW - target_len
        out[top_r + 1 : _BOTTOM_ROW, target_col] = int(paint)
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch or ambiguous layout."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(MIN_GROUPS, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, c + 1)) for c in counts]

    min_width = sum(2 * c for c in counts) + ZONE_GAP * (n_groups - 1)
    width = int(rng.integers(max(WIDTH_MIN, min_width), WIDTH_MAX + 1))
    regions = _layout_regions(rng, counts, width)

    palette = rng.choice(np.arange(*COLOR_RANGE), size=n_groups + 4, replace=False)
    legend_colors = [int(v) for v in palette[:n_groups]]
    cap_pool = [int(v) for v in palette[n_groups : n_groups + 2]]
    fill_pool = [int(v) for v in palette[n_groups + 2 : n_groups + 4]]

    legend_cols = [int((start + end) // 2) for start, end in regions]
    group_columns: list[list[int]] = []
    for count, region, legend_col in zip(counts, regions, legend_cols):
        cols = _choose_group_columns(rng, region, count, legend_col)
        if cols is None:
            return None
        group_columns.append(cols)

    # Verify the generator uses the same spatial ownership rule as the solver.
    for idx, cols in enumerate(group_columns):
        for col in cols:
            if _legend_owner(int(col), legend_cols) != idx:
                return None

    grid = np.zeros((GRID_H, width), dtype=int)
    group_lengths: list[list[int]] = []
    for idx, (paint, count, rank, cols) in enumerate(zip(legend_colors, counts, ranks, group_columns)):
        paste(grid, np.full((rank, 1), paint, dtype=int), 0, legend_cols[idx])

        lengths = _choose_fill_lengths(rng, count)
        colors = _bar_colors_for_group(rng, count, cap_pool, fill_pool)
        group_lengths.append(lengths)

        for col, fill_len, (cap, fill) in zip(cols, lengths, colors):
            paste(grid, _build_bar(cap, fill, fill_len), _MAX_CAP_ROW - fill_len, col)

    expected = _expected_output(grid, legend_colors, group_columns, group_lengths, ranks)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
