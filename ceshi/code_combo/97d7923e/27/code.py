# Combo variant of 97d7923e (index 27)
# combo_spec: D1/2+D2/0
# Name: 列位置区域分组+升序排名

import itertools
from pathlib import Path
import sys

import numpy as np

for _parent in Path(__file__).resolve().parents:
    if (_parent / "arc_utils.py").exists():
        _parent_str = str(_parent)
        if _parent_str not in sys.path:
            sys.path.append(_parent_str)
        break

from arc_utils import get_bg, neighbors4, paste


# Design analysis
# ---------------
# fusion_spec:
#   a_concretization_and_behavior_trace:
#     scene:
#       - Row 0 contains 2-3 slim legend stacks at different x-positions.
#       - Lower rows contain bottom-aligned one-column cap-fill-cap bars.
#       - Bars are grouped by horizontal region, not by cap color.
#     concrete_example:
#       - Suppose the left legend stack sits at column 3, has height 2, and color 6.
#       - Bars nearest column 3 might have fill lengths [1, 4, 6] and cap colors [2, 1, 2].
#       - Original task would ignore x-position, look for cap color 6, and rank longest-first.
#       - This combo instead ignores bar colors for grouping, sorts that left-region set as
#         [1, 4, 6], and repaints the length-4 bar to color 6 because height 2 means
#         "2nd shortest in this region".
#     visible_difference:
#       - The chosen bar can have a cap color unrelated to the legend color.
#       - Reversing the rank order changes which column is recolored even within the same region.
#   b_complexity_calibration:
#     - Keep only 2-3 regions so spatial partitioning is learnable from a few examples.
#     - Keep 2-3 bars per region so the ascending order is visible without overcrowding.
#     - Reuse a tiny global cap/fill palette across regions so bar colors cannot secretly
#       encode region identity.
#     - Force at least one region where ascending and descending select different bars.
#   c_implicit_assumptions_audit:
#     - Original _vertical_sandwiches(g, cap) assumed the legend already revealed one cap
#       color to search for. That breaks here; the solver must enumerate all bars first.
#     - Original _legend_entries only scanned a top-left cluster. That also breaks because
#       legend x-position now carries semantics, so every top-touching legend component across
#       the row must be parsed.
#     - Original repaint logic used one legend attribute for both matching and output color.
#       In the combo, x-position determines the candidate set while legend color is only the
#       paint to apply.
#     - Original descending sort in main is invalid. The combo must sort shortest-first.
#   d_placement_legality:
#     - Regions are laid out as disjoint horizontal intervals with at least one blank gap
#       between them, and bar columns are chosen inside those intervals with stride 2.
#     - Legend columns are separate from bar columns, so legend stacks never overlap bars and
#       never constrain bar height.
#     - All bars remain bottom-aligned and fit inside the fixed 10-row grid.
#   e_visual_observability:
#     - Legend colors are reserved for legends and do not appear on bar caps or fills.
#     - At least one cap color is reused across multiple regions, so color-based grouping is
#       visibly insufficient.
#     - Within each region, bar lengths are distinct, making shortest-first ranking visible.
#   f_visual_cleanliness:
#     - Legends occupy only row 0 and a few cells below it; bars occupy sparse single columns
#       below, keeping object types layered and easy to separate by eye.
#     - The region gaps prevent boundary ambiguity and keep nearest-legend ownership stable.
#   g_inference_of_change:
#     - The user can infer "which region" from legend placement because bars around each marker
#       get recolored with that marker's color across multiple examples.
#     - The user can infer "shortest-first" because at least one region in each sample would
#       pick a different bar under longest-first, and the generator rejects non-informative cases.
#   main_assumption_audit:
#     - _legend_owner assumes every bar has a unique nearest legend marker; the generator
#       verifies strict distance gaps for all placed bars.
#     - main assumes each region has at least k bars; generate_input chooses rank from that
#       region's bar count.
#     - main assumes only legend components touch row 0; generator keeps all bar top caps at
#       row >= 1 and legend colors disjoint from all bar colors.


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
            fill = int(col_vals[top_r + 1])
            if fill == bg or fill == cap:
                continue
            if np.all(col_vals[top_r + 1 : bot_r] == fill):
                segs.append((int(top_r), int(bot_r), int(c), int(cap), int(fill)))
    return segs


def _legend_entries(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    """
    Legend entries are the monochrome 4-connected components that touch row 0.

    Returns entries as (x_pos, legend_color, block_count), sorted by x_pos.
    block_count is the vertical height of that component.
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
        entries.append((int(x_pos), int(col), int(block_count)))

    entries.sort(key=lambda t: t[0])
    return entries


def _legend_owner(col: int, legend_xs: list[int]) -> int:
    """Index of the legend entry whose x-position is closest to `col` (left-biased on ties)."""
    return min(range(len(legend_xs)), key=lambda idx: (abs(int(col) - int(legend_xs[idx])), int(legend_xs[idx])))


def _seg_length(seg: tuple[int, int, int, int, int]) -> int:
    return int(seg[1] - seg[0] - 1)


def _region_sorted_bars(g: np.ndarray) -> tuple[list[tuple[int, int, int]], list[list[tuple[int, int, int, int, int]]]]:
    """Parse legend entries and return bars grouped by nearest legend, shortest-first."""
    bg = get_bg(g)
    legend = _legend_entries(g, bg)
    legend_xs = [int(x) for x, _paint, _k in legend]
    segs = _vertical_sandwiches(g)
    assert segs, "grid must contain vertical bars"

    regions: list[list[tuple[int, int, int, int, int]]] = []
    for idx in range(len(legend)):
        region = [s for s in segs if _legend_owner(int(s[2]), legend_xs) == idx]
        region.sort(key=lambda s: (_seg_length(s), s[0], s[2]))
        regions.append(region)
    return legend, regions


def _descending_region_output(g: np.ndarray) -> np.ndarray:
    """Reference output using the same spatial grouping but the original longest-first ranking."""
    out = np.array(g, dtype=int).copy()
    legend, regions = _region_sorted_bars(np.array(g, dtype=int))
    for (_x, paint, k), region in zip(legend, regions):
        assert 1 <= k <= len(region), f"legend rank {k} out of range for {len(region)} bars"
        ranked = sorted(region, key=lambda s: (-_seg_length(s), s[0], s[2]))
        top_r, bot_r, col, _cap, _fill = ranked[k - 1]
        out[top_r + 1 : bot_r, col] = int(paint)
    return out


def sample_metrics(grid: np.ndarray) -> dict[str, int]:
    """
    Quantify whether a sample really expresses the combo:
      - `rank_signal`: 1 iff shortest-first differs from longest-first under the same regions.
      - `shared_cap_colors`: number of cap colors that appear in more than one region.
      - `legend_bar_color_overlap`: number of legend colors reused on any bar pixel.
    """
    g = np.asarray(grid, dtype=int)
    legend, regions = _region_sorted_bars(g)

    cap_to_regions: dict[int, set[int]] = {}
    bar_colors = set()
    for idx, region in enumerate(regions):
        for _top, _bot, _col, cap, fill in region:
            bar_colors.add(int(cap))
            bar_colors.add(int(fill))
            cap_to_regions.setdefault(int(cap), set()).add(int(idx))

    shared_cap_colors = sum(1 for owners in cap_to_regions.values() if len(owners) > 1)
    legend_bar_color_overlap = sum(1 for _x, paint, _k in legend if int(paint) in bar_colors)
    rank_signal = int(not np.array_equal(main(g), _descending_region_output(g)))
    return {
        "region_count": int(len(legend)),
        "rank_signal": int(rank_signal),
        "shared_cap_colors": int(shared_cap_colors),
        "legend_bar_color_overlap": int(legend_bar_color_overlap),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor one bar per horizontal region using legend position and shortest-first ranking.

    Input structure:
    - Background cells dominate the grid. Several one-column legend stacks touch row 0 at
      separated x-positions, and each stack has its own paint color and height k.
    - Elsewhere are bottom-aligned vertical cap-fill-cap bars with solid interiors.
    - Bar cap/fill colors may repeat across different regions, so color is not a reliable
      grouping cue.

    Rule:
    - Use each legend stack's horizontal position as a region marker; a bar belongs to the
      nearest legend marker by column.
    - Inside that region, sort bars by interior length in ascending order (shortest = rank 1).
    - Recolor the interior of the rank-k bar to the legend stack's color, where k is the
      legend stack's height.

    Key insight:
    - The legend no longer matches bars by shared color. Its x-position chooses the candidate
      set, and its height chooses the k-th shortest member of that set.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    legend, regions = _region_sorted_bars(g)
    assert legend, "legend must be non-empty"

    for idx, ((_x, paint, k), region) in enumerate(zip(legend, regions)):
        assert region, f"legend region {idx} must have matching vertical bars"
        assert 1 <= k <= len(region), f"legend rank {k} out of range for {len(region)} bars"
        top_r, bot_r, col, _cap, _fill = region[k - 1]
        out[top_r + 1 : bot_r, col] = int(paint)

    return out


# --------------- generator (combo) ---------------

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
MAX_RETRIES = 500

_BOTTOM_ROW = GRID_H - 1
_MAX_CAP_ROW = GRID_H - 2

_FEASIBLE_COUNTS = {
    ng: [
        counts
        for counts in itertools.product(BARS_PER_GROUP, repeat=ng)
        if sum(counts) <= MAX_TOTAL_BARS and sum(2 * c for c in counts) + ZONE_GAP * (ng - 1) <= WIDTH_MAX
    ]
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
    return [int(c) for c in base[offset : offset + count]]


def _choose_fill_lengths(rng, count: int) -> list[int]:
    """Pick `count` distinct fill lengths for one region."""
    pool = list(range(1, MAX_FILL_LEN + 1))
    rng.shuffle(pool)
    return [int(v) for v in pool[:count]]


def _bar_colors_for_region(rng, count: int, cap_pool: list[int], fill_pool: list[int]) -> list[tuple[int, int]]:
    """Reuse a small global palette so region ownership cannot be inferred from color alone."""
    offset = int(rng.integers(len(cap_pool)))
    colors = []
    for i in range(count):
        cap = int(cap_pool[(offset + i) % len(cap_pool)])
        fill = int(fill_pool[(offset + i + 1) % len(fill_pool)])
        colors.append((cap, fill))
    return colors


def _rank_is_informative(count: int, rank: int) -> bool:
    """Return True iff shortest-first and longest-first pick different bars."""
    return 2 * rank != count + 1


def _distance_gap(col: int, legend_xs: list[int], owner: int) -> int:
    """How much farther the second-closest legend is than the closest one."""
    dists = sorted(abs(int(col) - int(x)) for x in legend_xs)
    if len(dists) < 2:
        return 99
    return int(dists[1] - dists[0])


def _has_unique_owners(group_columns: list[list[int]], legend_cols: list[int]) -> bool:
    """Every bar column must belong to its intended region with a strict nearest marker."""
    for idx, cols in enumerate(group_columns):
        for col in cols:
            if _legend_owner(int(col), legend_cols) != idx:
                return False
            if _distance_gap(int(col), legend_cols, idx) <= 0:
                return False
    return True


def _shared_cap_count(region_caps: list[list[int]]) -> int:
    cap_to_regions: dict[int, set[int]] = {}
    for idx, caps in enumerate(region_caps):
        for cap in caps:
            cap_to_regions.setdefault(int(cap), set()).add(int(idx))
    return sum(1 for owners in cap_to_regions.values() if len(owners) > 1)


def _expected_output(inp: np.ndarray, legend_colors: list[int], group_columns: list[list[int]], group_lengths: list[list[int]], ranks: list[int]) -> np.ndarray:
    """Compute the expected output by recoloring the rank-k shortest bar in each region."""
    out = inp.copy()
    for paint, cols, lengths, rank in zip(legend_colors, group_columns, group_lengths, ranks):
        ranked = sorted(zip(cols, lengths), key=lambda item: (item[1], item[0]))
        target_col, target_len = ranked[rank - 1]
        top_r = _MAX_CAP_ROW - target_len
        out[top_r + 1 : _BOTTOM_ROW, target_col] = int(paint)
    return out


def _descending_reference_output(inp: np.ndarray, legend_colors: list[int], group_columns: list[list[int]], group_lengths: list[list[int]], ranks: list[int]) -> np.ndarray:
    """Reference output for the same spatial grouping but longest-first ranking."""
    out = inp.copy()
    for paint, cols, lengths, rank in zip(legend_colors, group_columns, group_lengths, ranks):
        ranked = sorted(zip(cols, lengths), key=lambda item: (-item[1], item[0]))
        target_col, target_len = ranked[rank - 1]
        top_r = _MAX_CAP_ROW - target_len
        out[top_r + 1 : _BOTTOM_ROW, target_col] = int(paint)
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid combo sample."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(MIN_GROUPS, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, count + 1)) for count in counts]
    if not any(_rank_is_informative(count, rank) for count, rank in zip(counts, ranks)):
        return None

    min_width = sum(2 * c for c in counts) + ZONE_GAP * (n_groups - 1)
    width = int(rng.integers(max(WIDTH_MIN, min_width), WIDTH_MAX + 1))
    regions = _layout_regions(rng, counts, width)
    legend_cols = [int((start + end) // 2) for start, end in regions]

    group_columns: list[list[int]] = []
    for count, region, legend_col in zip(counts, regions, legend_cols):
        cols = _choose_group_columns(rng, region, count, legend_col)
        if cols is None:
            return None
        group_columns.append(cols)
    if not _has_unique_owners(group_columns, legend_cols):
        return None

    palette = rng.choice(np.arange(*COLOR_RANGE), size=n_groups + 4, replace=False)
    legend_colors = [int(v) for v in palette[:n_groups]]
    cap_pool = [int(v) for v in palette[n_groups : n_groups + 2]]
    fill_pool = [int(v) for v in palette[n_groups + 2 : n_groups + 4]]

    grid = np.zeros((GRID_H, width), dtype=int)
    group_lengths: list[list[int]] = []
    region_caps: list[list[int]] = []
    for idx, (paint, count, rank, cols) in enumerate(zip(legend_colors, counts, ranks, group_columns)):
        paste(grid, np.full((rank, 1), paint, dtype=int), 0, legend_cols[idx])

        lengths = _choose_fill_lengths(rng, count)
        colors = _bar_colors_for_region(rng, count, cap_pool, fill_pool)
        group_lengths.append(lengths)
        region_caps.append([cap for cap, _fill in colors])

        for col, fill_len, (cap, fill) in zip(cols, lengths, colors):
            paste(grid, _build_bar(cap, fill, fill_len), _MAX_CAP_ROW - fill_len, col)

    if _shared_cap_count(region_caps) == 0:
        return None

    expected = _expected_output(grid, legend_colors, group_columns, group_lengths, ranks)
    if not np.array_equal(main(grid), expected):
        return None

    if np.array_equal(expected, _descending_reference_output(grid, legend_colors, group_columns, group_lengths, ranks)):
        return None

    metrics = sample_metrics(grid)
    if metrics["shared_cap_colors"] <= 0 or metrics["rank_signal"] <= 0 or metrics["legend_bar_color_overlap"] != 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
