# Variant of 97d7923e
# Dimension: D5 — 目标结构：图例的编码方式 (symbol_instructions)
# Change: 图例在左上角：每个条目为一列，从第0行开始向下堆叠k个同色像素，k编码该帽色组内的目标排名；条目间间隔1-2列背景色... → 图例改为右下角
#
# Design analysis
# a) 具体化与行为追踪:
#    变体后的图例不再接触顶边，而是形成一个贴着底边和右边的竖条簇。每个图例条目仍是一列同色像素，
#    但现在从最后一行向上堆叠 k 个像素，k 仍表示该帽色组中按长度降序排序后的第 k 根条。
#    例: 原题会从 row 0 读到颜色 2 的高度 3，于是选帽色 2 组里第 3 长的条；变体则要先在底行最右侧
#    找到颜色 2 的向上堆叠高度 3，再执行同一排名逻辑。差异集中在“先定位并解码图例”的阶段。
# b) 复杂度校准:
#    只改图例锚点和读法，长度排序、同帽色分组、重着色规则不变，保持原题核心推理。为了让右下角图例
#    明显可见，生成器强制图例与条区横向分离，避免看起来像普通底部条形。
# c) 隐式假设审查:
#    原 `_legend_entries` 假设顶行必有非背景像素、图例从最左侧开始、图例可与条共列且仅需向右限界。
#    这些前提在变体里都失效: 顶行可能全空；图例改为从右缘回扫；图例因占据底行而不能与底对齐条共列。
#    `_vertical_sandwiches` 不依赖图例位置，因此保留。
# d) 放置合法性:
#    条仍然全部底对齐在 row 9；图例固定放在最右簇，中间留至少 3 列背景，因此不会与条重叠，也能让
#    底行扫描时把右下角图例簇与左侧条区明确分开。
# e) 视觉可观察性:
#    2-4 个样例里，人类可以直接看到“左上角线索消失、右下角出现新线索”。图例高度仍只有少量档位
#    (2-3 条/组，所以排名 1-3)，可从多个组的重复支持里推断其语义。
# f) 可视化整洁度:
#    图例只占右侧窄簇，条只占左侧区域，二者之间有宽空带，不会在同一区域叠层。
# g) 颜色变化可推断性:
#    颜色映射没有新增自由度。图例颜色仍直接对应帽色，输出仍只把某根条的内部填充改成该帽色。
# 9) main() 的新假设审查:
#    - 底行右端必须存在一个由 1-2 列间隔连接成的图例簇。
#    - 图例簇左侧与条区至少相隔 3 列背景，这样自右向左扫描遇到 >2 的缺口时就能停下。
#    - 每个图例颜色在图例簇内只出现一个 4 连通分量；生成器用不同帽色保证这一点。

import itertools
import sys

import numpy as np

sys.path.append('/home/lixinyu/snap/new/solvers')
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
    Legend is in the bottom-right: a cluster of non-bg pixels on the last row near the right edge,
    allowing small gaps (1-2 bg columns) between entries.

    Returns entries as (x_pos, legend_color, block_count), sorted by x_pos.
    block_count is the vertical height of the 4-connected component of that color that touches
    the bottom row within the legend cluster (i.e., how many stacked blocks from bottom upward).
    """
    h, w = g.shape
    bottom_pos = [int(c) for c in np.flatnonzero(g[h - 1, :] != bg)]
    assert bottom_pos, 'legend row must be non-empty'
    bottom_pos.sort(reverse=True)

    cols = [bottom_pos[0]]
    for c in bottom_pos[1:]:
        if cols[-1] - c <= 2:
            cols.append(c)
        else:
            break
    cols.sort()
    legend_min_c = min(cols)

    entries: list[tuple[int, int, int]] = []
    seen = np.zeros((h, w), dtype=bool)
    for c in cols:
        col = int(g[h - 1, c])
        if col == bg or seen[h - 1, c]:
            continue
        st = [(h - 1, c)]
        seen[h - 1, c] = True
        pix = []
        while st:
            r, cc = st.pop()
            pix.append((r, cc))
            for rr, ccc in neighbors4(r, cc):
                if 0 <= rr < h and legend_min_c <= ccc < w and (not seen[rr, ccc]) and int(g[rr, ccc]) == col:
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
    Recolor vertical bars by a rank legend anchored in the opposite corner.

    Input structure:
    - bg = background. Bottom-right: monochrome blocks touching the last row; each has color K and height k,
      stacked upward from the bottom edge.
    - Elsewhere: vertical bars capped top and bottom by a single color, with solid-color interiors.

    Rule:
    - For each legend entry (K, k), find vertical bars whose cap color is K.
    - Sort those bars by interior length, longest first.
    - Recolor the interior of the rank-k bar to K.

    Key insight: The important signal is not the corner itself but the upward stack height in the bottom-right legend,
    which encodes which length-ranked bar to recolor within the matching cap-color group.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, 'legend must be non-empty'

    for _x, cap, k in legend:
        segs = _vertical_sandwiches(g, cap)
        assert segs, f'legend cap {cap} must have matching vertical bars'
        segs.sort(key=lambda s: (-(s[1] - s[0] - 1), s[0], s[2]))
        assert 1 <= k <= len(segs), f'legend rank {k} out of range for {len(segs)} bars'
        top_r, bot_r, col, _fill = segs[k - 1]
        out[top_r + 1 : bot_r, col] = cap

    return out


# --------------- generator (O2) ---------------

# --- Configuration ---
GRID_H = 10
WIDTH_MIN = 10
WIDTH_MAX = 20
MAX_GROUPS = 3
MAX_TOTAL_BARS = 8
BARS_PER_GROUP = (2, 3)
MAX_FILL_LEN = 7
COLOR_RANGE = (1, 10)
COL_SPACING = 2
LEGEND_GAP_CHOICES = (1, 2)
LEGEND_GAP_TO_BARS = 3
MAX_RETRIES = 300

_BOTTOM_ROW = GRID_H - 1
_MAX_CAP_ROW = GRID_H - 2


def _min_bar_span(total_bars: int) -> int:
    return 1 + COL_SPACING * (total_bars - 1)


def _min_legend_span(n_groups: int) -> int:
    return n_groups + (n_groups - 1) * min(LEGEND_GAP_CHOICES)


_FEASIBLE_COUNTS = {
    ng: [
        list(c)
        for c in itertools.product(BARS_PER_GROUP, repeat=ng)
        if sum(c) <= MAX_TOTAL_BARS
        and max(WIDTH_MIN, _min_bar_span(sum(c)) + LEGEND_GAP_TO_BARS + _min_legend_span(ng)) <= WIDTH_MAX
    ]
    for ng in range(1, MAX_GROUPS + 1)
}


def _build_bar(cap: int, fill: int, fill_len: int) -> np.ndarray:
    """Return a (fill_len+2, 1) column sprite: cap / fill*N / cap."""
    bar = np.full((fill_len + 2, 1), fill, dtype=int)
    bar[0, 0] = cap
    bar[-1, 0] = cap
    return bar


def _assign_bar_columns(rng, n_groups: int, counts: list[int], width: int) -> list[list[int]]:
    """Distribute bar columns inside the left region, leaving at least one bg col between bars."""
    total_bars = sum(counts)
    base_span = _min_bar_span(total_bars)
    extra_width = width - base_span
    parts = rng.multinomial(extra_width, np.full(total_bars, 1.0 / total_bars))

    occupied = [int(parts[0])]
    for idx in range(total_bars - 1):
        occupied.append(occupied[-1] + COL_SPACING + int(parts[idx + 1]))

    group_columns = [[] for _ in range(n_groups)]
    owners = [i for i, count in enumerate(counts) for _ in range(count)]
    rng.shuffle(owners)
    for col, owner in zip(occupied, owners):
        group_columns[owner].append(int(col))

    return [sorted(cols) for cols in group_columns]


def _assign_legend_columns(gaps: list[int], width: int) -> list[int]:
    """Place the legend as a right-anchored cluster on the bottom row."""
    span = 1
    for gap in gaps:
        span += gap + 1
    start = width - span
    cols = [start]
    for gap in gaps:
        cols.append(cols[-1] + gap + 1)
    return [int(c) for c in cols]


def _choose_fill_lengths(rng, count: int) -> list[int]:
    """Pick `count` distinct fill lengths for one group."""
    vals = rng.choice(np.arange(1, MAX_FILL_LEN + 1), size=count, replace=False)
    return [int(v) for v in vals]


def _expected_output(
    inp: np.ndarray,
    caps: list[int],
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
) -> np.ndarray:
    """Compute the expected output by recoloring the rank-k bar's fill to cap."""
    out = inp.copy()
    for cap, cols, lengths, rank in zip(caps, group_columns, group_lengths, ranks):
        ranked = sorted(zip(cols, lengths), key=lambda item: -item[1])
        target_col, target_len = ranked[rank - 1]
        top_r = _MAX_CAP_ROW - target_len
        out[top_r + 1 : _BOTTOM_ROW, target_col] = cap
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    if not feasible:
        return None
    counts = feasible[int(rng.integers(len(feasible)))]
    ranks = [int(rng.integers(1, c + 1)) for c in counts]

    palette = rng.choice(np.arange(*COLOR_RANGE), size=2 * n_groups, replace=False)
    caps = [int(v) for v in palette[:n_groups]]
    fills = [int(v) for v in palette[n_groups:]]

    legend_gaps = [int(rng.choice(LEGEND_GAP_CHOICES)) for _ in range(n_groups - 1)]
    legend_span = n_groups + sum(legend_gaps)
    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, _min_bar_span(total_bars) + LEGEND_GAP_TO_BARS + legend_span)
    if min_width > WIDTH_MAX:
        return None
    width = int(rng.integers(min_width, WIDTH_MAX + 1))

    bar_region_width = width - LEGEND_GAP_TO_BARS - legend_span
    group_columns = _assign_bar_columns(rng, n_groups, counts, bar_region_width)
    legend_columns = _assign_legend_columns(legend_gaps, width)

    grid = np.zeros((GRID_H, width), dtype=int)
    group_lengths: list[list[int]] = []
    for cap, fill, count, rank, cols, legend_col in zip(caps, fills, counts, ranks, group_columns, legend_columns):
        paste(grid, np.full((rank, 1), cap, dtype=int), GRID_H - rank, legend_col)

        lengths = _choose_fill_lengths(rng, count)
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
    raise RuntimeError(f'failed to generate a valid input after {MAX_RETRIES} attempts')
