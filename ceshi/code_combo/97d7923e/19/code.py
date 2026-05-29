# Combo variant of 97d7923e (index 19)
# combo_spec: D2/2+D5/0
# Name: 位置序号编码排名+右下角图例
#
# Design analysis
# a) 具体化与行为追踪:
#    原题在左上角放若干竖直色块，某色块的高度 k 直接表示“同帽色条中第 k 长的那根要被改色”。
#    本融合版把图例移到右下角，并把每个图例条目压缩成单个像素: 最右列最底部的像素代表 rank=1，
#    它上方的下一个图例像素代表 rank=2，再上方代表 rank=3。例: 若右下角最右列从下到上依次是
#    红、黄、蓝，那么红帽组改第 1 长条，黄帽组改第 2 长条，蓝帽组改第 3 长条。相比原题，
#    解题者不能再读“高度”，而必须先定位右下角图例，再从位置顺序恢复排名。
# b) 复杂度校准:
#    为了让“位置序号编码”在样例里真正起作用，生成器强制使用 2-3 个图例条目，避免退化成只有
#    一个条目时 rank 永远是 1。每组仍只有 2-3 根条，rank 档位也因此稳定在 1-3，视觉上容易归纳。
# c) 隐式假设审查:
#    原 `_legend_entries` 依赖“顶行有图例”“图例高度编码 rank”“图例分散在多个列里”。这些前提
#    全部失效。新版本改成读取最右侧图例列，并把自底向上的序号当作 rank。`_vertical_sandwiches`
#    只关心竖条结构，与图例位置无关，可以保留。`main()` 里原先的 `k` 来自块高，现在必须来自位置。
# d) 放置合法性:
#    条仍全部底对齐，图例固定在最右列；条区与图例列之间至少留 3 列背景，所以不会重叠，也不会把
#    右下角图例误看成某根底对齐条。图例像素之间留 1-2 行背景，保证它们是彼此分离的单像素条目。
# e) 视觉可观察性:
#    样例里左上角不再有线索，而右下角出现一串离散彩色像素；输出总是让这些颜色对应帽色组中的第
#    1/2/3 长条分别被重着色。多组样例足以支持“顺序编码 rank”这一归纳，而不会误导成人眼去读高度。
# f) 可视化整洁度:
#    图例只占最右列的少量像素，主场景的条全部在左侧区域，中间保持空带，区域职责清晰，不会叠层。
# g) 变化的可推断性:
#    没有新增任意颜色映射。图例颜色仍直接指向对应的帽色组，变化仅是“该组里哪一根条被选中”，
#    而这个选择完全由右下角图例的相对位置提供结构线索。
# 9) main() 的新假设审查:
#    - 最右侧非背景列就是图例列，且其最底部图例像素接触最后一行。
#    - 图例条目是单像素，按自底向上顺序分别编码 rank=1,2,3...；相邻条目之间至少隔 1 行背景。
#    - 每个图例颜色对应一组存在的帽色条，且该组条数不少于该颜色在图例中的位置序号。
#    - 条区与图例列横向分离，因此最右列不会混入真正的底对齐条。

import itertools
import sys

import numpy as np

sys.path.append('/home/lixinyu/snap/new/solvers')
from arc_utils import get_bg, paste


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
    Legend is a bottom-right vertical list of single pixels in the rightmost legend column.

    Returns entries as (row_pos, legend_color, rank), ordered from bottom to top.
    The bottommost entry has rank 1, the next one upward has rank 2, and so on.
    """
    h, w = g.shape
    non_bg_cols = [int(c) for c in np.flatnonzero(np.any(g != bg, axis=0))]
    assert non_bg_cols, 'grid must contain non-background pixels'
    legend_c = max(non_bg_cols)

    rows = [int(r) for r in np.flatnonzero(g[:, legend_c] != bg)]
    assert rows, 'legend column must be non-empty'
    rows.sort(reverse=True)
    assert rows[0] == h - 1, 'bottommost legend entry must touch the last row'

    for upper, lower in zip(rows, rows[1:]):
        assert upper - lower >= 2, 'legend entries must be separated vertically'

    return [(r, int(g[r, legend_c]), rank) for rank, r in enumerate(rows, start=1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor vertical bars using a bottom-right legend whose order encodes rank.

    Input structure:
    - bg = background. In the bottom-right corner, the rightmost legend column contains several
      single-pixel color markers separated by blank rows; the bottom marker is first, then the next
      marker upward, and so on.
    - Elsewhere: vertical cap-fill-cap bars, grouped by shared cap color, all bottom-aligned.

    Rule:
    - Read the rightmost legend column from bottom to top. A marker's color K identifies one bar group,
      and that marker's ordinal position gives rank k (first marker -> rank 1, second -> rank 2, ...).
    - Within the K-capped bars, sort by interior length descending.
    - Recolor the interior of the rank-k bar to K.

    Key insight: The legend pixels no longer encode rank by their size. Each marker is only one cell tall;
    the hidden rank comes from where that marker sits in the bottom-right ordering.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, 'legend must be non-empty'

    for _row, cap, k in legend:
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
MIN_GROUPS = 2
MAX_GROUPS = 3
MAX_TOTAL_BARS = 8
BARS_PER_GROUP = (2, 3)
MAX_FILL_LEN = 7
COLOR_RANGE = (1, 10)
COL_SPACING = 2
LEGEND_ROW_GAP_CHOICES = (1, 2)
LEGEND_GAP_TO_BARS = 3
MAX_RETRIES = 300

_BOTTOM_ROW = GRID_H - 1
_MAX_CAP_ROW = GRID_H - 2


def _min_bar_span(total_bars: int) -> int:
    return 1 + COL_SPACING * (total_bars - 1)


def _legend_orders(counts: tuple[int, ...]) -> list[tuple[int, ...]]:
    """All bottom-to-top group orders whose ordinal ranks fit each group's bar count."""
    n_groups = len(counts)
    orders: list[tuple[int, ...]] = []
    for order in itertools.permutations(range(n_groups)):
        if all(rank <= counts[group_idx] for rank, group_idx in enumerate(order, start=1)):
            orders.append(tuple(int(v) for v in order))
    return orders


_FEASIBLE_COUNTS = {
    ng: [
        tuple(int(v) for v in counts)
        for counts in itertools.product(BARS_PER_GROUP, repeat=ng)
        if sum(counts) <= MAX_TOTAL_BARS
        and max(WIDTH_MIN, _min_bar_span(sum(counts)) + LEGEND_GAP_TO_BARS + 1) <= WIDTH_MAX
        and _legend_orders(tuple(int(v) for v in counts))
    ]
    for ng in range(MIN_GROUPS, MAX_GROUPS + 1)
}


def _build_bar(cap: int, fill: int, fill_len: int) -> np.ndarray:
    """Return a (fill_len+2, 1) column sprite: cap / fill*N / cap."""
    bar = np.full((fill_len + 2, 1), fill, dtype=int)
    bar[0, 0] = cap
    bar[-1, 0] = cap
    return bar


def _assign_bar_columns(rng, n_groups: int, counts: tuple[int, ...], width: int) -> list[list[int]]:
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


def _legend_rows(rng, n_groups: int) -> list[int]:
    """Place legend entries bottom-to-top in the final column, with 1-2 blank rows between them."""
    gaps = [int(rng.choice(LEGEND_ROW_GAP_CHOICES)) for _ in range(n_groups - 1)]
    rows = [_BOTTOM_ROW]
    for gap in gaps:
        rows.append(rows[-1] - gap - 1)
    assert rows[-1] >= 0, 'legend must fit within the grid height'
    return rows


def _choose_fill_lengths(rng, count: int) -> list[int]:
    """Pick `count` distinct fill lengths for one group."""
    vals = rng.choice(np.arange(1, MAX_FILL_LEN + 1), size=count, replace=False)
    return [int(v) for v in vals]


def _ranks_from_order(order: tuple[int, ...], n_groups: int) -> list[int]:
    ranks = [0] * n_groups
    for rank, group_idx in enumerate(order, start=1):
        ranks[group_idx] = rank
    return ranks


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
        ranked = sorted(zip(cols, lengths), key=lambda item: (-item[1], item[0]))
        target_col, target_len = ranked[rank - 1]
        top_r = _MAX_CAP_ROW - target_len
        out[top_r + 1 : _BOTTOM_ROW, target_col] = cap
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(MIN_GROUPS, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    if not feasible:
        return None
    counts = feasible[int(rng.integers(len(feasible)))]
    orders = _legend_orders(counts)
    legend_order = orders[int(rng.integers(len(orders)))]
    ranks = _ranks_from_order(legend_order, n_groups)

    palette = rng.choice(np.arange(*COLOR_RANGE), size=2 * n_groups, replace=False)
    caps = [int(v) for v in palette[:n_groups]]
    fills = [int(v) for v in palette[n_groups:]]

    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, _min_bar_span(total_bars) + LEGEND_GAP_TO_BARS + 1)
    width = int(rng.integers(min_width, WIDTH_MAX + 1))

    bar_region_width = width - LEGEND_GAP_TO_BARS - 1
    group_columns = _assign_bar_columns(rng, n_groups, counts, bar_region_width)
    legend_rows = _legend_rows(rng, n_groups)
    legend_col = width - 1

    grid = np.zeros((GRID_H, width), dtype=int)
    group_lengths: list[list[int]] = []
    for cap, fill, count, cols in zip(caps, fills, counts, group_columns):
        lengths = _choose_fill_lengths(rng, count)
        group_lengths.append(lengths)
        for col, fill_len in zip(cols, lengths):
            paste(grid, _build_bar(cap, fill, fill_len), _MAX_CAP_ROW - fill_len, col)

    for rank, group_idx in enumerate(legend_order, start=1):
        paste(grid, np.array([[caps[group_idx]]], dtype=int), legend_rows[rank - 1], legend_col)

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
