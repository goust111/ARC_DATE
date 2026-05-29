# Combo variant of 97d7923e (index 28)
# combo_spec: D3/3+D5/2
# Name: 第三色重着色+独立小矩形图例
#
# Design analysis
# a) 具体化与行为追踪
#    - 主体对象仍是原题的竖直 cap/fill/cap 单列条: 同组条共享帽色 K 与填充色 F, 按填充段长度降序排名。
#    - 融合后不再使用左上角连成一簇的单色柱状图例, 而是在上边缘分散放置若干彼此隔离的小矩形图例项。
#    - 每个图例项是一个独立的 kx2 矩形: 左列全是帽色 K, 右列全是目标色 T, 高度 k 编码排名。
#    - 例子: 顶边某处出现一个高 3 的双色小矩形, 左列为蓝, 右列为红。则在所有帽色=蓝的条中找到第 3 长的那根,
#      原题会把内部刷成蓝色, 本融合版改为刷成红色。差异同时体现在“图例位置/形状”与“输出颜色”两处。
# b) 复杂度校准
#    - 每组仍只有一个排名值和一个目标条, 排序规则未增加分支; 复杂度主要来自图例识别从“角落柱”变成“分散矩形”。
#    - 图例项数量仍与条组数量一致(1-4 组), 且所有图例都复用统一的双色矩形模板, 人类可以从少量样例中看到重复模式。
#    - 目标色 T 始终与帽色 K 不同, 保证该变体在输出上稳定可见, 不会退化成原题的“刷回帽色”。
# c) 隐式假设审查
#    - 原 solver 假设所有图例都在左上连续列内; 这里该前提失效, 必须改为扫描整条上边缘上的独立连通块。
#    - 原 solver 假设图例只提供一种颜色且该颜色同时负责匹配与重着色; 这里必须同时解析 K 与 T 两列。
#    - 原 solver 对条的识别只依赖“同色帽 + 纯色内部”, 这一点仍成立, 因而 `_vertical_sandwiches` 可以复用。
#    - 新 parser 额外假设图例项是触顶、宽度恰为 2 的实心矩形, 且左右两列各自纯色; 生成器按该模板固定构造。
# d) 放置合法性
#    - 每个图例项占用一根真实条的列作为左列(K 列), 右侧紧邻一列专门给目标色 T, 该右列不放任何条。
#    - 图例项之间至少隔着一根没有图例的条槽位, 这样它们在顶边不会横向粘连; 同时所有条整体压低到第 3 行以下,
#      令图例左右两侧在图例高度范围内也保持背景, 从而形成真正独立的小矩形区域。
#    - 与图例共列的那根条还要额外满足 `<= 7-k`, 以保证图例正下方至少保留一行背景, 避免与条顶帽连成一体。
# e) 视觉可观察性
#    - 解题者能先在上边缘找到分散的双色矩形, 再用左列颜色匹配对应条组, 再用矩形高度决定“第几长”。
#    - 排名只依赖肉眼可见的条长, 颜色作用由图例中的左右列直接提示, 不依赖颜色编号等不可视信息。
# f) 可视化整洁度
#    - 图例全部限定在上边缘, 条形主体仍位于下方, 两类结构以一行背景隔开, 层次清楚。
#    - 每个图例项都保持规整的矩形外形, 比零散单像素标记更容易与主体条区分。
# g) 变化的可推断性
#    - 颜色语义被拆成“左列 K 负责找组, 右列 T 负责上色”, 高度 k 继续承担排名语义, 三种信息各有固定载体。
#    - 生成器保证所有图例颜色两两不同, 且条的原始填充色不复用任何图例颜色, 使 K/T/F 三类角色更易分辨。
# 9) main 的新假设与生成器保证
#    - 假设所有图例连通块都触及顶边且不互相接触; 生成器把它们放在分散的顶边槽位并插入背景间隔。
#    - 假设每个图例项都是高度 1-3、宽度 2 的实心双色矩形; 生成器严格按此模板构造。
#    - 假设每个图例帽色 K 都至少对应 2 根条且排名 k 不超过该组条数; 生成器沿用原题的 2-3 根/组约束。
#    - 假设目标色 T 与帽色 K、原填充色 F 都不同; 生成器从剩余颜色池挑选 F, 并令图例双色全局互异。

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


def _legend_entries(g: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    """
    Parse the scattered top-edge legend rectangles.

    Returns entries as (x_pos, cap_color, target_color, rank), sorted by x_pos.
    Each entry is a solid top-touching rectangle of size (rank x 2):
      - left column: cap color K
      - right column: repaint color T
      - rectangle height rank encodes which ranked bar to select
    """
    h, w = g.shape
    top_cols = [int(c) for c in np.flatnonzero(g[0, :] != bg)]
    assert top_cols, 'legend row must be non-empty'

    seen = np.zeros((h, w), dtype=bool)
    entries: list[tuple[int, int, int, int]] = []
    for c in top_cols:
        if seen[0, c]:
            continue
        st = [(0, c)]
        seen[0, c] = True
        pix: list[tuple[int, int]] = []
        while st:
            r, cc = st.pop()
            pix.append((r, cc))
            for rr, ccc in neighbors4(r, cc):
                if 0 <= rr < h and 0 <= ccc < w and (not seen[rr, ccc]) and int(g[rr, ccc]) != bg:
                    seen[rr, ccc] = True
                    st.append((rr, ccc))

        rows = [r for r, _ in pix]
        cols = [cc for _, cc in pix]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        height = max_r - min_r + 1
        width = max_c - min_c + 1

        assert min_r == 0, 'legend components must touch the top edge'
        assert width == 2, 'legend components must be two columns wide'
        assert len(pix) == height * width, 'legend components must be solid rectangles'

        cap = int(g[0, min_c])
        target = int(g[0, min_c + 1])
        assert cap != bg and target != bg and cap != target, 'legend colors must be distinct non-bg values'
        assert np.all(g[:height, min_c] == cap), 'legend left stripe must be monochrome'
        assert np.all(g[:height, min_c + 1] == target), 'legend right stripe must be monochrome'

        if height < h:
            assert np.all(g[height, min_c : max_c + 1] == bg), 'legend must have a bg row beneath it'
        if min_c > 0:
            assert np.all(g[:height, min_c - 1] == bg), 'legend must have a bg column on the left'
        if max_c + 1 < w:
            assert np.all(g[:height, max_c + 1] == bg), 'legend must have a bg column on the right'

        entries.append((min_c, cap, target, height))

    entries.sort(key=lambda t: t[0])
    return entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use scattered two-color mini-rectangles to select and repaint ranked bars.

    Input structure:
    - Background is the dominant color. Along the top edge sit several isolated small rectangles, each separated from others by background.
    - Each legend rectangle is `k x 2`: its left column has cap color K, its right column has target color T, and its height k encodes a rank.
    - The main field contains bottom-aligned vertical cap-fill-cap bars. Bars in the same group share cap color K and have solid interiors.

    Rule:
    - For each legend entry (K, T, k), collect all bars whose two endpoints are color K.
    - Sort those bars by interior height from longest to shortest.
    - Recolor only the interior of the rank-k bar to T; the cap cells stay unchanged.

    Key insight: The legend is no longer a corner stack. You must first detect the isolated top-edge rectangles, then read three separate cues from each one: left color picks the bar group, rectangle height gives the rank, and right color supplies the repaint color.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, 'legend must be non-empty'

    for _x, cap, target, rank in legend:
        segs = _vertical_sandwiches(g, cap)
        assert segs, f'legend cap {cap} must have matching vertical bars'
        segs.sort(key=lambda s: (-(s[1] - s[0] - 1), s[0], s[2]))
        assert 1 <= rank <= len(segs), f'legend rank {rank} out of range for {len(segs)} bars'
        top_r, bot_r, col, _fill = segs[rank - 1]
        out[top_r + 1 : bot_r, col] = target

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
COL_SPACING = 2              # minimum column stride between actual bars
MAX_RETRIES = 300            # retry budget for generate_input

_BOTTOM_ROW = GRID_H - 1     # row index where all bars end
_MAX_CAP_ROW = GRID_H - 2    # highest row a bar's top cap can occupy
_ALL_NON_BG = list(range(*COLOR_RANGE))
_MIN_BAR_TOP_ROW = 3         # keep rows 0..2 free around top-edge legend rectangles
_BAR_MAX_FILL_LEN = _MAX_CAP_ROW - _MIN_BAR_TOP_ROW

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


def _actual_bar_columns(rng, total_bars: int, width: int) -> list[int]:
    """Choose bar columns with stride >= 2, allowing random leading/inter-bar slack."""
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    extra_width = width - min_width
    slack = rng.multinomial(extra_width, np.full(total_bars + 1, 1.0 / (total_bars + 1)))

    cols = [int(slack[0])]
    for idx in range(total_bars - 1):
        cols.append(cols[-1] + COL_SPACING + int(slack[idx + 1]))
    return cols


def _assign_columns(
    rng, n_groups: int, counts: list[int], width: int
) -> tuple[list[list[int]], list[int]]:
    """
    Distribute actual bar columns across groups and choose one legend-sharing anchor per group.

    Anchor bars occupy non-adjacent slot indices so their top-edge 2-column legend rectangles
    remain separated by at least one background column.
    """
    total_bars = sum(counts)
    actual_cols = _actual_bar_columns(rng, total_bars, width)

    anchor_slots = [idx for idx in range(0, total_bars, 2) if idx < total_bars - 1]
    assert len(anchor_slots) >= n_groups, 'need enough separated slots for legend anchors'
    chosen_anchor_slots = sorted(int(v) for v in rng.choice(anchor_slots, size=n_groups, replace=False))

    group_for_slot = [-1] * total_bars
    anchor_cols = [-1] * n_groups
    group_order = list(range(n_groups))
    rng.shuffle(group_order)
    for group, slot in zip(group_order, chosen_anchor_slots):
        group_for_slot[slot] = group
        anchor_cols[group] = actual_cols[slot]

    remaining_slots = [idx for idx, owner in enumerate(group_for_slot) if owner < 0]
    remaining_owners = [group for group, count in enumerate(counts) for _ in range(count - 1)]
    rng.shuffle(remaining_owners)
    for slot, owner in zip(remaining_slots, remaining_owners):
        group_for_slot[slot] = owner

    group_columns = [[] for _ in range(n_groups)]
    for slot, col in enumerate(actual_cols):
        group_columns[group_for_slot[slot]].append(int(col))
    for cols in group_columns:
        cols.sort()

    return group_columns, anchor_cols


def _choose_fill_lengths(rng, cols: list[int], anchor_col: int, rank: int) -> list[int]:
    """
    Pick distinct fill lengths for one group.

    The legend-sharing anchor bar must fit under a rank-high top-edge rectangle with one bg row.
    """
    count = len(cols)
    max_anchor_len = min(_BAR_MAX_FILL_LEN, MAX_FILL_LEN - rank)
    anchor_len = int(rng.integers(1, max_anchor_len + 1))
    pool = [length for length in range(1, _BAR_MAX_FILL_LEN + 1) if length != anchor_len]
    rng.shuffle(pool)

    by_col = {anchor_col: anchor_len}
    other_cols = [col for col in cols if col != anchor_col]
    for col, length in zip(other_cols, pool[: count - 1]):
        by_col[col] = int(length)
    return [by_col[col] for col in cols]


def _choose_fills(rng, n_groups: int, forbidden: set[int]) -> list[int]:
    """Pick group fill colors from non-legend colors, allowing repeats if needed."""
    pool = [color for color in _ALL_NON_BG if color not in forbidden]
    assert pool, 'at least one non-legend color must remain for bar fills'
    return [int(pool[int(rng.integers(len(pool)))]) for _ in range(n_groups)]


def _expected_output(
    inp: np.ndarray,
    caps: list[int],
    targets: list[int],
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
) -> np.ndarray:
    """Compute the expected output by recoloring the rank-k bar's fill to the legend target."""
    out = inp.copy()
    for cap, target, cols, lengths, rank in zip(caps, targets, group_columns, group_lengths, ranks):
        ranked = sorted(zip(cols, lengths), key=lambda item: -item[1])
        target_col, target_len = ranked[rank - 1]
        top_r = _MAX_CAP_ROW - target_len
        out[top_r + 1 : _BOTTOM_ROW, target_col] = target
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, count + 1)) for count in counts]

    legend_palette = rng.choice(np.array(_ALL_NON_BG), size=2 * n_groups, replace=False)
    caps = [int(v) for v in legend_palette[:n_groups]]
    targets = [int(v) for v in legend_palette[n_groups:]]
    fills = _choose_fills(rng, n_groups, set(map(int, legend_palette)))

    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    group_columns, anchor_cols = _assign_columns(rng, n_groups, counts, width)

    grid = np.zeros((GRID_H, width), dtype=int)
    group_lengths: list[list[int]] = []
    for cap, target, fill, rank, cols, anchor_col in zip(
        caps, targets, fills, ranks, group_columns, anchor_cols
    ):
        paste(grid, np.full((rank, 1), cap, dtype=int), 0, anchor_col)
        paste(grid, np.full((rank, 1), target, dtype=int), 0, anchor_col + 1)

        lengths = _choose_fill_lengths(rng, cols, anchor_col, rank)
        group_lengths.append(lengths)

        for col, fill_len in zip(cols, lengths):
            paste(grid, _build_bar(cap, fill, fill_len), _MAX_CAP_ROW - fill_len, col)

    expected = _expected_output(grid, caps, targets, group_columns, group_lengths, ranks)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f'failed to generate a valid input after {MAX_RETRIES} attempts')
