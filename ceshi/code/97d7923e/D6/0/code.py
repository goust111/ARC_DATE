# Variant of 97d7923e
# Dimension: D6 — 分组内排序规则 (meta)
# Change: 组内按填充长度降序排列（最长=rank 1），使用长度作为唯一排序键
#   → 按列位置排序替代长度排序（最左=rank 1），长度仅作为干扰信息保留

import itertools

import numpy as np

from arc_utils import get_bg, neighbors4, paste


# Design analysis for the D6 variant
# a) 具体化与行为追踪
#    - 输入仍由两部分组成：左上角图例 + 若干底对齐竖条。
#    - 每个图例项仍是一列单色块，颜色=某组 bar 的端点颜色 K，高度=k。
#    - 每组 bar 仍共享端点色 K 和填充色 F，bar 仍是 cap / fill... / cap 的竖直结构。
#    - 原题在每组内部按“填充长度从长到短”排名；本变体改成按“列坐标从左到右”排名。
#    - 具体追踪例子：
#      某组 K 有三根 bar，列为 c=2,7,10，填充长度分别为 2,6,4，legend 高度 k=2。
#      原题：长度排序后是 c=7 (6), c=10 (4), c=2 (2)，rank 2 选 c=10。
#      变体：位置排序后是 c=2, c=7, c=10，rank 2 选 c=7。
#      差异落在“选哪一列重着色”这一步，因此 solver 的排序键和 generator 的 oracle 都要同步修改。
#
# b) 复杂度校准
#    - 不增加新符号系统，不引入额外计数规则；只替换组内排名标准，复杂度与参考任务 31f7f899
#      的“按 x 位置定义槽位顺序”同量级，仍是单一空间排序规则。
#    - 为了让 D6 变化肉眼可见，保留组内不同长度，让长度继续作为显眼但误导性的干扰维度。
#    - 同时限制 rank 仍只取 1..3，bars/group 仍只取 2 或 3，避免自由度过大。
#
# c) 对原 main() 的隐式假设审查
#    - `_vertical_sandwiches()` 只负责找 bar，本身不依赖长度排序，保持不变。
#    - `_legend_entries()` 只负责读图例高度 k，也不依赖旧 D6，保持不变。
#    - 原 `segs.sort(key=lambda s: (-(length), row, col))` 明确假设“长度是主排序键”。
#      在变体中这个前提失效，必须改成按列号排序，否则 solver 会退回原题。
#    - 原 generator 的 `_expected_output()` 也以长度选目标列；若不改，oracle 会错误。
#    - 原 `_choose_fill_lengths()` 只保证长度各异和 legend 列能放下，本身仍然有效。
#
# d) 放置合法性
#    - 仍沿用原题的 10x(10..20) 网格、底对齐 bar、单列 bar、每列最多一个 bar。
#    - 每组最左侧 bar 仍与 legend 共列，且其长度受限，保证 legend 底部与 bar 顶 cap 之间至少隔 1 行背景。
#    - 所有 bar 列互异，`COL_SPACING=2` 保证列之间至少隔 1 列背景，不重叠不粘连。
#    - solver 依赖“同一列最多只存在一个有效 cap-fill-cap bar”；生成器继续保证这一点。
#
# e) 视觉可观察性
#    - 人类需要从少量样例中看出“图例高度对应第几根 bar”，因此排名必须是可见属性。
#    - 列位置是直接可见的；颜色 id、数组索引等不可见属性未参与排序。
#    - 为避免样例退化为原题，生成器强制至少有一个分组满足：
#      “按列位置选中的目标列” != “按长度排序选中的目标列”。
#
# f) 可视化整洁度
#    - 图例仍只占左上少量列；bar 仍是细长单列对象，结构简单清晰。
#    - 不叠加额外标记，不改变配色关系，因此视觉层次与原题一致，不会引入新噪声。
#
# g) 颜色变化可推断性
#    - 输出变化仍只有一种：被选中的 bar 内部由 fill 颜色改成 cap 颜色。
#    - 颜色映射线索完全由“同组端点色 K + 图例中的同色块”提供，不需要记忆任意颜色对应表。
#
# h) 防退化策略
#    - 测试 `main(input)==expected` 只能证明新 solver 与新 oracle 一致。
#    - 因此生成器额外检查：对同一输入，至少一个分组的新旧规则选中不同列。
#      这样样本在行为上确实体现了 D6 的变化，而不只是形式上改了排序键。
#
# i) 新 main() 的假设与 generator 保障
#    - 新 main() 假设：同色组内 bar 的列坐标可严格排序，且 legend 高度 k 不超过该组 bar 数。
#    - generator 保障：每组 2 或 3 根 bar，列唯一；rank 从 1..count 采样；因此排序与索引都合法。


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
    Recolor one vertical bar per color group using a legend-defined ordinal.

    Input structure:
    - The background surrounds a top-left legend made of one-column monochrome stacks touching row 0.
    - Elsewhere, each legend color appears as the cap color of several bottom-aligned vertical cap/fill/cap bars.

    Rule:
    - For each legend entry with color K and height k, collect all bars capped by K.
    - Order that color group's bars by column position from left to right.
    - Recolor the interior of the k-th bar in that spatial order to K.

    Key insight: The legend height is not a size match; it is an ordinal index into the left-to-right order of bars sharing the same cap color.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, "legend must be non-empty"

    for _x, cap, k in legend:
        segs = _vertical_sandwiches(g, cap)
        assert segs, f"legend cap {cap} must have matching vertical bars"
        segs.sort(key=lambda s: (s[2], s[0], s[1]))
        assert 1 <= k <= len(segs), f"legend rank {k} out of range for {len(segs)} bars"
        top_r, bot_r, col, _fill = segs[k - 1]
        out[top_r + 1 : bot_r, col] = cap

    return out


# --------------- generator (D6 variant) ---------------

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
    """Distribute bar columns across groups with random inter-group spacing.

    Returns one list of column indices per group; the leftmost column in each
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

    return [sorted(cols) for cols in group_columns]


def _choose_fill_lengths(rng, count: int, rank: int) -> list[int]:
    """Pick `count` distinct fill lengths for one group.

    The leftmost bar shares a column with the legend, so its top cap must stay
    below the legend with at least one background row in between.
    """
    max_leftmost_len = MAX_FILL_LEN - rank
    leftmost_len = int(rng.integers(1, max_leftmost_len + 1))
    pool = [length for length in range(1, MAX_FILL_LEN + 1) if length != leftmost_len]
    rng.shuffle(pool)
    return [leftmost_len] + pool[: count - 1]


def _position_target(cols: list[int], rank: int) -> int:
    """Column chosen by the new variant rule: k-th leftmost bar."""
    return sorted(int(c) for c in cols)[rank - 1]


def _length_target(cols: list[int], lengths: list[int], rank: int) -> int:
    """Column chosen by the original rule: k-th bar after length-desc sorting."""
    ranked = sorted(zip(cols, lengths), key=lambda item: (-int(item[1]), int(item[0])))
    return int(ranked[rank - 1][0])


def _expected_output(
    inp: np.ndarray,
    caps: list[int],
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
) -> np.ndarray:
    """Compute the expected output using left-to-right ranking within each group."""
    out = inp.copy()
    for cap, cols, lengths, rank in zip(caps, group_columns, group_lengths, ranks):
        target_col = _position_target(cols, rank)
        target_idx = cols.index(target_col)
        target_len = int(lengths[target_idx])
        top_r = _MAX_CAP_ROW - target_len
        out[top_r + 1 : _BOTTOM_ROW, target_col] = cap
    return out


def _has_variant_signal(group_columns: list[list[int]], group_lengths: list[list[int]], ranks: list[int]) -> bool:
    """Return True iff at least one group behaves differently from the original task."""
    for cols, lengths, rank in zip(group_columns, group_lengths, ranks):
        if _position_target(cols, rank) != _length_target(cols, lengths, rank):
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch or weak signal."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, count + 1)) for count in counts]

    # Avoid overly weak scenes where every legend height is 1.
    if max(ranks) == 1:
        return None

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

    if not _has_variant_signal(group_columns, group_lengths, ranks):
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
