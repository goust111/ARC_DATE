# Combo variant of 97d7923e (index 4)
# combo_spec: D2/0+D6/0
# Name: 升序排名+列位置排序

# Design analysis
# a) 具体化与行为追踪
#    - 场景仍由左上角图例和若干底对齐竖直 bar 组成。每组 bar 共享端点色 cap=K 与填充色 fill=F。
#    - 原题做法：同色组内先按填充长度从长到短排序，图例高度 k 选第 k 长的 bar。
#    - 本融合变体：长度完全退出排名；同色组只按列坐标从左到右排序，最左=rank 1，图例高度 k 选第 k 根。
#    - 追踪例子：某组 K 的三根 bar 位于列 [2, 7, 10]，长度 [6, 2, 4]，图例高度 k=1。
#      原题会选长度 6 的列 2；若 k=2 原题会选长度 4 的列 10。
#      融合后无论长度如何，排序都是 [2, 7, 10]；k=1 选列 2，k=2 选列 7，k=3 选列 10。
#    - 因而可见差异发生在“组内哪一列被重着色”，并且长度会保留下来作为干扰维度而不是排序依据。
#
# b) 复杂度校准
#    - 仍限制每组 bar 数为 2 或 3、图例高度为 1..bar_count，避免 ordinal 自由度过大。
#    - 保留组内互异长度，让肉眼能看到“长度很显眼但其实不决定目标”的反常性。
#    - 额外避免所有组都退化为 count=3 且 k=2；那样左右方向的 rank 2 相同，D2 的方向变化不可见。
#
# c) 隐式假设审查
#    - `_vertical_sandwiches()` 只依赖 cap-fill-cap 的单列结构，不关心排名规则，可原样复用。
#    - `_legend_entries()` 只把图例读取成 (颜色, 高度 k)，同样不依赖排名依据，可原样复用。
#    - 原 `main()` 中真正失效的是排序键：不能再按长度，必须改为按列号升序。
#    - 原 generator 的 oracle 也不能再通过长度选目标；需要改成“第 k 左”的列。
#    - 新增的融合风险是：如果样本恰好让左到右与右到左、或左到右与原题长度降序，选中同一列，
#      则某一维变化会在视觉上失效。生成器必须显式过滤这类退化样本。
#
# d) 放置合法性
#    - 仍使用 10x(10..20) 网格、单列 bar、底端固定在最后一行、列间至少隔 1 列背景。
#    - 每组最左 bar 与图例共列，因此其长度需满足 `fill_len <= MAX_FILL_LEN - rank`，
#      这样图例底部与该 bar 顶 cap 之间至少保留 1 行背景。
#    - 所有 bar 列唯一，避免 solver 在同一列中读到多个合法 sandwich。
#
# e) 视觉可观察性
#    - 排名依据只能来自肉眼可见的空间位置，不能用颜色值或数组索引等不可见因素。
#    - 本变体中“左到右”是直接可见的；长度仍然可见，但仅作为误导线索存在。
#    - 图例高度继续提供 ordinal k，因此观察者可从多个样例对比出“同色组内找第 k 左”的规律。
#
# f) 可视化整洁度
#    - 图例仍集中在左上角且为单列堆叠；主区域仍是稀疏的单列竖条，没有新增符号层。
#    - 颜色职责保持简单：cap 色负责分组与最终重着色，fill 色只作为未命中的内部颜色。
#
# g) 变化的可推断性
#    - 输出里唯一变化仍是某一根 bar 的内部被涂成 cap 色，颜色映射线索完全由“图例色=端点色”给出。
#    - 为确保融合行为真的发生，生成器要求：
#      1. 至少一组的“左到右目标列”不同于原题“长度降序目标列”。
#      2. 至少一组的“左到右目标列”不同于“右到左目标列”。
#      这样 D6 的排序基准变化与 D2 的方向变化都能在样本中观测到。
#
# h) 新 main() 的假设与 generator 保障
#    - 新 `main()` 假设：同色组 bar 列坐标唯一，图例高度 k 在 1..组内 bar 数范围内。
#    - generator 通过固定 2-3 根 bar/组、唯一列、合法 rank、闭环 oracle 校验来保证这些前提。

from pathlib import Path
import sys
import itertools

import numpy as np

_ROOT = Path("/home/lixinyu/snap/new")
for _path in (_ROOT, _ROOT / "solvers"):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.append(_path_str)

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor vertical bars by legend-driven ordinal selection using spatial order.

    Input structure:
    - The upper-left corner contains one-column legend stacks touching row 0; each stack has cap color K and height k.
    - The rest of the grid contains bottom-aligned vertical cap/fill/cap bars, with each color group sharing the same cap color K.

    Rule:
    - For each legend entry (K, k), collect all bars whose top and bottom caps are color K.
    - Order that group only by column position from left to right, so the leftmost bar is rank 1.
    - Recolor the interior cells of the rank-k bar to K.

    Key insight: The legend height is an ordinal into the left-to-right order of matching bars; bar length is visible but irrelevant noise.
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


# --------------- generator (combo D2/0 + D6/0) ---------------

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
        group_columns[owner].append(int(col))

    return [sorted(cols) for cols in group_columns]


def _choose_fill_lengths(rng, count: int, rank: int) -> list[int]:
    """Pick `count` distinct fill lengths for one group."""
    max_leftmost_len = MAX_FILL_LEN - rank
    leftmost_len = int(rng.integers(1, max_leftmost_len + 1))
    pool = [length for length in range(1, MAX_FILL_LEN + 1) if length != leftmost_len]
    rng.shuffle(pool)
    return [leftmost_len] + pool[: count - 1]


def _left_to_right_target(cols: list[int], rank: int) -> int:
    """Column chosen by the combo rule: rank-k in left-to-right order."""
    return sorted(int(c) for c in cols)[rank - 1]


def _right_to_left_target(cols: list[int], rank: int) -> int:
    """Reference target for the reversed spatial direction."""
    return sorted((int(c) for c in cols), reverse=True)[rank - 1]


def _length_desc_target(cols: list[int], lengths: list[int], rank: int) -> int:
    """Reference target for the original task: rank-k by descending fill length."""
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
        target_col = _left_to_right_target(cols, rank)
        target_idx = cols.index(target_col)
        target_len = int(lengths[target_idx])
        top_r = _MAX_CAP_ROW - target_len
        out[top_r + 1 : _BOTTOM_ROW, target_col] = cap
    return out


def _has_combo_signal(
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
) -> bool:
    """
    Require both component effects to be visible:
      - D6: left-to-right target differs from original length-desc target in at least one group.
      - D2 in the fused setting: left-to-right target differs from right-to-left target in at least one group.
    """
    differs_from_original = False
    differs_from_reverse = False
    for cols, lengths, rank in zip(group_columns, group_lengths, ranks):
        forward_col = _left_to_right_target(cols, rank)
        if forward_col != _length_desc_target(cols, lengths, rank):
            differs_from_original = True
        if forward_col != _right_to_left_target(cols, rank):
            differs_from_reverse = True
    return differs_from_original and differs_from_reverse


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch or weak signal."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, count + 1)) for count in counts]

    # Avoid scenes where every legend height is the trivial first ordinal.
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

    if not _has_combo_signal(group_columns, group_lengths, ranks):
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
