# Combo variant of 97d7923e (index 21)
# combo_spec: D6/0+D14/0
# Name: 列位置排序+顶部对齐

import itertools
import sys
from pathlib import Path

import numpy as np

_ROOT_DIR = Path(__file__).resolve().parents[4]
if str(_ROOT_DIR) not in sys.path:
    sys.path.append(str(_ROOT_DIR))

from arc_utils import get_bg, neighbors4, paste


# Design analysis for the combo variant
# a) 具体化与行为追踪
#    - 输入仍由左上角图例和若干单列竖条组成。图例每项颜色为 cap 色 K，高度为 k。
#    - 融合后每组 bar 不再底对齐，而是全部从同一 top-cap row 开始向下生长；fill 越长，
#      bottom cap 越低。组内排名也不再看长度，而是看列位置：最左=rank 1。
#    - 具体例子：某组 K 的三根 bar 位于列 [1, 6, 10]，fill 长度为 [2, 5, 3]，图例高度 k=2。
#      原题：先按长度排序，顺序是 col 6, 10, 1，所以 rank 2 选 col 10。
#      仅做顶部对齐但保留原排序时，选中的仍然是 col 10，只是三根 bar 共顶端。
#      融合后：三根 bar 先按列坐标排成 col 1, 6, 10，因此 rank 2 改为选 col 6。
#      差异同时体现在“视觉锚点变成顶部”和“被重着色的目标列改由空间顺序决定”。
#
# b) 复杂度校准
#    - 保持 1-4 个分组、每组 2-3 根 bar、图例高度 1..3，不引入额外对象类型。
#    - 组内 fill 长度仍保持互异，这样长度继续是显眼但误导性的干扰维度，能凸显 D6 的变化。
#    - 顶部对齐后可用长度上限变成动态值，因此仍把 bars/group 限在 2 或 3，保证每组都能
#      选出互异长度且版面不拥挤。
#
# c) 隐式假设审查
#    - `_vertical_sandwiches()` 只要求同列存在 cap/fill/cap 且 fill 为单色连续段，不依赖底对齐，
#      因此在融合后仍成立。
#    - `_legend_entries()` 只解析左上角接触 row 0 的图例簇，不依赖 bar 的锚定方向，也可保留。
#    - 原 `main()` 的核心假设是 `segs.sort(...length...)`，这个前提因 D6 失效，必须改成按列排序。
#    - 融合后 `main()` 还新增一个前提：同色组 bars 共享同一 top row；否则“顶部对齐”这一维度
#      就没有真正落实。generator 需要直接构造并稳定保证它。
#    - 原 generator 通过固定 bottom row 反推 top row；这一布局假设因 D14 失效，必须反过来用
#      固定 top row 推导 bottom row。
#
# d) 放置合法性
#    - 共享 top row 设为 `max(ranks) + 1`，即最高图例下方留 1 行背景，再开始放 bar。
#    - 这样图例列与同列 bar 之间至少隔 1 行空白，不接触、不重叠。
#    - `max_fill_len = min(MAX_FILL_LEN, GRID_H - bar_top_row - 2)` 保证 bottom cap 不越界。
#    - 列分配继续使用至少 1 列背景间隔，且每列只放一根 bar，避免对象粘连与误检。
#
# e) 视觉可观察性
#    - 人类能直接看出 bars 共享同一上边界、底部随长度变化，因此 D14 的几何变化是可见的。
#    - D6 的排序依据也完全可见：同色 bars 在水平方向上的先后顺序就是 rank 顺序。
#    - 排名不依赖颜色编号、生成顺序等不可视属性，只依赖像素位置这一显式线索。
#
# f) 可视化整洁度
#    - 图例仍压缩在左上角；bars 仍是一列宽的稀疏结构，阅读负担与原题相近。
#    - 全部 bars 使用统一 top anchor，使画面天然分成“上方图例 / 下方 bar 场”两个清晰区域。
#    - 不引入额外标记层，因此被重着色的目标 bar 在输出中也能直接定位。
#
# g) 变化的可推断性
#    - 输出变化仍只有一种：目标 bar 的 fill 变成该组 cap 色。
#    - 颜色线索由“图例色 K = bar 端点色 K”直接给出；长度只是干扰项，不承担映射职责。
#    - 为避免样本退化为“按长度排序也会得到同样输出”，生成器强制至少一个分组满足
#      左右顺序选中的列 != 长度顺序选中的列。
#
# h) 融合行为量化指标
#    - 定义 `signal_score` = 一个样本中，按列位置选中的目标列与按长度选中的目标列不同的分组数。
#    - 若 `signal_score == 0`，则 D6 在该样本里没有起作用，样本会退化；生成器直接丢弃。
#    - D14 的量化约束则是每个分组 `len({top_r}) == 1`，即同组 bars 的 top row 唯一。
#
# i) 新 `main()` 假设与 generator 保障
#    - `main()` 假设每个图例颜色都能匹配到至少一组有效 bar，且图例高度 k 不超过该组 bar 数。
#      generator 通过“每个图例正好对应一组 bars，rank 从 1..count 采样”来保证。
#    - `main()` 假设同组 bars 列坐标互异，便于左到右严格排序；generator 的列分配构造直接保证。
#    - `main()` 假设共享 top row。generator 把所有 bars 都贴到同一个 `bar_top_row` 上，并在
#      自校验中再次验证 `main(input) == expected`。


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
    Recolor one top-anchored vertical bar per legend-selected spatial rank.

    Input structure:
    - The background surrounds a top-left legend of one-column monochrome stacks touching row 0.
      Each stack uses cap color K and its height k gives that group's ordinal.
    - Elsewhere, the same cap colors appear on single-column cap/fill/cap bars whose top caps all
      sit on one shared row below the legend, while longer fills extend farther downward.

    Rule:
    - For each legend entry (K, k), collect every vertical bar capped by K.
    - Confirm that bars in that color group share one top row, then sort them from left to right.
    - Recolor the interior of the k-th bar in that left-to-right order to color K.

    Key insight: The legend height is an ordinal into the horizontal order of same-cap bars; bar
    length only affects how far each top-aligned bar grows downward and does not determine rank.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, "legend must be non-empty"

    for _x, cap, k in legend:
        segs = _vertical_sandwiches(g, cap)
        assert segs, f"legend cap {cap} must have matching vertical bars"
        top_rows = {top_r for top_r, _bot_r, _col, _fill in segs}
        assert len(top_rows) == 1, f"cap {cap} bars must share one top row in this variant"
        segs.sort(key=lambda s: (s[2], s[1], s[0]))
        assert 1 <= k <= len(segs), f"legend rank {k} out of range for {len(segs)} bars"
        top_r, bot_r, col, _fill = segs[k - 1]
        out[top_r + 1 : bot_r, col] = cap

    return out


# --------------- generator (combo variant) ---------------

# --- Configuration ---
GRID_H = 10                  # fixed grid height
WIDTH_MIN = 10               # minimum grid width
WIDTH_MAX = 20               # maximum grid width (inclusive)
MAX_GROUPS = 4               # max number of cap-color groups
MAX_TOTAL_BARS = 10          # max bars across all groups
BARS_PER_GROUP = (2, 3)      # allowed bars-per-group counts
MAX_FILL_LEN = 7             # nominal max interior fill length before layout clipping
COLOR_RANGE = (1, 10)        # non-bg color ids (exclusive upper)
COL_SPACING = 2              # minimum column stride between bars
MAX_RETRIES = 300            # retry budget for generate_input

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

    return [sorted(int(c) for c in cols) for cols in group_columns]


def _bar_top_row(ranks: list[int]) -> int:
    """Shared top-cap row: one blank row below the tallest legend stack."""
    return max(ranks) + 1


def _max_fill_len(bar_top_row: int) -> int:
    """Largest fill length that still leaves room for the bottom cap."""
    return min(MAX_FILL_LEN, GRID_H - bar_top_row - 2)


def _choose_fill_lengths(rng, count: int, max_fill_len: int) -> list[int]:
    """Pick `count` distinct fill lengths that fit below the shared top row."""
    assert count <= max_fill_len, f"need {count} distinct lengths, only {max_fill_len} fit"
    pool = list(range(1, max_fill_len + 1))
    rng.shuffle(pool)
    return [int(v) for v in pool[:count]]


def _position_target(cols: list[int], rank: int) -> int:
    """Column chosen by the combo rule: k-th leftmost bar."""
    return sorted(int(c) for c in cols)[rank - 1]


def _length_target(cols: list[int], lengths: list[int], rank: int) -> int:
    """Column chosen by the original ranking rule: k-th longest bar."""
    ranked = sorted(zip(cols, lengths), key=lambda item: (-int(item[1]), int(item[0])))
    return int(ranked[rank - 1][0])


def _signal_score(group_columns: list[list[int]], group_lengths: list[list[int]], ranks: list[int]) -> int:
    """How many groups change target column under position-based ranking."""
    score = 0
    for cols, lengths, rank in zip(group_columns, group_lengths, ranks):
        if _position_target(cols, rank) != _length_target(cols, lengths, rank):
            score += 1
    return score


def _expected_output(
    inp: np.ndarray,
    caps: list[int],
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
    bar_top_row: int,
) -> np.ndarray:
    """Compute the expected output using top alignment + left-to-right ranking."""
    out = inp.copy()
    for cap, cols, lengths, rank in zip(caps, group_columns, group_lengths, ranks):
        target_col = _position_target(cols, rank)
        target_idx = cols.index(target_col)
        target_len = int(lengths[target_idx])
        bottom_r = bar_top_row + target_len + 1
        out[bar_top_row + 1 : bottom_r, target_col] = cap
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch or weak signal."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, count + 1)) for count in counts]

    if max(ranks) == 1:
        return None

    bar_top_row = _bar_top_row(ranks)
    max_fill_len = _max_fill_len(bar_top_row)
    if max_fill_len < max(counts):
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

        lengths = _choose_fill_lengths(rng, count, max_fill_len)
        group_lengths.append(lengths)

        for col, fill_len in zip(cols, lengths):
            paste(grid, _build_bar(cap, fill, fill_len), bar_top_row, col)

    if _signal_score(group_columns, group_lengths, ranks) == 0:
        return None

    expected = _expected_output(grid, caps, group_columns, group_lengths, ranks, bar_top_row)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
