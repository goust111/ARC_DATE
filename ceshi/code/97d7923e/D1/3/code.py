# Variant of 97d7923e
# Dimension: D1 — 匹配条件：端点颜色分组 (match_cond)
# Change: s.endpoint_colors == d.color_group — 垂直条的顶部和底部帽色(cap color)与... → 端点颜色对匹配（顶底可不同色）

import itertools
from pathlib import Path
import sys

import numpy as np

for _parent in Path(__file__).resolve().parents:
    if (_parent / "arc_utils.py").exists():
        if str(_parent) not in sys.path:
            sys.path.append(str(_parent))
        break

from arc_utils import get_bg, paste


# Design analysis for the D1 variant
# a) 具体化与行为追踪
#    - 原题图例是左上角单色竖块，颜色 K 与条的上下端点都相同；块高 k 表示“第 k 长”。
#    - 本变体把每个图例条目改成重复 k 次的 2 像素竖标记：每个标记在同一列的 row0 / row1
#      分别放置 (top_color, bottom_color)。这样颜色对是可见的，重复次数 k 仍然承担“排名索引”。
#    - 主体条仍是单列竖条，但现在允许顶帽色和底帽色不同。匹配条件从“端点色 K”
#      变成有序对 (顶色, 底色)。
#    - 小例子：若图例有两个条目 `(2,5)` 重复 2 次、`(2,7)` 重复 1 次，那么它们共享顶色 2，
#      但 solver 必须把 bars 分别按 `(2,5)` 与 `(2,7)` 分组；若只按顶色 2 合并，第二组会选错。
# b) 复杂度校准
#    - 颜色对是少量离散类别，不引入不可见数值规则；每组仍只有 2-3 根条，长度排序复杂度与原题一致。
#    - 为了让“必须看颜色对”肉眼可见，生成器强制至少两组共享同一个顶色但底色不同。
# c) 隐式假设审查
#    - 原 `_vertical_sandwiches(g, cap)` 假设上下端点颜色相同；本变体必须改为读取条的顶色与底色。
#    - 原 `_legend_entries` 依赖“同色连通块高度=排名”；本变体不能再用连通块高度，需改为统计
#      左上角 2 像素标记的重复次数。
#    - 原 `out[top+1:bot, col] = cap` 默认内部只会涂成一个颜色；现在若继续无条件单色覆盖，
#      会丢失底色信息，因此输出改为把内部上半段涂顶色、下半段涂底色。
# d) 放置合法性
#    - 图例只占用 row0-row1；所有带图例的列都保证 row2 为空，再从 row3 及以下开始放 bar，
#      这样 solver 可以把一列中的上方图例 run 与下方 bar run 稳定分开。
#    - 每组条列在水平方向连续分配，列间至少隔 1 个背景列；图例只覆盖该组最左侧的 rank 根条。
# e) 视觉可观察性
#    - 解题者能直接看到：左上角每个图例条目不是单色柱，而是若干个重复的双色竖标记；
#      输出中被选中的条内部也出现对应的双色分层，因此“颜色对匹配”是显式可见的。
# f) 可视化整洁度
#    - 图例仍集中在左上角，bars 保持单列竖条，不引入额外装饰；共享列也只发生在图例正下方，
#      中间固定留 1 行背景，避免对象粘连。
# g) 颜色变化可推断性
#    - 输出颜色变化完全由该条自身的两端颜色决定：上半内部变顶色，下半内部变底色。
#      没有依赖颜色 id 的隐藏映射，也没有跨组任意换色。


def _column_runs(g: np.ndarray, col: int, bg: int) -> list[tuple[int, int]]:
    """Return contiguous non-bg row runs in one column as inclusive (start, end)."""
    rows = [int(r) for r in np.flatnonzero(g[:, col] != bg)]
    if not rows:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = rows[0]
    for r in rows[1:]:
        if r == prev + 1:
            prev = r
            continue
        runs.append((start, prev))
        start = prev = r
    runs.append((start, prev))
    return runs


def _vertical_bars(g: np.ndarray) -> list[tuple[int, int, int, int, int, int]]:
    """
    Find bar candidates as (top_r, bottom_r, col, top_color, bottom_color, fill_color).

    A valid bar is the lowest non-bg run in a column:
      - run length >= 3
      - interior is a solid single color
      - fill differs from background and from both endpoint colors
    """
    h, w = g.shape
    bg = get_bg(g)
    segs: list[tuple[int, int, int, int, int, int]] = []
    for c in range(w):
        runs = _column_runs(g, c, bg)
        if not runs:
            continue
        top_r, bottom_r = runs[-1]
        if bottom_r - top_r + 1 < 3:
            continue
        top_color = int(g[top_r, c])
        bottom_color = int(g[bottom_r, c])
        fill = int(g[top_r + 1, c])
        if fill in (bg, top_color, bottom_color):
            continue
        if np.all(g[top_r + 1 : bottom_r, c] == fill):
            segs.append((top_r, bottom_r, c, top_color, bottom_color, fill))
    return segs


def _legend_entries(g: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    """
    Read the top-left legend as repeated 2-pixel markers.

    Each legend token occupies one column with:
      - row 0: top endpoint color
      - row 1: bottom endpoint color

    Consecutive tokens with the same ordered pair, separated by at most one bg
    column, belong to the same legend entry. The number of repeated tokens is
    the rank index k.
    """
    h, w = g.shape
    assert h >= 2, 'legend needs two rows'
    top_pos = [int(c) for c in np.flatnonzero(g[0, :] != bg)]
    assert top_pos, 'legend row must be non-empty'
    top_pos.sort()

    entries: list[tuple[int, int, int, int]] = []
    start_c = top_pos[0]
    prev_c = top_pos[0]
    top_color = int(g[0, start_c])
    bottom_color = int(g[1, start_c])
    assert bottom_color != bg, 'legend token must have a bottom color'
    count = 1

    for c in top_pos[1:]:
        cur_top = int(g[0, c])
        cur_bottom = int(g[1, c])
        assert cur_bottom != bg, 'legend token must have a bottom color'
        same_entry = (c - prev_c <= 2) and (cur_top == top_color) and (cur_bottom == bottom_color)
        if same_entry:
            count += 1
        else:
            entries.append((start_c, top_color, bottom_color, count))
            start_c = c
            top_color = cur_top
            bottom_color = cur_bottom
            count = 1
        prev_c = c

    entries.append((start_c, top_color, bottom_color, count))
    return entries


def _paint_pair_fill(out: np.ndarray, top_r: int, bottom_r: int, col: int, top_color: int, bottom_color: int) -> None:
    """Fill the selected bar interior with a vertical split: upper half top_color, lower half bottom_color."""
    fill_len = bottom_r - top_r - 1
    split = (fill_len + 1) // 2
    out[top_r + 1 : top_r + 1 + split, col] = top_color
    out[top_r + 1 + split : bottom_r, col] = bottom_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor vertical bars by legend-driven endpoint-pair rank selection.

    Input structure:
    - bg = background. Top-left legend uses repeated 2-pixel columns: row 0 gives a bar's top cap color,
      row 1 gives its bottom cap color, and the number of repeated columns is the rank index.
    - Main grid contains single-column vertical bars with solid interiors; each bar may have different
      colors on its top and bottom caps.

    Rule:
    - For each legend entry ((top_color, bottom_color), k), gather bars whose ordered endpoint pair matches it.
    - Rank those bars by interior length (longest = rank 1).
    - Recolor the rank-k bar's interior using the legend pair: upper half becomes top_color, lower half becomes bottom_color.

    Key insight: The legend no longer names a group by one cap color; it specifies an ordered two-color endpoint signature,
    and the repeated token count is the index into the length-sorted bars of that exact signature.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, 'legend must be non-empty'

    bars = _vertical_bars(g)
    assert bars, 'input must contain vertical bars'

    grouped: dict[tuple[int, int], list[tuple[int, int, int, int, int, int]]] = {}
    for seg in bars:
        grouped.setdefault((seg[3], seg[4]), []).append(seg)

    for _x, top_color, bottom_color, k in legend:
        key = (top_color, bottom_color)
        segs = grouped.get(key, [])
        assert segs, f'legend pair {key} must have matching vertical bars'
        segs = sorted(segs, key=lambda s: (-(s[1] - s[0] - 1), s[0], s[2]))
        assert 1 <= k <= len(segs), f'legend rank {k} out of range for {len(segs)} bars'
        top_r, bottom_r, col, _top, _bottom, _fill = segs[k - 1]
        _paint_pair_fill(out, top_r, bottom_r, col, top_color, bottom_color)

    return out


# --------------- generator (O2) ---------------

# --- Configuration ---
GRID_H = 10
WIDTH_MIN = 10
WIDTH_MAX = 20
MIN_GROUPS = 2
MAX_GROUPS = 4
MAX_TOTAL_BARS = 10
BARS_PER_GROUP = (2, 3)
MAX_FILL_LEN = 7
MAX_SHARED_FILL_LEN = 5
COLOR_RANGE = (1, 10)
COL_SPACING = 2
MAX_RETRIES = 400

_BOTTOM_ROW = GRID_H - 1
_TOKEN_BOTTOM_ROW = 1
_MAX_CAP_ROW = GRID_H - 2

_FEASIBLE_COUNTS = {
    ng: [c for c in itertools.product(BARS_PER_GROUP, repeat=ng) if sum(c) <= MAX_TOTAL_BARS]
    for ng in range(MIN_GROUPS, MAX_GROUPS + 1)
}

_LENGTH_ASSIGNMENTS: dict[tuple[int, int], list[tuple[int, ...]]] = {}
for count in BARS_PER_GROUP:
    for rank in range(1, count + 1):
        vals: list[tuple[int, ...]] = []
        for perm in itertools.permutations(range(1, MAX_FILL_LEN + 1), count):
            if all(perm[i] <= MAX_SHARED_FILL_LEN for i in range(rank)):
                vals.append(tuple(int(v) for v in perm))
        _LENGTH_ASSIGNMENTS[(count, rank)] = vals


def _build_bar(top_color: int, fill: int, fill_len: int, bottom_color: int) -> np.ndarray:
    """Return a (fill_len+2, 1) column sprite: top / fill*N / bottom."""
    bar = np.full((fill_len + 2, 1), fill, dtype=int)
    bar[0, 0] = top_color
    bar[-1, 0] = bottom_color
    return bar


def _assign_group_columns(rng, counts: list[int], width: int) -> list[list[int]]:
    """Assign each group a left-to-right block of bar columns with fixed intra-group spacing."""
    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    assert width >= min_width
    extra_width = width - min_width

    # Extra width only widens the gaps between groups or the trailing margin.
    gap_bonus = rng.multinomial(extra_width, np.full(len(counts), 1.0 / len(counts)))
    group_columns: list[list[int]] = []
    cur = 0
    for gi, count in enumerate(counts):
        cols = [cur + COL_SPACING * j for j in range(count)]
        group_columns.append(cols)
        cur = cols[-1] + COL_SPACING + int(gap_bonus[gi])
    return group_columns


def _choose_pairs(rng, n_groups: int) -> list[tuple[int, int]]:
    """Pick unique ordered endpoint pairs with at least one repeated top color."""
    endpoint_palette = [int(v) for v in rng.choice(np.arange(*COLOR_RANGE), size=3, replace=False)]
    all_pairs = [(a, b) for a in endpoint_palette for b in endpoint_palette if a != b]

    for _ in range(200):
        rng.shuffle(all_pairs)
        pairs = list(all_pairs[:n_groups])
        top_counts = {}
        for top, _bottom in pairs:
            top_counts[top] = top_counts.get(top, 0) + 1
        if max(top_counts.values()) >= 2:
            return pairs
    raise RuntimeError('failed to sample endpoint pairs with a repeated top color')


def _random_lengths(rng, count: int, rank: int) -> list[int]:
    candidates = _LENGTH_ASSIGNMENTS[(count, rank)]
    choice = candidates[int(rng.integers(len(candidates)))]
    return [int(v) for v in choice]


def _ranked_target(cols: list[int], lengths: list[int], rank: int) -> tuple[int, int]:
    ranked = sorted(zip(cols, lengths), key=lambda item: (-item[1], _MAX_CAP_ROW - item[1], item[0]))
    return ranked[rank - 1]


def _expected_output(
    inp: np.ndarray,
    pairs: list[tuple[int, int]],
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
) -> np.ndarray:
    """Compute the expected output by recoloring the selected bar with a top/bottom split."""
    out = inp.copy()
    for (top_color, bottom_color), cols, lengths, rank in zip(pairs, group_columns, group_lengths, ranks):
        target_col, target_len = _ranked_target(cols, lengths, rank)
        top_r = _MAX_CAP_ROW - target_len
        _paint_pair_fill(out, top_r, _BOTTOM_ROW, target_col, top_color, bottom_color)
    return out


def _top_only_disagreement(
    pairs: list[tuple[int, int]],
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
) -> int:
    """
    Count legend entries whose selected column would change if bars were grouped only by top color.

    A positive value means the sample really exercises the D1 change.
    """
    actual_targets = {}
    for pair, cols, lengths, rank in zip(pairs, group_columns, group_lengths, ranks):
        actual_targets[pair] = _ranked_target(cols, lengths, rank)[0]

    by_top: dict[int, list[tuple[int, int]]] = {}
    for (top_color, _bottom_color), cols, lengths in zip(pairs, group_columns, group_lengths):
        for col, fill_len in zip(cols, lengths):
            by_top.setdefault(top_color, []).append((col, fill_len))

    wrong = 0
    for pair, rank in zip(pairs, ranks):
        merged = sorted(by_top[pair[0]], key=lambda item: (-item[1], _MAX_CAP_ROW - item[1], item[0]))
        wrong_col = merged[rank - 1][0]
        if wrong_col != actual_targets[pair]:
            wrong += 1
    return wrong


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch or weak variant signal."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(MIN_GROUPS, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, c + 1)) for c in counts]

    pairs = _choose_pairs(rng, n_groups)
    endpoint_colors = sorted({color for pair in pairs for color in pair})
    fill_pool = [c for c in range(*COLOR_RANGE) if c not in endpoint_colors]
    rng.shuffle(fill_pool)
    fills = [int(v) for v in fill_pool[:n_groups]]

    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    group_columns = _assign_group_columns(rng, counts, width)

    grid = np.zeros((GRID_H, width), dtype=int)
    group_lengths: list[list[int]] = []

    for (top_color, bottom_color), fill, count, rank, cols in zip(pairs, fills, counts, ranks, group_columns):
        lengths = _random_lengths(rng, count, rank)
        group_lengths.append(lengths)

        for token_col in cols[:rank]:
            grid[0, token_col] = top_color
            grid[_TOKEN_BOTTOM_ROW, token_col] = bottom_color

        for col, fill_len in zip(cols, lengths):
            top_r = _MAX_CAP_ROW - fill_len
            paste(grid, _build_bar(top_color, fill, fill_len, bottom_color), top_r, col)

    if _top_only_disagreement(pairs, group_columns, group_lengths, ranks) <= 0:
        return None

    expected = _expected_output(grid, pairs, group_columns, group_lengths, ranks)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f'failed to generate a valid input after {MAX_RETRIES} attempts')
