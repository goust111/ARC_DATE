# Combo variant of 97d7923e (index 11)
# combo_spec: D1/3+D4/2
# Name: 端点颜色对匹配+双层帽色

from __future__ import annotations

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


# Design analysis for this combo variant
# --------------------------------------
# a) 具体化与行为追踪
# - 原题图例是左上角单色竖块，高度 k 表示“第 k 长”；主体条是单列、单像素帽色的 cap-fill-cap。
# - 本融合把图例改成左上角两行编码的横向条目：每个条目由若干个连续 token 列组成，
#   token 的 row 0 / row 1 分别放 (top_color, bottom_color)，token 数量就是 rank k。
# - 主体条改成两列宽对象，每列完全相同；顶部和底部各有厚度 t in {2, 3} 的帽区，
#   中间仍是单色 fill。条的匹配键从单个 cap 色变成有序对 (top_color, bottom_color)。
# - 小例子：若图例条目 `(2, 5)` 宽 2、`(2, 7)` 宽 1，那么它们共享顶色 2，但必须分成
#   两组。若某组 bars 的 fill 长度分别为 5 和 3，则 `(2, 5), k=2` 选长度 3 的那根；
#   被选中后其 fill 矩形上半段变 2，下半段变 5，并且要一次性改完整个 2 列宽矩形。
# - 差异点因此有两层：先看 legend 的双色有序对去分组，再在厚帽 bar 中只度量两帽之间的
#   fill 长度做排序，最后把选中 fill 重绘成对应的双色分层。
#
# b) 复杂度校准
# - 颜色对只使用少量离散组合，且每个样例强制至少两组共享同一个顶色但底色不同，这样
#   “不能只看一个颜色”是可观察的。
# - 厚帽厚度只取 2 或 3，并且每组内部一致、不同组之间常常不同；这样 thick-cap 是稳定
#   结构，而不是噪声。bar 仍只按 fill 长度排序，避免引入额外不可见数值。
# - bars 统一做成 2 列宽，这样若还按原题的单列条去看，会把一个对象误读成两根重复条，
#   从而在检测和涂色上都失败，D4 变化就不是纯装饰。
#
# c) 隐式假设审查
# - 原 solver 假设 legend 的 rank 来自单色连通块高度；这里失效，因为 legend 现在是
#   两行双色 token 的水平重复，必须改成按“同一颜色对连续出现多少列”来读 rank。
# - 原 solver 假设一列里看到 cap-fill-cap 就是一根 bar；这里失效，因为真实对象是
#   两个完全相同的相邻列，且帽区厚度 >= 2。检测必须先锁定 2 列宽对象，再读取厚帽。
# - 原 solver 假设上下帽同色；这里失效，因为匹配键改成有序对 (top_color, bottom_color)。
# - 原 solver 直接把 fill 全部刷成单个 cap 色；这里也不够了，因为 legend 提供的是双色对。
#   新输出把选中 fill 的上半段刷成 top_color，下半段刷成 bottom_color。
# - 新 main() 额外依赖的前提：
#   1. 每个真实 bar 恰好由 2 个相同相邻列组成，且左右至少隔 1 个背景列。
#   2. 每列 bar 只有一个连续非背景 run；其顶部和底部单色 run 厚度相等且 >= 2。
#   3. legend 只占 row 0..1，并在其右侧留出至少 2 列空白后 bars 才开始。
#   4. 图例条目的 token 列在水平方向连续，组间只用 1 列背景分隔。
#
# d) 放置合法性
# - 生成器把 legend 放在左侧专用区域，bars 放在其右侧，之间至少留 2 列背景，避免顶行 bar
#   像素污染 legend 检测。
# - 每根 bar 固定占 2 列，相邻 bars 之间至少留 1 列背景；所有 bars 底部对齐到最后一行。
# - fill 长度受厚度约束：要求 top_r >= 1，因此 fill_len <= GRID_H - 1 - 2*t；这保证
#   bar 不碰 row 0，且不会越界。
#
# e) 视觉可观察性
# - 人类能直接看到 legend 不再是单色竖块，而是“两行双色、宽度代表 rank”的条目。
# - bars 也显然不是原题单列细条，而是 2 列宽、上下厚帽的对象；选中后 fill 出现上下双色
#   分层，和自身端点色、legend 颜色对都一致。
# - 样例中至少两组共享同一顶色但底色不同，所以只看顶色会选错；同时厚帽让“只能量中间 fill”
#   这一点也肉眼可见。
#
# f) 可视化整洁度
# - legend 与 bars 水平分区，避免对象堆叠；bar 之间固定留白，输出只改 fill 矩形，不改帽区，
#   所以结构边界始终清楚。
# - 每组颜色对唯一，fill 色不与任何端点色重复，降低视觉混淆。
#
# g) 变化的可推断性
# - 颜色变化不是任意映射：被改色的永远是与 legend 条目颜色对匹配的 bar，且变换后的上下
#   分层直接复用了该 bar 的顶帽色和底帽色。
# - 生成器额外拒绝两类退化样本：
#   1. 若按“只看顶色分组”仍会选中同一目标，则拒绝，确保 D1 真正在起作用。
#   2. 若按“单列单帽条”去解时输出与融合规则一样，则拒绝，确保 D4 不是伪变化。


def _legend_entries(g: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    """
    Read the top-left legend as horizontal runs of 2-row endpoint-pair tokens.

    Each token column has:
      - row 0: top endpoint color
      - row 1: bottom endpoint color

    Consecutive columns with the same ordered pair form one legend entry.
    The entry width is the rank index k.
    """
    h, w = g.shape
    assert h >= 2, "legend needs two rows"

    token_cols = [int(c) for c in np.flatnonzero(np.any(g[:2, :] != bg, axis=0))]
    assert token_cols, "legend must be non-empty"
    token_cols.sort()

    cols = [token_cols[0]]
    for c in token_cols[1:]:
        if c - cols[-1] <= 2:
            cols.append(c)
        else:
            break

    entries: list[tuple[int, int, int, int]] = []
    start_c = cols[0]
    prev_c = cols[0]
    top_color = int(g[0, start_c])
    bottom_color = int(g[1, start_c])
    assert top_color != bg and bottom_color != bg, "legend tokens must occupy both rows"
    width = 1

    for c in cols[1:]:
        cur_top = int(g[0, c])
        cur_bottom = int(g[1, c])
        assert cur_top != bg and cur_bottom != bg, "legend tokens must occupy both rows"
        same_entry = (c == prev_c + 1) and (cur_top == top_color) and (cur_bottom == bottom_color)
        if same_entry:
            width += 1
        else:
            entries.append((start_c, top_color, bottom_color, width))
            start_c = c
            top_color = cur_top
            bottom_color = cur_bottom
            width = 1
        prev_c = c

    entries.append((start_c, top_color, bottom_color, width))
    return entries


def _vertical_pair_bars(g: np.ndarray) -> list[tuple[int, int, int, int, int, int, int]]:
    """
    Find thick-capped 2-column bars as
    (fill_top, fill_bottom, left_col, top_color, bottom_color, fill_color, thickness),
    where fill_bottom is exclusive.
    """
    h, w = g.shape
    bg = get_bg(g)
    bars: list[tuple[int, int, int, int, int, int, int]] = []

    for c in range(w - 1):
        if not np.array_equal(g[:, c], g[:, c + 1]):
            continue
        if c > 0 and np.array_equal(g[:, c - 1], g[:, c]):
            continue
        if c + 2 < w and np.array_equal(g[:, c + 1], g[:, c + 2]):
            continue

        col_vals = g[:, c]
        rows = [int(r) for r in np.flatnonzero(col_vals != bg)]
        if not rows:
            continue
        if rows != list(range(rows[0], rows[-1] + 1)):
            continue

        top_r = rows[0]
        bottom_r = rows[-1]
        total_h = bottom_r - top_r + 1
        if total_h < 5:
            continue

        top_color = int(col_vals[top_r])
        fill_top = top_r
        while fill_top <= bottom_r and int(col_vals[fill_top]) == top_color:
            fill_top += 1
        if fill_top > bottom_r:
            continue
        thickness = fill_top - top_r
        if thickness < 2:
            continue

        bottom_color = int(col_vals[bottom_r])
        fill_bottom = bottom_r
        while fill_bottom >= top_r and int(col_vals[fill_bottom]) == bottom_color:
            fill_bottom -= 1
        fill_bottom += 1
        if bottom_r - fill_bottom + 1 != thickness:
            continue
        if fill_top >= fill_bottom:
            continue

        fill_color = int(col_vals[fill_top])
        if fill_color in (bg, top_color, bottom_color):
            continue
        if not np.all(col_vals[fill_top:fill_bottom] == fill_color):
            continue

        bars.append((fill_top, fill_bottom, c, top_color, bottom_color, fill_color, thickness))

    return bars


def _paint_pair_fill(
    out: np.ndarray,
    fill_top: int,
    fill_bottom: int,
    left_col: int,
    top_color: int,
    bottom_color: int,
) -> None:
    """Paint the selected 2-column fill rectangle with an upper/lower color split."""
    fill_len = fill_bottom - fill_top
    split = (fill_len + 1) // 2
    out[fill_top : fill_top + split, left_col : left_col + 2] = top_color
    out[fill_top + split : fill_bottom, left_col : left_col + 2] = bottom_color


def _ranked_target(
    bars: list[tuple[int, int, int, int, int, int, int]],
    rank: int,
) -> tuple[int, int, int, int, int, int, int]:
    ranked = sorted(bars, key=lambda b: (-(b[1] - b[0]), b[0], b[2]))
    return ranked[rank - 1]


def _top_only_disagreement(
    pairs: list[tuple[int, int]],
    group_bars: list[list[tuple[int, int, int, int, int, int, int]]],
    ranks: list[int],
) -> int:
    """Count legend entries whose target changes if bars are grouped only by top color."""
    actual_targets = {}
    for pair, bars, rank in zip(pairs, group_bars, ranks):
        actual_targets[pair] = _ranked_target(bars, rank)[2]

    by_top: dict[int, list[tuple[int, int, int, int, int, int, int]]] = {}
    for pair, bars in zip(pairs, group_bars):
        by_top.setdefault(pair[0], []).extend(bars)

    wrong = 0
    for pair, rank in zip(pairs, ranks):
        merged = sorted(by_top[pair[0]], key=lambda b: (-(b[1] - b[0]), b[0], b[2]))
        if merged[rank - 1][2] != actual_targets[pair]:
            wrong += 1
    return wrong


def _thin_column_output(grid: np.ndarray) -> np.ndarray:
    """
    Ablation used only for generator validation:
    keep the new legend, but pretend bars are old one-column one-pixel-cap objects.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    grouped: dict[tuple[int, int], list[tuple[int, int, int]]] = {}

    for c in range(g.shape[1]):
        rows = [int(r) for r in np.flatnonzero(g[:, c] != bg)]
        if not rows:
            continue
        if rows != list(range(rows[0], rows[-1] + 1)):
            continue
        top_r = rows[0]
        bottom_r = rows[-1]
        if bottom_r - top_r + 1 < 3:
            continue
        top_color = int(g[top_r, c])
        bottom_color = int(g[bottom_r, c])
        fill_color = int(g[top_r + 1, c])
        if fill_color in (bg, top_color, bottom_color):
            continue
        if not np.all(g[top_r + 1 : bottom_r, c] == fill_color):
            continue
        grouped.setdefault((top_color, bottom_color), []).append((top_r, bottom_r, c))

    for _x, top_color, bottom_color, rank in legend:
        segs = grouped.get((top_color, bottom_color), [])
        if not segs or not (1 <= rank <= len(segs)):
            continue
        segs = sorted(segs, key=lambda s: (-(s[1] - s[0] - 1), s[0], s[2]))
        top_r, bottom_r, c = segs[rank - 1]
        fill_len = bottom_r - top_r - 1
        split = (fill_len + 1) // 2
        out[top_r + 1 : top_r + 1 + split, c] = top_color
        out[top_r + 1 + split : bottom_r, c] = bottom_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor thick-capped vertical bars by legend-driven endpoint-pair rank selection.

    Input structure:
    - Background is 0. In the top-left corner, each legend entry is a horizontal run of 2-row tokens:
      row 0 stores a bar's top cap color, row 1 stores its bottom cap color, and the run width is rank k.
    - Elsewhere, each source object is a bottom-aligned 2-column bar whose two columns are identical:
      a top cap of thickness 2 or 3, a solid fill rectangle, and a bottom cap of the same thickness.
      The top and bottom cap colors may differ.

    Rule:
    - For each legend entry ((top_color, bottom_color), k), gather the bars whose ordered endpoint pair
      matches that legend pair exactly.
    - Rank those bars by the height of the fill region between the thick caps (longest = rank 1).
    - Recolor only the rank-k bar's fill rectangle: its upper half becomes top_color and its lower half
      becomes bottom_color.

    Key insight:
    - The legend no longer points to a single cap color. It names an ordered two-color signature, and its
      width is the rank index into the length-sorted bars of that exact signature. Thick hats are part of
      the bar structure but are not counted in the ranking length.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, "legend must be non-empty"

    bars = _vertical_pair_bars(g)
    assert bars, "input must contain thick endpoint-pair bars"

    grouped: dict[tuple[int, int], list[tuple[int, int, int, int, int, int, int]]] = {}
    for bar in bars:
        grouped.setdefault((bar[3], bar[4]), []).append(bar)

    for _x, top_color, bottom_color, rank in legend:
        key = (top_color, bottom_color)
        segs = grouped.get(key, [])
        assert segs, f"legend pair {key} must have matching bars"
        assert 1 <= rank <= len(segs), f"legend rank {rank} out of range for {len(segs)} bars"
        fill_top, fill_bottom, left_col, _top, _bottom, _fill, _thickness = _ranked_target(segs, rank)
        _paint_pair_fill(out, fill_top, fill_bottom, left_col, top_color, bottom_color)

    return out


# --------------- generator (O2) ---------------

# --- Configuration ---
GRID_H = 10
WIDTH_MIN = 16
WIDTH_MAX = 32
MIN_GROUPS = 2
MAX_GROUPS = 3
MAX_TOTAL_BARS = 6
BARS_PER_GROUP = (2, 3)
CAP_THICKNESS_CHOICES = (2, 3)
COLOR_RANGE = (1, 10)
TOP_MARGIN = 1
BAR_STRIDE = 3
LEGEND_GROUP_GAP = 1
LEGEND_BAR_GAP = 2
MAX_RETRIES = 500

_BOTTOM_ROW = GRID_H - 1

_FEASIBLE_COUNTS = {
    ng: [c for c in itertools.product(BARS_PER_GROUP, repeat=ng) if sum(c) <= MAX_TOTAL_BARS]
    for ng in range(MIN_GROUPS, MAX_GROUPS + 1)
}


def _build_bar(top_color: int, fill_color: int, fill_len: int, bottom_color: int, thickness: int) -> np.ndarray:
    """Return a thick-cap 2-column vertical sprite."""
    total_h = fill_len + 2 * thickness
    bar = np.full((total_h, 2), fill_color, dtype=int)
    bar[:thickness, :] = top_color
    bar[-thickness:, :] = bottom_color
    return bar


def _choose_pairs(rng: np.random.Generator, n_groups: int) -> list[tuple[int, int]]:
    """Pick unique ordered endpoint pairs with at least one repeated top color."""
    endpoint_palette = [int(v) for v in rng.choice(np.arange(*COLOR_RANGE), size=3, replace=False)]
    all_pairs = [(a, b) for a in endpoint_palette for b in endpoint_palette if a != b]

    for _ in range(200):
        rng.shuffle(all_pairs)
        pairs = [tuple(map(int, all_pairs[i])) for i in range(n_groups)]
        top_counts: dict[int, int] = {}
        for top_color, _bottom_color in pairs:
            top_counts[top_color] = top_counts.get(top_color, 0) + 1
        if max(top_counts.values()) >= 2:
            return pairs
    raise RuntimeError("failed to sample endpoint pairs with a repeated top color")


def _choose_thicknesses(rng: np.random.Generator, n_groups: int) -> list[int]:
    """Use both thickness levels across the sample whenever possible."""
    for _ in range(200):
        vals = [int(rng.choice(CAP_THICKNESS_CHOICES)) for _ in range(n_groups)]
        if n_groups == 1 or len(set(vals)) >= 2:
            return vals
    raise RuntimeError("failed to sample visible thickness variation")


def _choose_fill_lengths(rng: np.random.Generator, count: int, thickness: int) -> list[int]:
    """Pick distinct fill lengths that keep the full bar below row 0."""
    max_fill_len = GRID_H - TOP_MARGIN - 2 * thickness
    assert count <= max_fill_len
    pool = np.arange(1, max_fill_len + 1)
    rng.shuffle(pool)
    return [int(v) for v in pool[:count]]


def _legend_width(ranks: list[int]) -> int:
    return sum(ranks) + LEGEND_GROUP_GAP * (len(ranks) - 1)


def _assign_group_columns(
    rng: np.random.Generator,
    counts: list[int],
    width: int,
    legend_width: int,
) -> list[list[int]]:
    """Assign left columns for 2-wide bars, keeping groups in left-to-right blocks."""
    total_bars = sum(counts)
    bar_width = BAR_STRIDE * total_bars - 1
    min_width = max(WIDTH_MIN, legend_width + LEGEND_BAR_GAP + bar_width)
    assert width >= min_width

    extra_width = width - min_width
    gap_bonus = rng.multinomial(extra_width, np.full(len(counts) + 1, 1.0 / (len(counts) + 1)))

    cur = legend_width + LEGEND_BAR_GAP + int(gap_bonus[0])
    group_columns: list[list[int]] = []
    for gi, count in enumerate(counts):
        cols = [cur + BAR_STRIDE * j for j in range(count)]
        group_columns.append(cols)
        cur = cols[-1] + BAR_STRIDE + int(gap_bonus[gi + 1])
    return group_columns


def _paint_legend(grid: np.ndarray, pairs: list[tuple[int, int]], ranks: list[int]) -> None:
    cur = 0
    for (top_color, bottom_color), rank in zip(pairs, ranks):
        grid[0, cur : cur + rank] = top_color
        grid[1, cur : cur + rank] = bottom_color
        cur += rank + LEGEND_GROUP_GAP


def _expected_output(
    inp: np.ndarray,
    pairs: list[tuple[int, int]],
    group_bars: list[list[tuple[int, int, int, int, int, int, int]]],
    ranks: list[int],
) -> np.ndarray:
    out = inp.copy()
    for (top_color, bottom_color), bars, rank in zip(pairs, group_bars, ranks):
        fill_top, fill_bottom, left_col, _t, _b, _f, _thick = _ranked_target(bars, rank)
        _paint_pair_fill(out, fill_top, fill_bottom, left_col, top_color, bottom_color)
    return out


def _combo_metrics(
    grid: np.ndarray,
    pairs: list[tuple[int, int]],
    group_bars: list[list[tuple[int, int, int, int, int, int, int]]],
    ranks: list[int],
) -> tuple[int, int]:
    """Return (pair_disagreement_count, thin_ablation_diff_pixels)."""
    pair_disagreement = _top_only_disagreement(pairs, group_bars, ranks)
    combo_out = main(grid)
    thin_out = _thin_column_output(grid)
    thin_diff = int(np.count_nonzero(combo_out != thin_out))
    return pair_disagreement, thin_diff


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid fused sample. Returns None on weak behavior or oracle mismatch."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(MIN_GROUPS, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, c + 1)) for c in counts]
    pairs = _choose_pairs(rng, n_groups)
    thicknesses = _choose_thicknesses(rng, n_groups)

    endpoint_colors = sorted({color for pair in pairs for color in pair})
    fill_pool = [c for c in range(*COLOR_RANGE) if c not in endpoint_colors]
    rng.shuffle(fill_pool)
    fills = [int(v) for v in fill_pool[:n_groups]]

    legend_width = _legend_width(ranks)
    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, legend_width + LEGEND_BAR_GAP + (BAR_STRIDE * total_bars - 1))
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    group_columns = _assign_group_columns(rng, counts, width, legend_width)

    grid = np.zeros((GRID_H, width), dtype=int)
    _paint_legend(grid, pairs, ranks)

    group_bars: list[list[tuple[int, int, int, int, int, int, int]]] = []
    for (top_color, bottom_color), fill_color, count, cols, thickness in zip(
        pairs, fills, counts, group_columns, thicknesses
    ):
        lengths = _choose_fill_lengths(rng, count, thickness)
        bars: list[tuple[int, int, int, int, int, int, int]] = []
        for left_col, fill_len in zip(cols, lengths):
            bar = _build_bar(top_color, fill_color, fill_len, bottom_color, thickness)
            top_r = GRID_H - bar.shape[0]
            paste(grid, bar, top_r, left_col)
            fill_top = top_r + thickness
            fill_bottom = fill_top + fill_len
            bars.append((fill_top, fill_bottom, left_col, top_color, bottom_color, fill_color, thickness))
        group_bars.append(bars)

    pair_disagreement, thin_diff = _combo_metrics(grid, pairs, group_bars, ranks)
    if pair_disagreement <= 0 or thin_diff <= 0:
        return None

    expected = _expected_output(grid, pairs, group_bars, ranks)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
