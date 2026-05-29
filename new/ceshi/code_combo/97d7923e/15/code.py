# Combo variant of 97d7923e (index 15)
# combo_spec: D1/3+D10/2
# Name: 端点颜色对匹配+位置映射颜色

import itertools
from pathlib import Path
import sys

import numpy as np

for _parent in Path(__file__).resolve().parents:
    if (_parent / "arc_utils.py").exists():
        if str(_parent) not in sys.path:
            sys.path.append(str(_parent))
        break
    if (_parent / "solvers" / "arc_utils.py").exists():
        _solver_dir = _parent / "solvers"
        if str(_solver_dir) not in sys.path:
            sys.path.append(str(_solver_dir))
        break

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# a) 具体化与行为追踪
#    - 原题图例是单色竖块，颜色直接等于 bar 的帽色；solver 看到颜色 K 后，直接去找两端都是 K 的条。
#    - D1/3 要求 bars 改按有序端点对 `(top_cap, bottom_cap)` 分组，而不是按单一帽色分组。
#      如果两组 bars 共享同一个顶帽色但底帽色不同，只看顶帽色会把两组错误合并。
#    - D10/2 要求图例颜色不再等于 bars 的端点色；图例只通过“左到右位置”指向对应 bar 组。
#    - 融合后：左上 legend 由若干重复的 2 像素 token 组成，token 的 row0/row1 是“图例专用颜色对”
#      `(legend_top, legend_bottom)`，重复次数 k 仍然表示“第 k 长”。bars 本身使用另一套端点颜色对
#      `(bar_top, bar_bottom)`。第 i 个 legend 条目只控制第 i 个 bar 组；bar 组内部再按真实端点对分组。
#    - 小例子：若第 1 个 legend 条目是 `(8,6)` 重复 2 次，第 2 个条目是 `(8,4)` 重复 1 次，
#      但第 1、第 2 个 bar 组真实端点对分别是 `(2,5)` 与 `(2,7)`，solver 不能把 `(8,6)` 去和 bars
#      直接匹配；它必须先用“第 1 个 legend -> 第 1 个组”的位置对应，再在该组里使用真实端点对 `(2,5)`，
#      选出第 2 长的 bar，并把它的内部上半涂成 2、下半涂成 5。
# b) 复杂度校准
#    - 每个样例只保留 2-3 个组、每组 2-3 根条，长度排序仍然与原题同量级。
#    - 为了让“必须看端点对”可观察，生成器强制至少两组共享同一个顶帽色但底帽色不同，并进一步要求：
#      若错误地只按顶帽色合并，这些样例里至少有一个被选中的列会改变。
#    - 为了让“必须看位置映射”可观察，legend 使用与 bars 端点完全不重合的专用颜色；直接按颜色匹配必失败。
# c) 隐式假设审查
#    - 原 `_vertical_sandwiches(g, cap)` 假设上下端点同色；融合后不成立，因此改成读取每列最低 non-bg run，
#      并显式记录 `(top_color, bottom_color)`。
#    - 原 `_legend_entries` 依赖“单色连通块高度 == 排名”；融合后 legend 是重复的双色 token，
#      因此必须改为统计连续同色对 token 的个数。
#    - 原 `for legend_color in legend: segs = _vertical_sandwiches(g, legend_color)` 依赖图例色 == 帽色；
#      融合后该前提被 D10/2 打破，所以 main() 必须先把 bars 按真实端点对分组并按左到右排序，再与 legend
#      条目顺序配对。
#    - 原 `out[...] = cap` 只能生成单色内部；融合后若继续这样做，会丢失 D1/3 的“端点对”信息。
#      因此输出改为把选中 bar 的内部按垂直两段涂成 `(top_cap, bottom_cap)`。
# d) 放置合法性
#    - legend 只占用 row 0-1；所有承载 legend token 的 bar 列都限制 fill_len <= 5，使 bar 顶帽位于 row >= 3，
#      从而 row 2 保持背景，legend 与 bar 稳定分离。
#    - 每组 bar 列按固定步长 2 构成一个左到右连续区间，组间再留额外空隙；这样每组有唯一 leftmost column，
#      位置映射无歧义，且对象不会重叠。
#    - legend token 只放在对应组最左侧的前 `rank` 根条上，因此“legend 列位置 -> bar 组列范围”的对应关系
#      在像素上直接可见。
# e) 视觉可观察性
#    - 输入里能直接看到：上方是若干重复的双色 token，下方 bars 的端点颜色却是另一套颜色，说明 legend 颜色
#      不是直接答案。
#    - 同时至少两组共享顶帽色，迫使解题者继续看底帽色，才能把 bars 正确分成 `(top,bottom)` 组。
#    - 输出里被选中的条内部出现真实端点色的上下分层，给出了“控制颜色来自 bar 组而非 legend token”这一证据。
# f) 可视化整洁度
#    - 场景仍保持上方 legend、下方 bars 的两区布局，不引入额外装饰。
#    - 每组横向成块、组间留白，legend token 只覆盖各组左端若干列，不会与其他组的标记交错。
# g) 变化的可推断性
#    - 颜色角色分成三类：legend-only 颜色、bar endpoint 颜色、fill-only 颜色。legend 色从不出现在 bar 端点，
#      fill 色也不承担结构标记，因此少量样例就能支持“位置找组 + 端点对定色”的归纳。
# New main() assumptions and generator guarantees:
#    - legend 条目数 == bar 组数；generator 用同一个 `n_groups` 构造两者。
#    - 每个 bar 组都有唯一 leftmost column；main() 用它定义位置顺序，generator 的组布局保证该顺序稳定。
#    - 每个 legend 条目内部的 token 颜色对唯一；这样 `_legend_entries` 可以稳定切分条目。
#    - 每个组被选中的目标 bar 内部长度 >= 2；这不是 solver 必需条件，但 generator 强制它以便双色输出可见。


def _column_runs(g: np.ndarray, col: int, bg: int) -> list[tuple[int, int]]:
    """Return inclusive row ranges of contiguous non-background runs in one column."""
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
    Find bars as (top_r, bottom_r, col, top_color, bottom_color, fill_color).

    A valid bar is the lowest non-background run in a column, which leaves any
    legend token above it in a separate run.
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
        fill_color = int(g[top_r + 1, c])
        if fill_color in (bg, top_color, bottom_color):
            continue
        if np.all(g[top_r + 1 : bottom_r, c] == fill_color):
            segs.append((top_r, bottom_r, c, top_color, bottom_color, fill_color))
    return segs


def _legend_entries(g: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    """
    Read legend entries as repeated 2-pixel tokens.

    Each token occupies one column:
      - row 0: legend-top color
      - row 1: legend-bottom color

    Consecutive columns whose token pair matches, allowing the generator's
    one-cell horizontal gap, form one legend entry; the token count is rank k.
    """
    h, w = g.shape
    assert h >= 2, "legend needs two rows"
    top_pos = [int(c) for c in np.flatnonzero(g[0, :] != bg)]
    assert top_pos, "legend row must be non-empty"
    top_pos.sort()

    entries: list[tuple[int, int, int, int]] = []
    start_c = top_pos[0]
    prev_c = top_pos[0]
    legend_top = int(g[0, start_c])
    legend_bottom = int(g[1, start_c])
    assert legend_bottom != bg, "legend token must have a bottom color"
    count = 1

    for c in top_pos[1:]:
        cur_top = int(g[0, c])
        cur_bottom = int(g[1, c])
        assert cur_bottom != bg, "legend token must have a bottom color"
        same_entry = (c - prev_c <= 2) and (cur_top == legend_top) and (cur_bottom == legend_bottom)
        if same_entry:
            count += 1
        else:
            entries.append((start_c, legend_top, legend_bottom, count))
            start_c = c
            legend_top = cur_top
            legend_bottom = cur_bottom
            count = 1
        prev_c = c

    entries.append((start_c, legend_top, legend_bottom, count))
    return entries


def _ordered_pair_groups(
    bars: list[tuple[int, int, int, int, int, int]]
) -> list[tuple[int, tuple[int, int], list[tuple[int, int, int, int, int, int]]]]:
    """Group bars by ordered endpoint pair and return groups ordered by leftmost column."""
    grouped: dict[tuple[int, int], list[tuple[int, int, int, int, int, int]]] = {}
    for seg in bars:
        grouped.setdefault((seg[3], seg[4]), []).append(seg)

    groups: list[tuple[int, tuple[int, int], list[tuple[int, int, int, int, int, int]]]] = []
    for pair, segs in grouped.items():
        leftmost = min(seg[2] for seg in segs)
        groups.append((leftmost, pair, segs))
    groups.sort(key=lambda item: (item[0], item[1][0], item[1][1]))
    return groups


def _paint_pair_fill(
    out: np.ndarray, top_r: int, bottom_r: int, col: int, top_color: int, bottom_color: int
) -> None:
    """Recolor the interior with an upper top-color band and a lower bottom-color band."""
    fill_len = bottom_r - top_r - 1
    split = (fill_len + 1) // 2
    out[top_r + 1 : top_r + 1 + split, col] = top_color
    out[top_r + 1 + split : bottom_r, col] = bottom_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor vertical bars by legend-order endpoint-pair rank selection.

    Input structure:
    - The top area contains repeated 2-pixel legend tokens. Each token column has one legend-only color on row 0
      and another legend-only color on row 1; repeating the same token pair k times forms one legend entry.
    - The lower area contains one-column bars with solid interiors. Bars are grouped by their ordered endpoint pair,
      so a group's top cap and bottom cap may be different colors.

    Rule:
    - Read legend entries from left to right, and independently order the bar groups by the leftmost column where
      each endpoint pair first appears.
    - Pair the i-th legend entry with the i-th endpoint-pair group. Within that group, rank bars by interior length
      from longest to shortest.
    - Recolor the rank-k bar's interior using the paired group's real endpoint colors: the upper part becomes the
      group's top-cap color and the lower part becomes the group's bottom-cap color.

    Key insight:
    - The legend's colors are decoys. The useful information is the legend entry's position and repeat count: position
      chooses which endpoint-pair group to inspect, and repeat count chooses which length-ranked bar inside that group
      to repaint with the bar group's own endpoint colors.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, "legend must be non-empty"

    bars = _vertical_bars(g)
    assert bars, "input must contain vertical bars"

    groups = _ordered_pair_groups(bars)
    assert len(legend) == len(groups), "legend/group count mismatch"

    for (_legend_x, _legend_top, _legend_bottom, k), (_group_x, pair, segs) in zip(legend, groups):
        segs = sorted(segs, key=lambda s: (-(s[1] - s[0] - 1), s[0], s[2]))
        assert 1 <= k <= len(segs), f"legend rank {k} out of range for {len(segs)} bars"
        top_r, bottom_r, col, top_color, bottom_color, _fill = segs[k - 1]
        assert pair == (top_color, bottom_color)
        _paint_pair_fill(out, top_r, bottom_r, col, top_color, bottom_color)

    return out


# --------------- generator (O2) ---------------

# --- Configuration ---
GRID_H = 10
WIDTH_MIN = 10
WIDTH_MAX = 20
MIN_GROUPS = 2
MAX_GROUPS = 3
MAX_TOTAL_BARS = 9
BARS_PER_GROUP = (2, 3)
MAX_FILL_LEN = 7
MAX_SHARED_FILL_LEN = 5
MIN_TARGET_FILL_LEN = 2
COLOR_RANGE = (1, 10)
COL_SPACING = 2
MAX_RETRIES = 500

_BOTTOM_ROW = GRID_H - 1
_TOKEN_BOTTOM_ROW = 1
_MAX_CAP_ROW = GRID_H - 2

_FEASIBLE_COUNTS = {
    ng: [c for c in itertools.product(BARS_PER_GROUP, repeat=ng) if sum(c) <= MAX_TOTAL_BARS]
    for ng in range(MIN_GROUPS, MAX_GROUPS + 1)
}

_LENGTH_ASSIGNMENTS: dict[tuple[int, int], list[tuple[int, ...]]] = {}
for _count in BARS_PER_GROUP:
    for _rank in range(1, _count + 1):
        vals: list[tuple[int, ...]] = []
        for perm in itertools.permutations(range(1, MAX_FILL_LEN + 1), _count):
            if any(perm[i] > MAX_SHARED_FILL_LEN for i in range(_rank)):
                continue
            if sorted(perm, reverse=True)[_rank - 1] < MIN_TARGET_FILL_LEN:
                continue
            vals.append(tuple(int(v) for v in perm))
        _LENGTH_ASSIGNMENTS[(_count, _rank)] = vals


def _build_bar(top_color: int, fill_color: int, fill_len: int, bottom_color: int) -> np.ndarray:
    """Return a vertical sprite: top cap / fill / bottom cap."""
    bar = np.full((fill_len + 2, 1), fill_color, dtype=int)
    bar[0, 0] = top_color
    bar[-1, 0] = bottom_color
    return bar


def _assign_group_columns(rng, counts: list[int], width: int) -> list[list[int]]:
    """Assign each group a contiguous left-to-right block of bar columns with spacing 2."""
    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    assert width >= min_width
    extra_width = width - min_width

    gap_bonus = rng.multinomial(extra_width, np.full(len(counts), 1.0 / len(counts)))
    group_columns: list[list[int]] = []
    cur = 0
    for gi, count in enumerate(counts):
        cols = [cur + COL_SPACING * j for j in range(count)]
        group_columns.append(cols)
        cur = cols[-1] + COL_SPACING + int(gap_bonus[gi])
    return group_columns


def _choose_pairs_from_palette(
    rng, palette: list[int], n_groups: int, *, require_shared_top: bool
) -> list[tuple[int, int]]:
    """Sample unique ordered color pairs from a palette of three colors."""
    all_pairs = [(int(a), int(b)) for a in palette for b in palette if a != b]
    for _ in range(300):
        rng.shuffle(all_pairs)
        pairs = list(all_pairs[:n_groups])
        if not require_shared_top:
            return pairs
        top_counts: dict[int, int] = {}
        for top_color, _bottom_color in pairs:
            top_counts[top_color] = top_counts.get(top_color, 0) + 1
        if max(top_counts.values()) >= 2:
            return pairs
    raise RuntimeError("failed to sample ordered pairs with the required shared-top structure")


def _random_lengths(rng, count: int, rank: int) -> list[int]:
    candidates = _LENGTH_ASSIGNMENTS[(count, rank)]
    choice = candidates[int(rng.integers(len(candidates)))]
    return [int(v) for v in choice]


def _ranked_target(cols: list[int], lengths: list[int], rank: int) -> tuple[int, int]:
    ranked = sorted(zip(cols, lengths), key=lambda item: (-item[1], item[0]))
    target_col, target_len = ranked[rank - 1]
    return int(target_col), int(target_len)


def _expected_output(
    inp: np.ndarray,
    endpoint_pairs: list[tuple[int, int]],
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
) -> np.ndarray:
    """Compute the expected recolored output using the true endpoint-pair groups."""
    out = inp.copy()
    for (top_color, bottom_color), cols, lengths, rank in zip(endpoint_pairs, group_columns, group_lengths, ranks):
        target_col, target_len = _ranked_target(cols, lengths, rank)
        top_r = _MAX_CAP_ROW - target_len
        _paint_pair_fill(out, top_r, _BOTTOM_ROW, target_col, top_color, bottom_color)
    return out


def _top_only_disagreement(
    endpoint_pairs: list[tuple[int, int]],
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
) -> int:
    """
    Count legend entries whose chosen bar would change if bars were grouped only by top color.

    A positive value means the sample genuinely exercises the endpoint-pair grouping.
    """
    actual_targets: dict[tuple[int, int], int] = {}
    for pair, cols, lengths, rank in zip(endpoint_pairs, group_columns, group_lengths, ranks):
        actual_targets[pair] = _ranked_target(cols, lengths, rank)[0]

    by_top: dict[int, list[tuple[int, int]]] = {}
    for (top_color, _bottom_color), cols, lengths in zip(endpoint_pairs, group_columns, group_lengths):
        for col, fill_len in zip(cols, lengths):
            by_top.setdefault(top_color, []).append((int(col), int(fill_len)))

    wrong = 0
    for pair, rank in zip(endpoint_pairs, ranks):
        merged = sorted(by_top[pair[0]], key=lambda item: (-item[1], item[0]))
        wrong_col = merged[rank - 1][0]
        if wrong_col != actual_targets[pair]:
            wrong += 1
    return wrong


def _position_mapping_needed(
    legend_pairs: list[tuple[int, int]], endpoint_pairs: list[tuple[int, int]]
) -> int:
    """
    Count legend entries that cannot be matched to any bar group by direct color equality.

    In this combo, the desired value is the full group count.
    """
    endpoint_set = set(endpoint_pairs)
    return sum(1 for pair in legend_pairs if pair not in endpoint_set)


def _selected_lengths(group_columns: list[list[int]], group_lengths: list[list[int]], ranks: list[int]) -> list[int]:
    """Return the interior lengths of the actually selected bars, one per group."""
    return [_ranked_target(cols, lengths, rank)[1] for cols, lengths, rank in zip(group_columns, group_lengths, ranks)]


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid sample. Returns None on oracle mismatch or weak fusion signal."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(MIN_GROUPS, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, count + 1)) for count in counts]

    palette = rng.choice(np.arange(*COLOR_RANGE), size=6 + n_groups, replace=False)
    endpoint_palette = [int(v) for v in palette[:3]]
    legend_palette = [int(v) for v in palette[3:6]]
    fills = [int(v) for v in palette[6:]]

    endpoint_pairs = _choose_pairs_from_palette(rng, endpoint_palette, n_groups, require_shared_top=True)
    legend_pairs = _choose_pairs_from_palette(rng, legend_palette, n_groups, require_shared_top=False)

    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    group_columns = _assign_group_columns(rng, counts, width)

    grid = np.zeros((GRID_H, width), dtype=int)
    group_lengths: list[list[int]] = []

    for legend_pair, endpoint_pair, fill_color, count, rank, cols in zip(
        legend_pairs, endpoint_pairs, fills, counts, ranks, group_columns
    ):
        lengths = _random_lengths(rng, count, rank)
        group_lengths.append(lengths)

        for token_col in cols[:rank]:
            grid[0, token_col] = legend_pair[0]
            grid[_TOKEN_BOTTOM_ROW, token_col] = legend_pair[1]

        for col, fill_len in zip(cols, lengths):
            top_r = _MAX_CAP_ROW - fill_len
            paste(grid, _build_bar(endpoint_pair[0], fill_color, fill_len, endpoint_pair[1]), top_r, col)

    if _top_only_disagreement(endpoint_pairs, group_columns, group_lengths, ranks) <= 0:
        return None
    if _position_mapping_needed(legend_pairs, endpoint_pairs) != n_groups:
        return None
    if min(_selected_lengths(group_columns, group_lengths, ranks)) < MIN_TARGET_FILL_LEN:
        return None

    expected = _expected_output(grid, endpoint_pairs, group_columns, group_lengths, ranks)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
