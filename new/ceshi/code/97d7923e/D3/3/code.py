# Variant of 97d7923e
# Dimension: D3 — 变换操作：重着色规则 (transform)
# Change: 将匹配条的整个填充区域重着色为帽色(cap color)，即图例颜色覆盖填充色
#         → 重着色为图例指定的第三种颜色
#
# Design analysis
# a) 具体化与行为追踪
#    - 保留原题主体对象: 每组仍是若干根竖直的 cap/fill/cap 单列条，按同一 cap 颜色分组，按 fill 长度降序排名。
#    - 变体只改图例编码与输出颜色。每个图例项现在是“并排两列、等高”的双色竖块:
#      左列颜色 K = 匹配帽色，右列颜色 T = 目标重着色颜色，共同高度 k = 排名索引。
#    - 小例子: 若顶部出现 [蓝][红] 两列且高度为 2，则要在所有帽色=蓝的条里找第 2 长的那根，
#      原题会把其填充区刷成蓝色；本变体改为刷成红色。差异集中发生在最后一步赋值，排序逻辑不变。
# b) 复杂度校准
#    - 图例只新增一个相邻颜色列，没有引入额外对象种类或多分支规则。
#    - 同一图中仍有 1-4 组条，每组 2-3 根，能重复展示“左色负责匹配、右色负责重着色”的模式。
#    - 目标色与帽色始终不同，因此输出与原题的视觉差异稳定可见。
# c) 隐式假设审查
#    - 原 _legend_entries 只返回一种颜色并默认它既是匹配键也是输出色；变体后这个前提失效，必须改为同时解析 K 和 T。
#    - 原 main 中 `out[...] = cap` 依赖“输出色=cap”；变体后必须改为 `out[...] = target`。
#    - 原 _vertical_sandwiches 只依赖条两端是同色 cap、内部是纯色 fill，这一点在变体中仍成立，可复用。
# d) 放置合法性
#    - 仍让每组的第一根条所在列兼作图例左列(cap 列)；图例右列放在其右侧紧邻的空列，只占顶部若干格，不放条。
#    - 原列分配函数前 n_groups 根条固定落在偶数列 0,2,4,...，因此相邻奇数列 1,3,5,... 可安全承载图例右列，不越界也不与条重叠。
#    - 图例左列下方继续保留至少一行背景隔开 legend 与 bar 顶帽，避免 solver 把 legend 与 bar 误并成一个更长的对象。
# e) 视觉可观察性
#    - 排序仍只由肉眼可见的条长决定，颜色变化只由相邻双色图例提示，不使用任何颜色 id 的数值性质。
#    - 2-4 组样例中，解题者能直接看到“条端点颜色对应图例左列，条内部改成图例右列”。
# f) 可视化整洁度
#    - 新增颜色信息只出现在左上角 legend 区，主体条形区不增加额外覆盖层，因此整体布局仍然整齐。
#    - 图例各项共享统一的“两列同高”外观，避免局部标记过碎。
# g) 颜色变化可推断性
#    - 每个图例项只引入一对颜色映射 K→T，高度仍承担 rank 信息，颜色语义清晰。
#    - 生成器保证所有 legend 左/右列颜色全局互异，且条的 fill 颜色不复用任何 legend 颜色，减少误读。
# 9) main 的新假设与生成器保证
#    - 假设顶行最左连续非背景区完全由 legend 组成，且列数为偶数；生成器始终把 legend 放在最左端且只占这片连续区。
#    - 假设每个 legend 项恰是两列相邻、等高、不同色；生成器按此固定构造。
#    - 假设每个目标色 T 与该组 fill 不同，否则重着色不可见；生成器从非 legend 颜色池选 fill，保证这一点。

import itertools

import numpy as np

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


def _top_run_length(g: np.ndarray, col: int, color: int) -> int:
    """Return the length of the same-color run starting at row 0 in one column."""
    h = 0
    while h < g.shape[0] and int(g[h, col]) == color:
        h += 1
    return h


def _legend_entries(g: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    """
    Parse the top-left legend as adjacent column pairs.

    Returns entries as (x_pos, cap_color, target_color, rank), sorted by x_pos.
    Each entry occupies two neighboring top-touching columns:
      - left column: cap color K
      - right column: repaint color T
      - both columns have the same height k
    """
    h, w = g.shape
    top_pos = [int(c) for c in np.flatnonzero(g[0, :] != bg)]
    assert top_pos, 'legend row must be non-empty'
    top_pos.sort()

    cols = [top_pos[0]]
    for c in top_pos[1:]:
        if c == cols[-1] + 1:
            cols.append(c)
        else:
            break

    assert len(cols) % 2 == 0, 'legend must contain an even number of columns'

    entries: list[tuple[int, int, int, int]] = []
    for i in range(0, len(cols), 2):
        cap_col = cols[i]
        target_col = cols[i + 1]
        assert target_col == cap_col + 1, 'legend columns must form adjacent pairs'

        cap = int(g[0, cap_col])
        target = int(g[0, target_col])
        assert cap != bg and target != bg and cap != target, 'legend colors must be distinct non-bg values'

        rank = _top_run_length(g, cap_col, cap)
        target_rank = _top_run_length(g, target_col, target)
        assert rank == target_rank, 'paired legend columns must have the same height'

        entries.append((cap_col, cap, target, rank))

    return entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use a two-color legend to select ranked bars and repaint them.

    Input structure:
    - Background is the dominant color. The top-left legend is made of adjacent two-column entries touching row 0.
    - In each legend entry, the left column has cap color K, the right column has repaint color T, and both columns share height k.
    - The main field contains vertical cap-fill-cap bars whose endpoints are K and whose interiors are solid non-background colors.

    Rule:
    - For each legend entry (K, T, k), collect the bars capped by K.
    - Sort those bars by interior height from longest to shortest.
    - Recolor the interior of the rank-k bar to T while leaving its caps unchanged.

    Key insight: The legend height still encodes rank, but color meaning is split across the paired columns: the left color chooses the bar group and the right color supplies the new fill color.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, 'legend must be non-empty'

    for _x, cap, target, k in legend:
        segs = _vertical_sandwiches(g, cap)
        assert segs, f'legend cap {cap} must have matching vertical bars'
        segs.sort(key=lambda s: (-(s[1] - s[0] - 1), s[0], s[2]))
        assert 1 <= k <= len(segs), f'legend rank {k} out of range for {len(segs)} bars'
        top_r, bot_r, col, _fill = segs[k - 1]
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

_BOTTOM_ROW = GRID_H - 1    # row index where all bars end
_MAX_CAP_ROW = GRID_H - 2   # highest row a bar's top cap can occupy
_ALL_NON_BG = list(range(*COLOR_RANGE))

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
    """Distribute actual bar columns across groups with random inter-group spacing.

    The first column of each group doubles as the legend's left cap column.
    Its immediate right neighbor remains free of bars so the legend's target-color
    column can be placed there without changing the bar layout logic.
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

    return group_columns


def _choose_fill_lengths(rng, count: int, rank: int) -> list[int]:
    """Pick `count` distinct fill lengths for one group.

    The first length (legend-sharing bar column) is constrained so the bar fits
    below the legend block with one background row separating them.
    """
    max_legend_len = MAX_FILL_LEN - rank
    shared_len = int(rng.integers(1, max_legend_len + 1))
    pool = [length for length in range(1, MAX_FILL_LEN + 1) if length != shared_len]
    rng.shuffle(pool)
    return [shared_len] + pool[: count - 1]


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
    group_columns = _assign_columns(rng, n_groups, counts, width)

    grid = np.zeros((GRID_H, width), dtype=int)
    group_lengths: list[list[int]] = []
    for cap, target, fill, count, rank, cols in zip(caps, targets, fills, counts, ranks, group_columns):
        paste(grid, np.full((rank, 1), cap, dtype=int), 0, cols[0])
        paste(grid, np.full((rank, 1), target, dtype=int), 0, cols[0] + 1)

        lengths = _choose_fill_lengths(rng, count, rank)
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
