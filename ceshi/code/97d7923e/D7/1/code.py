# Variant of 97d7923e
# Dimension: D7 — 网格形状与尺寸 (grid_shape)
# Change: 输出与输入网格尺寸相同(10行 x 10-20列)，仅修改匹配条的填充色... → 输出仅包含被重着色的条

import itertools

import numpy as np

from arc_utils import get_bg, neighbors4, paste


# Design analysis for the D7 variant
# ----------------------------------
# variant_spec:
#   preserved_input:
#     - 输入仍是 10 x (10..20) 的黑底网格。
#     - 左上角图例仍由 1..4 个竖直单色块组成，颜色 K 对应某一组条的端点色，块高 k 表示排名。
#     - 主体仍是若干竖直 cap-fill-cap 条：顶部和底部为端点色 K，中间是统一填充色 F。
#   changed_output:
#     - 原题输出保留整张图，只把每组中排名第 k 的条内部改成 K。
#     - 变体输出改成“抽取赢家条”：先按原题规则找出每组被选中的那一条，再只保留这些条；
#       其余所有条和全部图例都清零为背景色。
#     - 因为被保留的是“改色后的条”，所以每个保留条会成为一根从 top cap 到 bottom cap 全部为 K 的实心竖线。
#   concrete_trace:
#     - 例：某个颜色 K 有 3 根条，内部长度分别是 6、4、2，图例高度 k=2。
#     - 原题：长度降序后选中长度 4 的那根，只把它内部的 4 个 F 改成 K，其余条和图例保持。
#     - 变体：仍选中长度 4 的那根，但输出网格先全清空，只在该列保留这一整根条，因此图上只剩一根纯 K 竖线。
#   complexity_calibration:
#     - 不改变匹配机制、排序机制、颜色角色或条的几何约束，只反转“输出保留什么”。
#     - 每组仍有 2..3 根条，保证每个样例都存在未被选中的条，避免退化成“新旧输出一样”。
#     - 图例仍保留 1..4 组，颜色种类有限，解题者能从少量样例看出“只留下每组胜出的那根条”。
#   implicit_assumptions_to_recheck:
#     - 原 main() 的 `out = g.copy()` 依赖“未命中像素默认保留”这个旧属性；变体下必须改成背景画布。
#     - 原 main() 仅修改 `top_r+1:bot_r`，因为端点本来就保留；变体下若只改内部会丢失条的端点，
#       所以输出阶段必须绘制 `top_r:bot_r+1` 整段。
#     - `_legend_entries()` 假设顶行左侧存在连续图例簇，且后面的顶行彩色像素不属于图例。生成器继续保证这一点。
#     - `_vertical_sandwiches()` 假设同色 cap 在同一列中按相邻出现配对，且内部是单一非背景非 cap 颜色。生成器保持不变。
#   placement_legality:
#     - legend 仍放在每组的第一列上方，条仍底对齐到最后一行。
#     - 共享图例列的那根条必须位于图例下方且至少隔 1 行背景；因此图例列条长仍限制为 `<= MAX_FILL_LEN - rank`。
#     - 组内长度保持互异，确保按长度排序没有歧义。
#   observability:
#     - 输入中同时出现图例、被选中条和未选中条；输出里图例全部消失，只有少数纯色竖线残留，
#       与原题“整图保留、局部改色”形成强可见对比。
#   new_main_assumptions_and_generator_support:
#     - `main()` 现在假设每个 legend entry 至少能选出 1 根合法条，且 rank 不超过该组条数；generator 直接按组数和 rank 构造。
#     - `main()` 按列保留完整胜出条，假设不同组不会共列；generator 的列分配继续保证列互斥。


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
    assert top_pos, 'legend row must be non-empty'
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


def _selected_bars(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return selected bars as (top_r, bottom_r, col, cap)."""
    bg = get_bg(g)
    legend = _legend_entries(g, bg)
    assert legend, 'legend must be non-empty'

    selected: list[tuple[int, int, int, int]] = []
    for _x, cap, k in legend:
        segs = _vertical_sandwiches(g, cap)
        assert segs, f'legend cap {cap} must have matching vertical bars'
        segs.sort(key=lambda s: (-(s[1] - s[0] - 1), s[0], s[2]))
        assert 1 <= k <= len(segs), f'legend rank {k} out of range for {len(segs)} bars'
        top_r, bot_r, col, _fill = segs[k - 1]
        selected.append((top_r, bot_r, col, cap))
    return selected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Isolate the legend-selected vertical bars.

    Input structure:
    - bg = background. Top-left: monochrome blocks touching row 0; each has color K and height k.
    - Main grid: vertical cap-fill-cap bars whose top/bottom endpoints share color K and whose interiors are solid.

    Rule:
    - For each legend entry (K, k), find all bars capped by K and sort them by interior length (longest = rank 1).
    - Select the rank-k bar and recolor its interior to K, exactly as in the original task.
    - Output only those selected recolored bars; erase every other bar and all legend pixels to background.

    Key insight: The legend still encodes a rank into each cap-color group, but the result is no longer an in-place recoloring of the whole scene; it is a filtered canvas that keeps only the winning bars.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = np.full_like(g, bg)

    for top_r, bot_r, col, cap in _selected_bars(g):
        out[top_r : bot_r + 1, col] = cap

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
COL_SPACING = 2              # minimum column stride between bars
MAX_RETRIES = 300            # retry budget for generate_input

_BOTTOM_ROW = GRID_H - 1    # row index where all bars end
_MAX_CAP_ROW = GRID_H - 2   # highest row a bar's top cap can occupy

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

    Returns one list of column indices per group; the first column in each
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

    return group_columns


def _choose_fill_lengths(rng, count: int, rank: int) -> list[int]:
    """Pick `count` distinct fill lengths for one group.

    The first length (legend column bar) is constrained so the bar fits
    below the legend block with a bg gap row.
    """
    max_legend_len = MAX_FILL_LEN - rank
    shared_len = int(rng.integers(1, max_legend_len + 1))
    pool = [length for length in range(1, MAX_FILL_LEN + 1) if length != shared_len]
    rng.shuffle(pool)
    return [shared_len] + pool[: count - 1]


def _expected_output(
    inp: np.ndarray,
    caps: list[int],
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
) -> np.ndarray:
    """Compute the expected output for the selective-output variant."""
    out = np.full_like(inp, get_bg(inp))
    for cap, cols, lengths, rank in zip(caps, group_columns, group_lengths, ranks):
        ranked = sorted(zip(cols, lengths), key=lambda item: -item[1])
        target_col, target_len = ranked[rank - 1]
        top_r = _MAX_CAP_ROW - target_len
        out[top_r : _BOTTOM_ROW + 1, target_col] = cap
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, count + 1)) for count in counts]

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
