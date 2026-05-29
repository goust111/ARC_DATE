# Variant of 97d7923e
# Dimension: D7 — 网格形状与尺寸 (grid_shape)
# Change: 输出与输入网格尺寸相同(10行 x 10-20列)，仅修改匹配条的填充色... → 输出将每组条提取为独立子网格

# Design analysis:
# spec_variant:
#   concrete_output:
#     - 输入仍保留原题结构：左上角是按高度编码排名的图例，主体是若干组同端点色的竖条。
#     - 输出不再是原图就地改色；每个图例颜色对应的一组竖条都会被抽出，重建成一个独立 frame。
#     - 每个 frame 采用统一布局：第 0 列是该组图例柱，第 1 列留空，之后把该组竖条按原输入中的左右顺序压紧排到偶数列。
#     - 各 frame 再按图例从左到右的顺序横向拼接，中间插入一列整高分隔色，形成“分帧输出”。
#   behavior_trace:
#     - 原题：找到颜色 K 的所有竖条，按长度降序排序，把第 k 名的填充改成 K，结果仍留在原网格中。
#     - 本变体：前半段完全相同，仍然由图例高度 k 选择第 k 名竖条；但后半段不再回写原图，而是只保留该颜色组，重组到单独 frame 中。
#     - 差异集中在最后的渲染阶段，因此 D7 是核心变化，原有对象识别与排名逻辑保持稳定。
#   complexity_calibration:
#     - 生成器固定使用 2-4 组而不是允许 1 组，避免出现“只有一个 frame”导致分帧规律不显眼的样例。
#     - 每组仍保留 2-3 根竖条，且组内长度互异；这样 rank 选择在视觉上足够明确，也不会引入额外歧义。
#     - frame 布局是单一固定模板，没有旋转、镜像或多种输出制式，避免少量样例无法归纳。
#   implicit_assumptions_audit:
#     - `_legend_entries` 依赖“顶行左侧存在一个紧凑图例簇”；生成器继续沿用原题的 legend 放置方式，确保该前提成立。
#     - `_vertical_sandwiches` 依赖“竖条是 cap/fill/cap 的纯色连续列”；生成器完全保留这一点。
#     - 原题 `out = g.copy()` 默认输出尺寸不变；变体下这个前提失效，因此 `main()` 必须改为从空白画布重建输出。
#     - 直接复制输入整列会把“图例与竖条共列”的那一列也整体复制，导致 frame 中重复出现图例；因此 frame 里的竖条必须根据段信息重绘，而不是整列搬运。
#   placement_legality:
#     - 输入放置仍遵守原题的非重叠条件：legend 在上方，竖条到底对齐，图例共享列的那根竖条与图例之间保留至少一行背景。
#     - 输出 frame 采用固定偶数列放条的重排规则，组内对象不会互相覆盖；frame 间也由整列分隔。
#   visual_observability:
#     - 每个样例至少有 2 组，因此一定出现多个 frame。
#     - 分隔列是统一的整高颜色，frame 边界肉眼可见。
#     - 组内仍保留原本的条高差、端点色和填充色，所以“图例选 rank，再抽出分组”可以从样例直接读出。
#   cleanliness_and_color_reasoning:
#     - 只有一个新的结构性色彩角色：frame 分隔列；它总是选用输入中未使用的备用颜色，不参与排序或选择。
#     - 真正的颜色变化仍然是原题规则：被图例选中的那根条，其内部填充变成端点色。

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


def _legend_entries(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    """
    Legend is in the top-left: a cluster of non-bg pixels on the top row near the left edge,
    allowing small gaps (typically 1 bg column) between entries.

    Returns entries as (x_pos, legend_color, block_count), sorted by x_pos.
    block_count is the vertical height of the 4-connected component of that color that touches
    the top row within the legend cluster.
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


def _choose_divider_color(g: np.ndarray, bg: int) -> int:
    used = set(map(int, g.ravel()))
    for color in range(1, 10):
        if color != bg and color not in used:
            return color
    raise AssertionError("need a spare non-bg color for frame separators")


def _build_group_frame(g: np.ndarray, bg: int, cap: int, rank: int) -> np.ndarray:
    segs = _vertical_sandwiches(g, cap)
    assert segs, f"legend cap {cap} must have matching vertical bars"

    ranked = sorted(segs, key=lambda s: (-(s[1] - s[0] - 1), s[0], s[2]))
    assert 1 <= rank <= len(ranked), f"legend rank {rank} out of range for {len(segs)} bars"
    target_col = ranked[rank - 1][2]

    ordered = sorted(segs, key=lambda s: s[2])
    frame = np.full((g.shape[0], 2 * len(ordered) + 1), bg, dtype=int)
    frame[:rank, 0] = cap

    for idx, (top_r, bot_r, src_col, fill) in enumerate(ordered):
        dest_col = 2 + 2 * idx
        frame[top_r, dest_col] = cap
        frame[bot_r, dest_col] = cap
        frame[top_r + 1 : bot_r, dest_col] = cap if src_col == target_col else fill

    return frame


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild each cap-color group into its own frame after legend-driven rank selection.

    Input structure:
    - Top-left: a compact legend cluster touching row 0; each monochrome entry has color K and height k.
    - Main field: vertical cap/fill/cap bars, where all bars in one group share the same cap color K.

    Rule:
    - For each legend entry (K, k), find every vertical bar capped by K.
    - Rank those bars by interior length (longest = rank 1) and recolor the rank-k bar's interior to K.
    - Rebuild that entire color group as a compact frame: legend column on the left, then the group's bars packed to the right in their original left-to-right order.
    - Concatenate all frames from left to right, inserting one full-height divider column between neighboring frames.

    Key insight: The legend height still acts as a rank index, but the answer is no longer an in-place recolor; the solver must extract each group and reassemble the output as separated frames.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    legend = _legend_entries(g, bg)
    assert legend, "legend must be non-empty"

    frames = [_build_group_frame(g, bg, cap, k) for _x, cap, k in legend]
    divider = _choose_divider_color(g, bg)

    out_w = sum(frame.shape[1] for frame in frames) + max(0, len(frames) - 1)
    out = np.full((g.shape[0], out_w), bg, dtype=int)

    cursor = 0
    for idx, frame in enumerate(frames):
        paste(out, frame, 0, cursor)
        cursor += frame.shape[1]
        if idx + 1 < len(frames):
            out[:, cursor] = divider
            cursor += 1

    return out


# --------------- generator (D7 frame-separation variant) ---------------

# --- Configuration ---
GRID_H = 10                  # fixed input height
WIDTH_MIN = 10               # minimum input width
WIDTH_MAX = 20               # maximum input width (inclusive)
MAX_GROUPS = 4               # max number of cap-color groups
MAX_TOTAL_BARS = 10          # max bars across all groups
BARS_PER_GROUP = (2, 3)      # allowed bars-per-group counts
MAX_FILL_LEN = 7             # max interior fill length of a bar
COLOR_RANGE = (1, 10)        # non-bg color ids (exclusive upper)
COL_SPACING = 2              # minimum column stride between bars
MAX_RETRIES = 300            # retry budget for generate_input

MIN_GROUPS = 2
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
    """Pick `count` distinct fill lengths for one group."""
    max_legend_len = MAX_FILL_LEN - rank
    shared_len = int(rng.integers(1, max_legend_len + 1))
    pool = [l for l in range(1, MAX_FILL_LEN + 1) if l != shared_len]
    rng.shuffle(pool)
    return [shared_len] + pool[: count - 1]


def _expected_frame(
    cap: int,
    fill: int,
    cols: list[int],
    lengths: list[int],
    rank: int,
) -> np.ndarray:
    frame = np.zeros((GRID_H, 2 * len(cols) + 1), dtype=int)
    frame[:rank, 0] = cap

    ranked = sorted(zip(cols, lengths), key=lambda item: (-item[1], item[0]))
    target_col, _target_len = ranked[rank - 1]

    for idx, (src_col, fill_len) in enumerate(sorted(zip(cols, lengths), key=lambda item: item[0])):
        dest_col = 2 + 2 * idx
        top_r = _MAX_CAP_ROW - fill_len
        frame[top_r, dest_col] = cap
        frame[top_r + 1 : _BOTTOM_ROW, dest_col] = cap if src_col == target_col else fill
        frame[_BOTTOM_ROW, dest_col] = cap

    return frame


def _expected_output(
    caps: list[int],
    fills: list[int],
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
    divider: int,
) -> np.ndarray:
    frames = [
        _expected_frame(cap, fill, cols, lengths, rank)
        for cap, fill, cols, lengths, rank in zip(caps, fills, group_columns, group_lengths, ranks)
    ]

    out_w = sum(frame.shape[1] for frame in frames) + len(frames) - 1
    out = np.zeros((GRID_H, out_w), dtype=int)

    cursor = 0
    for idx, frame in enumerate(frames):
        paste(out, frame, 0, cursor)
        cursor += frame.shape[1]
        if idx + 1 < len(frames):
            out[:, cursor] = divider
            cursor += 1

    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(MIN_GROUPS, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, c + 1)) for c in counts]

    palette = rng.choice(np.arange(*COLOR_RANGE), size=2 * n_groups, replace=False)
    caps = [int(v) for v in palette[:n_groups]]
    fills = [int(v) for v in palette[n_groups:]]
    divider = min(color for color in range(1, 10) if color not in set(caps + fills))

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

    expected = _expected_output(caps, fills, group_columns, group_lengths, ranks, divider)
    if expected.shape == grid.shape:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
