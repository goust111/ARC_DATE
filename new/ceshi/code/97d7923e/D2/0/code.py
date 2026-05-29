# Variant of 97d7923e
# Dimension: D2 — 匹配条件：排名与图例计数的对应 (match_cond)
# Change: s.length_rank_in_group == d.block_count — 图例条目的色块堆叠数k对应组内按长度... → 升序排名替代降序

# Design analysis
# spec_variant:
#   core_change:
#     preserved_scene: 图例仍放在左上角，主区域仍是按端点色分组的竖直 cap-fill-cap 柱条。
#     changed_match: 图例高度 k 不再指向“第 k 长”的柱条，而是指向“第 k 短”的柱条。
#   concrete_behavior_trace:
#     example_group:
#       cap: 2
#       legend_height: 1
#       bar_fill_lengths: [2, 5, 7]
#     original_rule:
#       sorted_desc: [7, 5, 2]
#       chosen_bar: 长度 7 的柱条被重着色
#     variant_rule:
#       sorted_asc: [2, 5, 7]
#       chosen_bar: 长度 2 的柱条被重着色
#     visible_difference:
#       effect: 同一个图例在输入完全不变时，会把另一根同色端点柱条的内部改成端点色
#       why_visible: 被染色的是不同列，且每组内部长度互异，肉眼可比较长短后推出“最短优先”
#   complexity_calibration:
#     kept_small: 仍只允许每组 2-3 根柱条、图例高度 1-3，避免排序规则过多档位
#     ambiguity_control:
#       - 每组柱条内部长度两两不同，避免“第 k 短”出现并列
#       - 图例色与填充色分离，避免颜色自身制造伪线索
#     observability_guard:
#       - 若所有组都落在 count=3 且 k=2，则升序/降序都会选中中位长度，变体不可见
#       - 生成器因此强制至少一个组满足 2*k != count+1，使新旧规则在该组上选中不同柱条
#   implicit_assumption_audit:
#     vertical_detection:
#       code: _vertical_sandwiches 只依赖“端点同色、内部纯色、背景不参与”
#       status: 变体不改变该前提，可原样保留
#     legend_parsing:
#       code: _legend_entries 只把图例高度读成整数 k，不关心 k 代表长短方向
#       status: 可原样保留
#     ranking_step:
#       old_dependency: main 中排序键把最长放前面
#       fix: 改为按长度升序，次级键保持 (top_row, col) 稳定顺序
#     generator_oracle:
#       old_dependency: _expected_output 用降序找 rank-k
#       fix: 同步改为升序，否则会把正确输入误判为 oracle mismatch
#   placement_legality:
#     grid: 高 10、宽 10-20、背景 0
#     bars:
#       - 所有柱条底端仍对齐到最后一行
#       - 图例与其同列柱条之间保持至少 1 行背景间隔
#       - 柱条列间距至少为 1 个背景列，不重叠
#     legality_reason:
#       - 共享列的那根柱条长度需满足 fill_len <= MAX_FILL_LEN - rank
#       - 这样柱条顶帽行一定在 legend 底下至少隔 1 行，solver 才会把图例和柱条视为两个独立结构
#   visual_cleanliness:
#     - 左上角只保留细窄图例列，主区域是稀疏单列柱条，区域职责清晰
#     - 每组共享端点色和填充色，跨组颜色对不复用，减少视觉混淆
#   color_inference:
#     - 颜色只承担“分组”与“被重着色成端点色”的可见角色
#     - 排序依据完全来自可见的柱条长度，不依赖颜色 id 等不可见数值
#   new_main_assumptions:
#     - 每个图例高度 k 仍需满足 1 <= k <= 该组柱条数
#     - 每组长度需互异，保证“第 k 短”唯一
#     - generate_input 明确按这些条件构造，并用 main(grid) == expected 做闭环验证

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
    Recolor vertical bars by legend-driven rank selection, using shortest-first order.

    Input structure:
    - Background cells dominate the grid; the upper-left legend contains one-column color stacks touching row 0.
    - Elsewhere, each candidate object is a vertical cap-fill-cap bar with one solid interior color and matching top/bottom caps.

    Rule:
    - For each legend entry (cap color K, stack height k), collect all bars whose caps are color K.
    - Rank those bars by interior length in ascending order (shortest = rank 1).
    - Recolor the interior of the rank-k bar to K.

    Key insight: The legend height is not a size to copy; it is an index into the shortest-to-longest ordering within the matching cap-color group.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, "legend must be non-empty"

    for _x, cap, k in legend:
        segs = _vertical_sandwiches(g, cap)
        assert segs, f"legend cap {cap} must have matching vertical bars"
        segs.sort(key=lambda s: (s[1] - s[0] - 1, s[0], s[2]))
        assert 1 <= k <= len(segs), f"legend rank {k} out of range for {len(segs)} bars"
        top_r, bot_r, col, _fill = segs[k - 1]
        out[top_r + 1 : bot_r, col] = cap

    return out


# --------------- generator (O2) ---------------

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
        group_columns[owner].append(col)

    return group_columns


def _choose_fill_lengths(rng, count: int, rank: int) -> list[int]:
    """Pick `count` distinct fill lengths for one group."""
    max_legend_len = MAX_FILL_LEN - rank
    shared_len = int(rng.integers(1, max_legend_len + 1))
    pool = [length for length in range(1, MAX_FILL_LEN + 1) if length != shared_len]
    rng.shuffle(pool)
    return [shared_len] + pool[: count - 1]


def _rank_is_informative(count: int, rank: int) -> bool:
    """Return True iff shortest-first and longest-first pick different bars."""
    return 2 * rank != count + 1


def _has_variant_signal(counts: list[int], ranks: list[int]) -> bool:
    """At least one group must visibly distinguish ascending from descending ranking."""
    return any(_rank_is_informative(count, rank) for count, rank in zip(counts, ranks))


def _expected_output(
    inp: np.ndarray,
    caps: list[int],
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
) -> np.ndarray:
    """Compute the expected output by recoloring the rank-k shortest bar's fill to cap."""
    out = inp.copy()
    for cap, cols, lengths, rank in zip(caps, group_columns, group_lengths, ranks):
        ranked = sorted(zip(cols, lengths), key=lambda item: (item[1], item[0]))
        target_col, target_len = ranked[rank - 1]
        top_r = _MAX_CAP_ROW - target_len
        out[top_r + 1 : _BOTTOM_ROW, target_col] = cap
    return out


def _descending_reference_output(
    inp: np.ndarray,
    caps: list[int],
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
) -> np.ndarray:
    """Reference output for the original longest-first rule, used only for degeneration checks."""
    out = inp.copy()
    for cap, cols, lengths, rank in zip(caps, group_columns, group_lengths, ranks):
        ranked = sorted(zip(cols, lengths), key=lambda item: (-item[1], item[0]))
        target_col, target_len = ranked[rank - 1]
        top_r = _MAX_CAP_ROW - target_len
        out[top_r + 1 : _BOTTOM_ROW, target_col] = cap
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch or weak variant signal."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, count + 1)) for count in counts]
    if not _has_variant_signal(counts, ranks):
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

    expected = _expected_output(grid, caps, group_columns, group_lengths, ranks)
    if not np.array_equal(main(grid), expected):
        return None

    if np.array_equal(expected, _descending_reference_output(grid, caps, group_columns, group_lengths, ranks)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
