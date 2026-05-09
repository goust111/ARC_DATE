# Combo variant of 1ae2feb7 (index 10)
# combo_spec: D9/1+D12/0
# Name: 双分隔条双侧色条

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for combo D9/1 + D12/0
# =======================================
# a) Concrete behavior / trace
#    原题只有一条近乎满高的竖分隔条，所有水平色条都在同一侧，另一侧整块为空。每条色条以
#    “长度 = 步长”的方式向空白侧周期打点，并按“离分隔条越近越先处理、不覆盖已有颜色”。
#    现在改成两条同色竖分隔条，把网格切成 [左色条区 | 中间目标区 | 右色条区] 三段，且左右外侧
#    都能放色条。左侧色条只向右打到中间区，右侧色条只向左打到中间区。若某一行左右两侧都有色条，
#    则两束射线会在中间区相遇并竞争同一批格子。
#    具体小例子：中间区宽 7，左侧某行有一条长度 3 的近条，则它尝试填 left_sep+1,left_sep+4,...
#    同行右侧若有一条长度 2 的近条，则它尝试填 right_sep-1,right_sep-3,...。若两侧都声明同一格，
#    先处理距离本侧分隔条更近的条；若距离并列，则固定左侧优先。生成器不会把“并列 tie-break”
#    当成核心规律，而是强制至少出现“不同距离导致的真实竞争”，让人能从样例看出几何优先关系。
#
# b) Complexity calibration
#    若只放双分隔条但多数行仍只来自单侧，会太接近 D9/1；若仅仅把左右两侧都用上但从不在同一行交汇，
#    又会接近 D12/0 的“分行独立发射”。因此生成器同时强制：
#    - 左右两侧在每个样本里都至少出现一次；
#    - 至少一条活动行两侧同时有色条；
#    - 至少一条双侧活动行在中间区存在双方都想填的共享目标格；
#    - 至少一条共享目标格的胜负由“距离更近”决定，而不是靠同距 tie-break。
#    这样融合后的差异在 2-4 个样例中能直接看见，同时每侧每行仍只保留 1-2 条相邻色条，自由度没有失控。
#
# c) Hidden assumptions broken in the original main()
#    - 原版只找一条分隔列；现在必须找两条同色候选列。
#    - 原版用 left_mass/right_mass 判断哪边是 bars；现在两边都可能是 bars，中间才是目标区。
#    - 原版只扫描分隔条一侧；现在要分别扫描 [0,left_sep) 和 (right_sep,w)。
#    - 原版只向单侧整块区域打点；现在左侧只能打到 (left_sep,right_sep) 内，右侧同理。
#    - 原版“近条优先”只在单侧成立；现在需要把左右两侧的操作并入一个排序，主键仍是“离本侧分隔条的距离”。
#
# d) Placement legality
#    生成器显式采样 left_width/right_width/mid_width，并只在外侧放条、中间清空输入，因此：
#    - 两条分隔条不会重叠且同色；
#    - 左右外侧各自有足够宽度容纳 1-2 条相邻色条与可选 1 格分隔条间隙；
#    - 中间区至少保留 5 列，保证双向打点和交汇可见；
#    - 分隔条缺口只允许出现在非活动行，避免破坏活动行的几何线索。
#
# e) Visual observability
#    解题者能直接看见两条同色竖线、左右外侧的水平色条，以及输出中只在中间区新增的双向点列。
#    至少一条行会同时出现左右来源的点，且至少一处共享目标格由更近的来源抢占，因此“近条优先”
#    仍能通过可见几何关系推断出来，不依赖颜色编号等不可见因素。
#
# f) Visual cleanliness
#    活动行之间仍保持至少 1 空行；中间区只承载一层最终着色，不叠加额外标记；同一行即使双侧都活跃，
#    也只靠左右外侧条和中间点列互动，结构清楚。
#
# g) Color inference
#    颜色没有重映射，点列直接继承各自色条颜色。为避免双侧同行太乱，生成器尽量让同行左右主色不同，
#    但规则本身不依赖具体颜色身份。
#
# h) Generator-side non-degeneracy checks
#    除了验证 main(inp) == out，还要求：
#    - 左右两侧都有活动行；
#    - 至少一条行属于双侧行；
#    - 至少一条双侧行在最终输出里两侧都保留了可见贡献；
#    - 至少一个中间格同时被左右声明，且左右最近声明距离不同。
#
# i) New solver assumptions guaranteed by generator
#    main() 现在依赖以下前提，generate_input() 都会构造保证：
#    - 恰有两条同色、近乎满高的竖分隔条；
#    - 输入中只有外侧会出现水平色条，中间区全空；
#    - 色条不跨越分隔条，且同一侧每行至多 1-2 条相邻 run；
#    - 非分隔条列不会伪装成“近乎满高的同色列”；
#    - 融合特征不是偶发，而是有统计上的强制出现。


def _find_sep_cols(g: np.ndarray, bg: int) -> tuple[int, int, int] | tuple[None, None, None]:
    h, w = g.shape
    candidates: list[tuple[int, int]] = []
    for c in range(w):
        col = g[:, c]
        vals = col[col != bg]
        if vals.size >= h - 1 and (vals == vals[0]).all():
            candidates.append((c, int(vals[0])))

    best: tuple[int, int, int] | None = None
    for i, (c0, color0) in enumerate(candidates):
        for c1, color1 in candidates[i + 1 :]:
            if color0 != color1:
                continue
            if best is None or (c1 - c0) > (best[1] - best[0]):
                best = (c0, c1, color0)
    if best is None:
        return None, None, None
    return best


def _iter_runs_1d(row: np.ndarray, c_lo: int, c_hi: int, bg: int):
    c = c_lo
    while c < c_hi:
        if int(row[c]) == bg:
            c += 1
            continue
        color = int(row[c])
        c0 = c
        while c < c_hi and int(row[c]) == color:
            c += 1
        yield c0, c - 1, color


def _collect_ops(
    g: np.ndarray,
    left_sep: int,
    right_sep: int,
    bg: int,
) -> list[tuple[int, int, int, int, int, int]]:
    ops: list[tuple[int, int, int, int, int, int]] = []
    h, w = g.shape
    for r in range(h):
        for c0, c1, color in _iter_runs_1d(g[r, :], 0, left_sep, bg):
            length = c1 - c0 + 1
            dist = (left_sep - 1) - c1
            ops.append((dist, 0, r, c0, length, color))
        for c0, c1, color in _iter_runs_1d(g[r, :], right_sep + 1, w, bg):
            length = c1 - c0 + 1
            dist = c0 - (right_sep + 1)
            ops.append((dist, 1, r, c0, length, color))
    return ops


def _positions_for_op(left_sep: int, right_sep: int, side: int, length: int) -> list[int]:
    if side == 0:
        return list(range(left_sep + 1, right_sep, length))
    return list(range(right_sep - 1, left_sep, -length))


def _paint_middle_from_op(
    out: np.ndarray,
    row: int,
    left_sep: int,
    right_sep: int,
    side: int,
    length: int,
    color: int,
    bg: int,
) -> int:
    hits = 0
    for cc in _positions_for_op(left_sep, right_sep, side, length):
        if int(out[row, cc]) == bg:
            out[row, cc] = color
            hits += 1
    return hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overview:
        Detect two same-colored vertical separators, read bars from both outer regions, and
        extend them into the middle region with the original periodic no-overwrite rule.

    Input structure:
        The grid contains two nearly solid same-color separator columns. Bars appear only in
        the outer left and/or outer right regions. The region between the separators is empty
        in the input and is the only area that can be painted.

    Rule:
        Every monochrome horizontal bar emits periodic dots toward the middle: left-side bars
        paint rightward from the left separator, right-side bars paint leftward from the right
        separator, with step size equal to the bar length. All bars are processed globally from
        nearest to farthest relative to their own separator; existing non-background cells are
        never overwritten.

    Key insight:
        This is the original row-wise periodic extension rule applied symmetrically from both
        sides of a double-separator layout, so the only real change is that the middle region
        can receive competing claims from left and right bars on the same row.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    left_sep, right_sep, _sep_color = _find_sep_cols(g, bg)
    if left_sep is None or right_sep is None:
        return g.copy()

    out = g.copy()
    ops = _collect_ops(g, left_sep, right_sep, bg)
    ops.sort(key=lambda t: (t[0], t[1], t[2], t[3], t[4], t[5]))

    for _dist, side, row, _c0, length, color in ops:
        if length <= 0:
            continue
        _paint_middle_from_op(out, row, left_sep, right_sep, side, length, color, bg)

    return out


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

SIDE_MIN, SIDE_MAX = 3, 5
MID_MIN, MID_MAX = 5, 10

PALETTE_MIN, PALETTE_MAX = 5, 8
ACTIVE_MIN, ACTIVE_MAX = 3, 5

P_TWO_BARS = 0.45
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5
P_SEP_COLUMN_GAP = 0.15

MAX_ATTEMPTS = 500


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    bar_pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(bar_pool)) + 1))
    palette = list(map(int, rng.choice(bar_pool, size=size, replace=False)))
    return sep_color, palette


def _choose_primary(rng: np.random.Generator, palette: list[int], exclude: set[int]) -> int:
    choices = [c for c in palette if c not in exclude]
    if not choices:
        choices = palette
    return int(rng.choice(choices))


def _sample_row_bars(
    rng: np.random.Generator,
    palette: list[int],
    primary: int,
    side_len: int,
) -> tuple[list[tuple[int, int]], int]:
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = side_len - sep_gap
    if capacity <= 0:
        raise ValueError("invalid side capacity")

    two_bars = capacity >= 2 and rng.random() < P_TWO_BARS
    if not two_bars:
        length = int(rng.integers(1, capacity + 1))
        return [(length, primary)], sep_gap

    near_len = int(rng.integers(1, capacity))
    far_len = int(rng.integers(1, capacity - near_len + 1))
    others = [c for c in palette if c != primary]
    colors = [primary, int(rng.choice(others))]
    if rng.random() < P_SWAP_COLORS:
        colors.reverse()
    return [(near_len, colors[0]), (far_len, colors[1])], sep_gap


def _place_side_bars(
    out: np.ndarray,
    row: int,
    sep_col: int,
    side: int,
    bars: list[tuple[int, int]],
    sep_gap: int,
) -> list[tuple[int, int, int, int, int]]:
    ops: list[tuple[int, int, int, int, int]] = []
    if side == 0:
        cursor = sep_col - sep_gap
        for length, color in bars:
            c0 = cursor - length
            c1 = cursor - 1
            out[row, c0 : c1 + 1] = color
            dist = (sep_col - 1) - c1
            ops.append((dist, side, row, length, color))
            cursor -= length
    else:
        cursor = sep_col + 1 + sep_gap
        for length, color in bars:
            c0 = cursor
            c1 = cursor + length - 1
            out[row, c0 : c1 + 1] = color
            dist = c0 - (sep_col + 1)
            ops.append((dist, side, row, length, color))
            cursor += length
    return ops


def _choose_row_states(rng: np.random.Generator, count: int) -> list[str]:
    states = ["B", "L", "R"]
    while len(states) < count:
        states.append(str(rng.choice(["B", "L", "R"])))
    rng.shuffle(states)
    return states


def _claim_maps(
    ops: list[tuple[int, int, int, int, int]],
    left_sep: int,
    right_sep: int,
) -> dict[int, dict[int, list[tuple[int, int, int]]]]:
    claims: dict[int, dict[int, list[tuple[int, int, int]]]] = {}
    for dist, side, row, length, color in ops:
        row_claims = claims.setdefault(row, {})
        for cc in _positions_for_op(left_sep, right_sep, side, length):
            row_claims.setdefault(cc, []).append((side, dist, color))
    return claims


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(H_MIN, H_MAX + 1))
    w = int(rng.integers(W_MIN, W_MAX + 1))

    feasible: list[tuple[int, int, int]] = []
    for left_width in range(SIDE_MIN, SIDE_MAX + 1):
        for right_width in range(SIDE_MIN, SIDE_MAX + 1):
            mid_width = w - left_width - right_width - 2
            if MID_MIN <= mid_width <= MID_MAX:
                feasible.append((left_width, right_width, mid_width))
    if not feasible:
        return None

    left_width, right_width, _mid_width = feasible[int(rng.integers(0, len(feasible)))]
    left_sep = left_width
    right_sep = w - right_width - 1

    sep_color, palette = _sample_palette(rng)

    max_active = min(ACTIVE_MAX, (h + 1) // 2)
    if ACTIVE_MIN > max_active:
        return None
    active_count = int(rng.integers(ACTIVE_MIN, max_active + 1))
    active_rows = _choose_active_rows(rng, h, active_count)
    active_row_set = set(active_rows)
    inactive_rows = [r for r in range(h) if r not in active_row_set]
    row_states = _choose_row_states(rng, active_count)

    out = np.zeros((h, w), dtype=int)
    out[:, left_sep] = sep_color
    out[:, right_sep] = sep_color

    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        out[gap_row, left_sep] = 0
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        out[gap_row, right_sep] = 0

    all_ops: list[tuple[int, int, int, int, int]] = []
    side_active = {0: 0, 1: 0}
    both_rows: list[int] = []

    for row, state in zip(active_rows, row_states):
        used_colors: set[int] = set()
        if state in ("L", "B"):
            primary = _choose_primary(rng, palette, used_colors)
            used_colors.add(primary)
            bars, sep_gap = _sample_row_bars(rng, palette, primary, left_width)
            all_ops.extend(_place_side_bars(out, row, left_sep, 0, bars, sep_gap))
            side_active[0] += 1
        if state in ("R", "B"):
            primary = _choose_primary(rng, palette, used_colors)
            used_colors.add(primary)
            bars, sep_gap = _sample_row_bars(rng, palette, primary, right_width)
            all_ops.extend(_place_side_bars(out, row, right_sep, 1, bars, sep_gap))
            side_active[1] += 1
        if state == "B":
            both_rows.append(row)

    all_ops.sort(key=lambda t: (t[0], t[1], t[2], t[3], t[4]))

    visible_hits: dict[int, dict[int, int]] = {row: {0: 0, 1: 0} for row in both_rows}
    for dist, side, row, length, color in all_ops:
        hits = _paint_middle_from_op(out, row, left_sep, right_sep, side, length, color, 0)
        if row in visible_hits:
            visible_hits[row][side] += hits

    claims = _claim_maps(all_ops, left_sep, right_sep)
    shared_cells = 0
    decisive_shared_cells = 0
    for row in both_rows:
        for entries in claims.get(row, {}).values():
            sides = {side for side, _dist, _color in entries}
            if sides != {0, 1}:
                continue
            shared_cells += 1
            left_best = min(dist for side, dist, _color in entries if side == 0)
            right_best = min(dist for side, dist, _color in entries if side == 1)
            if left_best != right_best:
                decisive_shared_cells += 1

    visible_both_rows = sum(
        1
        for row in both_rows
        if visible_hits[row][0] > 0 and visible_hits[row][1] > 0
    )

    inp = out.copy()
    inp[:, left_sep + 1 : right_sep] = 0

    if side_active[0] == 0 or side_active[1] == 0:
        return None
    if not both_rows:
        return None
    if visible_both_rows == 0:
        return None
    if shared_cells == 0 or decisive_shared_cells == 0:
        return None
    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
