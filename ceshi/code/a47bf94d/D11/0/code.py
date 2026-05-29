# Variant of a47bf94d
# Dimension: D11 — 通道数量与空间布局 (meta)
# Change: 4-5个通道，均匀列间距4-5，顶部和底部放置token行，管道竖直为主加横向交叉段... → 左右布局
#
# Design analysis for the D11 variant
# -----------------------------------
# source_task: None
#
# variant_spec:
#   core_change:
#     original_layout: token 位于顶部/底部，两端由纵向主干和若干横向 crossover 连接
#     new_layout: token 改为位于左/右两侧，两端由横向主干和若干纵向 crossover 连接
#   pixel_realization:
#     channels:
#       - 4 或 5 条通道
#       - 每条通道对应一条主水平线，中心行均匀间距 4 或 5
#       - 另有一条 bypass 行，位于最下方通道下方 2 格，用于实现置换分解
#     tokens:
#       - 左侧 token 中心列固定为 2 或 3
#       - 右侧 token 中心列镜像放置
#       - token 中心与 wire endpoint 始终相距 2 格，方向为水平向外
#       - 形状与原题保持一致：solid 3x3、X、diamond
#     wires:
#       - 主干由颜色 8 构成，整体以水平段为主
#       - crossover 发生在若干列上，以竖线连接两条活动通道
#       - 颜色 5 仍作为 crossing marker，但现在放在竖向 crossover 上下，包夹“直行穿越”的水平线
#       - 颜色 9 矩形仍嵌入 wire 中，只是其常见覆盖方向从“竖线段优先”旋转为“横线段优先”
#   concrete_trace:
#     original_local_step:
#       - 某条通道从顶部 endpoint 往下走
#       - 在 crossover 行横向移到另一列
#       - 最后到达底部 endpoint，对应置换后的列
#     variant_local_step:
#       - 某条通道从左侧 endpoint 往右走
#       - 在 crossover 列竖向移到另一行
#       - 最后到达右侧 endpoint，对应置换后的行
#     key_difference:
#       - solver 的 token/endpoint 关系不依赖上下，只依赖“endpoint 外侧 2 格有 token”
#       - generator 里所有“列=通道、行=crossover”的几何假设都必须旋转 90 度
#   complexity_calibration:
#     - 保留 4-5 条通道和非恒等置换，避免样例退化成无交叉直线
#     - 保留 0-3 个 9-rect、0-2 个 X 输入，避免变体只剩简单旋转
#     - crossover 列至少间隔 2，通道行至少间隔 4，保证 5-marker 与 token 可分辨
#   implicit_assumption_audit_for_main:
#     - _find_solid_3x3 / _find_x_3x3: 只依赖局部 3x3 形状，不依赖布局方向；仍成立
#     - track_mask 去掉 token 单元: 只要求 token 与 wire 不重叠；generator 继续保证
#     - endpoint 检测: 只要求 token 中心在 endpoint 外侧 2 格，方向属于四联通之一；左右布局仍成立
#     - _trace_wire: 只依赖正交路径、5/8/9 轨道和 9-rect 面穿越规则；水平主干同样成立
#     - 在 solid/X 分支中写入远端形状时，默认目标 3x3 区域为空或仅含互补预置 diamond；generator 保证不发生错误覆盖
#   placement_legality:
#     - 通道行由 top_margin + k * spacing 生成，bypass 行与最后一条通道相距 2，避免 5-marker 覆盖 bypass
#     - 左右 token 中心列选 2/3，保证 3x3 token 不越界
#     - crossover 列位于左右 endpoint 之间，并按 2 列最小步长排布
#     - 9-rect 仅在已有 wire/crossing/bg 上放置，并检查左右贯通入口；与已放置 9-rect 保持 1 格间距
#   visual_observability:
#     - 人眼能直接看到 token 从“左右两侧”而非“上下两侧”被连接
#     - 输出固定为左侧 diamond、右侧 X，配合横向主干，左右追踪关系清晰
#     - 颜色不参与排序或几何决策，只用于保持 token 身份
#   cleanliness:
#     - token 只在两侧边带出现
#     - wire 只在中部走廊出现
#     - 5 和 9 仍是局部标记，不与 token 大面积重叠
#   generator_guarantees_for_new_main_assumptions:
#     - 每条通道左右各有一个度为 1 的 endpoint
#     - 每个 token 中心恰好位于某个 endpoint 外侧 2 格
#     - 非恒等置换保证至少一个竖向 crossover，样本不会退化回纯平行直线

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]


def _find_solid_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for maximal 3x3 solid blocks."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            col = int(sub[0, 0])
            if col == bg or np.any(sub != col):
                continue
            maximal = True
            for rr in range(r0, r0 + 3):
                for cc in range(c0, c0 + 3):
                    for nr, nc in neighbors4(rr, cc):
                        if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                            if int(g[nr, nc]) == col:
                                maximal = False
                                break
                    if not maximal:
                        break
                if not maximal:
                    break
            if not maximal:
                continue
            found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _find_x_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for isolated 3x3 'X' patterns."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}

    mask = np.zeros((3, 3), dtype=bool)
    for rr, cc in [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]:
        mask[rr, cc] = True

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            col = int(sub[0, 0])
            if col == bg:
                continue
            if np.any(sub[mask] != col) or np.any(sub[~mask] != bg):
                continue
            isolated = True
            for dr0, dc0 in _X_OFF:
                rr, cc = r0 + 1 + dr0, c0 + 1 + dc0
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                        if int(g[nr, nc]) == col:
                            isolated = False
                            break
                if not isolated:
                    break
            if not isolated:
                continue
            found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=4)]


def _find_9rects(g: np.ndarray) -> list[tuple[set, dict]]:
    """Find color-9 rectangular blocks and their wire connections per face."""
    mask9 = g == 9
    if not mask9.any():
        return []
    h, w = g.shape
    rects = []
    for comp9 in components_mask(mask9, connectivity=4):
        cells = set((int(r), int(c)) for r, c in comp9)
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        faces: dict[str, list] = {}
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) not in cells and 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) in (5, 8):
                    face = "N" if nr < r0 else "S" if nr > r1 else "W" if nc < c0 else "E"
                    faces.setdefault(face, []).append((nr, nc))
        rects.append((cells, faces))
    return rects


_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIR2FACE = {(1, 0): "N", (-1, 0): "S", (0, 1): "W", (0, -1): "E"}
_FACE2DIR = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}


def _trace_wire(
    comp: set[tuple[int, int]],
    start: tuple[int, int],
    dir0: tuple[int, int],
    h: int,
    w: int,
    rects_9: list[tuple[set, dict]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Greedy wire following with 9-rect pass-through.

    At a color-9 rectangle: exit from opposite face (aligned with entry column/row).
    If no opposite exit, assert exactly one other exit face and use that.
    """
    cur, d, prev = start, dir0, None
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        if (cur, d) in visited:
            assert False, "wire trace looped — should not happen with 9-rect pass-through"
        visited.add((cur, d))

        jumped = False
        for cells9, faces9 in rects_9:
            if cur in cells9:
                entry_face = _DIR2FACE[d]
                opp = _OPP[entry_face]
                if opp in faces9:
                    exit_face = opp
                else:
                    others = [f for f in faces9 if f != entry_face]
                    assert len(others) == 1
                    exit_face = others[0]
                pts = faces9[exit_face]
                er, ec = cur
                if entry_face in ("N", "S"):
                    exit_cell = min(pts, key=lambda p: abs(p[1] - ec))
                else:
                    exit_cell = min(pts, key=lambda p: abs(p[0] - er))
                cur, d, prev = exit_cell, _FACE2DIR[exit_face], None
                jumped = True
                break
        if jumped:
            continue

        r, c = cur
        opts = [((nr, nc), nr - r, nc - c) for nr, nc in neighbors4(r, c) if (nr, nc) in comp]
        if not opts:
            return cur, d

        rev = (-d[0], -d[1])
        nonrev = [o for o in opts if (o[1], o[2]) != rev]
        opts2 = nonrev if nonrev else opts

        straight = [o for o in opts2 if (o[1], o[2]) == d]
        if straight:
            p, dr, dc = straight[0]
        else:
            perp = [o for o in opts2 if o[1] * d[0] + o[2] * d[1] == 0]
            if perp:
                p, dr, dc = sorted(perp, key=lambda x: (x[1], x[2]))[0]
            else:
                p, dr, dc = opts2[0]

        if p == prev:
            return cur, d

        prev = cur
        cur = p
        d = (dr, dc)

    assert False, "wire trace exceeded step limit"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move colored side tokens through a left-to-right wire layout, replacing each with its complementary 3x3 form.

    Input structure:
    - The background is the most common color. Non-background cells form a horizontally dominated wire network plus 3x3 tokens placed two cells outside left/right endpoints.

    Rule:
    - Any solid 3x3 token collapses in place into a diamond (the four cardinal neighbors around its center).
    - Every token attached to an endpoint sends its complementary pattern across the wire to the far endpoint: solid produces X, while X produces diamond.

    Key insight:
    - The important relation is geometric rather than top-vs-bottom: a token center sits exactly two cells outside an endpoint, and the wire transports the next shape in the cycle to the opposite side.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)

    track_mask = g != bg
    for (_, _), (_, (r0, c0)) in solids.items():
        track_mask[r0 : r0 + 3, c0 : c0 + 3] = False
    for cr, cc in xs.keys():
        for dr, dc in _X_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                track_mask[rr, cc2] = False

    comps = _components(track_mask)
    rects_9 = _find_9rects(g)

    out = g.copy()

    for (cr, cc), (col, (r0, c0)) in solids.items():
        out[r0 : r0 + 3, c0 : c0 + 3] = bg
        for dr, dc in _DIAMOND_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                out[rr, cc2] = col

    for pts in comps:
        comp = set(pts)
        if len(comp) < 4:
            continue

        comp_colors = {int(g[r, c]) for r, c in comp}

        deg = {p: 0 for p in comp}
        for r, c in comp:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in comp:
                    deg[(r, c)] += 1
        endpoints = [p for p in comp if deg[p] <= 1]
        if not endpoints:
            continue

        for ep in endpoints:
            er, ec = ep
            for odr, odc in _N4:
                cen = (er + 2 * odr, ec + 2 * odc)
                if not (0 <= cen[0] < h and 0 <= cen[1] < w):
                    continue

                if cen in solids:
                    col, _ = solids[cen]
                    if col in comp_colors:
                        continue
                    dir_in = (-odr, -odc)
                    end, dd = _trace_wire(comp, ep, dir_in, h, w, rects_9)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _X_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

                elif cen in xs:
                    col, _ = xs[cen]
                    if col in comp_colors:
                        continue
                    dir_in = (-odr, -odc)
                    end, dd = _trace_wire(comp, ep, dir_in, h, w, rects_9)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _DIAMOND_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

    return out


# --------------- generator (left_right_refine_9rect_xtoken) ---------------

_DIAMOND = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

CHANNEL_COUNTS = [4, 5]
CHANNEL_SPACING = [4, 5]
TOP_MARGIN_RANGE = (1, 2)
BYPASS_ROW_OFFSET = 2
TOKEN_COL_OPTIONS = [2, 3]
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
GRID_H_MIN = 20
GRID_H_MAX = 26
GRID_W_MIN = 22
GRID_W_MAX = 26
EP_TOKEN_GAP = 2

RECT_COUNT_WEIGHTS = [10, 40, 30, 20]
RECT_WIRE_COUNT_WEIGHTS = [50, 38, 12]
RECT_PARALLEL_PROB = 0.75
RECT_W_RANGE = (2, 5)
RECT_MAX_HEIGHT = 7
RECT_RETRY_FACTOR = 5
RECT_MARGIN_GAP = 1

X_INPUT_COUNT_WEIGHTS = [60, 30, 10]
FAR_DIAMOND_PROB = 0.33

GEN_MAX_RETRIES = 300


def _decompose_to_crossovers(perm):
    """Return a sequence of row-to-row moves, using -1 as the bypass row."""
    n = len(perm)
    visited = [False] * n
    crossovers = []
    for start in range(n):
        if visited[start] or perm[start] == start:
            visited[start] = True
            continue
        cycle = []
        i = start
        while not visited[i]:
            visited[i] = True
            cycle.append(i)
            i = perm[i]
        if len(cycle) < 2:
            continue
        crossovers.append((cycle[0], -1))
        for k in range(len(cycle) - 1, 0, -1):
            crossovers.append((cycle[k], cycle[(k + 1) % len(cycle)]))
        crossovers.append((-1, cycle[1]))
    return crossovers


def _stamp_token(grid, center_r, center_c, color, offsets):
    """Write a token pattern onto the grid."""
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def _try_place_9rect(grid, ch_rows, xover_cols, left_ep_col, right_ep_col, placed_rects):
    """Attempt to place a color-9 rectangle aligned to the left-right wiring."""
    h, w = grid.shape
    target_wires = random.choices([1, 2, 3], weights=RECT_WIRE_COUNT_WEIGHTS)[0]

    if target_wires >= 2 and random.random() < RECT_PARALLEL_PROB and len(ch_rows) >= 2:
        mode = "parallel"
    elif target_wires >= 2 and xover_cols:
        mode = "crossing"
    else:
        mode = "single"

    if mode == "parallel":
        j0 = random.randint(0, len(ch_rows) - 2)
        r_lo, r_hi = ch_rows[j0], ch_rows[j0 + 1]
        margin_t, margin_b = random.randint(0, 1), random.randint(0, 1)
    else:
        j0 = random.randint(0, len(ch_rows) - 1)
        r_lo = r_hi = ch_rows[j0]
        margin_t, margin_b = random.randint(0, 2), random.randint(0, 2)
        if margin_t + margin_b < 1:
            if random.random() < 0.5:
                margin_t = 1
            else:
                margin_b = 1
    rect_r0, rect_r1 = r_lo - margin_t, r_hi + margin_b

    if rect_r1 - rect_r0 + 1 > RECT_MAX_HEIGHT or rect_r0 < 0 or rect_r1 >= h:
        return None

    max_w = min(RECT_W_RANGE[1], right_ep_col - left_ep_col - 3)
    if max_w < RECT_W_RANGE[0]:
        return None
    rect_w = random.randint(RECT_W_RANGE[0], max_w)
    if mode == "crossing" and xover_cols:
        xc = random.choice(xover_cols)
        c_start = xc - random.randint(0, rect_w - 1)
    else:
        c_start = random.randint(left_ep_col + 2, right_ep_col - rect_w - 1)
    rect_c0, rect_c1 = c_start, c_start + rect_w - 1

    if rect_c0 <= left_ep_col or rect_c1 >= right_ep_col:
        return None

    wire_rows = sorted(r for r in ch_rows if rect_r0 <= r <= rect_r1)
    if not wire_rows:
        return None
    for r in wire_rows:
        if not np.all(grid[r, rect_c0 : rect_c1 + 1] == 8):
            return None

    sub = grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1]
    if not np.all((sub == 0) | (sub == 5) | (sub == 8)):
        return None

    gap = RECT_MARGIN_GAP
    for pr0, pc0, pr1, pc1 in placed_rects:
        if rect_r0 <= pr1 + gap and pr0 <= rect_r1 + gap and rect_c0 <= pc1 + gap and pc0 <= rect_c1 + gap:
            return None

    for r in wire_rows:
        if grid[r, rect_c0 - 1] != 8 or grid[r, rect_c1 + 1] != 8:
            return None

    saved = grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1].copy()
    grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] = 9

    n_cols, s_cols, w_rows, e_rows = set(), set(), set(), set()
    for c in range(rect_c0, rect_c1 + 1):
        if rect_r0 - 1 >= 0 and grid[rect_r0 - 1, c] in (5, 8):
            n_cols.add(c)
        if rect_r1 + 1 < h and grid[rect_r1 + 1, c] in (5, 8):
            s_cols.add(c)
    for r in range(rect_r0, rect_r1 + 1):
        if rect_c0 - 1 >= 0 and grid[r, rect_c0 - 1] in (5, 8):
            w_rows.add(r)
        if rect_c1 + 1 < w and grid[r, rect_c1 + 1] in (5, 8):
            e_rows.add(r)
    total_wires = len(w_rows & e_rows) + len(n_cols & s_cols)

    if total_wires < 1 or total_wires > 3:
        grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] = saved
        return None
    return (rect_r0, rect_c0, rect_r1, rect_c1)


def _build_wire_network(grid, ch_rows, bypass_row, left_ep_col, right_ep_col, crossovers, xover_cols):
    """Draw a left-to-right wire network. Returns (slot_wire, bypass_wire)."""
    slot_wire = list(range(len(ch_rows)))
    bypass_wire = None

    row_seg_start = {}
    for row in ch_rows:
        row_seg_start[row] = left_ep_col
        grid[row, left_ep_col] = 8

    for idx, (src_slot, dst_slot) in enumerate(crossovers):
        xcol = xover_cols[idx]

        if src_slot == -1:
            src_row, dst_row = bypass_row, ch_rows[dst_slot]
        elif dst_slot == -1:
            src_row, dst_row = ch_rows[src_slot], bypass_row
        else:
            src_row, dst_row = ch_rows[src_slot], ch_rows[dst_slot]

        active_rows = {ch_rows[j] for j in range(len(ch_rows)) if slot_wire[j] is not None}
        if bypass_wire is not None:
            active_rows.add(bypass_row)

        for r in active_rows:
            if r in row_seg_start:
                start = row_seg_start[r] + 1
                if start <= xcol - 1:
                    grid[r, start:xcol] = 8

        r_min, r_max = min(src_row, dst_row), max(src_row, dst_row)
        grid[r_min : r_max + 1, xcol] = 8

        for r in active_rows:
            grid[r, xcol] = 8

        for r in active_rows:
            if r == src_row or r == dst_row:
                continue
            if r_min < r < r_max:
                if r - 1 >= r_min:
                    grid[r - 1, xcol] = 5
                if r + 1 <= r_max:
                    grid[r + 1, xcol] = 5

        new_seg_start = {r: xcol for r in active_rows if r != src_row}
        new_seg_start[dst_row] = xcol
        row_seg_start = new_seg_start

        if src_slot == -1:
            slot_wire[dst_slot] = bypass_wire
            bypass_wire = None
        elif dst_slot == -1:
            bypass_wire = slot_wire[src_slot]
            slot_wire[src_slot] = None
        else:
            slot_wire[dst_slot] = slot_wire[src_slot]
            slot_wire[src_slot] = None

    for j, row in enumerate(ch_rows):
        if slot_wire[j] is not None and row in row_seg_start:
            start = row_seg_start[row] + 1
            if start <= right_ep_col:
                grid[row, start : right_ep_col + 1] = 8

    return slot_wire, bypass_wire


def _derive_input(grid, ch_rows, perm, token_colors, left_token_col, right_token_col):
    """Place input tokens around the completed left-right wire network."""
    inp = grid.copy()
    n = len(ch_rows)
    n_x = random.choices([0, 1, 2], weights=X_INPUT_COUNT_WEIGHTS)[0]
    x_channels = set(random.sample(range(n), min(n_x, n)))

    for i in range(n):
        col = token_colors[i]
        left_center = (ch_rows[i], left_token_col)
        right_center = (ch_rows[perm[i]], right_token_col)

        if i in x_channels:
            if random.choice([True, False]):
                near, far = left_center, right_center
            else:
                near, far = right_center, left_center
            _stamp_token(inp, near[0], near[1], col, _X)
            if random.random() < FAR_DIAMOND_PROB:
                _stamp_token(inp, far[0], far[1], col, _DIAMOND)
        else:
            side = random.choice([left_center, right_center])
            _stamp_token(inp, side[0], side[1], col, _SOLID)
    return inp


def _gen_attempt():
    n = random.choice(CHANNEL_COUNTS)
    spacing = random.choice(CHANNEL_SPACING)

    top_margin = random.randint(*TOP_MARGIN_RANGE)
    first_row = top_margin + 1
    ch_rows = [first_row + i * spacing for i in range(n)]
    bypass_row = ch_rows[-1] + BYPASS_ROW_OFFSET

    h = max(GRID_H_MIN, bypass_row + 1)
    if h > GRID_H_MAX:
        return None

    left_token_col = random.choice(TOKEN_COL_OPTIONS)
    left_ep_col = left_token_col + EP_TOKEN_GAP
    token_colors = random.sample(TOKEN_COLORS_POOL, n)

    while True:
        perm = list(range(n))
        random.shuffle(perm)
        if perm != list(range(n)):
            break

    crossovers = _decompose_to_crossovers(perm)
    n_xovers = len(crossovers)
    if n_xovers == 0:
        return None

    min_wire_cols = n_xovers * 2 + 2
    w_min = max(GRID_W_MIN, 2 * left_token_col + 5 + min_wire_cols)
    if w_min > GRID_W_MAX:
        return None
    w = random.randint(w_min, GRID_W_MAX)

    right_token_col = w - left_token_col - 1
    right_ep_col = right_token_col - EP_TOKEN_GAP
    if right_ep_col - left_ep_col < min_wire_cols:
        return None

    avail = right_ep_col - left_ep_col - 2
    step = max(2, avail // (n_xovers + 1))
    xover_cols = [left_ep_col + 1 + step * (s + 1) for s in range(n_xovers)]
    if xover_cols[-1] >= right_ep_col:
        xover_cols = [left_ep_col + 2 + s * 2 for s in range(n_xovers)]
    if xover_cols[-1] >= right_ep_col:
        return None

    grid = np.zeros((h, w), dtype=int)
    slot_wire, bypass_wire = _build_wire_network(
        grid, ch_rows, bypass_row, left_ep_col, right_ep_col, crossovers, xover_cols
    )

    inv_perm = [0] * n
    for i in range(n):
        inv_perm[perm[i]] = i
    if any(slot_wire[j] != inv_perm[j] for j in range(n)) or bypass_wire is not None:
        return None

    n_rects = random.choices(range(4), weights=RECT_COUNT_WEIGHTS)[0]
    placed_rects = []
    for _ in range(n_rects * RECT_RETRY_FACTOR):
        if len(placed_rects) >= n_rects:
            break
        rect = _try_place_9rect(grid, ch_rows, xover_cols, left_ep_col, right_ep_col, placed_rects)
        if rect is not None:
            placed_rects.append(rect)

    output = grid.copy()
    for i in range(n):
        _stamp_token(output, ch_rows[i], left_token_col, token_colors[i], _DIAMOND)
        _stamp_token(output, ch_rows[perm[i]], right_token_col, token_colors[i], _X)

    inp = _derive_input(grid, ch_rows, perm, token_colors, left_token_col, right_token_col)
    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input():
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
