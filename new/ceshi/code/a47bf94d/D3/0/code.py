# Variant of a47bf94d
# Dimension: D3 — 传输轨迹可视化 (trajectory)
# Change: 无轨迹涂色——管道网络在输入和输出中保持不变，不因token传输而涂色或改变... → 连续轨迹涂色

# Design analysis
# spec:
#   core_change:
#     old: 管道只负责连通与路由，输出中仍保持 5/8/9 的固定配色。
#     new: token 在传输时把经过的管道像素连续染成自己的颜色；最终输出中的管道变成多色轨迹图。
#   concrete_behavior:
#     objects:
#       tokens:
#         solid_3x3: 发送后原地变 diamond
#         x_3x3: 原地保留 X，同时向远端发送 diamond
#         diamond: 仍只是静止终态，不作为新输入 token 识别
#       wires:
#         8: 普通管道段
#         5: crossing 标记，仍参与路径，被经过时也会被染色
#         9: 矩形通道块；变体中不再始终保留 9，而是按 token 在块内的实际穿越条带被染色
#     visible_effect:
#       input: 管道仍是青色/灰色/栗色编码，和原题一样
#       output: 除 token 形状变化外，所有被 token 走过的线路都变成 token 本色，因此能直接从彩色轨迹看出哪一个 token 走到了哪一端
#   behavior_trace_example:
#     original_step:
#       - 顶部某列有一个红色 solid，邻接 endpoint
#       - solver 找到同一 connected wire component，沿 8/5/9 追到另一端
#       - 输出只在起点把 solid 变 diamond，并在远端盖一个红色 X；中间 wire 不变
#     variant_step:
#       - 同样找到红色 solid 和同一条路径
#       - 除上述 token 变换外，还把 endpoint、拐弯处、crossing 的 5、以及 9-rect 中对应穿越条带，全部重涂成红色
#       - 差异只发生在“route application”环节；token 检测、wire 拓扑、远端补全规则不变
#   complexity_calibration:
#     - 不引入新的 token 类别、排序规则或颜色映射，避免超过 D3 所需复杂度
#     - 轨迹颜色直接继承 token 自身颜色，解题者可用“同色 token + 同色 trail + 同色远端补块”做配对
#     - 保留 4-5 个通道、非恒等置换、0-3 个 9-rect，使样例中有重复证据支持“彩色轨迹就是路由”
#   implicit_assumption_audit:
#     - 原 `_trace_wire()` 只需返回终点；D3 下必须返回完整路径，否则无法给中间管道着色
#     - 原来 route 顺序无关；现在 crossing 共享像素可能被多个 token 覆盖，所以必须固定遍历顺序，避免输出不稳定
#     - 原来进入 9-rect 后可直接跳到出口；现在必须显式补出矩形内部的直线条带，否则“连续轨迹”在 9 区域会断裂
#     - 原来 `out = g.copy()` 保持 wire 颜色无害；现在后续 route 会覆盖 wire，因此必须只覆盖轨迹格，不碰 token 格
#   placement_legality:
#     - 仍沿用原布局：token 中心与 endpoint 相距 2 格，因此轨迹只会染到 wire，不会与 token 图案重叠
#     - 9-rect 仍只放在合法 wire 区域；新增的矩形内条带着色只发生在 solver 输出，不影响输入可解析性
#     - generator 继续用 oracle 验证，并额外要求输出中必须存在足量染色 wire 像素，防止退化成原题
#   visual_observability:
#     - 颜色变化只依赖“哪一个 token 走过这里”，不依赖颜色 id 的数值大小或隐藏排序
#     - 输出相对输入的最大新增信息就是彩色管道，2-3 组样例足以看出 token 会沿线留下同色路径
#   cleanliness:
#     - 不叠加额外符号；仍只用 ARC 原色
#     - 共享 crossing 像素若被后处理 token 覆盖，数量很少，主干轨迹仍大面积可见
#   generator_guarantees:
#     - 非恒等置换保留，确保存在真实传输
#     - 每个样本要求 `main(input)` 与“无轨迹版本 plain output”在 wire 区上有明显差异
#     - 额外检查染色 wire 像素数与染色颜色数，确保 D3 控制变量确实生效

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4
import random

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
    comps = [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=4)]
    return sorted(comps, key=lambda pts: min(pts))


def _find_9rects(g: np.ndarray) -> list[dict]:
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
        faces: dict[str, list[tuple[int, int]]] = {}
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) not in cells and 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) in (5, 8):
                    face = "N" if nr < r0 else "S" if nr > r1 else "W" if nc < c0 else "E"
                    faces.setdefault(face, []).append((nr, nc))
        rects.append({"cells": cells, "faces": faces, "bbox": (r0, c0, r1, c1)})
    return rects


_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIR2FACE = {(1, 0): "N", (-1, 0): "S", (0, 1): "W", (0, -1): "E"}
_FACE2DIR = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    """Inclusive axis-aligned segment between two cells."""
    if a == b:
        return [a]
    if a[0] == b[0]:
        step = 1 if b[1] > a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] > a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    raise AssertionError("9-rect trail must be axis-aligned")


def _trace_wire(
    comp: set[tuple[int, int]],
    start: tuple[int, int],
    dir0: tuple[int, int],
    h: int,
    w: int,
    rect_lookup: dict[tuple[int, int], dict],
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]:
    """
    Greedy wire following with 9-rect pass-through and explicit trail capture.

    At a color-9 rectangle: choose the same exit face as the original rule, but
    also fill the internal straight strip so the painted trajectory stays continuous.
    """
    cur, d, prev = start, dir0, None
    trail = [start]
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        state = (cur, d)
        if state in visited:
            assert False, "wire trace looped — should not happen with 9-rect pass-through"
        visited.add(state)

        rect = rect_lookup.get(cur)
        if rect is not None:
            faces = rect["faces"]
            entry_face = _DIR2FACE[d]
            opp = _OPP[entry_face]
            if opp in faces:
                exit_face = opp
            else:
                others = [f for f in faces if f != entry_face]
                assert len(others) == 1
                exit_face = others[0]
            pts = faces[exit_face]
            er, ec = cur
            if entry_face in ("N", "S"):
                exit_cell = min(pts, key=lambda p: (abs(p[1] - ec), abs(p[0] - er), p))
            else:
                exit_cell = min(pts, key=lambda p: (abs(p[0] - er), abs(p[1] - ec), p))
            exit_dir = _FACE2DIR[exit_face]
            inside_end = (exit_cell[0] - exit_dir[0], exit_cell[1] - exit_dir[1])
            rect_strip = _line_cells(cur, inside_end)
            trail.extend(rect_strip[1:])
            trail.append(exit_cell)
            cur, d, prev = exit_cell, exit_dir, None
            continue

        r, c = cur
        opts = [((nr, nc), nr - r, nc - c) for nr, nc in neighbors4(r, c) if (nr, nc) in comp]
        if not opts:
            return cur, d, trail

        rev = (-d[0], -d[1])
        nonrev = [o for o in opts if (o[1], o[2]) != rev]
        opts2 = nonrev if nonrev else opts

        straight = [o for o in opts2 if (o[1], o[2]) == d]
        if straight:
            p, dr, dc = straight[0]
        else:
            perp = [o for o in opts2 if o[1] * d[0] + o[2] * d[1] == 0]
            if perp:
                p, dr, dc = sorted(perp, key=lambda x: (x[1], x[2], x[0]))[0]
            else:
                p, dr, dc = sorted(opts2, key=lambda x: (x[1], x[2], x[0]))[0]

        if p == prev:
            return cur, d, trail

        prev = cur
        cur = p
        d = (dr, dc)
        trail.append(cur)

    assert False, "wire trace exceeded step limit"


def _paint_trail(out: np.ndarray, trail: list[tuple[int, int]], color: int, protected: set[tuple[int, int]]) -> None:
    """Recolor wire pixels along a traced route while keeping token cells intact."""
    for r, c in trail:
        if (r, c) not in protected:
            out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route endpoint tokens through pipe networks and display the traveled path in token color.

    Input structure:
    - Background cells surround one or more orthogonal wire components made of 8/5/9, plus 3×3 tokens near wire endpoints.
    - Tokens are either solid 3×3 blocks or sparse X patterns; any pre-existing diamonds are passive destination hints.

    Rule:
    - A solid token collapses in place into a diamond, while an X token stays as an X.
    - Each endpoint token sends its complementary 3×3 pattern to the far endpoint of the same wire route.
    - The full route it traverses is recolored to the token's color, including crossing markers and the traversed strip inside any 9-rectangle.

    Key insight:
    - The original endpoint-to-endpoint transmission still determines where complements appear, but now the route itself becomes visible: follow the same-colored trail to pair source token, pipe path, and far-end completion.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)

    track_mask = g != bg
    protected: set[tuple[int, int]] = set()
    for (_cr, _cc), (_col, (r0, c0)) in solids.items():
        track_mask[r0 : r0 + 3, c0 : c0 + 3] = False
        for rr in range(r0, r0 + 3):
            for cc in range(c0, c0 + 3):
                protected.add((rr, cc))
    for cr, cc in xs.keys():
        for dr, dc in _X_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                track_mask[rr, cc2] = False
                protected.add((rr, cc2))

    comps = _components(track_mask)
    rects_9 = _find_9rects(g)
    rect_lookup: dict[tuple[int, int], dict] = {}
    for rect in rects_9:
        for cell in rect["cells"]:
            rect_lookup[cell] = rect

    out = g.copy()

    for (_cr, _cc), (col, (r0, c0)) in sorted(solids.items()):
        out[r0 : r0 + 3, c0 : c0 + 3] = bg
        for dr, dc in _DIAMOND_OFF:
            rr, cc2 = r0 + 1 + dr, c0 + 1 + dc
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
        endpoints = sorted(p for p in comp if deg[p] <= 1)
        if not endpoints:
            continue

        route_jobs: list[tuple[tuple[int, int], int, tuple[int, int], tuple[int, int], str]] = []
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
                    route_jobs.append((cen, col, ep, (-odr, -odc), "solid"))
                elif cen in xs:
                    col, _ = xs[cen]
                    if col in comp_colors:
                        continue
                    route_jobs.append((cen, col, ep, (-odr, -odc), "x"))

        route_jobs.sort(key=lambda item: (item[0], item[1], item[2], item[4]))
        for cen, col, ep, dir_in, kind in route_jobs:
            end, dd, trail = _trace_wire(comp, ep, dir_in, h, w, rect_lookup)
            _paint_trail(out, trail, col, protected)
            dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
            if kind == "solid":
                for dr, dc in _X_OFF:
                    rr, cc2 = dest[0] + dr, dest[1] + dc
                    if 0 <= rr < h and 0 <= cc2 < w:
                        out[rr, cc2] = col
            else:
                for dr, dc in _DIAMOND_OFF:
                    rr, cc2 = dest[0] + dr, dest[1] + dc
                    if 0 <= rr < h and 0 <= cc2 < w:
                        out[rr, cc2] = col

    return out


# --------------- generator (refine_9rect_xtoken_trails) ---------------

_DIAMOND = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

CHANNEL_COUNTS = [4, 5]
CHANNEL_SPACING = [4, 5]
LEFT_MARGIN_RANGE = (2, 3)
BYPASS_COL_OFFSET = 2
TOKEN_ROW_OPTIONS = [2, 3]
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
GRID_H_RANGE = (20, 26)
GRID_W_MIN = 22
GRID_W_MAX = 26
EP_TOKEN_GAP = 2

RECT_COUNT_WEIGHTS = [10, 40, 30, 20]
RECT_WIRE_COUNT_WEIGHTS = [50, 38, 12]
RECT_PARALLEL_PROB = 0.75
RECT_H_RANGE = (2, 5)
RECT_MAX_WIDTH = 7
RECT_RETRY_FACTOR = 5
RECT_MARGIN_GAP = 1

X_INPUT_COUNT_WEIGHTS = [60, 30, 10]
FAR_DIAMOND_PROB = 0.33

MIN_TRAIL_PIXELS = 18
MIN_TRAIL_COLORS = 2
GEN_MAX_RETRIES = 300


def _decompose_to_crossovers(perm):
    """Decompose permutation into sequential crossover moves using bypass for cycles."""
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
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def _try_place_9rect(grid, ch_cols, xover_rows, top_ep_row, bot_ep_row, n, placed_rects):
    """Attempt to place one color-9 rectangular block. Returns (r0, c0, r1, c1) or None."""
    h, w = grid.shape
    target_wires = random.choices([1, 2, 3], weights=RECT_WIRE_COUNT_WEIGHTS)[0]

    if target_wires >= 2 and random.random() < RECT_PARALLEL_PROB and n >= 2:
        mode = "parallel"
    elif target_wires >= 2 and xover_rows:
        mode = "crossing"
    else:
        mode = "single"

    if mode == "parallel":
        j0 = random.randint(0, n - 2)
        c_lo, c_hi = ch_cols[j0], ch_cols[j0 + 1]
        margin_l, margin_r = random.randint(0, 1), random.randint(0, 1)
    else:
        j0 = random.randint(0, n - 1)
        c_lo = c_hi = ch_cols[j0]
        margin_l, margin_r = random.randint(0, 2), random.randint(0, 2)
        if margin_l + margin_r < 1:
            if random.random() < 0.5:
                margin_l = 1
            else:
                margin_r = 1
    rect_c0, rect_c1 = c_lo - margin_l, c_hi + margin_r

    if rect_c1 - rect_c0 + 1 > RECT_MAX_WIDTH or rect_c0 < 0 or rect_c1 >= w:
        return None

    rect_h = random.randint(RECT_H_RANGE[0], min(RECT_H_RANGE[1], bot_ep_row - top_ep_row - 4))
    if mode == "crossing" and xover_rows:
        xr = random.choice(xover_rows)
        r_start = xr - random.randint(0, rect_h - 1)
    else:
        r_start = random.randint(top_ep_row + 2, bot_ep_row - rect_h - 1)
    rect_r0, rect_r1 = r_start, r_start + rect_h - 1

    wire_cols = sorted(c for c in ch_cols if rect_c0 <= c <= rect_c1)
    if not wire_cols:
        return None
    for c in wire_cols:
        if not np.all(grid[rect_r0 : rect_r1 + 1, c] == 8):
            return None

    sub = grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1]
    if not np.all((sub == 0) | (sub == 5) | (sub == 8)):
        return None

    gap = RECT_MARGIN_GAP
    for pr0, pc0, pr1, pc1 in placed_rects:
        if rect_r0 <= pr1 + gap and pr0 <= rect_r1 + gap and rect_c0 <= pc1 + gap and pc0 <= rect_c1 + gap:
            return None

    if rect_r0 <= top_ep_row or rect_r1 >= bot_ep_row:
        return None
    for c in wire_cols:
        if grid[rect_r0 - 1, c] != 8 or grid[rect_r1 + 1, c] != 8:
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
    total_wires = len(n_cols & s_cols) + len(w_rows & e_rows)

    if total_wires < 1 or total_wires > 3:
        grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] = saved
        return None
    return (rect_r0, rect_c0, rect_r1, rect_c1)


def _build_wire_network(grid, n, ch_cols, bypass_col, top_ep_row, bot_ep_row, crossovers, xover_rows):
    """Draw the wire network onto grid. Returns (slot_wire, bypass_wire) after all crossovers."""
    slot_wire = list(range(n))
    bypass_wire = None

    col_seg_start = {}
    for j in range(n):
        col_seg_start[ch_cols[j]] = top_ep_row
        grid[top_ep_row, ch_cols[j]] = 8

    for xover_idx, (src_slot, dst_slot) in enumerate(crossovers):
        xrow = xover_rows[xover_idx]

        if src_slot == -1:
            src_col, dst_col = bypass_col, ch_cols[dst_slot]
        elif dst_slot == -1:
            src_col, dst_col = ch_cols[src_slot], bypass_col
        else:
            src_col, dst_col = ch_cols[src_slot], ch_cols[dst_slot]

        active_cols = {ch_cols[j] for j in range(n) if slot_wire[j] is not None}
        if bypass_wire is not None:
            active_cols.add(bypass_col)

        for c in active_cols:
            if c in col_seg_start:
                start = col_seg_start[c] + 1
                if start <= xrow - 1:
                    grid[start:xrow, c] = 8

        c_min, c_max = min(src_col, dst_col), max(src_col, dst_col)
        grid[xrow, c_min : c_max + 1] = 8

        for c in active_cols:
            grid[xrow, c] = 8

        for c in active_cols:
            if c == src_col or c == dst_col:
                continue
            if c_min < c < c_max:
                if c - 1 >= c_min:
                    grid[xrow, c - 1] = 5
                if c + 1 <= c_max:
                    grid[xrow, c + 1] = 5

        new_seg_start = {c: xrow for c in active_cols if c != src_col}
        new_seg_start[dst_col] = xrow
        col_seg_start = new_seg_start

        if src_slot == -1:
            slot_wire[dst_slot] = bypass_wire
            bypass_wire = None
        elif dst_slot == -1:
            bypass_wire = slot_wire[src_slot]
            slot_wire[src_slot] = None
        else:
            slot_wire[dst_slot] = slot_wire[src_slot]
            slot_wire[src_slot] = None

    for j in range(n):
        if slot_wire[j] is not None:
            c = ch_cols[j]
            if c in col_seg_start:
                start = col_seg_start[c] + 1
                if start <= bot_ep_row:
                    grid[start : bot_ep_row + 1, c] = 8

    return slot_wire, bypass_wire


def _derive_input(grid, n, ch_cols, perm, token_colors, top_token_row, bot_token_row):
    """Derive the input grid from the wire-only grid by placing source tokens."""
    inp = grid.copy()
    n_x = random.choices([0, 1, 2], weights=X_INPUT_COUNT_WEIGHTS)[0]
    x_channels = set(random.sample(range(n), min(n_x, n)))

    for i in range(n):
        col = token_colors[i]
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])

        if i in x_channels:
            if random.choice([True, False]):
                near, far = top_center, bot_center
            else:
                near, far = bot_center, top_center
            _stamp_token(inp, near[0], near[1], col, _X)
            if random.random() < FAR_DIAMOND_PROB:
                _stamp_token(inp, far[0], far[1], col, _DIAMOND)
        else:
            side = random.choice([top_center, bot_center])
            _stamp_token(inp, side[0], side[1], col, _SOLID)
    return inp


def _build_plain_output(grid, n, ch_cols, perm, token_colors, top_token_row, bot_token_row):
    """Original-style output without trail recoloring, used as the generator baseline."""
    out = grid.copy()
    for i in range(n):
        _stamp_token(out, top_token_row, ch_cols[i], token_colors[i], _DIAMOND)
        _stamp_token(out, bot_token_row, ch_cols[perm[i]], token_colors[i], _X)
    return out


def _trail_metrics(inp: np.ndarray, out: np.ndarray, plain: np.ndarray) -> dict[str, int]:
    wire_mask = plain == 5
    wire_mask |= plain == 8
    wire_mask |= plain == 9
    recolored = wire_mask & (out != plain)
    painted_colors = {int(out[r, c]) for r, c in np.argwhere(recolored)}
    rect_recolored = int(np.count_nonzero((plain == 9) & (out != plain)))
    return {
        "trail_pixels": int(np.count_nonzero(recolored)),
        "trail_colors": len(painted_colors),
        "rect_trail_pixels": rect_recolored,
        "input_nonbg": int(np.count_nonzero(inp)),
    }


def _gen_attempt():
    n = random.choice(CHANNEL_COUNTS)
    spacing = random.choice(CHANNEL_SPACING)

    left_margin = random.randint(*LEFT_MARGIN_RANGE)
    first_col = left_margin + 1
    ch_cols = [first_col + i * spacing for i in range(n)]
    bypass_col = ch_cols[-1] + BYPASS_COL_OFFSET

    w = max(bypass_col + 3, GRID_W_MIN)
    if w > GRID_W_MAX:
        return None

    top_token_row = random.choice(TOKEN_ROW_OPTIONS)
    top_ep_row = top_token_row + EP_TOKEN_GAP
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

    min_wire_rows = n_xovers * 2 + 2
    h_min = top_ep_row + min_wire_rows + top_token_row + 3
    if h_min > GRID_H_RANGE[1]:
        return None
    h = random.randint(max(GRID_H_RANGE[0], h_min), GRID_H_RANGE[1])

    bot_token_row = h - top_token_row - 1
    bot_ep_row = bot_token_row - EP_TOKEN_GAP
    if bot_ep_row - top_ep_row < min_wire_rows:
        return None

    avail = bot_ep_row - top_ep_row - 2
    step = max(2, avail // (n_xovers + 1))
    xover_rows = [top_ep_row + 1 + step * (s + 1) for s in range(n_xovers)]
    if xover_rows[-1] >= bot_ep_row:
        xover_rows = [top_ep_row + 2 + s * 2 for s in range(n_xovers)]
    if xover_rows[-1] >= bot_ep_row:
        return None

    grid = np.zeros((h, w), dtype=int)
    slot_wire, bypass_wire = _build_wire_network(
        grid, n, ch_cols, bypass_col, top_ep_row, bot_ep_row, crossovers, xover_rows
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
        result = _try_place_9rect(grid, ch_cols, xover_rows, top_ep_row, bot_ep_row, n, placed_rects)
        if result is not None:
            placed_rects.append(result)

    plain_output = _build_plain_output(grid, n, ch_cols, perm, token_colors, top_token_row, bot_token_row)
    inp = _derive_input(grid, n, ch_cols, perm, token_colors, top_token_row, bot_token_row)
    try:
        out = main(inp)
    except Exception:
        return None

    wire_mask = np.isin(plain_output, [5, 8, 9])
    if not np.array_equal(out[~wire_mask], plain_output[~wire_mask]):
        return None

    metrics = _trail_metrics(inp, out, plain_output)
    if metrics["trail_pixels"] < MIN_TRAIL_PIXELS:
        return None
    if metrics["trail_colors"] < MIN_TRAIL_COLORS:
        return None
    return inp


def generate_input():
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
