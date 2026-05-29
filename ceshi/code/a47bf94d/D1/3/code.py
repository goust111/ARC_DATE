# Variant of a47bf94d
# Dimension: D1 — 管道路径追踪方式 (path)
# Change: 管道追踪（pipeline_trace）——沿青色(color 8)正交管道线段从一个端点追踪到另一个端点，遇到color-9矩形块时穿过（从入口面到对面出口面） → 方向标记引导路径

# Design analysis
# =================
# 1) 具体化与行为追踪
#    - 原题里，wire tracer 在 8/5/9 网络上默认“尽量直行”；5 只是给人看的“直行提示”。
#      这意味着只要进入一个十字路口，solver 不看局部标记也能靠“保持方向”得到正确路径。
#    - 这个变体把“保持方向”变成不可靠：主通道列改为整列贯通，因此在发生换道的行上，
#      原来 src/dst 两端的弯折点不再是死胡同式的 L/T，而是带有“继续直行”选项的 T 路口。
#      如果还沿用原 solver 的 greedy-straight，就会在这些点继续向下/向上，走错通道。
#    - 为了让人类可观察，我们在真正需要拐弯的 junction cell 上放置有颜色的箭头标记：
#      color 4 = 向左支路这一侧拐；color 5 = 向右支路这一侧拐。
#      同色对角装饰像素放在 NW/SW/NE/SE 四个象限之一，用来区分是“上侧弯”还是“下侧弯”。
#      四种 corner 类型因此是：
#        * 4 + NW 对角装饰 => {N,W}
#        * 4 + SW 对角装饰 => {S,W}
#        * 5 + NE 对角装饰 => {N,E}
#        * 5 + SE 对角装饰 => {S,E}
#      tracer 到达这样的 junction 时，不再默认直行，而是读取该 corner 对应的两条边；若当前是从其中一边进入，
#      则必须从另一边离开。
#    - 一个具体小例子：假设某颜色 token 从第 2 列通道要被送到第 4 列通道。
#      原逻辑：到达 row=r 的 src_col 时，src_col 在 r 下方没有竖线，所以只能转入横线。
#      新逻辑：src_col 整列都存在，来到 (r, src_col) 时既可以继续直行，也可以右转；
#      只有看到 color-5 + NE 对角装饰，才能知道这里应连接 {N,E}，即从上方进入后右转。
#      同一根横线在 dst_col 处再由 color-4/5 标记决定何处向下接入。
#
# 2) 复杂度校准
#    - 仍保留原题 3x3 token、9-rect、置换路由、output-first 验证。
#    - 只把 D1 改为“读标记转向”。为了避免过度自由，turn marker 只有两种颜色、四种角向类型；
#      这些类型会在一个样例中重复出现，不依赖颜色 id 的隐式排序。
#    - token 颜色池改为 {1,2,3,6,7}，刻意避开 4/5，避免“wire marker 颜色”和“token 颜色”冲突，
#      否则原 solver 里基于颜色集合的无害假设会失效。
#
# 3) 隐式假设审查（与变体直接相关的 main() 逻辑）
#    - 原 _trace_wire:
#      * 假设路口上“直行优先”总是正确。
#      * 假设 src/dst 弯折点没有直行选项，所以即使不看标记也不会误走。
#      这些在新变体中都不成立，因为主通道列是连续竖线。
#    - 原 track_mask = non-bg minus token cells:
#      * 如果直接保留，会把对角箭头装饰像素也算成轨道，产生许多孤立小 component。
#      新版只把 8/9 和“真正位于轨道上的 marker center”纳入 track_mask；对角装饰只作为视觉/解码信号。
#    - 原 _find_9rects 只把 (5,8) 视为 wire:
#      * 新版必须把 turn-marker center 的颜色 4/5 也视作 wire 邻接，否则 9-rect 紧邻箭头时会漏掉面连接。
#    - 原 comp_colors 检查：
#      * 原来它只是防 token 颜色意外出现在 wire 里；现在若 marker 与 token 复用颜色会误跳过合法 token。
#      因此生成器直接保证 token 不用 4/5，solver 也不再依赖这个颜色过滤。
#
# 4) 放置合法性
#    - token 中心仍离 endpoint 2 格，3x3 token 不越界。
#    - 主通道列全部贯通，横向 crossover row 之间至少隔 2 行；这样对角箭头装饰不会压到另一条横线。
#    - 9-rect 只允许覆盖 0/8 区域，不允许吞掉 4/5 marker；否则 turn rule 会被抹掉，solver 无法恢复。
#    - bypass 列仍无 token；当某段路径临时进入 bypass 时，只给 bypass 画该占用区间的竖线，并在区间端点
#      补 1-cell stub，保证这些 junction 也有“直行/转向”的局部选择。
#
# 5) 视觉可观察性
#    - 观察者能看到：原来 5 号灰色“直行夹子”消失了，取而代之的是 4/5 号彩色箭头拐角。
#    - 同一种 marker 颜色在多个路口重复出现；对角装饰固定落在某个象限，能从少量样例归纳出 corner 含义。
#    - continuous vertical columns 让“为什么需要箭头”在图上可见：没有箭头时，许多 junction 明显存在直行分支。
#
# 6) 可视化整洁度
#    - 箭头装饰只占一个对角像素，不直接铺在主线中央，因此 wire 主体仍清晰。
#    - token 颜色避开 4/5，marker 与 token 区域不会混成同类对象。
#    - 9-rect 仍保持纯色矩形块，不叠加箭头。
#
# 7) 颜色变化可推断性
#    - 本变体不引入新的颜色映射；颜色只承担“token 身份”与“junction turn family”两类固定角色。
#    - 4/5 只用于箭头，不与 token 复用；解题者无需猜测颜色置换。
#
# 8) 新 solver 的输入假设与 generator 保证
#    - 假设每个 turn-marker center 都能由“中心色 + 同色对角装饰”唯一解码。
#      generator 通过专门的 `_place_turn_marker()` 保证。
#    - 假设被标记的 junction 的实际路径只会从该 corner 的两条边之一进入。
#      generator 仍沿原 permutation/cycle 逻辑布线，只是把 channel 列改为连续竖线，因此这个假设成立。
#    - 假设 9-rect 不覆盖 marker center。
#      `_try_place_9rect()` 显式限制矩形内部只能含 0/8。

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]

WIRE_COLOR = 8
TURN_LEFT_COLOR = 4
TURN_RIGHT_COLOR = 5
TURN_COLORS = {TURN_LEFT_COLOR, TURN_RIGHT_COLOR}
WIRE_LIKE_COLORS = {WIRE_COLOR, 9, TURN_LEFT_COLOR, TURN_RIGHT_COLOR}

_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIR2FACE = {(1, 0): "N", (-1, 0): "S", (0, 1): "W", (0, -1): "E"}
_FACE2DIR = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}

_TURN_MARKER_SPEC = {
    ("N", "W"): (TURN_LEFT_COLOR, (-1, -1)),
    ("S", "W"): (TURN_LEFT_COLOR, (1, -1)),
    ("N", "E"): (TURN_RIGHT_COLOR, (-1, 1)),
    ("S", "E"): (TURN_RIGHT_COLOR, (1, 1)),
}
_TURN_DECODER = {
    TURN_LEFT_COLOR: [(("N", "W"), (-1, -1)), (("S", "W"), (1, -1))],
    TURN_RIGHT_COLOR: [(("N", "E"), (-1, 1)), (("S", "E"), (1, 1))],
}


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


def _is_wire_like(v: int) -> bool:
    return int(v) in WIRE_LIKE_COLORS


def _find_turn_markers(g: np.ndarray) -> dict[tuple[int, int], tuple[str, str]]:
    """Return {marker_center: (face_a, face_b)} for valid junction markers."""
    h, w = g.shape
    markers: dict[tuple[int, int], tuple[str, str]] = {}
    for r in range(h):
        for c in range(w):
            col = int(g[r, c])
            if col not in TURN_COLORS:
                continue
            for pair, (dr, dc) in _TURN_DECODER[col]:
                rr, cc = r + dr, c + dc
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if int(g[rr, cc]) != col:
                    continue
                ok = True
                for face in pair:
                    sdr, sdc = _FACE2DIR[face]
                    nr, nc = r + sdr, c + sdc
                    if not (0 <= nr < h and 0 <= nc < w and _is_wire_like(int(g[nr, nc]))):
                        ok = False
                        break
                if ok:
                    markers[(r, c)] = pair
                    break
    return markers


def _find_9rects(g: np.ndarray) -> list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]]:
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
                if (nr, nc) not in cells and 0 <= nr < h and 0 <= nc < w and _is_wire_like(int(g[nr, nc])):
                    face = "N" if nr < r0 else "S" if nr > r1 else "W" if nc < c0 else "E"
                    faces.setdefault(face, []).append((nr, nc))
        rects.append((cells, faces))
    return rects


def _trace_wire(
    comp: set[tuple[int, int]],
    start: tuple[int, int],
    dir0: tuple[int, int],
    h: int,
    w: int,
    rects_9: list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]],
    turn_markers: dict[tuple[int, int], tuple[str, str]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Follow a wire network with marker-guided turns and color-9 pass-through blocks.

    At a marked junction, the color+diagonal arrow selects a corner pair and overrides
    the default "keep straight" behavior whenever the tracer enters from one arm of that corner.
    """
    cur, d, prev = start, dir0, None
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 6):
        if (cur, d) in visited:
            raise AssertionError("wire trace looped under turn-marker routing")
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

        if cur in turn_markers:
            pair = turn_markers[cur]
            entry_face = _DIR2FACE[d]
            if entry_face in pair:
                exit_face = pair[0] if pair[1] == entry_face else pair[1]
                nd = _FACE2DIR[exit_face]
                nxt = (cur[0] + nd[0], cur[1] + nd[1])
                if nxt in comp:
                    prev = cur
                    cur = nxt
                    d = nd
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

    raise AssertionError("wire trace exceeded step limit")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Send each token through the cyan network, but read colored arrow junctions before deciding where to turn.

    Input structure:
    - Background cells are empty. Colored 3x3 tokens sit two cells away from wire endpoints.
    - The wire system uses color 8 for ordinary pipe, color 9 for pass-through blocks, and colors 4/5 for turn-marker junction cells plus their diagonal arrow decoration.

    Rule:
    - A solid 3x3 token collapses in place to a diamond and emits an X of the same color to the far endpoint.
    - An X token stays put and emits a diamond to the far endpoint.
    - When a tracer reaches a marked junction, it must follow the corner encoded by the marker instead of blindly continuing straight.

    Key insight:
    - The token transformation is unchanged from the base task; the new challenge is path reading.
    - Continuous channel columns make many junctions ambiguous, so the correct route comes from decoding the local arrow markers, not from a generic straight-ahead heuristic.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)
    turn_markers = _find_turn_markers(g)

    track_mask = np.zeros((h, w), dtype=bool)
    track_mask[g == WIRE_COLOR] = True
    track_mask[g == 9] = True
    for p in turn_markers:
        track_mask[p] = True

    for _, (_, (r0, c0)) in solids.items():
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
                    dir_in = (-odr, -odc)
                    end, dd = _trace_wire(comp, ep, dir_in, h, w, rects_9, turn_markers)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _X_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

                elif cen in xs:
                    col, _ = xs[cen]
                    dir_in = (-odr, -odc)
                    end, dd = _trace_wire(comp, ep, dir_in, h, w, rects_9, turn_markers)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _DIAMOND_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

    return out


# --------------- generator (turn_marked_continuous_columns) ---------------

_DIAMOND = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

CHANNEL_COUNTS = [4, 5]
CHANNEL_SPACING = [4, 5]
LEFT_MARGIN_RANGE = (2, 3)
BYPASS_COL_OFFSET = 2
TOKEN_ROW_OPTIONS = [2, 3]
TOKEN_COLORS_POOL = [1, 2, 3, 6, 7]
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

GEN_MAX_RETRIES = 300


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
    """Decompose a permutation into slot moves using the extra bypass column."""
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


def _stamp_token(grid: np.ndarray, center_r: int, center_c: int, color: int, offsets: list[tuple[int, int]]) -> None:
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def _place_turn_marker(grid: np.ndarray, r: int, c: int, pair: tuple[str, str]) -> bool:
    color, (dr, dc) = _TURN_MARKER_SPEC[pair]
    rr, cc = r + dr, c + dc
    h, w = grid.shape
    if not (0 <= rr < h and 0 <= cc < w):
        return False
    if grid[rr, cc] not in (0, color):
        return False
    grid[r, c] = color
    grid[rr, cc] = color
    return True


def _try_place_9rect(
    grid: np.ndarray,
    ch_cols: list[int],
    xover_rows: list[int],
    top_ep_row: int,
    bot_ep_row: int,
    n: int,
    placed_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    """Attempt to place one color-9 rectangular block over plain 8-cells only."""
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

    max_h = min(RECT_H_RANGE[1], bot_ep_row - top_ep_row - 4)
    if max_h < RECT_H_RANGE[0]:
        return None
    rect_h = random.randint(RECT_H_RANGE[0], max_h)
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
        if not np.all(grid[rect_r0 : rect_r1 + 1, c] == WIRE_COLOR):
            return None

    sub = grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1]
    if not np.all((sub == 0) | (sub == WIRE_COLOR)):
        return None

    gap = RECT_MARGIN_GAP
    for pr0, pc0, pr1, pc1 in placed_rects:
        if rect_r0 <= pr1 + gap and pr0 <= rect_r1 + gap and rect_c0 <= pc1 + gap and pc0 <= rect_c1 + gap:
            return None

    if rect_r0 <= top_ep_row or rect_r1 >= bot_ep_row:
        return None
    for c in wire_cols:
        if grid[rect_r0 - 1, c] != WIRE_COLOR or grid[rect_r1 + 1, c] != WIRE_COLOR:
            return None

    saved = grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1].copy()
    grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] = 9

    n_cols, s_cols, w_rows, e_rows = set(), set(), set(), set()
    for c in range(rect_c0, rect_c1 + 1):
        if rect_r0 - 1 >= 0 and _is_wire_like(int(grid[rect_r0 - 1, c])):
            n_cols.add(c)
        if rect_r1 + 1 < h and _is_wire_like(int(grid[rect_r1 + 1, c])):
            s_cols.add(c)
    for r in range(rect_r0, rect_r1 + 1):
        if rect_c0 - 1 >= 0 and _is_wire_like(int(grid[r, rect_c0 - 1])):
            w_rows.add(r)
        if rect_c1 + 1 < w and _is_wire_like(int(grid[r, rect_c1 + 1])):
            e_rows.add(r)
    total_wires = len(n_cols & s_cols) + len(w_rows & e_rows)

    if total_wires < 1 or total_wires > 3:
        grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] = saved
        return None
    return (rect_r0, rect_c0, rect_r1, rect_c1)


def _build_wire_network(
    grid: np.ndarray,
    n: int,
    ch_cols: list[int],
    bypass_col: int,
    top_ep_row: int,
    bot_ep_row: int,
    crossovers: list[tuple[int, int]],
    xover_rows: list[int],
) -> tuple[list[int | None], int | None]:
    """
    Draw continuous channel columns plus marker-guided crossover rows.

    Channel columns are fully vertical, so marked junctions genuinely need guidance:
    at those cells the tracer can keep going straight unless it reads the local arrow.
    """
    for c in ch_cols:
        grid[top_ep_row : bot_ep_row + 1, c] = WIRE_COLOR

    slot_wire: list[int | None] = list(range(n))
    bypass_wire: int | None = None
    bypass_entry_row: int | None = None

    def paint_bypass_vertical(r0: int, r1: int) -> None:
        for rr in range(r0, r1 + 1):
            if grid[rr, bypass_col] == 0:
                grid[rr, bypass_col] = WIRE_COLOR

    for xrow, (src_slot, dst_slot) in zip(xover_rows, crossovers):
        if src_slot == -1:
            src_col, dst_col = bypass_col, ch_cols[dst_slot]
        elif dst_slot == -1:
            src_col, dst_col = ch_cols[src_slot], bypass_col
        else:
            src_col, dst_col = ch_cols[src_slot], ch_cols[dst_slot]

        c_min, c_max = min(src_col, dst_col), max(src_col, dst_col)
        grid[xrow, c_min : c_max + 1] = WIRE_COLOR

        if dst_col > src_col:
            src_pair = ("N", "E")
            dst_pair = ("S", "W")
        else:
            src_pair = ("N", "W")
            dst_pair = ("S", "E")

        assert _place_turn_marker(grid, xrow, src_col, src_pair)
        assert _place_turn_marker(grid, xrow, dst_col, dst_pair)

        if src_slot == -1:
            assert bypass_wire is not None and bypass_entry_row is not None
            paint_bypass_vertical(bypass_entry_row, xrow)
            if bypass_entry_row - 1 >= top_ep_row:
                if grid[bypass_entry_row - 1, bypass_col] == 0:
                    grid[bypass_entry_row - 1, bypass_col] = WIRE_COLOR
            if xrow + 1 <= bot_ep_row:
                if grid[xrow + 1, bypass_col] == 0:
                    grid[xrow + 1, bypass_col] = WIRE_COLOR
            slot_wire[dst_slot] = bypass_wire
            bypass_wire = None
            bypass_entry_row = None
        elif dst_slot == -1:
            bypass_wire = slot_wire[src_slot]
            bypass_entry_row = xrow
            slot_wire[src_slot] = None
        else:
            slot_wire[dst_slot] = slot_wire[src_slot]
            slot_wire[src_slot] = None

    if bypass_wire is not None and bypass_entry_row is not None:
        paint_bypass_vertical(bypass_entry_row, bot_ep_row)
        if bypass_entry_row - 1 >= top_ep_row:
            if grid[bypass_entry_row - 1, bypass_col] == 0:
                grid[bypass_entry_row - 1, bypass_col] = WIRE_COLOR

    return slot_wire, bypass_wire


def _derive_input(
    grid: np.ndarray,
    n: int,
    ch_cols: list[int],
    perm: list[int],
    token_colors: list[int],
    top_token_row: int,
    bot_token_row: int,
) -> np.ndarray:
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


def _gen_attempt() -> np.ndarray | None:
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

    output = grid.copy()
    for i in range(n):
        _stamp_token(output, top_token_row, ch_cols[i], token_colors[i], _DIAMOND)
        _stamp_token(output, bot_token_row, ch_cols[perm[i]], token_colors[i], _X)

    inp = _derive_input(grid, n, ch_cols, perm, token_colors, top_token_row, bot_token_row)
    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
