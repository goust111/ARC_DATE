# Combo variant of a47bf94d (index 2)
# combo_spec: D1/3+D2/2
# Name: 方向标记引导+颜色编码交叉
#
# Design analysis
# ===============
# 1) 具体化与行为追踪
#    - 基础题里，token 变形规则不复杂，关键是沿青色网络把“互补形”送到另一端。原 solver 在交叉处几乎总能靠
#      “尽量直行”完成追踪，因为老式 5 标记只是提示，不真正改写局部路由。
#    - 这个融合版把交叉处拆成两层信号：
#      * 第一层是箭头中心（4/5 号）+ 同色对角装饰，表示“如果只看默认方向，应当按哪个转接槽方向转”。
#        这里不再编码单个拐角，而是编码整条转接槽的方向家族：
#          color 4 + "/" 对角装饰 => 左到右交换槽（down->right, left->down, up->left, right->up）
#          color 5 + "\" 对角装饰 => 右到左交换槽（down->left, right->down, up->right, left->up）
#      * 第二层是交叉两侧成对的颜色标记：
#          blue(1) = 直行穿过，忽略箭头默认转向
#          red(2)  = 执行箭头给出的转向
#    - 一个具体例子：某条水平槽从左往右穿过中间通道列 c。
#      若 c 处是红标记，则自上而下的信号会在箭头中心右转进入水平槽，原本在水平槽上的信号会向下落到该列；
#      若 c 处是蓝标记，则即使中心箭头看起来在提示右转，垂直信号仍保持向下，水平信号仍保持向右。
#      因而“箭头提供默认 turn family，颜色决定是否真的执行这次 turn”，蓝色就是显式的优先级覆盖案例。
#
# 2) 复杂度校准
#    - 箭头家族只保留两类（左右交换槽），交叉颜色也只保留两类（直行 / 转弯），总离散档位很少，便于从 2-4 个样例归纳。
#    - 为了让两层规则都可见，生成器要求每题至少出现 2 个彩色交叉，且 red / blue 都实际出现；同时要求蓝色覆盖会改变
#      最终端点配对，否则这题就退化成“只读箭头”。
#    - token 颜色池避开箭头颜色 4/5，只与 red/blue 有轻度复用；这种复用通过结构区分：
#      token 始终是 3x3 图案并远离端点，交叉标记始终成对夹在管道两侧。
#
# 3) 隐式假设审查
#    - 原 _trace_wire 依赖“交叉处直行优先”即可正确；在本变体里，纯箭头交叉和红色交叉都可能要求立刻离开当前方向，
#      因此必须先识别箭头中心，再看是否有红/蓝覆盖。
#    - 原 track_mask = 非背景减 token，会把箭头对角装饰误当轨道支路。新版只把 8/9、红蓝交叉夹子和“解码出的箭头中心”
#      放入轨道；对角装饰只承担可视解码作用。
#    - 原 comp_colors 检查假设 token 颜色绝不会混入线网。这里 blue/red 可能与 token 颜色重合，所以不能再用
#      “颜色出现在 component 里就跳过该 token”的过滤。
#    - 原 9-rect 识别只把 (5,8) 当作管道邻接。新版必须把 1/2/4/5 都视为 wire-like，否则矩形贴着箭头或交叉时会漏掉面连接。
#
# 4) 放置合法性
#    - 通道列仍按 4-5 列间距布置，水平槽只在 crossover row 出现；这样交叉夹子 (c-1,c+1) 不会和相邻通道混叠。
#    - 箭头对角装饰在同一方向家族内允许放在上侧或下侧；生成器按“上侧优先，否则下侧”的方式放置，避免相邻 crossover row
#      时对角装饰互相压住。
#    - color-9 矩形只允许覆盖 0/8，不允许吞掉箭头中心、红蓝夹子或对角装饰，否则人类无法恢复两层路由线索。
#    - bypass 列没有 token，只在被某个 cycle 暂时占用的区间内补竖线，因此不会额外制造假端点。
#
# 5) 视觉可观察性
#    - 观察者可以同时看到：交叉中心出现彩色箭头、两侧再出现红/蓝成对夹子。相同箭头家族在多个交叉重复出现，
#      但红蓝会把这些交叉分成“执行箭头转向”和“无视箭头直行”两类。
#    - 当同一条水平槽上同时有红和蓝时，输出里的 token 排列会显示：有些列真的交换了信号，有些列虽然也有箭头却没有交换。
#      这正是“颜色优先覆盖默认箭头”的可视证据。
#
# 6) 可视化整洁度
#    - 主线仍保持 8/9 的大块结构；箭头只占交叉中心和一个对角像素，红蓝标记只占中心左右两格，没有额外铺满局部区域。
#    - token、箭头、交叉夹子、9-rect 各自的几何角色固定，视觉上容易分层：3x3 token、正交主线、矩形块、交叉提示。
#
# 7) 变化的可推断性
#    - 颜色变化只承担两类局部语义：4/5 是箭头方向家族，1/2 是“是否执行该箭头”的控制位。不存在隐藏的颜色算术或排序。
#    - 蓝色交叉是关键：它与红色交叉共享同类箭头，但输出行为不同，因此解题者能从少量样例学到优先级而不是把箭头当唯一规则。
#
# 8) 新 solver 假设与 generator 保证
#    - solver 假设每个箭头中心都带有同家族的单个对角装饰，generator 的 _place_arrow_marker() 保证这一点。
#    - solver 假设每个带红蓝夹子的交叉一定也是一个已解码箭头中心；generator 在每个被彩色控制的交叉上总是先放箭头再放夹子。
#    - solver 假设蓝色交叉上保持直行时，当前方向前方确实还有轨道；continuous columns + 水平槽骨架保证这一点。
#
# 9) 退化检查
#    - 生成器额外计算“arrow-only 路由”和“融合规则路由”的底部落点差异；若二者完全一致，则说明蓝色覆盖没有实际生效，
#      这类样本会被丢弃。

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

PASS_MARK = 1
TURN_MARK = 2
MARKER_COLORS = {PASS_MARK, TURN_MARK}

RIGHT_ARROW_COLOR = 4
LEFT_ARROW_COLOR = 5
ARROW_COLORS = {RIGHT_ARROW_COLOR, LEFT_ARROW_COLOR}

WIRE_COLOR = 8
RECT_COLOR = 9
WIRE_LIKE_COLORS = {WIRE_COLOR, RECT_COLOR, PASS_MARK, TURN_MARK, RIGHT_ARROW_COLOR, LEFT_ARROW_COLOR}

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID_OFF = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]

_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIR2FACE = {(1, 0): "N", (-1, 0): "S", (0, 1): "W", (0, -1): "E"}
_FACE2DIR = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}

_ARROW_COLOR = {1: RIGHT_ARROW_COLOR, -1: LEFT_ARROW_COLOR}
_ARROW_DECOR = {
    1: [(-1, 1), (1, -1)],   # "/" family => left->right exchange lane
    -1: [(-1, -1), (1, 1)],  # "\" family => right->left exchange lane
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
            if maximal:
                found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _find_x_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for isolated 3x3 X patterns."""
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
            if isolated:
                found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=4)]


def _is_wire_like(v: int) -> bool:
    return int(v) in WIRE_LIKE_COLORS


def _find_arrow_markers(g: np.ndarray) -> dict[tuple[int, int], int]:
    """Return {center: hdir} for decoded arrow-family junction markers."""
    h, w = g.shape
    markers: dict[tuple[int, int], int] = {}
    for r in range(h):
        for c in range(w):
            col = int(g[r, c])
            hdir = 1 if col == RIGHT_ARROW_COLOR else -1 if col == LEFT_ARROW_COLOR else 0
            if hdir == 0:
                continue
            neigh_count = 0
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and _is_wire_like(int(g[nr, nc])):
                    neigh_count += 1
            if neigh_count < 2:
                continue
            for dr, dc in _ARROW_DECOR[hdir]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == col:
                    markers[(r, c)] = hdir
                    break
    return markers


def _find_crossing_controls(
    track_mask: np.ndarray,
    g: np.ndarray,
    arrow_markers: dict[tuple[int, int], int],
) -> dict[tuple[int, int], int]:
    """Return {arrow_center: PASS_MARK|TURN_MARK} for controlled interior crossings."""
    h, w = g.shape
    comp = {(int(r), int(c)) for r, c in np.argwhere(track_mask)}
    controls: dict[tuple[int, int], int] = {}
    for r, c in arrow_markers:
        if r <= 0 or r >= h - 1 or c <= 0 or c >= w - 1:
            continue
        mark = int(g[r, c - 1])
        if mark not in MARKER_COLORS or int(g[r, c + 1]) != mark:
            continue
        needed = {(r, c), (r, c - 1), (r, c + 1), (r - 1, c), (r + 1, c)}
        if needed.issubset(comp):
            controls[(r, c)] = mark
    return controls


def _find_9rects(g: np.ndarray) -> list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]]:
    """Find color-9 rectangular blocks and their wire contacts on each face."""
    mask9 = g == RECT_COLOR
    if not mask9.any():
        return []
    h, w = g.shape
    rects = []
    for comp9 in components_mask(mask9, connectivity=4):
        cells = {(int(r), int(c)) for r, c in comp9}
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


def _turn_dir_at_arrow(d: tuple[int, int], hdir: int) -> tuple[int, int] | None:
    if hdir == 1:
        mapping = {
            (1, 0): (0, 1),
            (0, 1): (1, 0),
            (-1, 0): (0, -1),
            (0, -1): (-1, 0),
        }
    else:
        mapping = {
            (1, 0): (0, -1),
            (0, -1): (1, 0),
            (-1, 0): (0, 1),
            (0, 1): (-1, 0),
        }
    return mapping.get(d)


def _trace_wire(
    comp: set[tuple[int, int]],
    start: tuple[int, int],
    dir0: tuple[int, int],
    h: int,
    w: int,
    rects_9: list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]],
    arrow_markers: dict[tuple[int, int], int],
    crossing_controls: dict[tuple[int, int], int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Follow one wire component using arrow-family defaults, red/blue overrides, and 9-rect pass-through.

    Pure arrow junctions always apply their turn family. At interior crossings, blue keeps the
    current direction while red executes that same arrow-family turn.
    """
    cur, d, prev = start, dir0, None
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 8):
        state = (cur, d)
        if state in visited:
            raise AssertionError("wire trace looped under arrow+color routing")
        visited.add(state)

        jumped = False
        for cells9, faces9 in rects_9:
            if cur in cells9:
                entry_face = _DIR2FACE[d]
                opp = _OPP[entry_face]
                if opp in faces9:
                    exit_face = opp
                else:
                    others = [face for face in faces9 if face != entry_face]
                    if len(others) != 1:
                        raise AssertionError("ambiguous 9-rect routing")
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

        arrow_hdir = arrow_markers.get(cur)
        if arrow_hdir is not None:
            control = crossing_controls.get(cur)
            should_turn = control != PASS_MARK
            if should_turn:
                nd = _turn_dir_at_arrow(d, arrow_hdir)
                if nd is not None:
                    nxt = (cur[0] + nd[0], cur[1] + nd[1])
                    if nxt in comp and nxt != prev:
                        prev, cur, d = cur, nxt, nd
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
    Route colored 3x3 tokens through pipe networks whose default turns come from arrow junctions, but whose crossings can overrule those arrows.

    Input structure:
    - Background cells surround a cyan pipe system with optional maroon 9-rect pass-through blocks.
    - Tokens appear as 3x3 solids or 3x3 X-patterns two cells away from degree-1 endpoints.
    - Some junction centers are colored arrow markers (4/5) with a matching diagonal decoration; some of those same junctions also carry blue/red side markers that control the crossing behavior.

    Rule:
    - A solid 3x3 token collapses in place to a diamond and emits an X of the same color to the opposite endpoint.
    - An X token stays in place and emits a diamond to the opposite endpoint.
    - At an arrow-only junction, the signal follows the arrow's turn family. At an arrow+color crossing, red executes that turn while blue forces straight-through motion instead.

    Key insight:
    - The token-shape transformation is still the original complementary solid/X/diamond cycle.
    - The new challenge is two-layer routing: arrows define the default local turn map, but blue crossings visibly override that default. Correct endpoint pairing therefore depends on learning both the arrow family and the color priority.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)
    arrow_markers = _find_arrow_markers(g)

    track_mask = np.zeros((h, w), dtype=bool)
    track_mask[g == WIRE_COLOR] = True
    track_mask[g == RECT_COLOR] = True
    track_mask[(g == PASS_MARK) | (g == TURN_MARK)] = True
    for p in arrow_markers:
        track_mask[p] = True

    for _, (_, (r0, c0)) in solids.items():
        track_mask[r0 : r0 + 3, c0 : c0 + 3] = False
    for cr, cc in xs:
        for dr, dc in _X_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                track_mask[rr, cc2] = False

    comps = _components(track_mask)
    rects_9 = _find_9rects(g)
    crossing_controls = _find_crossing_controls(track_mask, g, arrow_markers)

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
                    end, dd = _trace_wire(comp, ep, (-odr, -odc), h, w, rects_9, arrow_markers, crossing_controls)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _X_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

                elif cen in xs:
                    col, _ = xs[cen]
                    end, dd = _trace_wire(comp, ep, (-odr, -odc), h, w, rects_9, arrow_markers, crossing_controls)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _DIAMOND_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

    return out


# --- Generator ---

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

RECT_COUNT_WEIGHTS = [12, 42, 28, 18]
RECT_WIRE_COUNT_WEIGHTS = [55, 33, 12]
RECT_PARALLEL_PROB = 0.75
RECT_H_RANGE = (2, 5)
RECT_MAX_WIDTH = 7
RECT_RETRY_FACTOR = 5
RECT_MARGIN_GAP = 1

X_INPUT_COUNT_WEIGHTS = [60, 30, 10]
FAR_DIAMOND_PROB = 0.33
ROW_TURN_PROB = 0.58

GEN_MAX_RETRIES = 500


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
    """Decompose a permutation into slot moves using the extra bypass slot."""
    n = len(perm)
    visited = [False] * n
    crossovers: list[tuple[int, int]] = []
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


def _slot_to_col(slot: int, ch_cols: list[int], bypass_col: int) -> int:
    return bypass_col if slot == len(ch_cols) else ch_cols[slot]


def _place_arrow_marker(grid: np.ndarray, r: int, c: int, hdir: int) -> bool:
    color = _ARROW_COLOR[hdir]
    h, w = grid.shape
    for dr, dc in _ARROW_DECOR[hdir]:
        rr, cc = r + dr, c + dc
        if not (0 <= rr < h and 0 <= cc < w):
            continue
        if grid[rr, cc] not in (0, color):
            continue
        grid[r, c] = color
        grid[rr, cc] = color
        return True
    return False


def _simulate_row(slot_wire: list[int | None], src: int, dst: int, markers: dict[int, int]) -> list[int | None]:
    slot_wire = slot_wire.copy()
    step = 1 if dst > src else -1
    runner = slot_wire[src]
    if runner is None or slot_wire[dst] is not None:
        raise AssertionError("invalid row move")
    slot_wire[src] = None
    for j in range(src + step, dst, step):
        if slot_wire[j] is None:
            continue
        if markers.get(j) == TURN_MARK:
            runner, slot_wire[j] = slot_wire[j], runner
    slot_wire[dst] = runner
    return slot_wire


def _bottom_slots(slot_wire: list[int | None]) -> list[int]:
    n = len(slot_wire)
    out = [0] * n
    for slot, identity in enumerate(slot_wire):
        if identity is None:
            continue
        out[identity] = slot
    return out


def _simulate_rows(n: int, row_specs: list[dict[str, object]], *, force_all_turn: bool = False) -> list[int | None]:
    slot_wire: list[int | None] = list(range(n)) + [None]
    for spec in row_specs:
        markers = dict(spec["markers"])  # type: ignore[arg-type]
        if force_all_turn:
            markers = {slot: TURN_MARK for slot in markers}
        slot_wire = _simulate_row(slot_wire, int(spec["src"]), int(spec["dst"]), markers)
    if slot_wire[n] is not None:
        raise AssertionError("bypass must be empty at the bottom")
    return slot_wire[:n]


def _build_wire_network(
    grid: np.ndarray,
    n: int,
    ch_cols: list[int],
    bypass_col: int,
    top_ep_row: int,
    bot_ep_row: int,
    crossovers: list[tuple[int, int]],
    xover_rows: list[int],
) -> tuple[list[int | None], list[dict[str, object]]]:
    """
    Draw continuous channel columns plus arrow-marked row transfers.

    Each interior crossing gets both an arrow-family center and a red/blue control pair.
    Blue keeps straight; red executes the arrow family's swap turn.
    """
    for c in ch_cols:
        grid[top_ep_row : bot_ep_row + 1, c] = WIRE_COLOR

    slot_wire: list[int | None] = list(range(n)) + [None]
    hole_slot = n
    bypass_entry_row: int | None = None
    row_specs: list[dict[str, object]] = []

    for xrow, (src_raw, dst_raw) in zip(xover_rows, crossovers, strict=True):
        src = n if src_raw == -1 else src_raw
        dst = n if dst_raw == -1 else dst_raw
        if dst != hole_slot:
            raise AssertionError("row destination must match the current hole slot")

        src_col = _slot_to_col(src, ch_cols, bypass_col)
        dst_col = _slot_to_col(dst, ch_cols, bypass_col)
        c_min, c_max = sorted((src_col, dst_col))
        grid[xrow, c_min : c_max + 1] = WIRE_COLOR

        hdir = 1 if dst_col > src_col else -1
        if not _place_arrow_marker(grid, xrow, src_col, hdir):
            raise AssertionError("failed to place source arrow marker")
        if not _place_arrow_marker(grid, xrow, dst_col, hdir):
            raise AssertionError("failed to place destination arrow marker")

        markers: dict[int, int] = {}
        step = 1 if dst > src else -1
        for j in range(src + step, dst, step):
            if slot_wire[j] is None:
                continue
            c = _slot_to_col(j, ch_cols, bypass_col)
            mark = TURN_MARK if random.random() < ROW_TURN_PROB else PASS_MARK
            markers[j] = mark
            grid[xrow, c - 1] = mark
            grid[xrow, c + 1] = mark
            if not _place_arrow_marker(grid, xrow, c, hdir):
                raise AssertionError("failed to place interior arrow marker")

        row_specs.append({"src": src, "dst": dst, "hdir": hdir, "markers": markers.copy()})
        slot_wire = _simulate_row(slot_wire, src, dst, markers)
        hole_slot = src

        if dst == n:
            bypass_entry_row = xrow
        if src == n:
            if bypass_entry_row is None or bypass_entry_row >= xrow:
                raise AssertionError("invalid bypass interval")
            if bypass_entry_row + 1 <= xrow - 1:
                grid[bypass_entry_row + 1 : xrow, bypass_col] = WIRE_COLOR
            bypass_entry_row = None

    if bypass_entry_row is not None:
        raise AssertionError("bypass cannot stay occupied at the bottom")
    if slot_wire[n] is not None:
        raise AssertionError("bypass must be empty after all rows")
    return slot_wire[:n], row_specs


def _try_place_9rect(
    grid: np.ndarray,
    ch_cols: list[int],
    xover_rows: list[int],
    top_ep_row: int,
    bot_ep_row: int,
    n: int,
    placed_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    """Attempt to place one color-9 rectangle without swallowing arrow/color controls."""
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
    if rect_c0 < 0 or rect_c1 >= w or rect_c1 - rect_c0 + 1 > RECT_MAX_WIDTH:
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

    for pr0, pc0, pr1, pc1 in placed_rects:
        if rect_r0 <= pr1 + RECT_MARGIN_GAP and pr0 <= rect_r1 + RECT_MARGIN_GAP and rect_c0 <= pc1 + RECT_MARGIN_GAP and pc0 <= rect_c1 + RECT_MARGIN_GAP:
            return None

    if rect_r0 <= top_ep_row or rect_r1 >= bot_ep_row:
        return None
    for c in wire_cols:
        if grid[rect_r0 - 1, c] != WIRE_COLOR or grid[rect_r1 + 1, c] != WIRE_COLOR:
            return None

    saved = sub.copy()
    grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] = RECT_COLOR

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


def _place_tokens_and_output(
    wire_grid: np.ndarray,
    ch_cols: list[int],
    token_colors: list[int],
    top_token_row: int,
    bot_token_row: int,
    bottom_slots: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    inp = wire_grid.copy()
    out = wire_grid.copy()
    n = len(ch_cols)

    n_x = random.choices([0, 1, 2], weights=X_INPUT_COUNT_WEIGHTS)[0]
    x_ids = set(random.sample(range(n), min(n_x, n)))

    for identity in range(n):
        col = token_colors[identity]
        top_center = (top_token_row, ch_cols[identity])
        bot_center = (bot_token_row, ch_cols[bottom_slots[identity]])
        near_top = random.choice([True, False])
        near = top_center if near_top else bot_center
        far = bot_center if near_top else top_center

        if identity in x_ids:
            _stamp_token(inp, near[0], near[1], col, _X_OFF)
            if random.random() < FAR_DIAMOND_PROB:
                _stamp_token(inp, far[0], far[1], col, _DIAMOND_OFF)
            _stamp_token(out, near[0], near[1], col, _X_OFF)
            _stamp_token(out, far[0], far[1], col, _DIAMOND_OFF)
        else:
            _stamp_token(inp, near[0], near[1], col, _SOLID_OFF)
            _stamp_token(out, near[0], near[1], col, _DIAMOND_OFF)
            _stamp_token(out, far[0], far[1], col, _X_OFF)

    return inp, out


def _gen_attempt() -> tuple[np.ndarray, dict[str, int]] | None:
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
    if not crossovers:
        return None

    min_wire_rows = len(crossovers) * 2 + 2
    h_min = top_ep_row + min_wire_rows + top_token_row + 3
    if h_min > GRID_H_RANGE[1]:
        return None
    h = random.randint(max(GRID_H_RANGE[0], h_min), GRID_H_RANGE[1])

    bot_token_row = h - top_token_row - 1
    bot_ep_row = bot_token_row - EP_TOKEN_GAP
    if bot_ep_row - top_ep_row < min_wire_rows:
        return None

    avail = bot_ep_row - top_ep_row - 2
    step = max(2, avail // (len(crossovers) + 1))
    xover_rows = [top_ep_row + 1 + step * (k + 1) for k in range(len(crossovers))]
    if xover_rows[-1] >= bot_ep_row:
        xover_rows = [top_ep_row + 2 + k * 2 for k in range(len(crossovers))]
    if xover_rows[-1] >= bot_ep_row:
        return None

    grid = np.zeros((h, w), dtype=int)
    try:
        slot_wire, row_specs = _build_wire_network(
            grid, n, ch_cols, bypass_col, top_ep_row, bot_ep_row, crossovers, xover_rows
        )
    except AssertionError:
        return None

    marker_values = [mark for spec in row_specs for mark in dict(spec["markers"]).values()]  # type: ignore[arg-type]
    red_count = sum(mark == TURN_MARK for mark in marker_values)
    blue_count = sum(mark == PASS_MARK for mark in marker_values)
    if red_count == 0 or blue_count == 0 or len(marker_values) < 2:
        return None

    actual_bottom = _bottom_slots(slot_wire)
    arrow_only_bottom = _bottom_slots(_simulate_rows(n, row_specs, force_all_turn=True))
    priority_effect = sum(int(a != b) for a, b in zip(actual_bottom, arrow_only_bottom))
    if priority_effect == 0:
        return None
    if actual_bottom == list(range(n)):
        return None

    placed_rects: list[tuple[int, int, int, int]] = []
    n_rects = random.choices(range(4), weights=RECT_COUNT_WEIGHTS)[0]
    for _ in range(n_rects * RECT_RETRY_FACTOR):
        if len(placed_rects) >= n_rects:
            break
        rect = _try_place_9rect(grid, ch_cols, xover_rows, top_ep_row, bot_ep_row, n, placed_rects)
        if rect is not None:
            placed_rects.append(rect)

    inp, expected = _place_tokens_and_output(grid, ch_cols, token_colors, top_token_row, bot_token_row, actual_bottom)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    metrics = {
        "marked_crossings": len(marker_values),
        "red_crossings": red_count,
        "blue_crossings": blue_count,
        "priority_effect": priority_effect,
        "rects": len(placed_rects),
    }
    return inp, metrics


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")


def generate_with_metrics() -> tuple[np.ndarray, dict[str, int]]:
    """Helper for diagnostics of the fused arrow+crossing-control behavior."""
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
