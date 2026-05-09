# Variant of a47bf94d
# Dimension: D2 — 管道交叉处理规则 (obstacle)
# Change: 交叉标记（crossing markers）——垂直管道通过水平管道的交叉处，用一对5号像素标记穿行管道的交叉点前后，表示“直行不转弯”；
#         改为颜色编码交叉行为：交叉标记改用两种颜色，blue=直行穿过该交叉，red=在该交叉与水平转接槽交换路线。
#
# Design analysis
# a) 具体化与行为追踪
#    - 原题的交叉行本质上是“一个水平转接槽穿过若干竖直通道”；5 只是视觉提示，solver 几乎不读取它。
#    - 本变体把每个中间交叉点改成真正起作用的彩色标记对：在交叉中心左右各放一个同色 marker。
#      blue marker 表示两条信号都保持当前方向；red marker 表示水平信号与竖直信号在该交叉处交换路线。
#    - 具体小例子：一条信号从左侧沿水平槽前进，遇到某列竖线。
#      原题：无论看到什么 marker，solver 都会贪心直走，直到槽尾才下落。
#      变体：若该列是 blue，水平信号继续前进，竖线继续下行；若是 red，水平信号在此下落，原本竖线上的信号接管水平槽继续前进。
#      因而同一条水平槽上红蓝序列会改变通道排列，输入/输出里能直接看到颜色-行为映射。
# b) 复杂度校准
#    - 只用两种 marker 颜色，行为档位只有两类：straight / swap-turn，便于 2-4 个样例归纳。
#    - token 颜色池排除 {1,2}，避免 marker 颜色与 token 颜色重叠造成“这是不是同一类对象”的歧义。
#    - 每个样例要求至少两个被标记交叉，并且 red/blue 都实际出现，防止退化成原题。
# c) 隐式假设审查
#    - 原 main() 的 _trace_wire 依赖“交叉永远贪心直行”；变体下这不成立，所以必须显式识别 marker 与交叉方向。
#    - 原生成器默认 5 只是装饰；变体里 marker 会改写最终 permutation，因此 generator 不能先定 permutation 再忽略中间颜色，
#      而要直接模拟红蓝交叉对 slot 顺序的影响。
#    - 原 comp_colors 依赖 token 颜色不出现在管道里；保留该假设，因此 token pool 改为 {3,4,5,6,7}。
# d) 放置合法性
#    - 通道列间距维持 4-5，保证 marker 对 (c-1,c+1) 不会与相邻交叉重叠。
#    - 仍使用“空槽+bypass”的行式转接骨架；每次水平槽只在当前空槽与源槽之间绘制，故无越界、无自交。
#    - 9-rect 仍可放置，但禁止覆盖彩色交叉 marker，否则 D2 的核心线索会被吞掉。
# e) 视觉可观察性
#    - 红蓝 marker 直接贴在交叉两侧，且同一行常同时出现两种颜色；肉眼可对比“蓝处穿过、红处换线”的结果。
#    - 输出中的 token 互换位置由这些红蓝交叉决定，不依赖任何不可见排序或颜色 id 运算。
# f) 可视化整洁度
#    - 交叉规则只在原本的水平转接行上生效，不额外叠新图层；9-rect 避开 marker，避免交叉区域视觉混叠。
# g) 颜色变化可推断性
#    - 颜色只用于交叉规则，不参与 token 变色；marker 颜色种类固定为两种且在同图重复。
# h) 生成器验证指标
#    - 统计每题的 marked_crossings、red_crossings、blue_crossings；要求两色都非零且总交叉数 >= 2。
# i) main() 的新假设与 generator 保证
#    - main() 假设每个彩色交叉都落在一个水平转接槽内部，且该槽两端一端“只接上方”、另一端“只接下方”。
#      generator 的 row builder 恰好按这个骨架构造，因此 _find_marked_crossings 能恢复槽方向。

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

PASS_MARK = 1
TURN_MARK = 2
MARKER_COLORS = {PASS_MARK, TURN_MARK}
WIRE_COLORS = {8, PASS_MARK, TURN_MARK}

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID_OFF = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]


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


def _find_9rects(g: np.ndarray) -> list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]]:
    """Find color-9 rectangular blocks and the wire contacts on each face."""
    mask9 = g == 9
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
                if (nr, nc) not in cells and 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) in WIRE_COLORS:
                    face = "N" if nr < r0 else "S" if nr > r1 else "W" if nc < c0 else "E"
                    faces.setdefault(face, []).append((nr, nc))
        rects.append((cells, faces))
    return rects


_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIR2FACE = {(1, 0): "N", (-1, 0): "S", (0, 1): "W", (0, -1): "E"}
_FACE2DIR = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}


def _find_marked_crossings(track_mask: np.ndarray, g: np.ndarray) -> dict[tuple[int, int], tuple[int, int]]:
    """
    Return {(r,c): (marker_color, hdir)} for marked interior crossings.

    hdir is +1 if the horizontal transfer lane runs left -> right from its
    above-only end to its below-only end, else -1.
    """
    h, w = g.shape
    comp = {(int(r), int(c)) for r, c in np.argwhere(track_mask)}
    crossings: dict[tuple[int, int], tuple[int, int]] = {}

    for r in range(h):
        for c in range(1, w - 1):
            col = int(g[r, c - 1])
            if col not in MARKER_COLORS or int(g[r, c + 1]) != col:
                continue
            if (r, c) not in comp or (r, c - 1) not in comp or (r, c + 1) not in comp:
                continue
            if (r - 1, c) not in comp or (r + 1, c) not in comp:
                continue

            c0 = c
            while (r, c0 - 1) in comp:
                c0 -= 1
            c1 = c
            while (r, c1 + 1) in comp:
                c1 += 1
            if c0 == c1:
                continue

            left_up = (r - 1, c0) in comp
            left_down = (r + 1, c0) in comp
            right_up = (r - 1, c1) in comp
            right_down = (r + 1, c1) in comp

            if left_up and not left_down and right_down and not right_up:
                hdir = 1
            elif left_down and not left_up and right_up and not right_down:
                hdir = -1
            elif right_up and not right_down and left_down and not left_up:
                hdir = -1
            elif right_down and not right_up and left_up and not left_down:
                hdir = 1
            else:
                continue
            crossings[(r, c)] = (col, hdir)
    return crossings


def _turn_dir_at_crossing(d: tuple[int, int], hdir: int) -> tuple[int, int] | None:
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
    crossings: dict[tuple[int, int], tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Follow one wire component using color-coded crossing rules and 9-rect pass-through.

    Blue crossing markers preserve direction; red crossing markers swap the horizontal
    transfer lane with the vertical lane according to the lane orientation recovered
    from the segment endpoints.
    """
    cur, d, prev = start, dir0, None
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 6):
        state = (cur, d)
        if state in visited:
            raise AssertionError("wire trace looped")
        visited.add(state)

        jumped = False
        for cells9, faces9 in rects_9:
            if cur in cells9:
                entry_face = _DIR2FACE[d]
                opp = _OPP[entry_face]
                if opp in faces9:
                    exit_face = opp
                else:
                    others = [f for f in faces9 if f != entry_face]
                    if len(others) != 1:
                        raise AssertionError("ambiguous 9-rect turn")
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

        red_rule = crossings.get(cur)
        if red_rule is not None and red_rule[0] == TURN_MARK:
            nd = _turn_dir_at_crossing(d, red_rule[1])
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
    Route colored 3x3 tokens through pipe networks whose crossings are controlled by marker colors.

    Input structure:
    - Background cells surround orthogonal wire networks, optional maroon 9-rect pass-through blocks,
      and token patterns near degree-1 wire endpoints.
    - A 3x3 solid block is an active sender; a 3x3 X-pattern is also an active sender.
      Blue/red marker pairs appear around some wire crossings and encode how that crossing behaves.

    Rule:
    - Every solid 3x3 token collapses in place to a diamond.
    - Each sender emits its complementary 3x3 pattern to the opposite endpoint of its wire:
      solid -> X, X -> diamond.
    - At a blue-marked crossing the signal keeps its current direction; at a red-marked crossing
      the signal swaps between the vertical pipe and the horizontal transfer lane.

    Key insight:
    - The wire layout is no longer solved by plain greedy tracing: the marker color at each crossing
      changes which channel continues downward, so the red/blue sequence determines the endpoint pairing.
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
    crossings = _find_marked_crossings(track_mask, g)

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
                    end, dd = _trace_wire(comp, ep, (-odr, -odc), h, w, rects_9, crossings)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _X_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

                elif cen in xs:
                    col, _ = xs[cen]
                    if col in comp_colors:
                        continue
                    end, dd = _trace_wire(comp, ep, (-odr, -odc), h, w, rects_9, crossings)
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
TOKEN_COLORS_POOL = [3, 4, 5, 6, 7]
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
GEN_MAX_RETRIES = 400


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
    """Original empty-slot scaffold: channel indices plus bypass (-1)."""
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


def _choose_row_markers(slot_wire: list[int | None], src: int, dst: int) -> dict[int, int]:
    step = 1 if dst > src else -1
    mids = [j for j in range(src + step, dst, step) if slot_wire[j] is not None]
    markers: dict[int, int] = {}
    for j in mids:
        markers[j] = TURN_MARK if random.random() < 0.6 else PASS_MARK
    return markers


def _build_wire_network(
    grid: np.ndarray,
    n: int,
    ch_cols: list[int],
    bypass_col: int,
    top_ep_row: int,
    bot_ep_row: int,
    crossovers: list[tuple[int, int]],
    xover_rows: list[int],
) -> tuple[list[int | None], dict[tuple[int, int], dict[int, int]]]:
    """
    Draw the row-transfer scaffold and simulate red/blue crossing effects.

    Returns:
    - final slot_wire over channel slots + bypass
    - row marker metadata keyed by (src_slot, dst_slot)
    """
    hole_slot = n
    slot_wire: list[int | None] = list(range(n)) + [None]
    col_seg_start = {ch_cols[j]: top_ep_row for j in range(n)}
    for c in ch_cols:
        grid[top_ep_row, c] = 8

    row_markers: dict[tuple[int, int], dict[int, int]] = {}

    for xrow, (src_slot_raw, dst_slot_raw) in zip(xover_rows, crossovers, strict=True):
        src = n if src_slot_raw == -1 else src_slot_raw
        dst = n if dst_slot_raw == -1 else dst_slot_raw
        if dst != hole_slot:
            raise AssertionError("scaffold lost the unique empty slot")

        active_cols = {_slot_to_col(i, ch_cols, bypass_col) for i, v in enumerate(slot_wire) if v is not None}
        for c in active_cols:
            start = col_seg_start[c] + 1
            if start <= xrow - 1:
                grid[start:xrow, c] = 8

        src_col = _slot_to_col(src, ch_cols, bypass_col)
        dst_col = _slot_to_col(dst, ch_cols, bypass_col)
        c_min, c_max = sorted((src_col, dst_col))
        grid[xrow, c_min : c_max + 1] = 8
        for c in active_cols:
            grid[xrow, c] = 8

        markers = _choose_row_markers(slot_wire, src, dst)
        step = 1 if dst > src else -1
        for j in range(src + step, dst, step):
            if slot_wire[j] is None:
                continue
            c = _slot_to_col(j, ch_cols, bypass_col)
            mark = markers[j]
            grid[xrow, c - 1] = mark
            grid[xrow, c + 1] = mark
        row_markers[(src, dst)] = markers

        slot_wire = _simulate_row(slot_wire, src, dst, markers)
        hole_slot = src
        col_seg_start = {
            _slot_to_col(i, ch_cols, bypass_col): xrow
            for i, v in enumerate(slot_wire)
            if v is not None
        }

    for i in range(n):
        if slot_wire[i] is not None:
            c = ch_cols[i]
            start = col_seg_start[c] + 1
            if start <= bot_ep_row:
                grid[start : bot_ep_row + 1, c] = 8

    if slot_wire[n] is not None:
        raise AssertionError("bypass must be empty at the bottom")
    return slot_wire[:n], row_markers


def _try_place_9rect(
    grid: np.ndarray,
    ch_cols: list[int],
    xover_rows: list[int],
    top_ep_row: int,
    bot_ep_row: int,
    n: int,
    placed_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    """Attempt to place one color-9 rectangle without swallowing colored crossing markers."""
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
            margin_l = 1 if random.random() < 0.5 else 0
            margin_r = 1 - margin_l
    rect_c0, rect_c1 = c_lo - margin_l, c_hi + margin_r
    if rect_c0 < 0 or rect_c1 >= w or rect_c1 - rect_c0 + 1 > RECT_MAX_WIDTH:
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
    if np.any((sub == PASS_MARK) | (sub == TURN_MARK)):
        return None
    if not np.all((sub == 0) | (sub == 8)):
        return None

    for pr0, pc0, pr1, pc1 in placed_rects:
        if rect_r0 <= pr1 + RECT_MARGIN_GAP and pr0 <= rect_r1 + RECT_MARGIN_GAP and rect_c0 <= pc1 + RECT_MARGIN_GAP and pc0 <= rect_c1 + RECT_MARGIN_GAP:
            return None

    if rect_r0 <= top_ep_row or rect_r1 >= bot_ep_row:
        return None
    for c in wire_cols:
        if grid[rect_r0 - 1, c] != 8 or grid[rect_r1 + 1, c] != 8:
            return None

    saved = sub.copy()
    grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] = 9

    n_cols, s_cols, w_rows, e_rows = set(), set(), set(), set()
    for c in range(rect_c0, rect_c1 + 1):
        if rect_r0 - 1 >= 0 and grid[rect_r0 - 1, c] in WIRE_COLORS:
            n_cols.add(c)
        if rect_r1 + 1 < h and grid[rect_r1 + 1, c] in WIRE_COLORS:
            s_cols.add(c)
    for r in range(rect_r0, rect_r1 + 1):
        if rect_c0 - 1 >= 0 and grid[r, rect_c0 - 1] in WIRE_COLORS:
            w_rows.add(r)
        if rect_c1 + 1 < w and grid[r, rect_c1 + 1] in WIRE_COLORS:
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
        bottom_center = (bot_token_row, ch_cols[bottom_slots[identity]])
        near_top = random.choice([True, False])
        near = top_center if near_top else bottom_center
        far = bottom_center if near_top else top_center

        if identity in x_ids:
            _stamp_token(inp, near[0], near[1], col, _X_OFF)
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
        slot_wire, row_markers = _build_wire_network(
            grid, n, ch_cols, bypass_col, top_ep_row, bot_ep_row, crossovers, xover_rows
        )
    except AssertionError:
        return None

    marker_values = [m for markers in row_markers.values() for m in markers.values()]
    red_count = sum(m == TURN_MARK for m in marker_values)
    blue_count = sum(m == PASS_MARK for m in marker_values)
    if red_count == 0 or blue_count == 0 or len(marker_values) < 2:
        return None

    placed_rects: list[tuple[int, int, int, int]] = []
    n_rects = random.choices(range(4), weights=RECT_COUNT_WEIGHTS)[0]
    for _ in range(n_rects * RECT_RETRY_FACTOR):
        if len(placed_rects) >= n_rects:
            break
        rect = _try_place_9rect(grid, ch_cols, xover_rows, top_ep_row, bot_ep_row, n, placed_rects)
        if rect is not None:
            placed_rects.append(rect)

    bottom_slots = [slot_wire.index(i) for i in range(n)]
    inp, expected = _place_tokens_and_output(grid, ch_cols, token_colors, top_token_row, bot_token_row, bottom_slots)
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
    """Helper for variant diagnostics."""
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
