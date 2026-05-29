# Combo variant of a47bf94d (index 1)
# combo_spec: D1/2+D2/0
# Name: BFS激活+反射交叉

from __future__ import annotations

from collections import deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

# Design analysis
# a) 具体化与行为追踪:
#    - 保留原题的 token 词汇: solid 3x3、X、diamond，以及 token 中心距 endpoint 恰好 2 格的几何关系。
#    - 第一处融合改动来自 D1/2: token 不再沿一条贪心路径送到唯一终点，而是把 endpoint 当作信号源，在整张管道图上做 BFS 扩散。
#    - 第二处融合改动来自 D2/0: color-9 矩形不再表示“直穿到对面”，而是局部反射器。
#      这里采用可见几何两档规则:
#        wide(2x3): N<->W, S<->E
#        tall(3x2): N<->E, S<->W
#    - 一个具体链路示例:
#      source 从顶部竖干进入 wide 反射器，原题会 N->S 直穿；融合后会 N->W 转到上方总线。
#      BFS 随后沿总线同时点亮多个上侧叶子，并在更左侧 tall 反射器处 W->S 下折到下方总线，再继续扩散到多个底部叶子。
#      因而“单终点搬运”变成了“经镜面改向的多终点扩散”。
#
# b) 复杂度校准:
#    - 每个样本固定只有一个真实源 token，避免多源颜色在共享叶子上互相覆盖。
#    - 但终点叶子保留 6-8 个，足以让 BFS 扩散与原题 1 对 1 配对在视觉上明显不同。
#    - generator 采用两种稳定模板:
#        1) tall+tall 链，展示“上方入射后沿总线扩散，再向下折返”；
#        2) wide+tall 链，展示“左侧入射先被抬到总线，再向下折返”。
#      这样 wide/tall 两类镜面都能在样本中真实参与路由，而不会退化成只见一种分支。
#
# c) 隐式假设审查:
#    - 原 main() 假设“到达某点后继续贪心追一条路”；分叉后这个前提失效，所以求解器必须显式建图并做 BFS。
#    - 原 main() 还假设 9 块只需 entry_face -> opposite_face；融合后需要读取 9 块 bbox 的长宽比并改为相邻面的成对连接。
#    - endpoint 识别仍依赖“真实可见管道(5/8)局部度数为 1”，因此 generator 不能让 token 接到镜面外壁或多叉节点上。
#    - X/solid 检测仍依赖 token 周围 3x3 不被同色额外像素粘连，因此 generator 需要把 token 区域完全留空。
#
# d) 放置合法性:
#    - token 行、endpoint 行、bus 行和反射器行彼此错开，保证 3x3 token、8 号管道和 9 号矩形不重叠。
#    - 反射器只覆盖原本的 {0,8} 区域；generator 只连接真正参与传输的镜面面，避免额外 degree-1 假端点混入输出。
#    - 所有 endpoint 都恰好有一个普通管道邻居，保证 solver 的 outward-direction 推断成立。
#
# e) 视觉可观察性:
#    - 输入里只有一个有色源 token，输出里同色互补形状在多个叶子同时出现，人类容易看出“一个源扩散到许多端点”。
#    - 9 号矩形的长宽比仍然可见地编码反射类型；两个模板都刻意固定一条清晰的反射链，让求解者看懂“镜面会把 BFS 前沿拐弯”。
#    - generator 额外拒绝“把 9 当直穿也得到同一终点集合”的退化样本，确保镜面规则真的决定可达叶子。
#
# f) 可视化整洁度:
#    - 结构采用单连通骨架: 顶部总线、底部总线、两块串联镜面和若干竖向叶子，避免密集交叉噪声。
#    - 镜面只保留任务真正需要的入射/出射面，避免无意义短支路被 BFS 误识别为额外叶子。
#
# g) 变化的可推断性:
#    - 颜色不参与路由或反射，只标识源 token 身份；所有输出变化都由“源形状 + BFS 扩散 + 镜面转向”决定。
#    - solid 仍在原地坍缩成 diamond，X 仍保持 X；远端统一落下互补形状，所以颜色规律不会掺入额外条件。
#
# h) main() 的新假设与 generator 保证:
#    - 假设每个 9 组件都是单个矩形且宽高不相等；generator 固定生成 2x3 和 3x2 两类。
#    - 假设每个 token 只挂在普通 degree-1 endpoint 上；generator 只在预留的 top/bottom leaf 位置放 token。
#    - 假设源 token 至多一个；generator 固定单源。

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID_OFF = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]

_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_TALL_PAIR = {"N": "E", "E": "N", "S": "W", "W": "S"}
_WIDE_PAIR = {"N": "W", "W": "N", "S": "E", "E": "S"}


def _find_solid_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0, c0))} for maximal solid 3x3 blocks."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0:r0 + 3, c0:c0 + 3]
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
    """Return {center: (color, (r0, c0))} for isolated 3x3 X patterns."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    mask = np.zeros((3, 3), dtype=bool)
    for rr, cc in [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]:
        mask[rr, cc] = True

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0:r0 + 3, c0:c0 + 3]
            col = int(sub[0, 0])
            if col == bg:
                continue
            if np.any(sub[mask] != col) or np.any(sub[~mask] != bg):
                continue
            isolated = True
            for dr, dc in _X_OFF:
                rr, cc = r0 + 1 + dr, c0 + 1 + dc
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


def _find_9rects(g: np.ndarray) -> list[dict[str, object]]:
    """Find color-9 rectangles, their face contacts, and aspect-ratio orientation."""
    mask9 = g == 9
    if not mask9.any():
        return []
    h, w = g.shape
    rects: list[dict[str, object]] = []
    for comp9 in components_mask(mask9, connectivity=4):
        cells = {(int(r), int(c)) for r, c in comp9}
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        faces: dict[str, list[tuple[int, int]]] = {}
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in cells or not (0 <= nr < h and 0 <= nc < w):
                    continue
                if int(g[nr, nc]) not in (5, 8):
                    continue
                face = "N" if nr < r0 else "S" if nr > r1 else "W" if nc < c0 else "E"
                faces.setdefault(face, []).append((nr, nc))
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        orientation = "tall" if hh > ww else "wide" if ww > hh else "square"
        rects.append(
            {
                "cells": cells,
                "faces": faces,
                "orientation": orientation,
            }
        )
    return rects


def _pick_face_match(face: str, src: tuple[int, int], dst_pts: list[tuple[int, int]]) -> tuple[int, int]:
    sr, sc = src
    if face in ("N", "S"):
        return min(dst_pts, key=lambda p: abs(p[1] - sc))
    return min(dst_pts, key=lambda p: abs(p[0] - sr))


def _build_local_wire_adj(g: np.ndarray) -> dict[tuple[int, int], set[tuple[int, int]]]:
    """Undirected adjacency on visible wire pixels only (5/8, excluding 9 teleport behavior)."""
    nodes = {(int(r), int(c)) for r, c in np.argwhere((g == 5) | (g == 8))}
    adj: dict[tuple[int, int], set[tuple[int, int]]] = {p: set() for p in nodes}
    for r, c in nodes:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in nodes:
                adj[(r, c)].add((nr, nc))
    return adj


def _build_flow_graph(g: np.ndarray, *, reflect_mode: bool) -> dict[tuple[int, int], set[tuple[int, int]]]:
    """
    Build directed flow graph on visible wire pixels.

    Normal 5/8 adjacencies are bidirectional.
    A 9-rectangle contributes extra face-to-face transitions:
    - reflect_mode=True: tall => N<->E, S<->W; wide => N<->W, S<->E
    - reflect_mode=False: emulate the original straight-through interpretation N<->S, W<->E
    """
    adj = _build_local_wire_adj(g)

    for rect in _find_9rects(g):
        faces: dict[str, list[tuple[int, int]]] = rect["faces"]  # type: ignore[assignment]
        orientation = rect["orientation"]
        if reflect_mode and orientation == "tall":
            mapping = _TALL_PAIR
        elif reflect_mode and orientation == "wide":
            mapping = _WIDE_PAIR
        else:
            mapping = _OPP

        for face, pts in faces.items():
            exit_face = mapping.get(face)
            if exit_face is None or exit_face not in faces:
                continue
            for p in pts:
                q = _pick_face_match(face, p, faces[exit_face])
                if p == q:
                    continue
                adj.setdefault(p, set()).add(q)
    return adj


def _clear_box(out: np.ndarray, center: tuple[int, int], bg: int) -> None:
    cr, cc = center
    h, w = out.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                out[rr, cc2] = bg


def _paint_shape(
    out: np.ndarray,
    center: tuple[int, int],
    color: int,
    offsets: list[tuple[int, int]],
    bg: int,
) -> None:
    _clear_box(out, center, bg)
    h, w = out.shape
    for dr, dc in offsets:
        rr, cc = center[0] + dr, center[1] + dc
        if 0 <= rr < h and 0 <= cc < w:
            out[rr, cc] = color


def _endpoint_outdir(ep: tuple[int, int], local_adj: dict[tuple[int, int], set[tuple[int, int]]]) -> tuple[int, int] | None:
    nbrs = list(local_adj.get(ep, ()))
    if len(nbrs) != 1:
        return None
    nr, nc = nbrs[0]
    return ep[0] - nr, ep[1] - nc


def _attached_center(ep: tuple[int, int], outdir: tuple[int, int] | None) -> tuple[int, int] | None:
    if outdir is None:
        return None
    return ep[0] + 2 * outdir[0], ep[1] + 2 * outdir[1]


def _reachable(start: tuple[int, int], adj: dict[tuple[int, int], set[tuple[int, int]]]) -> set[tuple[int, int]]:
    seen = {start}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nxt in adj.get(cur, ()):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood a branching pipe graph from one endpoint token, with maroon rectangles turning the flow by reflection.

    Input structure:
    - Background 0 surrounds a connected 8/5 wire skeleton with one solid or X token attached two cells outside a degree-1 endpoint.
    - Color-9 rectangles are embedded in that skeleton and expose wire contacts on their sides; their aspect ratio matters.

    Rule:
    - A solid source collapses in place to a diamond, while an X source stays an X.
    - The source endpoint launches a BFS signal through the pipe network. Plain 8/5 links are traversed normally, but each 9-rectangle redirects the signal by a 90-degree reflection: wide blocks pair N<->W and S<->E; tall blocks pair N<->E and S<->W.
    - Every other reachable endpoint receives the source token's complementary pattern: solid emits X, X emits diamond.

    Key insight:
    - The task is neither simple endpoint pairing nor shape copying in place. You must infer a branching reachability graph, then replace each maroon block's old straight-through semantics with the correct reflective face pairing before expanding the BFS frontier.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)
    local_adj = _build_local_wire_adj(g)
    flow_adj = _build_flow_graph(g, reflect_mode=True)
    endpoints = [p for p, nbrs in flow_adj.items() if len(nbrs) == 1 and len(local_adj.get(p, ())) == 1]
    endpoint_dirs = {ep: _endpoint_outdir(ep, local_adj) for ep in endpoints}

    out = g.copy()
    source_by_ep: dict[tuple[int, int], tuple[int, str]] = {}

    for ep in endpoints:
        center = _attached_center(ep, endpoint_dirs[ep])
        if center is None:
            continue
        if center in solids:
            col, _ = solids[center]
            source_by_ep[ep] = (col, "solid")
            _paint_shape(out, center, col, _DIAMOND_OFF, bg)
        elif center in xs:
            col, _ = xs[center]
            source_by_ep[ep] = (col, "x")
            _paint_shape(out, center, col, _X_OFF, bg)

    for src_ep, (col, kind) in source_by_ep.items():
        reached = _reachable(src_ep, flow_adj)
        dest_offsets = _X_OFF if kind == "solid" else _DIAMOND_OFF
        for ep in endpoints:
            if ep == src_ep or ep not in reached or ep in source_by_ep:
                continue
            center = _attached_center(ep, endpoint_dirs[ep])
            if center is None:
                continue
            _paint_shape(out, center, col, dest_offsets, bg)

    return out


CHANNEL_COUNTS = [3, 4]
CHANNEL_SPACING = [4, 5]
LEFT_MARGIN_RANGE = (4, 4)
RIGHT_MARGIN = 3
TOKEN_ROW_OPTIONS = [2, 3]
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
GRID_H_RANGE = (20, 26)
GRID_W_MIN = 22
GRID_W_MAX = 26
EP_TOKEN_GAP = 2
TOP_BUS_GAP_RANGE = (4, 5)
BOTTOM_BUS_GAP_RANGE = (4, 5)
SOURCE_SHAPE_WEIGHTS = [1, 1]  # solid, x
GEN_MAX_RETRIES = 300


def _stamp_token(grid: np.ndarray, center_r: int, center_c: int, color: int, offsets: list[tuple[int, int]]) -> None:
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def _place_tall_reflector(grid: np.ndarray, mid_r: int, left_c: int) -> dict[str, tuple[int, int]]:
    """3x2 reflector: bbox rows mid_r-1..mid_r+1, cols left_c..left_c+1."""
    grid[mid_r - 1:mid_r + 2, left_c:left_c + 2] = 9
    return {
        "N": (mid_r - 2, left_c),
        "E": (mid_r, left_c + 2),
        "S": (mid_r + 2, left_c),
        "W": (mid_r, left_c - 1),
    }


def _place_wide_reflector(grid: np.ndarray, top_r: int, mid_c: int) -> dict[str, tuple[int, int]]:
    """2x3 reflector: bbox rows top_r..top_r+1, cols mid_c-1..mid_c+1."""
    grid[top_r:top_r + 2, mid_c - 1:mid_c + 2] = 9
    return {
        "N": (top_r - 1, mid_c),
        "E": (top_r + 1, mid_c + 2),
        "S": (top_r + 2, mid_c),
        "W": (top_r, mid_c - 2),
    }


def _draw_segment(grid: np.ndarray, p0: tuple[int, int], p1: tuple[int, int]) -> None:
    r0, c0 = p0
    r1, c1 = p1
    if r0 == r1:
        c_lo, c_hi = sorted((c0, c1))
        grid[r0, c_lo:c_hi + 1] = 8
    elif c0 == c1:
        r_lo, r_hi = sorted((r0, r1))
        grid[r_lo:r_hi + 1, c0] = 8
    else:
        raise ValueError("segments must be orthogonal")


def _collect_reachable_token_centers(
    wire: np.ndarray,
    source_ep: tuple[int, int],
    token_centers: list[tuple[int, int]],
    *,
    reflect_mode: bool,
) -> set[tuple[int, int]]:
    local_adj = _build_local_wire_adj(wire)
    flow_adj = _build_flow_graph(wire, reflect_mode=reflect_mode)
    endpoints = [p for p, nbrs in flow_adj.items() if len(nbrs) == 1 and len(local_adj.get(p, ())) == 1]
    endpoint_dirs = {ep: _endpoint_outdir(ep, local_adj) for ep in endpoints}
    center_by_ep: dict[tuple[int, int], tuple[int, int]] = {}
    valid_centers = set(token_centers)
    for ep in endpoints:
        center = _attached_center(ep, endpoint_dirs[ep])
        if center in valid_centers:
            center_by_ep[ep] = center
    reached = _reachable(source_ep, flow_adj)
    return {center_by_ep[ep] for ep in endpoints if ep in reached and ep != source_ep and ep in center_by_ep}


def _expected_output(
    wire: np.ndarray,
    source_center: tuple[int, int],
    dest_centers: list[tuple[int, int]],
    color: int,
    source_kind: str,
) -> np.ndarray:
    out = wire.copy()
    if source_kind == "solid":
        _paint_shape(out, source_center, color, _DIAMOND_OFF, 0)
        dest_offsets = _X_OFF
    else:
        _paint_shape(out, source_center, color, _X_OFF, 0)
        dest_offsets = _DIAMOND_OFF
    for center in dest_centers:
        _paint_shape(out, center, color, dest_offsets, 0)
    return out


def _build_tall_chain_layout(
    n: int,
    spacing: int,
    left_margin: int,
    top_token_row: int,
    top_ep_row: int,
    top_bus_row: int,
    bot_token_row: int,
    bot_ep_row: int,
    bot_bus_row: int,
    h: int,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None:
    source_col = left_margin
    cols = [source_col + spacing * (i + 1) for i in range(n)]
    tall1_left_c = source_col
    tall2_left_c = cols[-1] + 2
    w = max(tall2_left_c + RIGHT_MARGIN + 1, GRID_W_MIN)
    if w > GRID_W_MAX:
        return None
    if tall1_left_c < 2 or tall2_left_c + 2 >= w:
        return None

    grid = np.zeros((h, w), dtype=int)
    tall1_faces = _place_tall_reflector(grid, top_bus_row, tall1_left_c)
    tall2_faces = _place_tall_reflector(grid, top_bus_row, tall2_left_c)

    _draw_segment(grid, (top_ep_row, source_col), tall1_faces["N"])
    for c in cols:
        _draw_segment(grid, (top_ep_row, c), (top_bus_row, c))
    _draw_segment(grid, tall1_faces["E"], tall2_faces["W"])
    _draw_segment(grid, tall2_faces["S"], (bot_bus_row, tall2_faces["S"][1]))
    _draw_segment(grid, (bot_bus_row, cols[0]), (bot_bus_row, tall2_faces["S"][1]))
    for c in cols:
        _draw_segment(grid, (bot_bus_row, c), (bot_ep_row, c))

    source_center = (top_token_row, source_col)
    source_ep = (top_ep_row, source_col)
    dest_centers = [(top_token_row, c) for c in cols] + [(bot_token_row, c) for c in cols]
    return grid, source_center, source_ep, dest_centers


def _build_wide_entry_layout(
    n: int,
    spacing: int,
    left_margin: int,
    top_token_row: int,
    top_ep_row: int,
    top_bus_row: int,
    bot_token_row: int,
    bot_ep_row: int,
    bot_bus_row: int,
    h: int,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None:
    source_col = left_margin
    source_row = top_bus_row + 1
    wide1_mid_c = source_col + 3
    cols = [source_col + spacing * (i + 1) for i in range(n)]
    tall2_left_c = cols[-1] + 2
    w = max(tall2_left_c + RIGHT_MARGIN + 1, GRID_W_MIN)
    if w > GRID_W_MAX:
        return None
    if source_col - 3 < 0 or source_row + 1 >= h:
        return None
    if tall2_left_c + 2 >= w:
        return None

    grid = np.zeros((h, w), dtype=int)
    wide1_faces = _place_wide_reflector(grid, source_row, wide1_mid_c)
    tall2_faces = _place_tall_reflector(grid, top_bus_row, tall2_left_c)

    _draw_segment(grid, (source_row, source_col), wide1_faces["W"])
    for c in cols:
        _draw_segment(grid, (top_ep_row, c), (top_bus_row, c))
    _draw_segment(grid, wide1_faces["N"], tall2_faces["W"])
    _draw_segment(grid, tall2_faces["S"], (bot_bus_row, tall2_faces["S"][1]))
    _draw_segment(grid, (bot_bus_row, cols[0]), (bot_bus_row, tall2_faces["S"][1]))
    for c in cols:
        _draw_segment(grid, (bot_bus_row, c), (bot_ep_row, c))

    source_center = (source_row, source_col - 2)
    source_ep = (source_row, source_col)
    dest_centers = [(top_token_row, c) for c in cols] + [(bot_token_row, c) for c in cols]
    return grid, source_center, source_ep, dest_centers


def _gen_attempt() -> np.ndarray | None:
    layout_kind = random.choice(["tall_chain", "wide_entry"])
    n = random.choice(CHANNEL_COUNTS)
    spacing = random.choice(CHANNEL_SPACING)
    left_margin = random.randint(*LEFT_MARGIN_RANGE)

    top_token_row = random.choice(TOKEN_ROW_OPTIONS)
    top_ep_row = top_token_row + EP_TOKEN_GAP

    h = random.randint(*GRID_H_RANGE)
    bot_token_row = h - top_token_row - 1
    bot_ep_row = bot_token_row - EP_TOKEN_GAP

    top_bus_row = top_ep_row + random.randint(*TOP_BUS_GAP_RANGE)
    bot_bus_row = bot_ep_row - random.randint(*BOTTOM_BUS_GAP_RANGE)
    if bot_bus_row - top_bus_row < 4:
        return None

    if layout_kind == "tall_chain":
        layout = _build_tall_chain_layout(
            n,
            spacing,
            left_margin,
            top_token_row,
            top_ep_row,
            top_bus_row,
            bot_token_row,
            bot_ep_row,
            bot_bus_row,
            h,
        )
    else:
        layout = _build_wide_entry_layout(
            n,
            spacing,
            left_margin,
            top_token_row,
            top_ep_row,
            top_bus_row,
            bot_token_row,
            bot_ep_row,
            bot_bus_row,
            h,
        )
    if layout is None:
        return None
    grid, source_center, source_ep, dest_centers = layout

    reflect_reached = _collect_reachable_token_centers(
        grid,
        source_ep,
        [source_center] + dest_centers,
        reflect_mode=True,
    )
    straight_reached = _collect_reachable_token_centers(
        grid,
        source_ep,
        [source_center] + dest_centers,
        reflect_mode=False,
    )

    expect_reached = set(dest_centers)
    if reflect_reached != expect_reached:
        return None
    if straight_reached == reflect_reached:
        return None
    if len(reflect_reached.symmetric_difference(straight_reached)) < 2:
        return None

    color = random.choice(TOKEN_COLORS_POOL)
    source_kind = random.choices(["solid", "x"], weights=SOURCE_SHAPE_WEIGHTS)[0]

    inp = grid.copy()
    if source_kind == "solid":
        _stamp_token(inp, source_center[0], source_center[1], color, _SOLID_OFF)
    else:
        _stamp_token(inp, source_center[0], source_center[1], color, _X_OFF)

    expected = _expected_output(grid, source_center, dest_centers, color, source_kind)
    try:
        if np.array_equal(main(inp), expected):
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
