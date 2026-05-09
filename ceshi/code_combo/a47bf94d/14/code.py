# Combo variant of a47bf94d (index 14)
# combo_spec: D11/2+D17/0
# Name: 不等间距通道+可变尺寸token

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4


# Design analysis for combo D11/2 + D17/0
# a) 具体化与行为追踪
#    原题里每个 token 都是固定 3x3，中心离端点固定 2 格，通道列距也几乎恒定。
#    融合后把这三件事绑在一起改：
#    1. token 只取两档尺寸：3x3 与 5x5；
#    2. 同一条物理通道两端保持同尺寸，所以“尺寸”是通道属性，不是一次性的输入噪声；
#    3. 相邻通道中心距不再统一，而是由相邻尺寸决定：
#       3-3 对应 gap 4，3-5 对应 gap 5，5-5 对应 gap 6。
#    这样大 token 会稳定出现在更宽的通道带里，尺寸-间距对应关系在同一张图里就能观察到。
#    具体小例子：左到右尺寸 [3,3,5,5] 时，列中心可为 [4,8,13,19]，gap 依次是 [4,5,6]。
#    若第 3 条 5x5 通道在底端放一个实心块，solver 应把它在原位拆成同尺寸的十字轮廓，
#    并沿线路在顶端生成同尺寸的对角 X；原题只会看“离端点 2 格的 3x3”，这里必须改成
#    “离端点 2 或 3 格，并按检测到的尺寸套用对应模板”。
#
# b) 复杂度校准
#    只保留两档尺寸，且每个样例都同时出现 3x3 与 5x5，避免自由度过多。
#    通道数保持 4-5 条，尺寸配置固定为 [3,3,5,5] 或 [3,3,3,5,5]，这样同一尺寸会重复出现，
#    人眼能在 2-4 个样例里看出“大 token 的通道更宽、离端点更远”。
#
# c) 隐式假设审查
#    原 main() 中与本变体直接冲突的假设有：
#    - `_find_solid_3x3` / `_find_x_3x3` 假设 token 只有 3x3；
#    - `cen = ep + 2 * dir` 假设中心距端点固定 2；
#    - 生成器的 `CHANNEL_SPACING` 假设所有列距统一。
#    新实现逐项改掉：
#    - 用 8 连通分彩色组件，再按 bbox=3/5 和相对坐标精确分类 solid / X / diamond；
#    - 对每个端点同时尝试 dist=2 和 dist=3，并要求找到的 token 尺寸与 dist 一致；
#    - 生成器按尺寸对决定 gap，所以 spacing 不再是单值常量。
#    叠加后的新冲突是：如果置换把 3x3 token 送进 5x5 通道，会出现“尺寸-通道属性”矛盾。
#    为避免这个冲突，生成器只在同尺寸组内部做置换。
#
# d) 放置合法性
#    gap 设计为 4/5/6，可保证 3x3 与 5x5 bbox 之间始终至少隔 1 列；
#    上下端点共用一条端点行，尺寸差异通过中心到端点距离 2/3 体现，因此布线仍可复用原题的
#    单层正交网络；9 矩形只嵌在导线区域，不能碰 token bbox。
#
# e) 视觉可观察性
#    这个融合不依赖颜色值排序。人类直接能看到：
#    - 大 token 本身更大；
#    - 大 token 离端点更远；
#    - 大 token 聚集的通道间距也更大。
#    这三条在输入和输出里都会重复出现，不需要任何隐藏数字规则。
#
# f) 可视化整洁度
#    通道只做上下布局，不引入额外干扰元素；尺寸只用两档；相邻 bbox 保持 1 列空白；
#    这样即使有 9 号矩形和 5 号交叉标记，顶/底 token 区仍然清楚可读。
#
# g) 变化可推断性与新假设
#    本变体不引入颜色映射。solver 的新增假设是：
#    - 非导线颜色组件都是 3x3 或 5x5 的 solid / X / diamond；
#    - 同尺寸组内置换，故某条线两端的 token 尺寸一致；
#    - token 只会在端点法向外 dist=size//2+1 的位置出现。
#    生成器显式保证了这些条件，因此 solver 的判定链条稳定。


_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIR2FACE = {(1, 0): "N", (-1, 0): "S", (0, 1): "W", (0, -1): "E"}
_FACE2DIR = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}

TOKEN_SIZES = (3, 5)
WIRE_COLORS = {5, 8, 9}

CHANNEL_SIZE_PROFILES = {
    4: [3, 3, 5, 5],
    5: [3, 3, 3, 5, 5],
}
PAIR_GAP = {
    (3, 3): 4,
    (3, 5): 5,
    (5, 3): 5,
    (5, 5): 6,
}
PERMUTATIONS = {
    4: [
        [1, 0, 3, 2],
    ],
    5: [
        [2, 1, 0, 4, 3],
        [1, 0, 2, 4, 3],
    ],
}

LEFT_MARGIN_RANGE = (2, 3)
BYPASS_COL_OFFSET = 3
TOP_EP_ROW_OPTIONS = [6, 7]
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
GRID_H_RANGE = (26, 30)
GRID_W_MAX = 30

RECT_COUNT_WEIGHTS = [25, 45, 25, 5]  # 0,1,2,3
RECT_WIRE_COUNT_WEIGHTS = [50, 38, 12]
RECT_PARALLEL_PROB = 0.75
RECT_H_RANGE = (2, 5)
RECT_MAX_WIDTH = 10
RECT_RETRY_FACTOR = 6
RECT_MARGIN_GAP = 1

X_INPUT_COUNT_WEIGHTS = [55, 35, 10]  # 0,1,2
FAR_DIAMOND_PROB = 0.4
GEN_MAX_RETRIES = 300


def _radius(size: int) -> int:
    return int(size) // 2


def _endpoint_distance(size: int) -> int:
    return _radius(size) + 1


def _solid_offsets(size: int) -> list[tuple[int, int]]:
    r = _radius(size)
    return [(dr, dc) for dr in range(-r, r + 1) for dc in range(-r, r + 1)]


def _x_offsets(size: int) -> list[tuple[int, int]]:
    r = _radius(size)
    out = []
    for d in range(-r, r + 1):
        out.append((d, d))
        if d != 0:
            out.append((d, -d))
    return sorted(set(out))


def _diamond_offsets(size: int) -> list[tuple[int, int]]:
    r = _radius(size)
    out = []
    for d in range(1, r + 1):
        out.extend([(d, 0), (-d, 0), (0, d), (0, -d)])
    return out


def _shape_offsets(kind: str, size: int) -> list[tuple[int, int]]:
    if kind == "solid":
        return _solid_offsets(size)
    if kind == "x":
        return _x_offsets(size)
    if kind == "diamond":
        return _diamond_offsets(size)
    raise ValueError(f"unknown kind: {kind}")


def _stamp_pattern(grid: np.ndarray, center: tuple[int, int], color: int, kind: str, size: int) -> None:
    cr, cc = center
    for dr, dc in _shape_offsets(kind, size):
        rr, cc2 = cr + dr, cc + dc
        if 0 <= rr < grid.shape[0] and 0 <= cc2 < grid.shape[1]:
            grid[rr, cc2] = color


def _clear_bbox(grid: np.ndarray, center: tuple[int, int], size: int, bg: int) -> None:
    r = _radius(size)
    cr, cc = center
    grid[cr - r : cr + r + 1, cc - r : cc + r + 1] = bg


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=4)]


def _classify_token_component(
    pts: list[tuple[int, int]],
) -> tuple[str, int, tuple[int, int], tuple[int, int, int, int], set[tuple[int, int]]] | None:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w or h not in TOKEN_SIZES:
        return None
    size = h
    cr, cc = r0 + _radius(size), c0 + _radius(size)
    rel = {(r - cr, c - cc) for r, c in pts}
    for kind in ("solid", "x", "diamond"):
        shape = set(_shape_offsets(kind, size))
        if rel == shape:
            return kind, size, (cr, cc), (r0, c0, r1, c1), set(pts)
    return None


def _find_tokens(g: np.ndarray, bg: int) -> dict[tuple[int, int], dict[str, object]]:
    tokens: dict[tuple[int, int], dict[str, object]] = {}
    colors = sorted(set(map(int, g.flatten())) - {int(bg)} - WIRE_COLORS)
    for col in colors:
        for comp in components_mask(g == int(col), connectivity=8):
            pts = [(int(r), int(c)) for r, c in comp]
            info = _classify_token_component(pts)
            if info is None:
                continue
            kind, size, center, bbox, cells = info
            tokens[center] = {
                "shape": kind,
                "size": int(size),
                "color": int(col),
                "center": center,
                "bbox": bbox,
                "cells": cells,
            }
    return tokens


def _find_9rects(g: np.ndarray) -> list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]]:
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
                if (nr, nc) not in cells and 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) in (5, 8):
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
) -> tuple[tuple[int, int], tuple[int, int]]:
    cur, d, prev = start, dir0, None
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        if (cur, d) in visited:
            raise AssertionError("wire trace looped")
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
                    if len(others) != 1:
                        raise AssertionError("invalid 9-rect routing")
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

    raise AssertionError("wire trace exceeded step limit")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route variable-size endpoint tokens through the wire network and rebuild the scaled complementary shape at the far end.

    Input structure:
    - The most common color is background. Functional wires still use colors 8, 5, and 9, but endpoint tokens can now be either 3x3 or 5x5.
    - A token sits outward from a degree-1 wire endpoint by a size-dependent distance: 2 cells for 3x3 and 3 cells for 5x5.
    - Token shapes are solid squares, diagonal X patterns, or already-settled cardinal crosses; larger channels are spaced farther apart horizontally.

    Rule:
    - A solid token is cleared from its bounding box and replaced in place by a same-size cardinal cross.
    - An X token stays where it is.
    - Every source token also sends its complementary same-size shape to the opposite endpoint of its wire path: solid emits X, X emits the cardinal cross.

    Key insight:
    - The old shortcut "look exactly 2 cells away for a 3x3 token" no longer works; the solver must first recover token size from the visible component geometry, then use that size to infer both endpoint distance and the correct scaled template.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    tokens = _find_tokens(g, bg)
    solids = {center: tok for center, tok in tokens.items() if tok["shape"] == "solid"}
    xs = {center: tok for center, tok in tokens.items() if tok["shape"] == "x"}

    track_mask = g != bg
    for tok in tokens.values():
        for r, c in tok["cells"]:
            track_mask[r, c] = False

    comps = _components(track_mask)
    rects_9 = _find_9rects(g)

    out = g.copy()

    for tok in solids.values():
        _clear_bbox(out, tok["center"], int(tok["size"]), int(bg))
        _stamp_pattern(out, tok["center"], int(tok["color"]), "diamond", int(tok["size"]))

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
                for size in TOKEN_SIZES:
                    dist = _endpoint_distance(size)
                    cen = (er + dist * odr, ec + dist * odc)
                    tok = solids.get(cen) or xs.get(cen)
                    if tok is None or int(tok["size"]) != int(size):
                        continue
                    col = int(tok["color"])
                    if col in comp_colors:
                        continue
                    dir_in = (-odr, -odc)
                    end, dd = _trace_wire(comp, ep, dir_in, h, w, rects_9)
                    dest = (end[0] + dist * dd[0], end[1] + dist * dd[1])
                    out_kind = "x" if tok["shape"] == "solid" else "diamond"
                    _stamp_pattern(out, dest, col, out_kind, size)

    return out


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
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


def _build_channel_cols(sizes: list[int], left_margin: int) -> list[int]:
    cols = [int(left_margin) + 1]
    for i in range(len(sizes) - 1):
        cols.append(cols[-1] + PAIR_GAP[(int(sizes[i]), int(sizes[i + 1]))])
    return cols


def _slot_centers(top_ep_row: int, bot_ep_row: int, sizes: list[int], cols: list[int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    top = []
    bot = []
    for size, col in zip(sizes, cols):
        dist = _endpoint_distance(size)
        top.append((top_ep_row - dist, col))
        bot.append((bot_ep_row + dist, col))
    return top, bot


def _try_place_9rect(
    grid: np.ndarray,
    ch_cols: list[int],
    xover_rows: list[int],
    top_ep_row: int,
    bot_ep_row: int,
    n: int,
    placed_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
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
            margin_l = 1
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
        if not np.all(grid[rect_r0 : rect_r1 + 1, c] == 8):
            return None

    sub = grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1]
    if not np.all((sub == 0) | (sub == 5) | (sub == 8)):
        return None

    for pr0, pc0, pr1, pc1 in placed_rects:
        if rect_r0 <= pr1 + RECT_MARGIN_GAP and pr0 <= rect_r1 + RECT_MARGIN_GAP and rect_c0 <= pc1 + RECT_MARGIN_GAP and pc0 <= rect_c1 + RECT_MARGIN_GAP:
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
    return rect_r0, rect_c0, rect_r1, rect_c1


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
    slot_wire: list[int | None] = list(range(n))
    bypass_wire: int | None = None

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


def _build_expected_from_specs(
    wire_grid: np.ndarray,
    sizes: list[int],
    perm: list[int],
    token_colors: list[int],
    top_centers: list[tuple[int, int]],
    bot_centers: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = wire_grid.copy()
    out = wire_grid.copy()

    n = len(sizes)
    n_x = random.choices([0, 1, 2], weights=X_INPUT_COUNT_WEIGHTS)[0]
    x_channels = set(random.sample(range(n), min(n_x, n)))

    for i in range(n):
        size = int(sizes[i])
        color = int(token_colors[i])
        top_center = top_centers[i]
        bot_center = bot_centers[perm[i]]
        source_is_top = random.choice([True, False])
        source_center = top_center if source_is_top else bot_center
        far_center = bot_center if source_is_top else top_center

        if i in x_channels:
            _stamp_pattern(inp, source_center, color, "x", size)
            _stamp_pattern(out, source_center, color, "x", size)
            _stamp_pattern(out, far_center, color, "diamond", size)
            if random.random() < FAR_DIAMOND_PROB:
                _stamp_pattern(inp, far_center, color, "diamond", size)
        else:
            _stamp_pattern(inp, source_center, color, "solid", size)
            _stamp_pattern(out, source_center, color, "diamond", size)
            _stamp_pattern(out, far_center, color, "x", size)

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([4, 5])
    sizes = list(CHANNEL_SIZE_PROFILES[n])
    perm = list(random.choice(PERMUTATIONS[n]))
    left_margin = random.randint(*LEFT_MARGIN_RANGE)
    ch_cols = _build_channel_cols(sizes, left_margin)
    bypass_col = ch_cols[-1] + BYPASS_COL_OFFSET

    max_right = max(c + _radius(s) for c, s in zip(ch_cols, sizes))
    w = max(max_right + 3, bypass_col + 3)
    if w > GRID_W_MAX:
        return None

    top_ep_row = random.choice(TOP_EP_ROW_OPTIONS)
    crossovers = _decompose_to_crossovers(perm)
    n_xovers = len(crossovers)
    if n_xovers == 0:
        return None

    min_wire_rows = n_xovers * 2 + 2
    h_min = top_ep_row + min_wire_rows + 6
    if h_min > GRID_H_RANGE[1]:
        return None
    h = random.randint(max(GRID_H_RANGE[0], h_min), GRID_H_RANGE[1])
    bot_ep_row = h - 6
    if bot_ep_row - top_ep_row < min_wire_rows:
        return None

    top_centers, bot_centers = _slot_centers(top_ep_row, bot_ep_row, sizes, ch_cols)
    for (tr, tc), size in zip(top_centers, sizes):
        r = _radius(size)
        if tr - r < 0 or tc - r < 0 or tc + r >= w:
            return None
    for (br, bc), size in zip(bot_centers, sizes):
        r = _radius(size)
        if br + r >= h or bc - r < 0 or bc + r >= w:
            return None

    avail = bot_ep_row - top_ep_row - 2
    step = max(2, avail // (n_xovers + 1))
    xover_rows = [top_ep_row + 1 + step * (k + 1) for k in range(n_xovers)]
    if xover_rows[-1] >= bot_ep_row:
        xover_rows = [top_ep_row + 2 + 2 * k for k in range(n_xovers)]
    if xover_rows[-1] >= bot_ep_row:
        return None

    grid = np.zeros((h, w), dtype=int)
    slot_wire, bypass_wire = _build_wire_network(grid, n, ch_cols, bypass_col, top_ep_row, bot_ep_row, crossovers, xover_rows)

    inv_perm = [0] * n
    for i in range(n):
        inv_perm[perm[i]] = i
    if any(slot_wire[j] != inv_perm[j] for j in range(n)) or bypass_wire is not None:
        return None

    n_rects = random.choices(range(4), weights=RECT_COUNT_WEIGHTS)[0]
    placed_rects: list[tuple[int, int, int, int]] = []
    for _ in range(n_rects * RECT_RETRY_FACTOR):
        if len(placed_rects) >= n_rects:
            break
        rect = _try_place_9rect(grid, ch_cols, xover_rows, top_ep_row, bot_ep_row, n, placed_rects)
        if rect is not None:
            placed_rects.append(rect)

    token_colors = random.sample(TOKEN_COLORS_POOL, n)
    inp, expected = _build_expected_from_specs(grid, sizes, perm, token_colors, top_centers, bot_centers)
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
