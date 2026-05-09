# Variant of a47bf94d
# Dimension: D2 — 管道交叉处理规则 (obstacle)
# Change: 交叉标记（crossing markers）——垂直管道通过水平管道的交叉处，用一对5号像素标记穿行管道的交叉点前后，表示'直行不转弯'；9号矩形块作为宽接合体直穿对面 → 反射镜代替直穿

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

# Design analysis
# a) 具体化与行为追踪:
#    - 变体只改 color-9 矩形块的含义。它们仍然嵌在 8/5 管道网络里，但不再表示“从入射面直穿到对面”。
#    - 我把 9 号块限制成两种非正方形反射镜:
#      tall(3x2): N<->E, S<->W
#      wide(2x3): N<->W, S<->E
#      也就是“长边朝竖”与“长边朝横”分别编码两种手性，规则完全由可见几何决定。
#    - 这些 9 号块只会放在原本有 5 标记的真实十字交叉点上。原题在这里会让竖线直行穿过水平线；
#      本变体会把从上方来的信号折到一侧，同时把对应侧向来的信号折到下方，等价于在交叉处做一次局部换轨。
#    - 小例子: 某行原本有一条从左到右的水平跨线，列 c 处有一根上下贯通的竖线，且 (r,c-1)/(r,c+1) 是 5 标记。
#      原逻辑中，从北侧进入会从南侧出去；现在若该处盖上 tall 反射镜，则从北进入会改为从东出去，
#      从西进入则改为从南出去。差异恰好发生在“交叉处理”这一环。
# b) 复杂度校准:
#    - 只有 tall / wide 两类反射镜，避免隐式多档规则过多。
#    - 每个样例至少放一个 9 号反射镜，且 generator 会拒绝“反射版路由与原直穿版完全相同”的网络，确保 D2 变化可观察。
#    - 保留原题的 token 形状、颜色集合、上下端点布局和大部分 wire 生成方式，避免无关维度一起漂移。
# c) 原 main() 的隐式假设审查:
#    - _find_9rects 原先只需要收集 face 邻接；现在还额外依赖 bbox 长宽比，所以 generator 必须禁用正方形 9 块。
#    - _trace_wire 原先默认“有对面就直穿，否则唯一拐弯”。该前提在变体中失效，必须在进入 9 块时先按镜面规则选出口。
#    - 原追踪用 entry column/row 对齐对面出口；现在出口可能是相邻面，所以出口点选择改成“按出口面的横/纵坐标贴近当前点”。
#    - 主循环仍然假设 token 中心距 endpoint 为 2 格，且 token 颜色不与 wire 颜色冲突；这些在变体里仍成立。
# d) 放置合法性:
#    - 9 号镜面只覆盖原始交叉附近的 {0,5,8} 区域，不覆盖 token。
#    - 每个镜面都要暴露 N/S/E/W 四个可见连接面；generator 对四个方向逐一检查。
#    - 镜面彼此保留 1 格安全间距，避免 9 号连成更大组件导致手性歧义。
#    - 放置后会重新追踪 top endpoints，只有在“全部一一映射到 bottom endpoints、无回顶/环路/断路”时才接受。
# e) 视觉可观察性:
#    - 解题者只需看 9 号矩形的朝向: 高于宽时，一条路向右折； 宽于高时，一条路向左折。
#    - 颜色不参与排序、转向或映射；颜色只是 token 身份。
# f) 可视化整洁度:
#    - 仍保留大部分 8 号竖线和少量 5 号普通直穿标记；9 号块只替换少数关键交叉，不让整张图变成大块 maroon 噪声。
#    - 只用 3x2 / 2x3 两种矩形，视觉上稳定且容易比较。
# g) 颜色变化可推断性:
#    - 无新增颜色映射。颜色始终沿实际路由搬运，变的只是终点位置与补全形状。
# 9) 新假设审查:
#    - solver 现在假设每个 9 组件都能由 bbox 长宽比唯一判定手性；generator 通过只生成 3x2 / 2x3 保证这一点。
#    - solver 假设需要反射的 9 块位于可四向接线的位置；generator 用原 5 标记交叉点作为锚点并显式校验四个面。
#    - solver 允许仍存在未改动的普通 5 标记直穿交叉；generator 也保留它们，因此“镜面”和“普通直穿”会在同题中并存。

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]

_DIR2FACE = {(1, 0): "N", (-1, 0): "S", (0, 1): "W", (0, -1): "E"}
_FACE2DIR = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}
_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_TALL_MIRROR = {"N": "E", "E": "N", "S": "W", "W": "S"}
_WIDE_MIRROR = {"N": "W", "W": "N", "S": "E", "E": "S"}


def _find_solid_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for maximal 3x3 solid blocks."""
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
    """Return {center: (color, (r0,c0))} for isolated 3x3 X patterns."""
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


def _find_9rects(g: np.ndarray) -> list[dict[str, object]]:
    """Find color-9 rectangular mirror blocks and their wire connections per face."""
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
                if (nr, nc) not in cells and 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) in (5, 8):
                    face = "N" if nr < r0 else "S" if nr > r1 else "W" if nc < c0 else "E"
                    faces.setdefault(face, []).append((nr, nc))
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh > ww:
            orientation = "tall"
        elif ww > hh:
            orientation = "wide"
        else:
            orientation = "square"
        rects.append(
            {
                "cells": cells,
                "faces": faces,
                "bbox": (r0, c0, r1, c1),
                "orientation": orientation,
            }
        )
    return rects


def _rect_exit_face(rect: dict[str, object], entry_face: str, reflect_mode: bool) -> str:
    faces = rect["faces"]
    orientation = rect["orientation"]
    if reflect_mode and orientation == "tall" and {"N", "S", "E", "W"}.issubset(faces):
        return _TALL_MIRROR[entry_face]
    if reflect_mode and orientation == "wide" and {"N", "S", "E", "W"}.issubset(faces):
        return _WIDE_MIRROR[entry_face]

    opp = _OPP[entry_face]
    if opp in faces:
        return opp
    others = [f for f in faces if f != entry_face]
    assert len(others) == 1
    return others[0]


def _pick_exit_cell(cur: tuple[int, int], exit_face: str, pts: list[tuple[int, int]]) -> tuple[int, int]:
    er, ec = cur
    if exit_face in ("N", "S"):
        return min(pts, key=lambda p: abs(p[1] - ec))
    return min(pts, key=lambda p: abs(p[0] - er))


def _trace_wire(
    comp: set[tuple[int, int]],
    start: tuple[int, int],
    dir0: tuple[int, int],
    h: int,
    w: int,
    rects_9: list[dict[str, object]],
    *,
    reflect_mode: bool,
    used_rects: set[int] | None = None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Follow one pipe path, optionally interpreting color-9 rectangles as mirrors.

    reflect_mode=True:
      tall 9-rect: N<->E and S<->W
      wide 9-rect: N<->W and S<->E
    reflect_mode=False:
      emulate the original pass-through rule for degeneration checks.
    """
    cur, d, prev = start, dir0, None
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        state = (cur, d)
        if state in visited:
            assert False, "wire trace looped"
        visited.add(state)

        jumped = False
        for rect_idx, rect in enumerate(rects_9):
            if cur in rect["cells"]:
                entry_face = _DIR2FACE[d]
                exit_face = _rect_exit_face(rect, entry_face, reflect_mode)
                pts = rect["faces"][exit_face]
                exit_cell = _pick_exit_cell(cur, exit_face, pts)
                cur, d, prev = exit_cell, _FACE2DIR[exit_face], None
                if used_rects is not None:
                    used_rects.add(rect_idx)
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


def _build_track_mask(
    g: np.ndarray,
    bg: int,
    solids: dict[tuple[int, int], tuple[int, tuple[int, int]]],
    xs: dict[tuple[int, int], tuple[int, tuple[int, int]]],
) -> np.ndarray:
    track_mask = g != bg
    for _, (_, (r0, c0)) in solids.items():
        track_mask[r0:r0 + 3, c0:c0 + 3] = False
    for cr, cc in xs.keys():
        for dr, dc in _X_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < g.shape[0] and 0 <= cc2 < g.shape[1]:
                track_mask[rr, cc2] = False
    return track_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route endpoint tokens through pipe networks whose maroon rectangles act as quarter-turn mirrors.

    Input structure:
    - Background cells are empty; non-background cells form 8/5 wire networks plus nearby 3x3 endpoint tokens.
    - Tokens are either solid 3x3 blocks or X patterns, and color-9 rectangles may sit inside the wires.

    Rule:
    - Every solid 3x3 token collapses in place to a diamond (the four cardinal neighbors around its center).
    - Each endpoint token sends its complementary pattern to the far endpoint reached by the pipe: solid emits X, X emits diamond.
    - A color-9 rectangle is a mirror, not a straight junction: tall rectangles turn N<->E and S<->W, while wide rectangles turn N<->W and S<->E.

    Key insight:
    - Token shapes still cycle toward diamond/X complements, but endpoint pairing must now be inferred by reading each 9-rectangle's aspect-ratio-controlled reflection rather than assuming opposite-face pass-through.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)
    track_mask = _build_track_mask(g, bg, solids, xs)
    comps = _components(track_mask)
    rects_9 = _find_9rects(g)

    out = g.copy()

    for (cr, cc), (col, (r0, c0)) in solids.items():
        out[r0:r0 + 3, c0:c0 + 3] = bg
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
                    end, dd = _trace_wire(comp, ep, (-odr, -odc), h, w, rects_9, reflect_mode=True)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _X_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

                elif cen in xs:
                    col, _ = xs[cen]
                    if col in comp_colors:
                        continue
                    end, dd = _trace_wire(comp, ep, (-odr, -odc), h, w, rects_9, reflect_mode=True)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _DIAMOND_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

    return out


# --- Layout configuration ---
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

# --- Reflective 9-rect placement ---
REFLECTOR_COUNT_WEIGHTS = [70, 30]  # 1 or 2 reflectors
REFLECTOR_MARGIN_GAP = 1

# --- Input derivation ---
X_INPUT_COUNT_WEIGHTS = [60, 30, 10]
FAR_DIAMOND_PROB = 0.33

# --- Generation retry ---
GEN_MAX_RETRIES = 400


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
    """Decompose a permutation into sequential crossover moves using the bypass column."""
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
    """Draw the base wire network with ordinary 5-marked straight crossings."""
    slot_wire: list[int | None] = list(range(n))
    bypass_wire: int | None = None

    col_seg_start: dict[int, int] = {}
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
        grid[xrow, c_min:c_max + 1] = 8

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
                    grid[start:bot_ep_row + 1, c] = 8

    return slot_wire, bypass_wire


def _find_reflector_candidates(grid: np.ndarray, xover_rows: list[int]) -> list[tuple[int, int]]:
    """Crossings still marked by the original pair of 5-cells are mirror candidates."""
    h, w = grid.shape
    candidates = []
    seen = set()
    for r in xover_rows:
        if not (1 <= r < h - 1):
            continue
        for c in range(1, w - 1):
            if grid[r, c] != 8:
                continue
            if grid[r - 1, c] != 8 or grid[r + 1, c] != 8:
                continue
            if grid[r, c - 1] != 5 or grid[r, c + 1] != 5:
                continue
            seen.add((r, c))
    candidates.extend(sorted(seen))
    return candidates


def _reflector_bbox(center: tuple[int, int], orientation: str) -> tuple[int, int, int, int]:
    r, c = center
    if orientation == "tall":
        return r - 1, c, r + 1, c + 1
    return r, c - 1, r + 1, c + 1


def _bbox_overlaps(b0: tuple[int, int, int, int], b1: tuple[int, int, int, int], gap: int) -> bool:
    r0, c0, r1, c1 = b0
    rr0, cc0, rr1, cc1 = b1
    return r0 <= rr1 + gap and rr0 <= r1 + gap and c0 <= cc1 + gap and cc0 <= c1 + gap


def _can_place_reflector(
    grid: np.ndarray,
    center: tuple[int, int],
    orientation: str,
    placed_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    r, c = center
    r0, c0, r1, c1 = _reflector_bbox(center, orientation)
    if not (1 <= r0 and 1 <= c0 and r1 < h - 1 and c1 < w - 1):
        return None
    if any(_bbox_overlaps((r0, c0, r1, c1), prev, REFLECTOR_MARGIN_GAP) for prev in placed_rects):
        return None

    sub = grid[r0:r1 + 1, c0:c1 + 1]
    if not np.all((sub == 0) | (sub == 5) | (sub == 8)):
        return None

    if orientation == "tall":
        checks = [(r0 - 1, c), (r1 + 1, c), (r, c0 - 1), (r, c1 + 1)]
    else:
        checks = [(r0 - 1, c), (r1 + 1, c), (r, c0 - 1), (r, c1 + 1)]
    if any(grid[rr, cc] not in (5, 8) for rr, cc in checks):
        return None
    return r0, c0, r1, c1


def _place_reflector(grid: np.ndarray, center: tuple[int, int], orientation: str) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _reflector_bbox(center, orientation)
    grid[r0:r1 + 1, c0:c1 + 1] = 9

    for rr in range(r0, r1 + 1):
        for cc in (c0 - 1, c1 + 1):
            if 0 <= cc < grid.shape[1] and grid[rr, cc] == 5:
                grid[rr, cc] = 8
    for cc in range(c0, c1 + 1):
        for rr in (r0 - 1, r1 + 1):
            if 0 <= rr < grid.shape[0] and grid[rr, cc] == 5:
                grid[rr, cc] = 8

    return r0, c0, r1, c1


def _route_channel_mapping(
    wire_grid: np.ndarray,
    top_ep_row: int,
    bot_ep_row: int,
    ch_cols: list[int],
    *,
    reflect_mode: bool,
) -> tuple[list[int], set[int], list[int]] | None:
    """Trace top endpoints to bottom endpoints; return mapping and reflector usage."""
    h, w = wire_grid.shape
    track_mask = wire_grid != 0
    comps = [set(comp) for comp in _components(track_mask)]
    rects_9 = _find_9rects(wire_grid)
    col_to_idx = {c: i for i, c in enumerate(ch_cols)}

    def find_comp(ep: tuple[int, int]) -> set[tuple[int, int]] | None:
        for comp in comps:
            if ep in comp:
                return comp
        return None

    mapping: list[int] = []
    used_rects: set[int] = set()
    per_channel_hits: list[int] = []
    for c in ch_cols:
        ep = (top_ep_row, c)
        comp = find_comp(ep)
        if comp is None:
            return None
        local_hits: set[int] = set()
        end, dd = _trace_wire(
            comp,
            ep,
            (1, 0),
            h,
            w,
            rects_9,
            reflect_mode=reflect_mode,
            used_rects=local_hits,
        )
        if end[0] != bot_ep_row or end[1] not in col_to_idx or dd != (1, 0):
            return None
        mapping.append(col_to_idx[end[1]])
        per_channel_hits.append(len(local_hits))
        used_rects |= local_hits

    if sorted(mapping) != list(range(len(ch_cols))):
        return None
    return mapping, used_rects, per_channel_hits


def _derive_input(
    grid: np.ndarray,
    n: int,
    ch_cols: list[int],
    mapping: list[int],
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
        bot_center = (bot_token_row, ch_cols[mapping[i]])

        if i in x_channels:
            near, far = (top_center, bot_center) if random.choice([True, False]) else (bot_center, top_center)
            _stamp_token(inp, near[0], near[1], col, _X_OFF)
            if random.random() < FAR_DIAMOND_PROB:
                _stamp_token(inp, far[0], far[1], col, _DIAMOND_OFF)
        else:
            side = top_center if random.choice([True, False]) else bot_center
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
    xover_rows = [top_ep_row + 1 + step * (s + 1) for s in range(len(crossovers))]
    if xover_rows[-1] >= bot_ep_row:
        xover_rows = [top_ep_row + 2 + s * 2 for s in range(len(crossovers))]
    if xover_rows[-1] >= bot_ep_row:
        return None

    grid = np.zeros((h, w), dtype=int)
    slot_wire, bypass_wire = _build_wire_network(grid, n, ch_cols, bypass_col, top_ep_row, bot_ep_row, crossovers, xover_rows)

    inv_perm = [0] * n
    for i in range(n):
        inv_perm[perm[i]] = i
    if any(slot_wire[j] != inv_perm[j] for j in range(n)) or bypass_wire is not None:
        return None

    candidates = _find_reflector_candidates(grid, xover_rows)
    if not candidates:
        return None

    target_reflectors = random.choices([1, 2], weights=REFLECTOR_COUNT_WEIGHTS)[0]
    random.shuffle(candidates)
    placed_rects: list[tuple[int, int, int, int]] = []
    placed = 0
    for center in candidates:
        orientations = ["tall", "wide"]
        random.shuffle(orientations)
        for orientation in orientations:
            bbox = _can_place_reflector(grid, center, orientation, placed_rects)
            if bbox is None:
                continue
            placed_rects.append(_place_reflector(grid, center, orientation))
            placed += 1
            break
        if placed >= target_reflectors:
            break
    if placed == 0:
        return None

    try:
        routed = _route_channel_mapping(grid, top_ep_row, bot_ep_row, ch_cols, reflect_mode=True)
    except Exception:
        return None
    if routed is None:
        return None
    mapping, used_rects, per_channel_hits = routed
    if mapping == list(range(n)):
        return None
    if not used_rects or max(per_channel_hits) == 0:
        return None

    try:
        straight = _route_channel_mapping(grid, top_ep_row, bot_ep_row, ch_cols, reflect_mode=False)
    except Exception:
        straight = None
    if straight is not None and straight[0] == mapping:
        return None

    output = grid.copy()
    for i in range(n):
        _stamp_token(output, top_token_row, ch_cols[i], token_colors[i], _DIAMOND_OFF)
        _stamp_token(output, bot_token_row, ch_cols[mapping[i]], token_colors[i], _X_OFF)

    inp = _derive_input(grid, n, ch_cols, mapping, token_colors, top_token_row, bot_token_row)
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
