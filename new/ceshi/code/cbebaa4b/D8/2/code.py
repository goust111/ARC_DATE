# Variant of cbebaa4b
# Dimension: D8 — 连接器像素的颜色与角色 (meta)
# Change: 连接器统一为红色，红色仅作为连接功能标记，不承载其他语义
#   → 连接器为双像素对（方向编码）

# Design analysis for the D8 variant
# ----------------------------------
# a) 具体化与行为追踪
# design:
#   connector_form:
#     - 每个连接点不再是“独立红像素集合”，而是恰好一个 4-连通的红色双像素 domino。
#     - 同一行的红 domino 表示 horizontal port；同一列的红 domino 表示 vertical port。
#     - 端口仍然长在对象边界上，并通过同色主体像素连接回该对象。
#   input/output_effect:
#     - 输入里，散落 piece 和 anchor 上都能直接看到红色 2 格一组的连接器。
#     - 输出里，装配时只能让一个完整红 domino 与另一个完整红 domino 完全重合。
#     - “只重合其中 1 个红像素”或“把一组的两个像素分别搭到两个不同连接点上”都非法。
#   trace_example:
#     - 原逻辑：anchor 若有红像素 {(5,6),(5,7)}，solver 会把它们当两个独立目标坐标。
#     - 旧 solver 可尝试让某 piece 的某个红像素对准 (5,6)，再看其余红像素是否也碰巧可行。
#     - 新逻辑：必须把 piece 的某个 local domino 整体平移后恰好变成 {(5,6),(5,7)}；
#       若平移后只覆盖 {(5,6)} 而另一格落到别处，则整次放置直接判非法。
#
# b) 复杂度校准
# design:
#   kept_simple:
#     - 方向类型只保留两档：horizontal / vertical，避免规则过散。
#     - 每个 side 最多一个 port，仍保持原题“树状拼装”的整体复杂度。
#   enough_signal:
#     - 生成器要求每题同时出现 horizontal port 与 vertical port。
#     - 另外至少要有一个节点同时拥有两种方向的 port，避免样例长期退化成纯横向或纯纵向链。
#
# c) 隐式假设审查（原 main 的变体相关依赖）
# audit:
#   old_extract_reds:
#     - 原代码把所有 arr==2 的坐标平铺成 reds 列表，默认“每个红像素就是一个独立端口”。
#     - 变体后此假设失效，必须先按红色 4-连通块提取 port。
#   old_total_ports:
#     - 原代码用 len(reds) 统计“端口数”；现在应该用 len(ports)。
#   old_move_generation:
#     - 原代码枚举“piece 的某个红像素对准 assembly 的某个未配对红像素”。
#     - 变体后必须枚举“piece 的某个红 domino 对准 assembly 的某个未配对红 domino”。
#   old_feasibility:
#     - 原代码只看某个全局红坐标是否累计超过 2。
#     - 变体后还要禁止 1 格重叠：若一个 local domino 的两个格子里只有 1 格碰到已有红像素，则非法。
#   old_try_place:
#     - _try_place 本身允许红对红重叠，这仍可保留；但前提是 DFS 先完成 port-level 合法性筛选。
#
# d) 放置合法性
# design:
#   solver_side:
#     - DFS 状态同时追踪：
#       1. port_counts：某个全局红 domino 已被多少个原始 port 覆盖
#       2. red_counts：某个单独红坐标已被多少次覆盖
#     - 对每个待放 piece 的每个 port：
#       * 若两个格子都不碰已有红格：这是新暴露端口，允许。
#       * 若两个格子都碰已有红格，且恰好对应一个 count==1 的完整 port：允许配对。
#       * 若只碰 1 格，或 2 格碰到的不是同一个未配对 port：非法。
#   generator_side:
#     - output-first 仍成立：先长出 assembled tree，再打散得到输入。
#     - assembled 校验也使用同样的完整-port 规则，避免生成器偷偷产出“像素级能重合、端口级不合法”的图。
#
# e) 视觉可观察性
# design:
#   visible_rule:
#     - 人类在样例里能直接看到红 connector 总是 2 格一组，而且有横放与竖放两类。
#     - 输出中的连接也保留这两个红格的整体重合，因此方向约束是可见的，不依赖颜色 id 的隐藏排序。
#
# f) 可视化整洁度
# design:
#   cleanliness:
#     - 仍沿用原题的单连通 piece、2 格散布间距、树状输出，避免过多局部叠加。
#     - port 只占边界上的连续两格，不引入额外标记层。
#
# g) 颜色变化可推断性
# design:
#   colors:
#     - 未新增颜色映射；红色仍只表示连接器，只是其几何形态现在承载方向语义。
#
# h) 变体有效性指标（供测试统计）
# metrics:
#   - horizontal_ports: 样本中的横向 port 数
#   - vertical_ports: 样本中的纵向 port 数
#   - orthogonal_nodes: 至少一个节点同时含 horizontal + vertical port
#   这些指标直接衡量“方向编码”是否真实出现在样本中，而不是退化回旧题的无方向红点。
#
# i) 新 main() 的隐式假设与 generator 保证
# assumptions:
#   - 每个红色 4-连通块恰好 2 格，且只能是水平或垂直 domino。
#   - 同一 piece 的不同 port 彼此分离，不会合并成更大的红块。
#   - 合法输出中所有重叠红格都来自完整 port 对接，不存在半重叠。
# generator_guarantees:
#   - _port_cells / _make_piece / _make_anchor 保证每个 side 只生成一个 2 格红 domino。
#   - _assemble 用与 solver 相同的 port-level 约束验证 assembled 输出。
#   - _variant_metrics / _variant_structure_ok 保证方向信息在样本中可见。

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


def _normalize_cells(cells) -> tuple[tuple[int, int], tuple[int, int]]:
    pts = tuple(sorted((int(r), int(c)) for r, c in cells))
    assert len(pts) == 2, "each connector must contain exactly two red cells"
    return pts


def _port_orientation(
    cells: tuple[tuple[int, int], tuple[int, int]]
) -> str:
    (r0, c0), (r1, c1) = cells
    if r0 == r1 and abs(c0 - c1) == 1:
        return "h"
    if c0 == c1 and abs(r0 - r1) == 1:
        return "v"
    raise AssertionError("connector cells must form a horizontal or vertical domino")


def _shift_port(
    cells: tuple[tuple[int, int], tuple[int, int]], top: int, left: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    return tuple((r + top, c + left) for r, c in cells)


def _extract_ports(arr: np.ndarray) -> list[dict]:
    ports = []
    for pts in components_mask(arr == 2, connectivity=4):
        cells = _normalize_cells(pts)
        ports.append({"cells": cells, "orientation": _port_orientation(cells)})
    ports.sort(key=lambda p: p["cells"])
    return ports


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        reds = np.argwhere(arr == 2)
        ports = _extract_ports(arr)
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "reds": [(int(r), int(c)) for r, c in reds],
                "ports": ports,
                "shape": (int(arr.shape[0]), int(arr.shape[1])),
                "area": int(np.sum(arr != bg)),
            }
        )
    return pieces


def _try_place(
    canvas: np.ndarray, arr: np.ndarray, top: int, left: int, bg: int
) -> np.ndarray | None:
    h, w = canvas.shape
    ph, pw = arr.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return None

    sub = canvas[top : top + ph, left : left + pw]
    mask = arr != bg
    if not np.any(mask):
        return None

    conflict = mask & (sub != bg) & ~(((arr == 2) & (sub == 2)) | (arr == sub))
    if np.any(conflict):
        return None

    out = canvas.copy()
    out_sub = out[top : top + ph, left : left + pw]
    out_sub[mask] = arr[mask]
    out[top : top + ph, left : left + pw] = out_sub
    return out


def _init_port_states(
    piece: dict, top: int, left: int
) -> tuple[
    dict[tuple[tuple[int, int], tuple[int, int]], int],
    dict[tuple[int, int], int],
]:
    port_counts: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    red_counts: dict[tuple[int, int], int] = {}
    for port in piece["ports"]:
        gkey = _shift_port(port["cells"], top, left)
        port_counts[gkey] = port_counts.get(gkey, 0) + 1
        for cell in gkey:
            red_counts[cell] = red_counts.get(cell, 0) + 1
    return port_counts, red_counts


def _placement_overlap_score(
    piece: dict,
    top: int,
    left: int,
    port_counts: dict[tuple[tuple[int, int], tuple[int, int]], int],
    red_counts: dict[tuple[int, int], int],
) -> int | None:
    unpaired_ports = {k for k, v in port_counts.items() if v == 1}
    occupied_reds = {k for k, v in red_counts.items() if v > 0}
    overlap = 0

    for port in piece["ports"]:
        gkey = _shift_port(port["cells"], top, left)
        if any(red_counts.get(cell, 0) >= 2 for cell in gkey):
            return None

        hits = sum(cell in occupied_reds for cell in gkey)
        if hits == 1:
            return None
        if hits == 2:
            if port_counts.get(gkey, 0) != 1 or gkey not in unpaired_ports:
                return None
            overlap += 1

    return overlap


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the scattered sprites by docking red domino connectors onto matching red dominoes.

    Input structure:
    - The background is black (0). One sprite is a yellow anchor containing a 3x3 or 4x5 block of color 4.
    - Every connection site is a red 2-pixel domino on a sprite boundary.
    - A horizontal red domino encodes a horizontal docking site; a vertical red domino encodes a vertical docking site.

    Rule:
    - Keep the anchor fixed and translate the remaining sprites onto the same canvas.
    - A move is legal only when one or more full red dominoes on the moving sprite exactly overlap exposed red dominoes of the same orientation.
    - Partial red overlaps are forbidden: one red cell cannot dock by itself, and a domino cannot split across two different connection sites.
    - When all sprites are placed, every red domino must be paired exactly once.

    Key insight:
    - The red marks are no longer independent pixels; each 2-cell red component is a single typed port.
    - Solving the task requires matching whole domino ports, using their horizontal/vertical arrangement as the constraint that disambiguates valid attachments.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"
    assert all(
        len(port["cells"]) == 2 for p in pieces for port in p["ports"]
    ), "all connectors must be 2-cell dominoes"

    anchors = [p for p in pieces if np.any(p["arr"] == 4)]
    assert len(anchors) == 1, "input must have exactly one yellow (4) anchor piece"
    anchor = anchors[0]
    rest = [p for p in pieces if p is not anchor]

    canvas = np.full((h, w), bg, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, bg)
    assert placed is not None, "anchor piece must fit on canvas at its original position"

    total_ports = int(sum(len(p["ports"]) for p in pieces))
    assert total_ports % 2 == 0, "total connector ports must be even for complete pairing"
    target_unique_ports = total_ports // 2

    port_counts, red_counts = _init_port_states(anchor, ar0, ac0)
    solutions: list[np.ndarray] = []

    def dfs(
        cur: np.ndarray,
        remaining: list[dict],
        pc: dict[tuple[tuple[int, int], tuple[int, int]], int],
        rc: dict[tuple[int, int], int],
    ) -> None:
        if len(solutions) >= 2:
            return

        if not remaining:
            if (
                len(pc) == target_unique_ports
                and all(v == 2 for v in pc.values())
                and all(v == 2 for v in rc.values())
            ):
                solutions.append(cur)
            return

        if len(pc) > target_unique_ports:
            return

        unpaired = [k for k, v in pc.items() if v == 1]
        if not unpaired:
            return

        best_moves = None
        for target in unpaired:
            torient = _port_orientation(target)
            moves = []
            (tr0, tc0), (tr1, tc1) = target
            for idx, piece in enumerate(remaining):
                ph, pw = piece["shape"]
                for port in piece["ports"]:
                    if port["orientation"] != torient:
                        continue
                    (pr0, pc0), (pr1, pc1) = port["cells"]
                    dtop0, dleft0 = tr0 - pr0, tc0 - pc0
                    dtop1, dleft1 = tr1 - pr1, tc1 - pc1
                    if dtop0 != dtop1 or dleft0 != dleft1:
                        continue
                    top, left = dtop0, dleft0
                    if top < 0 or left < 0 or top + ph > h or left + pw > w:
                        continue

                    overlap = _placement_overlap_score(piece, top, left, pc, rc)
                    if overlap is None or overlap <= 0:
                        continue
                    moves.append((idx, top, left, overlap))

            if not moves:
                return
            if best_moves is None or len(moves) < len(best_moves):
                best_moves = moves
                if len(best_moves) == 1:
                    break

        assert best_moves is not None, "there must be at least one feasible connector move"

        dedup: dict[tuple[int, int, int], int] = {}
        for idx, top, left, ov in best_moves:
            key = (idx, top, left)
            dedup[key] = max(dedup.get(key, 0), ov)
        moves2 = [(idx, top, left, ov) for (idx, top, left), ov in dedup.items()]
        moves2.sort(key=lambda x: (-x[3], -remaining[x[0]]["area"], x[1], x[2], x[0]))

        for idx, top, left, _ov in moves2:
            piece = remaining[idx]
            nxt = _try_place(cur, piece["arr"], top, left, bg)
            if nxt is None:
                continue

            next_pc = dict(pc)
            next_rc = dict(rc)
            for port in piece["ports"]:
                gkey = _shift_port(port["cells"], top, left)
                next_pc[gkey] = next_pc.get(gkey, 0) + 1
                for cell in gkey:
                    next_rc[cell] = next_rc.get(cell, 0) + 1

            next_remaining = remaining[:idx] + remaining[idx + 1 :]
            dfs(nxt, next_remaining, next_pc, next_rc)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, port_counts, red_counts)
    assert len(solutions) == 1, "domino-port assembly must have exactly one valid solution"
    return solutions[0]


def _legacy_pixel_main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces
    anchors = [p for p in pieces if np.any(p["arr"] == 4)]
    assert len(anchors) == 1
    anchor = anchors[0]
    rest = [p for p in pieces if p is not anchor]

    canvas = np.full((h, w), bg, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, bg)
    assert placed is not None

    total_reds = int(sum(len(p["reds"]) for p in pieces))
    assert total_reds % 2 == 0
    target_unique_red = total_reds // 2

    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor["reds"]:
        red_counts[(ar0 + rr, ac0 + cc)] = red_counts.get((ar0 + rr, ac0 + cc), 0) + 1

    solutions: list[np.ndarray] = []

    def dfs(cur: np.ndarray, remaining: list[dict], rc: dict[tuple[int, int], int]) -> None:
        if len(solutions) >= 2:
            return
        if not remaining:
            if len(rc) == target_unique_red and all(v == 2 for v in rc.values()):
                solutions.append(cur)
            return
        if len(rc) > target_unique_red:
            return

        unpaired = [k for k, v in rc.items() if v == 1]
        if not unpaired:
            return
        rc1 = {k for k, v in rc.items() if v == 1}

        best_moves = None
        for target in unpaired:
            moves = []
            tr, tc = target
            for idx, piece in enumerate(remaining):
                ph, pw = piece["shape"]
                for pr, pc in piece["reds"]:
                    top = tr - pr
                    left = tc - pc
                    if top < 0 or left < 0 or top + ph > h or left + pw > w:
                        continue
                    bad = False
                    overlap = 0
                    for rr, cc in piece["reds"]:
                        gr, gc = top + rr, left + cc
                        prev = rc.get((gr, gc), 0)
                        if prev >= 2:
                            bad = True
                            break
                        if (gr, gc) in rc1:
                            overlap += 1
                    if not bad:
                        moves.append((idx, top, left, overlap))
            if not moves:
                return
            if best_moves is None or len(moves) < len(best_moves):
                best_moves = moves
                if len(best_moves) == 1:
                    break

        assert best_moves is not None
        dedup: dict[tuple[int, int, int], int] = {}
        for idx, top, left, ov in best_moves:
            key = (idx, top, left)
            dedup[key] = max(dedup.get(key, 0), ov)
        moves2 = [(idx, top, left, ov) for (idx, top, left), ov in dedup.items()]
        moves2.sort(key=lambda x: (-x[3], -remaining[x[0]]["area"], x[1], x[2], x[0]))

        for idx, top, left, _ov in moves2:
            piece = remaining[idx]
            nxt = _try_place(cur, piece["arr"], top, left, bg)
            if nxt is None:
                continue
            next_rc = dict(rc)
            for rr, cc in piece["reds"]:
                gr, gc = top + rr, left + cc
                next_rc[(gr, gc)] = next_rc.get((gr, gc), 0) + 1
            dfs(nxt, remaining[:idx] + remaining[idx + 1 :], next_rc)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, red_counts)
    assert len(solutions) == 1
    return solutions[0]


# --------------- generator (o5x) ---------------

# ── Tunable configuration ──────────────────────────────────────────────
GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 24

PIECE_COUNT_RANGE = (4, 6)
PIECE_H_RANGE = (4, 7)
PIECE_W_RANGE = (4, 8)
PIECE_DIM_CANDIDATES = 12
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
PIECE_CAP = 2
BAR_SKIP_PROB = 0.35

ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]
ANCHOR_CAP_CHOICES = [1, 2]

SCATTER_PADDING = 2

GEN_RETRY_LIMIT = 400

SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}
SIDE_ORIENTATION = {"top": "h", "bottom": "h", "left": "v", "right": "v"}


def _new_node(kind, h, w, color, top=0, left=0, cap=PIECE_CAP):
    return {
        "kind": kind,
        "shape": (int(h), int(w)),
        "color": int(color),
        "top": int(top),
        "left": int(left),
        "cap": int(cap),
        "sides": set(),
        "ports": {},
    }


def _ensure_port(node, side):
    if side in node["ports"]:
        return
    h, w = node["shape"]
    span = h if side in ("left", "right") else w
    node["ports"][side] = random.randint(1, span - 3)


def _port_cells(node, side):
    h, w = node["shape"]
    k = node["ports"][side]
    if side == "top":
        return [(0, k), (0, k + 1)]
    if side == "bottom":
        return [(h - 1, k), (h - 1, k + 1)]
    if side == "left":
        return [(k, 0), (k + 1, 0)]
    return [(k, w - 1), (k + 1, w - 1)]


def _global_port(node, side):
    tr, tc = node["top"], node["left"]
    return [(tr + r, tc + c) for r, c in _port_cells(node, side)]


def _bbox_of(node):
    h, w = node["shape"]
    return (node["top"], node["left"], node["top"] + h - 1, node["left"] + w - 1)


def _rects_overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _union_bbox(nodes):
    boxes = [_bbox_of(n) for n in nodes]
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _draw_path(arr, start, end, color, order):
    r, c = start
    er, ec = end
    axes = (
        [(r, er, c, True), (c, ec, er, False)]
        if order == 0
        else [(c, ec, r, False), (r, er, ec, True)]
    )
    for a_start, a_end, fixed, is_row in axes:
        step = 1 if a_end >= a_start else -1
        for v in range(a_start, a_end + step, step):
            rr, cc = (v, fixed) if is_row else (fixed, v)
            if arr[rr, cc] != 2:
                arr[rr, cc] = color


def _paint_bar(arr, side, color, used):
    h, w = arr.shape
    is_horizontal = side in ("top", "bottom")
    fixed = 0 if side in ("top", "left") else (h - 1 if side == "bottom" else w - 1)
    span = w if is_horizontal else h
    lo_conflict = "left" if is_horizontal else "top"
    hi_conflict = "right" if is_horizontal else "bottom"
    for i in range(span):
        if (i == 0 and lo_conflict in used) or (i == span - 1 and hi_conflict in used):
            continue
        if is_horizontal:
            arr[fixed, i] = color
        else:
            arr[i, fixed] = color


def _bar_endpoint(shape, side, hub):
    h, w = shape
    hr, hc = hub
    if side in ("top", "bottom"):
        row = 0 if side == "top" else h - 1
        return (row, min(max(hc, 1), w - 2))
    col = 0 if side == "left" else w - 1
    return (min(max(hr, 1), h - 2), col)


def _make_piece(node):
    h, w = node["shape"]
    color = node["color"]
    used = set(node["sides"])
    arr = np.zeros((h, w), dtype=int)
    hub = (random.randint(1, h - 2), random.randint(1, w - 2))

    bar_sides = [s for s in SIDES if s not in used]
    if len(bar_sides) > 1 and random.random() < BAR_SKIP_PROB:
        bar_sides.remove(random.choice(bar_sides))
    for side in bar_sides:
        _paint_bar(arr, side, color, used)

    terminals = []
    for side in used:
        dr, dc = INWARD[side]
        for r, c in _port_cells(node, side):
            arr[r, c] = 2
            rr, cc = r + dr, c + dc
            arr[rr, cc] = color
            terminals.append((rr, cc))

    arr[hub] = color
    order = random.randint(0, 1)
    for t in terminals:
        _draw_path(arr, t, hub, color, order)
    for side in bar_sides:
        _draw_path(arr, hub, _bar_endpoint((h, w), side, hub), color, 1 - order)

    if len(components_mask(arr != 0, connectivity=4)) != 1:
        return None
    if len(_extract_ports(arr)) != len(used):
        return None
    return arr


def _make_anchor(node):
    bh, bw = node["block"]
    arr = np.zeros(node["shape"], dtype=int)
    arr[1 : 1 + bh, 1 : 1 + bw] = 4
    for side in node["sides"]:
        for r, c in _port_cells(node, side):
            arr[r, c] = 2
    if len(_extract_ports(arr)) != len(node["sides"]):
        return None
    return arr


def _candidate_fits(nodes, parent_i, child):
    cb = _bbox_of(child)
    if not _rects_overlap(cb, _bbox_of(nodes[parent_i])):
        return False
    for i, other in enumerate(nodes):
        if i == parent_i:
            continue
        if _rects_overlap(cb, _bbox_of(other)):
            return False
    r0, c0, r1, c1 = _union_bbox(nodes + [child])
    return (r1 - r0 + 1) <= MAX_ASSEMBLY_DIM and (c1 - c0 + 1) <= MAX_ASSEMBLY_DIM


def _variant_metrics(nodes) -> dict[str, int]:
    horizontal_ports = 0
    vertical_ports = 0
    orthogonal_nodes = 0
    for node in nodes:
        orients = {SIDE_ORIENTATION[s] for s in node["sides"]}
        horizontal_ports += sum(SIDE_ORIENTATION[s] == "h" for s in node["sides"])
        vertical_ports += sum(SIDE_ORIENTATION[s] == "v" for s in node["sides"])
        if len(orients) == 2:
            orthogonal_nodes += 1
    return {
        "horizontal_ports": horizontal_ports,
        "vertical_ports": vertical_ports,
        "orthogonal_nodes": orthogonal_nodes,
    }


def _variant_structure_ok(nodes) -> bool:
    m = _variant_metrics(nodes)
    return (
        m["horizontal_ports"] > 0
        and m["vertical_ports"] > 0
        and m["orthogonal_nodes"] > 0
    )


def _grow_tree():
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    anchor = _new_node("anchor", bh + 2, bw + 2, 4, cap=random.choice(ANCHOR_CAP_CHOICES))
    anchor["block"] = (bh, bw)
    nodes = [anchor]
    n_pieces = random.randint(*PIECE_COUNT_RANGE)
    colors = random.sample(PIECE_COLORS, n_pieces)

    for i in range(n_pieces):
        placed = False
        dims = [
            (random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE))
            for _ in range(PIECE_DIM_CANDIDATES)
        ]
        random.shuffle(dims)
        for ph, pw in dims:
            parents = [j for j, n in enumerate(nodes) if len(n["sides"]) < n["cap"]]
            random.shuffle(parents)
            for parent_i in parents:
                parent = nodes[parent_i]
                sides = [s for s in SIDES if s not in parent["sides"]]
                random.shuffle(sides)
                for side in sides:
                    child = _new_node("piece", ph, pw, colors[i])
                    incoming = OPPOSITE[side]
                    _ensure_port(parent, side)
                    _ensure_port(child, incoming)
                    pr, pc = _global_port(parent, side)[0]
                    cr, cc = _port_cells(child, incoming)[0]
                    child["top"] = pr - cr
                    child["left"] = pc - cc
                    child["sides"].add(incoming)
                    if not _candidate_fits(nodes, parent_i, child):
                        continue
                    parent["sides"].add(side)
                    nodes.append(child)
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    if not _variant_structure_ok(nodes):
        return None
    return nodes


def _place_node_on_canvas(out, node, sr, sc, port_counts, red_counts):
    arr = node["arr"]
    top, left = node["top"] + sr, node["left"] + sc
    node["pos"] = (top, left)
    mask = arr != 0
    rows, cols = np.nonzero(mask)
    g_rows, g_cols = rows + top, cols + left
    vals = arr[rows, cols]
    cur_vals = out[g_rows, g_cols]
    conflicts = (cur_vals != 0) & ~((cur_vals == 2) & (vals == 2))
    if np.any(conflicts):
        return False

    local_ports = _extract_ports(arr)
    piece = {"ports": local_ports}
    overlap = _placement_overlap_score(piece, top, left, port_counts, red_counts)
    if overlap is None:
        return False

    for port in local_ports:
        gkey = _shift_port(port["cells"], top, left)
        port_counts[gkey] = port_counts.get(gkey, 0) + 1
        for cell in gkey:
            red_counts[cell] = red_counts.get(cell, 0) + 1

    paint_mask = cur_vals == 0
    out[g_rows[paint_mask], g_cols[paint_mask]] = vals[paint_mask]
    return True


def _assemble(nodes, h, w):
    for node in nodes:
        node["arr"] = _make_anchor(node) if node["kind"] == "anchor" else _make_piece(node)
        if node["arr"] is None:
            return None

    r0, c0, r1, c1 = _union_bbox(nodes)
    if (r1 - r0 + 1) > h or (c1 - c0 + 1) > w:
        return None
    sr = random.randint(-r0, h - 1 - r1)
    sc = random.randint(-c0, w - 1 - c1)

    out = np.zeros((h, w), dtype=int)
    all_port_counts = {}
    all_red_counts = {}
    for node in nodes:
        ok = _place_node_on_canvas(out, node, sr, sc, all_port_counts, all_red_counts)
        if not ok:
            return None

    total_ports = sum(len(_extract_ports(n["arr"])) for n in nodes)
    if len(all_port_counts) * 2 != total_ports:
        return None
    if any(v != 2 for v in all_port_counts.values()):
        return None
    if any(v != 2 for v in all_red_counts.values()):
        return None
    if len(components_mask(out != 0, connectivity=4)) != 1:
        return None
    return out


def _scatter(nodes, h, w):
    inp = np.zeros((h, w), dtype=int)
    anchor = nodes[0]
    paste(inp, anchor["arr"], *anchor["pos"], transparent=0)
    rest = nodes[1:]
    random.shuffle(rest)
    for node in rest:
        try:
            r, c = random_free_location(
                inp,
                node["arr"],
                transparent=0,
                padding=SCATTER_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, node["arr"], r, c, transparent=0)
    return inp


def _gen_attempt():
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    nodes = _grow_tree()
    if nodes is None:
        return None
    out = _assemble(nodes, h, w)
    if out is None:
        return None
    inp = _scatter(nodes, h, w)
    if inp is None:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cbebaa4b D8 input")
