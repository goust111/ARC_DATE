# Combo variant of cbebaa4b (index 20)
# combo_spec: D8/1+D12/0
# Name: 连接器同色于主体 + 所有精灵同色
#
# 设计分析
# a) 具体化与行为追踪
#    - 原题把连接器画成红色(2)，solver 直接把红色像素当作“可重叠端口”，装配时靠 red-on-red
#      重叠完成连接；不同主体颜色还能帮助区分不同精灵。
#    - 本融合变体同时取消这两条线索：连接器改为与主体同色，且所有非锚点精灵都用同一种主体色。
#      因而输出里不再出现红色像素，非锚点也不再靠颜色区分。
#    - 如果仍沿用原题的“端口重叠”语义，那么同色主体会在同坐标发生不可区分的融合，solver 也无法从颜色
#      中分离出连接器；因此装配语义必须同步改成“端口正向相邻贴合、不允许重叠”。
#    - 具体小例子：原题里锚点顶部两个红端口可能与子块底部两个红端口重叠到同一坐标；这里则改为锚点顶部
#      两个黄色端口与子块底部两个蓝色端口上下相邻。可连接性只由“边界上恰好长为 2 的短边段”决定。
# b) 复杂度校准
#    - 所有可移动精灵同色后，若允许任意边界装饰碰到外框，就会让端口和普通外边混在一起，视觉上无法推断。
#      因此生成器继续把真正端口限制为单边上恰好两个连续边界像素，并排除角点。
#    - 我保留原题的多样 bbox、高低宽不同、不同连接侧、不同内部路径与边框条纹，从而让几何仍足够区分。
#    - 不引入旋转/翻转等额外自由度，否则“失去颜色线索”与“变换自由度增加”叠加后会显著扩大歧义。
# c) 隐式假设审查
#    - 原 `_extract_pieces()` 假设 `arr == 2` 就能读出端口；这里改成几何识别：只把边界上连续 2 格、内侧
#      紧邻主体、且不占角的短边段当作端口。
#    - 原 `_try_place()` 默认允许 red-on-red 重叠；融合后必须完全禁止像素重叠，否则同色主体会产生伪合法
#      摆放。
#    - 原 DFS 统计“每个红坐标被覆盖两次”；这里改成维护“当前装配体仍然开放的端口集合”，并验证新块只在
#      这些端口外侧发生接触。
#    - 原 generator 可给每个 piece 独立着色；这里固定成单一 `COMMON_PIECE_COLOR`，并让锚点端口也与锚点
#      主体同色，避免退化回红色连接器版本。
# d) 放置合法性
#    - 组装态中任意两个精灵都不能重叠。
#    - 非父子节点在局部布局阶段保持至少 1 格净空，防止无意接触；真正的连接只允许发生在选定端口外侧。
#    - `_assemble()` 会再次验证每个已声明端口外侧都恰有占用像素，从而保证输出确实形成整棵连接树。
# e) 视觉可观察性
#    - 人类能直接看到：除了黄色锚点外，其余所有块都是同一种颜色，而且没有红色像素。
#    - 能用于配对的唯一稳定信号是“边缘上短短的 2 像素突出段”，而不是颜色值、生成顺序或隐含编号。
# f) 可视化整洁度
#    - 端口只出现在边界短段，普通边界装饰通常更长，因此信号较干净。
#    - 散点输入保留至少 2 格 Chebyshev 间隔，避免同色块在输入中粘连；输出中则只通过少量端口接触。
# g) 变化的可推断性
#    - 融合后的颜色变化规则非常少：非锚点统一成一个颜色，连接器与各自主体同色。颜色不再编码对象身份，
#      只是强调“颜色线索已失效，必须看几何端口”。
# h) 新 solver 假设与 generator 对应保证
#    - solver 假设每个有效端口都能被 `_extract_side_port()` 唯一识别；generator 通过 `_make_piece()` /
#      `_make_anchor()` 对每个已用 side 逐边校验，确保这一点成立。
#    - solver 假设除合法端口贴合外不存在其他 4 邻接；generator 在 `_candidate_fits()` 和 `_assemble()`
#      中都显式检查，避免意外接触。
#    - solver 假设锚点仍是唯一含黄色(4)的组件；generator 始终只在锚点上使用 4。

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, neighbors4, paste, random_free_location


SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}
OUTWARD = {"top": (-1, 0), "bottom": (1, 0), "left": (0, -1), "right": (0, 1)}


def _side_coords(shape: tuple[int, int], side: str, idxs: list[int]) -> list[tuple[int, int]]:
    h, w = shape
    if side == "top":
        return [(0, i) for i in idxs]
    if side == "bottom":
        return [(h - 1, i) for i in idxs]
    if side == "left":
        return [(i, 0) for i in idxs]
    return [(i, w - 1) for i in idxs]


def _extract_side_port(arr: np.ndarray, side: str, bg: int) -> dict | None:
    h, w = arr.shape
    if side == "top":
        idxs = np.flatnonzero(arr[0] != bg).tolist()
        if not idxs:
            return None
        if len(idxs) != 2 or idxs[1] != idxs[0] + 1 or idxs[0] == 0 or idxs[1] == w - 1:
            return None
        if np.any(arr[1, idxs] == bg):
            return None
    elif side == "bottom":
        idxs = np.flatnonzero(arr[h - 1] != bg).tolist()
        if not idxs:
            return None
        if len(idxs) != 2 or idxs[1] != idxs[0] + 1 or idxs[0] == 0 or idxs[1] == w - 1:
            return None
        if np.any(arr[h - 2, idxs] == bg):
            return None
    elif side == "left":
        idxs = np.flatnonzero(arr[:, 0] != bg).tolist()
        if not idxs:
            return None
        if len(idxs) != 2 or idxs[1] != idxs[0] + 1 or idxs[0] == 0 or idxs[1] == h - 1:
            return None
        if np.any(arr[idxs, 1] == bg):
            return None
    else:
        idxs = np.flatnonzero(arr[:, w - 1] != bg).tolist()
        if not idxs:
            return None
        if len(idxs) != 2 or idxs[1] != idxs[0] + 1 or idxs[0] == 0 or idxs[1] == h - 1:
            return None
        if np.any(arr[idxs, w - 2] == bg):
            return None
    return {"side": side, "cells": _side_coords((h, w), side, idxs)}


def _extract_ports(arr: np.ndarray, bg: int) -> list[dict]:
    ports = []
    for side in SIDES:
        port = _extract_side_port(arr, side, bg)
        if port is not None:
            ports.append(port)
    return ports


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        ports = _extract_ports(arr, bg)
        cells = [(int(r), int(c)) for r, c in np.argwhere(arr != bg)]
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "ports": ports,
                "shape": (int(arr.shape[0]), int(arr.shape[1])),
                "area": int(np.sum(arr != bg)),
                "cells": cells,
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
    if np.any(sub[mask] != bg):
        return None

    out = canvas.copy()
    out_sub = out[top : top + ph, left : left + pw]
    out_sub[mask] = arr[mask]
    out[top : top + ph, left : left + pw] = out_sub
    return out


def _port_key(side: str, cells: list[tuple[int, int]]) -> tuple[int, int, int, int, str]:
    (r0, c0), (r1, c1) = cells
    return (int(r0), int(c0), int(r1), int(c1), side)


def _global_port_cells(port: dict, top: int, left: int) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in port["cells"]]


def _analyze_placement(
    canvas: np.ndarray,
    piece: dict,
    top: int,
    left: int,
    open_ports: dict[tuple[int, int, int, int, str], dict],
    bg: int,
) -> tuple[list[tuple[int, int, int, int, str]], dict[tuple[int, int, int, int, str], dict]] | None:
    placed = _try_place(canvas, piece["arr"], top, left, bg)
    if placed is None:
        return None

    h, w = canvas.shape
    piece_globals = {(top + r, left + c) for r, c in piece["cells"]}
    matched_keys: list[tuple[int, int, int, int, str]] = []
    new_ports: dict[tuple[int, int, int, int, str], dict] = {}
    allowed_contacts: set[tuple[int, int, int, int]] = set()

    for port in piece["ports"]:
        cells = _global_port_cells(port, top, left)
        dr, dc = OUTWARD[port["side"]]
        nbrs = [(r + dr, c + dc) for r, c in cells]

        if any(not (0 <= nr < h and 0 <= nc < w) for nr, nc in nbrs):
            return None
        occupied = [canvas[nr, nc] != bg for nr, nc in nbrs]
        if any(occupied):
            if not all(occupied):
                return None
            key = _port_key(OPPOSITE[port["side"]], nbrs)
            if key not in open_ports:
                return None
            matched_keys.append(key)
            for (r, c), (nr, nc) in zip(cells, nbrs):
                allowed_contacts.add((r, c, nr, nc))
        else:
            key = _port_key(port["side"], cells)
            new_ports[key] = {"side": port["side"], "cells": cells}

    for r, c in piece_globals:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in piece_globals:
                continue
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if canvas[nr, nc] == bg:
                continue
            if (r, c, nr, nc) in allowed_contacts:
                continue
            return None

    if len(set(matched_keys)) != len(matched_keys):
        return None
    return matched_keys, new_ports


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered monochrome sprites around the fixed yellow anchor by matching same-color edge tabs geometrically.

    Input structure:
    - bg = black (0). Several disconnected sprites are scattered on the grid.
    - Exactly one sprite is the anchor: it contains a yellow (4) block, and its connector tabs are yellow too.
    - Every non-anchor sprite uses the same single body color, and every usable connector is a same-color 2-pixel edge segment.

    Rule:
    - Keep the anchor fixed and translate the other sprites only.
    - A move is valid when one of a sprite's edge tabs lands immediately outside an open tab on the growing assembly, with no overlap and no extra side contact anywhere else.
    - Every tab must be paired exactly once, producing one fully connected assembled structure.

    Key insight:
    - Color no longer distinguishes either pieces or connectors. The only reliable cue is geometric: connectors are the short exposed 2-pixel boundary tabs, so the puzzle is solved by pairing compatible edge positions rather than by chasing a special connector color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"

    anchors = [p for p in pieces if np.any(p["arr"] == 4)]
    assert len(anchors) == 1, "input must have exactly one yellow (4) anchor piece"
    anchor = anchors[0]
    rest = [p for p in pieces if p is not anchor]

    canvas = np.full((h, w), bg, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, bg)
    assert placed is not None, "anchor piece must fit on canvas at its original position"

    open_ports: dict[tuple[int, int, int, int, str], dict] = {}
    for port in anchor["ports"]:
        cells = _global_port_cells(port, ar0, ac0)
        open_ports[_port_key(port["side"], cells)] = {"side": port["side"], "cells": cells}

    solutions: list[np.ndarray] = []

    def dfs(cur: np.ndarray, remaining: list[dict], cur_open: dict[tuple[int, int, int, int, str], dict]) -> None:
        if len(solutions) >= 2:
            return
        if not remaining:
            if not cur_open:
                solutions.append(cur)
            return
        if not cur_open:
            return

        ports_left = sum(len(p["ports"]) for p in remaining)
        if len(cur_open) > ports_left:
            return

        best_moves = None
        for target_key, target in cur_open.items():
            moves = []
            desired = [
                (r + OUTWARD[target["side"]][0], c + OUTWARD[target["side"]][1])
                for r, c in target["cells"]
            ]
            for idx, piece in enumerate(remaining):
                for port in piece["ports"]:
                    if port["side"] != OPPOSITE[target["side"]]:
                        continue
                    top = desired[0][0] - port["cells"][0][0]
                    left = desired[0][1] - port["cells"][0][1]
                    analysis = _analyze_placement(cur, piece, top, left, cur_open, bg)
                    if analysis is None:
                        continue
                    matched_keys, new_ports = analysis
                    if target_key not in matched_keys:
                        continue
                    moves.append((idx, top, left, len(matched_keys), len(new_ports)))
            if not moves:
                return
            if best_moves is None or len(moves) < len(best_moves):
                best_moves = moves
                if len(best_moves) == 1:
                    break

        assert best_moves is not None, "there must be at least one feasible connector move"
        dedup: dict[tuple[int, int, int], tuple[int, int]] = {}
        for idx, top, left, matched_count, new_count in best_moves:
            key = (idx, top, left)
            score = (matched_count, -new_count)
            if key not in dedup or score > dedup[key]:
                dedup[key] = score
        moves2 = [
            (idx, top, left, score[0], -score[1])
            for (idx, top, left), score in dedup.items()
        ]
        moves2.sort(key=lambda x: (-x[3], x[4], -remaining[x[0]]["area"], x[1], x[2], x[0]))

        for idx, top, left, _matched_count, _new_count in moves2:
            piece = remaining[idx]
            next_remaining = remaining[:idx] + remaining[idx + 1 :]
            analysis = _analyze_placement(cur, piece, top, left, cur_open, bg)
            if analysis is None:
                continue
            matched_keys, new_ports = analysis
            nxt = _try_place(cur, piece["arr"], top, left, bg)
            if nxt is None:
                continue
            next_open = dict(cur_open)
            for key in matched_keys:
                next_open.pop(key, None)
            for key, port_obj in new_ports.items():
                if key in next_open:
                    raise AssertionError("new port cannot duplicate an existing open port")
                next_open[key] = port_obj
            dfs(nxt, next_remaining, next_open)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, open_ports)
    assert len(solutions) == 1, "jigsaw assembly must have exactly one valid solution"
    return solutions[0]


# --------------- generator (o5x) ---------------

# Canvas
GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 24

# Pieces
PIECE_COUNT_RANGE = (4, 6)
PIECE_H_RANGE = (4, 7)
PIECE_W_RANGE = (4, 8)
PIECE_DIM_CANDIDATES = 12
COMMON_PIECE_COLOR = 1
PIECE_CAP = 2
BAR_SKIP_PROB = 0.35

# Anchor
ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]
ANCHOR_CAP_CHOICES = [1, 2]

# Scatter
SCATTER_PADDING = 2

# Generation
GEN_RETRY_LIMIT = 300


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


def _expand_bbox(b, pad):
    return (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)


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
            arr[rr, cc] = color


def _interior_endpoint(shape, side, hub):
    h, w = shape
    hr, hc = hub
    if side == "top":
        return (1, min(max(hc, 1), w - 2))
    if side == "bottom":
        return (h - 2, min(max(hc, 1), w - 2))
    if side == "left":
        return (min(max(hr, 1), h - 2), 1)
    return (min(max(hr, 1), h - 2), w - 2)


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


def _make_piece(node):
    h, w = node["shape"]
    color = node["color"]
    used = set(node["sides"])
    arr = np.zeros((h, w), dtype=int)
    hub = (random.randint(1, h - 2), random.randint(1, w - 2))
    arr[hub] = color

    bar_sides = [s for s in SIDES if s not in used]
    if len(bar_sides) > 1 and random.random() < BAR_SKIP_PROB:
        bar_sides.remove(random.choice(bar_sides))
    for side in bar_sides:
        _paint_bar(arr, side, color, used)

    terminals = []
    for side in used:
        dr, dc = INWARD[side]
        for r, c in _port_cells(node, side):
            arr[r, c] = color
            rr, cc = r + dr, c + dc
            arr[rr, cc] = color
            terminals.append((rr, cc))

    for side in bar_sides:
        terminals.append(_interior_endpoint((h, w), side, hub))

    order = random.randint(0, 1)
    for t in terminals:
        _draw_path(arr, t, hub, color, order if random.random() < 0.5 else 1 - order)

    if len(components_mask(arr != 0, connectivity=4)) != 1:
        return None
    for side in used:
        if _extract_side_port(arr, side, 0) is None:
            return None
    for side in SIDES:
        if side not in used and _extract_side_port(arr, side, 0) is not None:
            return None
    return arr


def _make_anchor(node):
    bh, bw = node["block"]
    h, w = node["shape"]
    arr = np.zeros((h, w), dtype=int)
    arr[1 : 1 + bh, 1 : 1 + bw] = 4
    for side in node["sides"]:
        for r, c in _port_cells(node, side):
            arr[r, c] = 4
    for side in node["sides"]:
        if _extract_side_port(arr, side, 0) is None:
            return None
    return arr


def _candidate_fits(nodes, parent_i, child):
    cb = _bbox_of(child)
    if _rects_overlap(cb, _bbox_of(nodes[parent_i])):
        return False
    for i, other in enumerate(nodes):
        ob = _bbox_of(other)
        if i == parent_i:
            continue
        if _rects_overlap(_expand_bbox(cb, 1), ob):
            return False
    r0, c0, r1, c1 = _union_bbox(nodes + [child])
    return (r1 - r0 + 1) <= MAX_ASSEMBLY_DIM and (c1 - c0 + 1) <= MAX_ASSEMBLY_DIM


def _grow_tree():
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    anchor = _new_node("anchor", bh + 2, bw + 2, 4, cap=random.choice(ANCHOR_CAP_CHOICES))
    anchor["block"] = (bh, bw)
    nodes = [anchor]
    n_pieces = random.randint(*PIECE_COUNT_RANGE)

    for _ in range(n_pieces):
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
                    child = _new_node("piece", ph, pw, COMMON_PIECE_COLOR)
                    incoming = OPPOSITE[side]
                    _ensure_port(parent, side)
                    _ensure_port(child, incoming)
                    pr, pc = _global_port(parent, side)[0]
                    cr, cc = _port_cells(child, incoming)[0]
                    dr, dc = OUTWARD[side]
                    child["top"] = pr + dr - cr
                    child["left"] = pc + dc - cc
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
    return nodes


def _place_node_on_canvas(out, node, sr, sc):
    arr = node["arr"]
    top, left = node["top"] + sr, node["left"] + sc
    node["pos"] = (top, left)
    mask = arr != 0
    rows, cols = np.nonzero(mask)
    g_rows, g_cols = rows + top, cols + left
    if np.any(out[g_rows, g_cols] != 0):
        return False
    out[g_rows, g_cols] = arr[rows, cols]
    return True


def _tighten_node_sprite(node):
    arr = node["arr"]
    pts = np.argwhere(arr != 0)
    r0, c0, r1, c1 = bbox_coords(pts)
    if (r0, c0, r1, c1) == (0, 0, arr.shape[0] - 1, arr.shape[1] - 1):
        return
    node["arr"] = arr[r0 : r1 + 1, c0 : c1 + 1].copy()
    node["top"] += int(r0)
    node["left"] += int(c0)
    node["shape"] = (int(r1 - r0 + 1), int(c1 - c0 + 1))
    new_ports = {}
    for side, k in node["ports"].items():
        new_ports[side] = int(k - c0) if side in ("top", "bottom") else int(k - r0)
    node["ports"] = new_ports


def _assemble(nodes, h, w):
    for node in nodes:
        node["arr"] = _make_anchor(node) if node["kind"] == "anchor" else _make_piece(node)
        if node["arr"] is None:
            return None
        _tighten_node_sprite(node)

    r0, c0, r1, c1 = _union_bbox(nodes)
    if (r1 - r0 + 1) > h or (c1 - c0 + 1) > w:
        return None
    sr = random.randint(-r0, h - 1 - r1)
    sc = random.randint(-c0, w - 1 - c1)

    out = np.zeros((h, w), dtype=int)
    for node in nodes:
        if not _place_node_on_canvas(out, node, sr, sc):
            return None

    for node in nodes:
        top, left = node["pos"]
        for side in node["sides"]:
            cells = [(top + r, left + c) for r, c in _port_cells(node, side)]
            dr, dc = OUTWARD[side]
            nbrs = [(r + dr, c + dc) for r, c in cells]
            if any(not (0 <= nr < h and 0 <= nc < w) for nr, nc in nbrs):
                return None
            if any(out[nr, nc] == 0 for nr, nc in nbrs):
                return None
    if len(components_mask(out != 0, connectivity=4)) != 1:
        return None
    if np.any(out == 2):
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
    if np.array_equal(solved, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cbebaa4b input")
