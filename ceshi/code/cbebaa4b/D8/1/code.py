# Variant of cbebaa4b
# Dimension: D8 — 连接器像素的颜色与角色 (meta)
# Change: 连接器统一为红色，红色仅作为连接功能标记，不承载其他语义
#   → 连接器颜色与精灵主体色相同

# Design analysis for this D8 variant
# a) Concrete behavior:
#    - 原题里每个连接器像素都是红色(2)，solver 直接把颜色 2 当作“可配对且可重叠”的端口。
#    - 本变体里，端口不再统一染成红色，而是和该精灵主体同色；因此不同精灵之间不能再靠“同色重叠”完成连接，
#      否则不同主体色会在同一坐标冲突。
#    - 为保持任务核心仍是“围绕黄色锚点逐步装配散落精灵”，装配方式改成：两个 2 像素端口在输出里正向贴合相邻，
#      而不是重叠到同一坐标。这样连接器仍然是拼接口，但识别依据变成了“bbox 边缘上突出的 2 像素端口”。
#    - 具体小例子：原题中锚点顶部红色端口 [(5,8),(5,9)] 与子块底部红色端口重叠到同一对坐标；
#      现在则是子块底部同色端口放到 [(4,8),(4,9)]，与锚点顶部黄色端口在上下相邻的两对像素形成连接。
# b) Complexity calibration:
#    - 如果仍允许普通装饰条碰到 bbox 边界，那么“同色连接器”会和主体边界混在一起，视觉上看不出端口。
#    - 因此生成器保持原题的边框/条形主体，但把连接位置约束成“某条边上的 2 像素短段”；solver 只把这种短段识别为端口，
#      而把完整长边框视为普通主体。这样人眼仍可从“短双像素段”识别端口。
#    - 颜色仍控制在原题的少量 ARC 调色板内，非锚点颜色保持互异，避免相邻件同色导致视觉粘连。
# c) Hidden assumptions audit:
#    - 原 _extract_pieces() 用 arr == 2 提取端口；这里必须改成几何检测：检查 bbox 四边是否存在连续 2 像素边界短段。
#    - 原 _try_place() 允许 red-on-red overlap；这里必须改成完全禁止重叠，并单独验证“只有端口对端口可以发生邻接”。
#    - 原 dfs() 统计每个红色坐标被覆盖两次；这里改成维护“当前装配体上尚未配对的开放端口”。
#    - 原生成器把 red(2) 直接写进 piece/anchor；这里要改为写入主体色，同时把相对摆放从重叠改成相邻。
# d) Placement legality:
#    - 组装态中任意两个精灵不得重叠。
#    - 非父节点之间强制留至少 1 格净空，避免意外邻接；父子节点只允许在选中的 2 像素端口处正向贴合。
#    - solver 在尝试放置时逐像素检查：新块与现有装配体的接触只能来自该块端口的外侧，且外侧必须正好对应一个开放端口。
# e) Visual observability:
#    - 人类在样例中会直接看到：输入/输出都没有红色像素；每个可连接位置表现为“与主体同色的边缘双像素短段”。
#    - 输出里各块保持原色，不存在额外颜色映射；变化规律集中在“把散落的同色端口贴到一起”。
# f) Visual cleanliness:
#    - 端口仍是边缘上的短段，而普通边框通常是一整条长边，因此边缘信号依然干净；内部路径允许一定变化，但不会覆盖端口。
#    - 装配后对象只通过少量端口相连，不会出现大面积同区块互压或多层覆盖。
# g) Color inference:
#    - 这里没有颜色替换规则；颜色只保留对象身份。关键可推断点是“连接器失去统一红色，但仍由边缘位置承担连接角色”。

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


def _boundary_coords(shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out = set()
    for c in range(w):
        out.add((0, c))
        out.add((h - 1, c))
    for r in range(h):
        out.add((r, 0))
        out.add((r, w - 1))
    return out


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
    Reassemble scattered monochrome-tab sprites around the central yellow anchor by matching edge tabs geometrically.

    Input structure:
    - bg = black (0). Several colored sprites are scattered on the grid.
    - One anchor sprite contains a yellow (4) rectangle and its connector tabs are yellow too.
    - Every usable connector is a same-color 2-pixel edge segment; longer edge runs belong to the sprite body and are not connectors.

    Rule:
    - Keep the anchor fixed and translate the other sprites one by one.
    - A move is legal when one of the sprite's edge tabs lands immediately adjacent to an open tab on the growing assembly, with no overlap and no other accidental touch.
    - All tabs must be paired exactly once, yielding one fully assembled connected structure.

    Key insight:
    - Color no longer identifies connectors. The signal is geometric: connectors are the short exposed edge tabs, so the puzzle is solved by pairing compatible boundary protrusions rather than by chasing red pixels.
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
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
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
