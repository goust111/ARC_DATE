# Variant of cbebaa4b
# Dimension: D2 — 源对象的结构与属性 (src)
# Change: 散布在背景上的彩色精灵，每个精灵包含主体颜色像素和红色连接器像素，形状不规则... → 精灵分为两类（主精灵和辅助精灵）

"""
design_analysis:
  variant_core:
    original: 所有非锚点精灵都是同一类，只要红色连接器能配对，就可以直接接到当前连通体上的任何未配对红端口。
    new: 非锚点精灵按主体颜色分成两类。主精灵（蓝/天蓝）只能直接接到黄色锚点；辅助精灵（粉/橙）只能接到主精灵暴露出来的端口上。
    preserved:
      - 仍然是 output-first 生成：先造完整拼装结构，再把各精灵打散得到输入。
      - 仍然只允许平移，不允许旋转/翻转。
      - 仍然通过红色双像素连接器进行重叠拼接。
  concrete_visualization:
    anchor:
      - 一个黄色 3x3 或 4x5 实心块，边缘带若干个红色双像素端口。
    main_sprites:
      - 主体色只能来自 {1, 8}，在输出中形成紧贴锚点的“内环”。
      - 每个主精灵至少有 2 个端口：1 个接锚点，1-2 个留给辅助精灵。
    aux_sprites:
      - 主体色只能来自 {6, 7}，在输出中形成外层“叶子”。
      - 每个辅助精灵只有 1 个端口，因此只能作为末端挂到主精灵上。
  behavior_trace:
    original_flow:
      - 原 solver 选一个未配对红像素，然后尝试把任意剩余碎片平移到那里，只要整体红像素两两配对即可。
    new_flow:
      - 先把锚点的开放端口全部视为“只接受主精灵”的槽位。
      - 主精灵放下后，它剩余的端口再变成“只接受辅助精灵”的槽位。
      - 辅助精灵不会再生成新的可接槽位，因此装配深度固定为两层。
    key_difference:
      - 旧逻辑在几何上不区分“接锚点的碎片”和“挂在主碎片上的碎片”；新逻辑把这个差异外显为颜色组，并在搜索时显式分阶段处理。
  complexity_calibration:
    - 每个样例固定同时包含主精灵和辅助精灵，且辅助精灵数 >= 2，避免退化回“所有碎片都直接接锚点”的星形原题子情况。
    - 主/辅颜色只用两组固定调色板：主精灵 {1,8}，辅助精灵 {6,7}。颜色种类少，便于 2-4 个样例中看出“颜色决定层级”。
    - 主精灵一定至少有 2 个端口、辅助精灵一定只有 1 个端口。这个结构差异不是规则本体，但能帮助人类更快确认两层层级。
  implicit_assumption_review:
    - 原 _try_place 允许 arr == sub 的同色重叠，这在原题“每个碎片颜色唯一”时基本无害；本变体允许同组颜色重复后，这会错误地把两个同色精灵叠在一起，所以这里必须改成“只有红对红才允许重叠”。
    - 原 solver 以单个红像素为单位回溯；本变体把连接器稳定限制为边界上的 2 像素端口，因此 solver 改为按“端口对”匹配，避免把主/辅阶段信息打散到单像素层面。
    - 新 solver 默认：主精灵颜色只来自 MAIN_COLORS、辅助精灵颜色只来自 AUX_COLORS、辅助精灵只有一个端口。generator 会显式保证这些前提成立。
  placement_legality:
    - 生成 assembled output 时，每个新节点只允许与父节点 bbox 重叠，其余节点 bbox 一律不能相交。
    - 画到整张网格时仍只允许红像素彼此重叠，任何非红覆盖都判非法。
    - 打散输入时沿用随机空位放置，保持对象之间至少 2 格的切比雪夫间距。
  visual_observability:
    - 人类在输出里能直接看到：蓝/天蓝都围着黄色锚点，粉/橙都挂在这些蓝/天蓝结构的外侧。
    - 颜色分组是可见属性，不依赖颜色 id 的数值大小，也不使用任何不可见排序。
  cleanliness:
    - 仍保持一块锚点、若干离散精灵、黑色背景，重叠只发生在最终输出中的红端口处，不会出现多层非红覆盖。
  anti_degeneration_check:
    - generator 额外运行一个“忽略主/辅类别的 relaxed solver”。
    - 只有当 relaxed solver 出现多解或给出不同拼装时，样例才接受；这样可量化地确认“颜色分组约束”确实在起作用，而不是仅仅生成了原题也能唯一解释的特例。
"""

from __future__ import annotations

import random
from itertools import permutations

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


MAIN_COLORS = (1, 8)
AUX_COLORS = (6, 7)
ALL_PIECE_COLORS = MAIN_COLORS + AUX_COLORS


def _body_colors(arr: np.ndarray, bg: int) -> list[int]:
    return sorted({int(v) for v in arr.flatten() if int(v) not in {bg, 2}})


def _extract_ports(arr: np.ndarray) -> list[dict]:
    ports = []
    for comp in components_mask(arr == 2, connectivity=4):
        cells = sorted((int(r), int(c)) for r, c in comp)
        if len(cells) != 2:
            raise AssertionError("each connector must be a 2-cell red port")
        (r0, c0), (r1, c1) = cells
        if abs(r0 - r1) + abs(c0 - c1) != 1:
            raise AssertionError("red port cells must be adjacent")
        if r0 == r1:
            side = "top" if r0 == 0 else "bottom" if r0 == arr.shape[0] - 1 else None
        else:
            side = "left" if c0 == 0 else "right" if c0 == arr.shape[1] - 1 else None
        if side is None:
            raise AssertionError("red ports must lie on the piece boundary")
        ports.append({"cells": tuple(cells), "side": side})
    ports.sort(key=lambda p: (p["side"], p["cells"]))
    return ports


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        colors = _body_colors(arr, bg)
        ports = _extract_ports(arr)
        if colors == [4]:
            role = "anchor"
            color = 4
        else:
            assert len(colors) == 1, "non-anchor pieces must have a single body color"
            color = int(colors[0])
            if color in MAIN_COLORS:
                role = "main"
            elif color in AUX_COLORS:
                role = "aux"
            else:
                raise AssertionError("piece color must belong to one of the two visible groups")
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "ports": ports,
                "shape": (int(arr.shape[0]), int(arr.shape[1])),
                "area": int(np.sum(arr != bg)),
                "role": role,
                "color": int(color),
            }
        )
    return pieces


def _port_id(cells: frozenset[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(cells))


def _translated_port(port: dict, top: int, left: int) -> frozenset[tuple[int, int]]:
    return frozenset((int(top + r), int(left + c)) for r, c in port["cells"])


def _align_port(target: frozenset[tuple[int, int]], local_cells: tuple[tuple[int, int], ...]) -> tuple[int, int] | None:
    local = list(local_cells)
    target_cells = list(target)
    for perm in permutations(target_cells):
        top = int(perm[0][0] - local[0][0])
        left = int(perm[0][1] - local[0][1])
        moved = frozenset((top + r, left + c) for r, c in local)
        if moved == target:
            return top, left
    return None


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

    conflict = mask & (sub != bg) & ~((arr == 2) & (sub == 2))
    if np.any(conflict):
        return None

    out = canvas.copy()
    out_sub = out[top : top + ph, left : left + pw]
    out_sub[mask] = arr[mask]
    out[top : top + ph, left : left + pw] = out_sub
    return out


def _candidate_moves(
    canvas: np.ndarray,
    remaining: list[dict],
    open_ports: dict[tuple[tuple[int, int], ...], dict],
    occupied_reds: set[tuple[int, int]],
    bg: int,
    enforce_groups: bool,
) -> tuple[tuple[tuple[int, int], ...], list[tuple[int, int, int, int]]] | None:
    best_target = None
    best_moves = None
    for target_id, target in open_ports.items():
        moves = []
        for idx, piece in enumerate(remaining):
            if enforce_groups and target["accept"] is not None and piece["role"] != target["accept"]:
                continue
            for port_idx, port in enumerate(piece["ports"]):
                pos = _align_port(target["cells"], port["cells"])
                if pos is None:
                    continue
                top, left = pos
                bad = False
                used = 0
                for other_port in piece["ports"]:
                    moved = _translated_port(other_port, top, left)
                    if moved == target["cells"]:
                        used += 1
                        continue
                    if any(cell in occupied_reds for cell in moved):
                        bad = True
                        break
                if bad or used != 1:
                    continue
                nxt = _try_place(canvas, piece["arr"], top, left, bg)
                if nxt is None:
                    continue
                moves.append((idx, port_idx, top, left))
        if not moves:
            return None
        moves.sort(key=lambda x: (-remaining[x[0]]["area"], x[2], x[3], x[1], x[0]))
        if best_moves is None or len(moves) < len(best_moves):
            best_target = target_id
            best_moves = moves
            if len(best_moves) == 1:
                break
    if best_target is None or best_moves is None:
        return None
    return best_target, best_moves


def _search_solution(g: np.ndarray, enforce_groups: bool, max_solutions: int = 2) -> list[np.ndarray]:
    g = np.asarray(g, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"

    anchors = [p for p in pieces if p["role"] == "anchor"]
    assert len(anchors) == 1, "input must have exactly one yellow anchor piece"
    anchor = anchors[0]
    mains = [p for p in pieces if p["role"] == "main"]
    auxes = [p for p in pieces if p["role"] == "aux"]
    if enforce_groups:
        assert mains and auxes, "variant inputs must contain both main and auxiliary sprites"
        assert all(len(p["ports"]) >= 2 for p in mains), "main sprites must expose at least two ports"
        assert all(len(p["ports"]) == 1 for p in auxes), "auxiliary sprites must be single-port leaves"

    canvas = np.full((h, w), bg, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, bg)
    assert placed is not None, "anchor must fit at its original location"

    remaining = mains + auxes if enforce_groups else [p for p in pieces if p is not anchor]
    occupied_reds: set[tuple[int, int]] = set()
    open_ports: dict[tuple[tuple[int, int], ...], dict] = {}
    for port in anchor["ports"]:
        cells = _translated_port(port, ar0, ac0)
        pid = _port_id(cells)
        occupied_reds |= set(cells)
        open_ports[pid] = {"cells": cells, "accept": "main" if enforce_groups else None}

    solutions: list[np.ndarray] = []

    def dfs(cur: np.ndarray, rem: list[dict], ports: dict, reds: set[tuple[int, int]]) -> None:
        if len(solutions) >= max_solutions:
            return
        if not rem:
            if not ports:
                solutions.append(cur)
            return
        if not ports:
            return

        step = _candidate_moves(cur, rem, ports, reds, bg, enforce_groups)
        if step is None:
            return
        target_id, moves = step
        target = ports[target_id]
        for idx, port_idx, top, left in moves:
            piece = rem[idx]
            nxt = _try_place(cur, piece["arr"], top, left, bg)
            if nxt is None:
                continue

            next_ports = dict(ports)
            del next_ports[target_id]
            next_reds = set(reds)
            for j, port in enumerate(piece["ports"]):
                moved = _translated_port(port, top, left)
                next_reds |= set(moved)
                if j == port_idx:
                    continue
                pid = _port_id(moved)
                accept = None
                if enforce_groups and piece["role"] == "main":
                    accept = "aux"
                next_ports[pid] = {"cells": moved, "accept": accept}

            next_rem = rem[:idx] + rem[idx + 1 :]
            dfs(nxt, next_rem, next_ports, next_reds)
            if len(solutions) >= max_solutions:
                return

    dfs(placed, remaining, open_ports, occupied_reds)
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble a two-tier connector puzzle where only one color group may touch the anchor directly.

    Input structure:
    - bg = black (0). A yellow anchor sprite stays fixed while several colored sprites are scattered around it.
    - Blue / azure sprites are the main sprites; pink / orange sprites are auxiliary sprites.
    - Every sprite carries one or more red (2) two-pixel boundary connectors.

    Rule:
    - First attach each main sprite to a red connector on the yellow anchor.
    - Any leftover connectors exposed by those placed main sprites can then receive auxiliary sprites.
    - Only red pixels may overlap, and every connector pair must be consumed exactly once in the final assembly.

    Key insight:
    - The red pixels still define the geometric joins, but color now reveals a hierarchy: cool-colored pieces build the inner ring around the anchor, and warm-colored pieces can only hang off that ring.
    """
    sols = _search_solution(np.asarray(grid, dtype=int), enforce_groups=True, max_solutions=2)
    assert len(sols) == 1, "hierarchical assembly must have exactly one valid solution"
    return sols[0]


# --------------- generator ---------------

# Canvas
GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 24

# Pieces
PIECE_COUNT_RANGE = (4, 6)
PIECE_H_RANGE = (4, 7)
PIECE_W_RANGE = (4, 8)
PIECE_DIM_CANDIDATES = 12
BAR_SKIP_PROB = 0.35

# Anchor
ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]

# Scatter / retries
SCATTER_PADDING = 2
GEN_RETRY_LIMIT = 400

# Geometry
SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}


def _new_node(kind: str, h: int, w: int, color: int, top: int = 0, left: int = 0, cap: int = 1) -> dict:
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


def _ensure_port(node: dict, side: str) -> None:
    if side in node["ports"]:
        return
    h, w = node["shape"]
    span = h if side in ("left", "right") else w
    node["ports"][side] = random.randint(1, span - 3)


def _port_cells(node: dict, side: str) -> list[tuple[int, int]]:
    h, w = node["shape"]
    k = node["ports"][side]
    if side == "top":
        return [(0, k), (0, k + 1)]
    if side == "bottom":
        return [(h - 1, k), (h - 1, k + 1)]
    if side == "left":
        return [(k, 0), (k + 1, 0)]
    return [(k, w - 1), (k + 1, w - 1)]


def _global_port(node: dict, side: str) -> list[tuple[int, int]]:
    tr, tc = node["top"], node["left"]
    return [(tr + r, tc + c) for r, c in _port_cells(node, side)]


def _bbox_of(node: dict) -> tuple[int, int, int, int]:
    h, w = node["shape"]
    return (node["top"], node["left"], node["top"] + h - 1, node["left"] + w - 1)


def _rects_overlap(a, b) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _union_bbox(nodes: list[dict]) -> tuple[int, int, int, int]:
    boxes = [_bbox_of(n) for n in nodes]
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _draw_path(arr: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: int, order: int) -> None:
    r, c = start
    er, ec = end
    axes = [(r, er, c, True), (c, ec, er, False)] if order == 0 else [(c, ec, r, False), (r, er, ec, True)]
    for a_start, a_end, fixed, is_row in axes:
        step = 1 if a_end >= a_start else -1
        for v in range(a_start, a_end + step, step):
            rr, cc = (v, fixed) if is_row else (fixed, v)
            if arr[rr, cc] != 2:
                arr[rr, cc] = color


def _paint_bar(arr: np.ndarray, side: str, color: int, used: set[str]) -> None:
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


def _bar_endpoint(shape: tuple[int, int], side: str, hub: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    hr, hc = hub
    if side in ("top", "bottom"):
        row = 0 if side == "top" else h - 1
        return (row, min(max(hc, 1), w - 2))
    col = 0 if side == "left" else w - 1
    return (min(max(hr, 1), h - 2), col)


def _make_piece(node: dict) -> np.ndarray | None:
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


def _make_anchor(node: dict) -> np.ndarray:
    bh, bw = node["block"]
    h, w = node["shape"]
    arr = np.zeros((h, w), dtype=int)
    arr[1 : 1 + bh, 1 : 1 + bw] = 4
    for side in node["sides"]:
        for r, c in _port_cells(node, side):
            arr[r, c] = 2
    return arr


def _candidate_fits(nodes: list[dict], parent_i: int, child: dict) -> bool:
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


def _assign_group_colors(count: int, palette: tuple[int, ...]) -> list[int]:
    colors = list(palette)
    while len(colors) < count:
        colors.append(random.choice(palette))
    random.shuffle(colors)
    return colors[:count]


def _choose_piece_counts() -> tuple[int, int]:
    total = random.randint(*PIECE_COUNT_RANGE)
    if total == 4:
        n_main = 2
    elif total == 5:
        n_main = 2
    else:
        n_main = random.choice([2, 3])
    n_aux = total - n_main
    return n_main, n_aux


def _distribute_aux(n_main: int, n_aux: int) -> list[int]:
    counts = [1] * n_main
    extra = n_aux - n_main
    while extra > 0:
        choices = [i for i, v in enumerate(counts) if v < 2]
        if not choices:
            raise AssertionError("aux distribution exceeded per-main cap")
        counts[random.choice(choices)] += 1
        extra -= 1
    random.shuffle(counts)
    return counts


def _attach_child(nodes: list[dict], parent_i: int, child: dict) -> bool:
    parent = nodes[parent_i]
    sides = [s for s in SIDES if s not in parent["sides"]]
    random.shuffle(sides)
    for side in sides:
        incoming = OPPOSITE[side]
        _ensure_port(parent, side)
        _ensure_port(child, incoming)
        pr, pc = _global_port(parent, side)[0]
        cr, cc = _port_cells(child, incoming)[0]
        child["top"] = pr - cr
        child["left"] = pc - cc
        child["sides"].add(incoming)
        if not _candidate_fits(nodes, parent_i, child):
            child["sides"].remove(incoming)
            continue
        parent["sides"].add(side)
        return True
    return False


def _grow_hierarchy() -> list[dict] | None:
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    n_main, n_aux = _choose_piece_counts()
    aux_counts = _distribute_aux(n_main, n_aux)

    anchor = _new_node("anchor", bh + 2, bw + 2, 4, cap=n_main)
    anchor["block"] = (bh, bw)
    nodes = [anchor]
    main_nodes: list[dict] = []
    main_colors = _assign_group_colors(n_main, MAIN_COLORS)
    aux_colors = _assign_group_colors(n_aux, AUX_COLORS)

    for i in range(n_main):
        placed = False
        dims = [(random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE)) for _ in range(PIECE_DIM_CANDIDATES)]
        random.shuffle(dims)
        for ph, pw in dims:
            child = _new_node("main", ph, pw, main_colors[i], cap=1 + aux_counts[i])
            if _attach_child(nodes, 0, child):
                nodes.append(child)
                main_nodes.append(child)
                placed = True
                break
        if not placed:
            return None

    aux_idx = 0
    for main_i, aux_count in enumerate(aux_counts, start=1):
        for _ in range(aux_count):
            placed = False
            dims = [(random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE)) for _ in range(PIECE_DIM_CANDIDATES)]
            random.shuffle(dims)
            for ph, pw in dims:
                child = _new_node("aux", ph, pw, aux_colors[aux_idx], cap=1)
                if _attach_child(nodes, main_i, child):
                    nodes.append(child)
                    aux_idx += 1
                    placed = True
                    break
            if not placed:
                return None

    if len(nodes) != 1 + n_main + n_aux:
        return None
    return nodes


def _place_node_on_canvas(out: np.ndarray, node: dict, sr: int, sc: int) -> dict | None:
    arr = node["arr"]
    top, left = node["top"] + sr, node["left"] + sc
    node["pos"] = (top, left)
    red_counts = {}
    mask = arr != 0
    rows, cols = np.nonzero(mask)
    g_rows, g_cols = rows + top, cols + left
    vals = arr[rows, cols]
    cur_vals = out[g_rows, g_cols]
    conflicts = (cur_vals != 0) & ~((cur_vals == 2) & (vals == 2))
    if np.any(conflicts):
        return None
    red_mask = vals == 2
    for gr, gc in zip(g_rows[red_mask], g_cols[red_mask]):
        key = (int(gr), int(gc))
        red_counts[key] = red_counts.get(key, 0) + 1
    paint_mask = cur_vals == 0
    out[g_rows[paint_mask], g_cols[paint_mask]] = vals[paint_mask]
    return red_counts


def _assemble(nodes: list[dict], h: int, w: int) -> np.ndarray | None:
    for node in nodes:
        if node["kind"] == "anchor":
            node["arr"] = _make_anchor(node)
        else:
            node["arr"] = _make_piece(node)
        if node["arr"] is None:
            return None

    r0, c0, r1, c1 = _union_bbox(nodes)
    if (r1 - r0 + 1) > h or (c1 - c0 + 1) > w:
        return None
    sr = random.randint(-r0, h - 1 - r1)
    sc = random.randint(-c0, w - 1 - c1)

    out = np.zeros((h, w), dtype=int)
    all_red_counts = {}
    for node in nodes:
        rc = _place_node_on_canvas(out, node, sr, sc)
        if rc is None:
            return None
        for k, v in rc.items():
            all_red_counts[k] = all_red_counts.get(k, 0) + v

    total_ports = sum(int(np.sum(n["arr"] == 2)) for n in nodes)
    if len(all_red_counts) * 2 != total_ports:
        return None
    if any(v != 2 for v in all_red_counts.values()):
        return None
    if len(components_mask(out != 0, connectivity=4)) != 1:
        return None
    return out


def _scatter(nodes: list[dict], h: int, w: int) -> np.ndarray | None:
    inp = np.zeros((h, w), dtype=int)
    anchor = nodes[0]
    paste(inp, anchor["arr"], *anchor["pos"], transparent=0)
    rest = nodes[1:]
    random.shuffle(rest)
    for node in rest:
        try:
            r, c = random_free_location(inp, node["arr"], transparent=0, padding=SCATTER_PADDING, padding_connectivity=8)
        except ValueError:
            return None
        paste(inp, node["arr"], r, c, transparent=0)
    return inp


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    nodes = _grow_hierarchy()
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

    relaxed = _search_solution(inp, enforce_groups=False, max_solutions=2)
    if len(relaxed) == 1 and np.array_equal(relaxed[0], out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hierarchical cbebaa4b input")
