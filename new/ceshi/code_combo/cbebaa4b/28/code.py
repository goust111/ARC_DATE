# Combo variant of cbebaa4b (index 28)
# combo_spec: D2/2+D3/0+D11/2
# Name: 两类精灵 + 多锚点 + 干扰精灵

from __future__ import annotations

import random
from itertools import permutations

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


"""
design_analysis:
  fused_rule:
    original:
      - 只有一个黄色锚点，所有非锚点精灵都属于同一类，并且所有组件都必须参与最终拼装。
    new:
      - 同时存在 2-3 个固定黄色锚点。
      - 非锚点精灵按主体色分成主/辅两类：主精灵只能直接接黄色锚点，辅助精灵只能接在同组主精灵暴露出的端口上。
      - 输入里额外混入 1-2 个带红像素的干扰精灵；它们不会出现在输出里。
  concrete_visualization:
    anchors:
      - 三种锚点原型固定绑定三套“锚点形状 -> 主色 -> 辅色”规则：
        square 3x3 -> blue(1) / pink(6)
        wide 4x5 -> teal(8) / orange(7)
        tall 5x4 -> maroon(9) / green(3)
      - 每个样例选其中 2-3 种，因此解题者能同时看见“哪个颜色对属于哪个锚点形状”和“主色包内层、辅色挂外层”。
    real_group:
      - 每个锚点簇至少有 1 个主精灵和 1 个辅助精灵；部分簇再多 1 个辅助叶子。
      - 主精灵至少两个双像素红端口：一个接锚点，其余端口只给辅助精灵使用。
      - 辅助精灵恰好一个双像素红端口，因此只能做叶子。
    distractors:
      - 干扰精灵使用与真精灵相同的颜色池，但红像素故意做成内部/破碎布局，不构成合法边界双像素端口。
      - 因此它们看起来像“也带连接器的碎片”，但无法参与任何合法拼装。
  behavior_trace:
    small_example:
      - 原题里：唯一黄色锚点的一个未配对红端口，可以尝试让任意剩余真碎片去覆盖。
      - 融合后：若看到 4x5 黄色宽锚点的开放端口，它只接受 teal(8) 主精灵；该主精灵放下后，暴露出的端口又只接受 orange(7) 辅助精灵。
      - 同时输入中可能还有一个 pink(6) 干扰块带红像素，但这些红像素不在边界双像素端口上，所以不会进入合法候选。
    difference:
      - 决策从“哪块能几何对齐”变成“三重分类 + 分配”：
        1) 先排除无效干扰精灵。
        2) 再判断剩余真精灵是主还是辅。
        3) 最后依据锚点形状绑定的颜色对，把主/辅精灵送去对应锚点簇。
  complexity_calibration:
    - 锚点数控制在 2-3，真实精灵总数控制在 5-6，避免搜索爆炸。
    - 每个锚点簇只用固定的一对主/辅颜色，既减少记忆负担，也让少量样例中能反复看到同一规则。
    - 至少一个锚点簇拥有两个辅助叶子，避免退化成“每个锚点只有一根单边连接”的过简情况。
  implicit_assumption_review:
    - 原版 `len(anchors) == 1` 失效：这里必须接收多个锚点，并把它们全部作为初始 frontier。
    - 原版假设所有组件都参与拼装：这里 DFS 必须允许剩余组件被视为干扰项，因此终止条件改成“没有开放端口”而不是“remaining 为空”。
    - 原版 `_try_place` 允许同色非红重叠，这在这里会把同组重复颜色的碎片错误叠在一起，因此合法重叠收紧为仅允许 red-on-red。
    - 原版按单个红像素计数；融合后若继续这样做，会把“主端口只能接辅端口”“锚点只接主端口”的信息打散，所以 solver 改成按 2 像素端口匹配。
    - 新 solver 额外假设：
      - 真精灵的红连接器一定是边界上的 2 像素端口。
      - 主色/辅色与锚点形状的绑定关系固定。
      - 辅助精灵都是单端口叶子。
      - generator 会显式保证这些前提。
  placement_legality:
    - generator 继续采用 output-first：先在局部坐标造好每个锚点簇，再把整个簇贴到全局输出。
    - 每个新节点在局部组装时只允许与父节点 bbox 相交，其余节点 bbox 不能重叠。
    - 贴到全局输出时，各锚点簇之间保留至少 2 格间距；输入散布时所有组件继续保持 Chebyshev padding=2。
    - 干扰精灵除了不越界/不重叠外，还要通过“对所有真组件逐个尝试单红对齐都失败”的检查，确保它们真的不能附着。
  observability_and_cleanliness:
    - 人类可直接看到三种可见线索共同作用：
      - 黄色锚点不止一个。
      - 同一锚点形状对应固定主/辅颜色对。
      - 干扰精灵在输入里出现但在输出里消失。
    - 输出仍然是多个分离的干净连通簇，重叠只发生在红端口，不会产生不可读的多层覆盖。
  anti_degeneration_check:
    - generator 除了验证 `main(input) == output`，还额外运行两个 relaxed solver：
      - 保留主/辅层级但忽略锚点分组。
      - 同时忽略主/辅层级和锚点分组。
    - 只有当这两个 relaxed 版本都不能唯一复原同一输出时，样例才接受；这样可以量化确认“正确锚点分配”和“主辅层级”至少有一个在真正起作用，而不是退化成纯几何原题。
"""


GROUP_SPECS = (
    {"name": "square", "block": (3, 3), "main": 1, "aux": 6},
    {"name": "wide", "block": (4, 5), "main": 8, "aux": 7},
    {"name": "tall", "block": (5, 4), "main": 9, "aux": 3},
)
BLOCK_TO_GROUP = {spec["block"]: idx for idx, spec in enumerate(GROUP_SPECS)}
MAIN_COLOR_TO_GROUP = {spec["main"]: idx for idx, spec in enumerate(GROUP_SPECS)}
AUX_COLOR_TO_GROUP = {spec["aux"]: idx for idx, spec in enumerate(GROUP_SPECS)}
ROLE_COLORS = tuple(spec["main"] for spec in GROUP_SPECS) + tuple(spec["aux"] for spec in GROUP_SPECS)


def _body_colors(arr: np.ndarray, bg: int) -> list[int]:
    return sorted({int(v) for v in arr.flatten() if int(v) not in {bg, 2}})


def _extract_ports(arr: np.ndarray) -> list[dict] | None:
    ports = []
    red_mask = arr == 2
    if not np.any(red_mask):
        return []
    for comp in components_mask(red_mask, connectivity=4):
        cells = sorted((int(r), int(c)) for r, c in comp)
        if len(cells) != 2:
            return None
        (r0, c0), (r1, c1) = cells
        if abs(r0 - r1) + abs(c0 - c1) != 1:
            return None
        if r0 == r1:
            side = "top" if r0 == 0 else "bottom" if r0 == arr.shape[0] - 1 else None
        else:
            side = "left" if c0 == 0 else "right" if c0 == arr.shape[1] - 1 else None
        if side is None:
            return None
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
        role = "decoy"
        group = None
        if colors == [4] and ports is not None:
            yellow_pts = np.argwhere(arr == 4)
            yr0, yc0, yr1, yc1 = bbox_coords(yellow_pts)
            block = (int(yr1 - yr0 + 1), int(yc1 - yc0 + 1))
            group = BLOCK_TO_GROUP.get(block)
            if group is not None:
                role = "anchor"
        elif len(colors) == 1 and ports is not None:
            color = int(colors[0])
            if color in MAIN_COLOR_TO_GROUP and len(ports) >= 2:
                role = "main"
                group = MAIN_COLOR_TO_GROUP[color]
            elif color in AUX_COLOR_TO_GROUP and len(ports) == 1:
                role = "aux"
                group = AUX_COLOR_TO_GROUP[color]

        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "ports": ports or [],
                "shape": (int(arr.shape[0]), int(arr.shape[1])),
                "area": int(np.sum(arr != bg)),
                "role": role,
                "group": group,
            }
        )
    return pieces


def _port_id(cells: frozenset[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(cells))


def _translated_port(port: dict, top: int, left: int) -> frozenset[tuple[int, int]]:
    return frozenset((int(top + r), int(left + c)) for r, c in port["cells"])


def _align_port(
    target: frozenset[tuple[int, int]], local_cells: tuple[tuple[int, int], ...]
) -> tuple[int, int] | None:
    local = list(local_cells)
    for perm in permutations(list(target)):
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
    enforce_role: bool,
    enforce_group: bool,
) -> tuple[tuple[tuple[int, int], ...], list[tuple[int, int, int, int]]] | None:
    best_target = None
    best_moves = None
    for target_id, target in open_ports.items():
        moves = []
        for idx, piece in enumerate(remaining):
            if enforce_role and target["accept"] is not None and piece["role"] != target["accept"]:
                continue
            if enforce_group and target["group"] is not None and piece["group"] != target["group"]:
                continue
            for port_idx, port in enumerate(piece["ports"]):
                pos = _align_port(target["cells"], port["cells"])
                if pos is None:
                    continue
                top, left = pos
                used = 0
                bad = False
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
        moves.sort(
            key=lambda x: (
                -len(remaining[x[0]]["ports"]),
                -remaining[x[0]]["area"],
                x[2],
                x[3],
                x[1],
                x[0],
            )
        )
        if best_moves is None or len(moves) < len(best_moves):
            best_target = target_id
            best_moves = moves
            if len(best_moves) == 1:
                break
    if best_target is None or best_moves is None:
        return None
    return best_target, best_moves


def _search_solution(
    g: np.ndarray,
    *,
    enforce_role: bool,
    enforce_group: bool,
    max_solutions: int = 2,
) -> list[np.ndarray]:
    g = np.asarray(g, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    anchors = [p for p in pieces if p["role"] == "anchor"]
    assert 2 <= len(anchors) <= 3, "input must have 2 or 3 valid yellow anchors"

    real_pieces = [p for p in pieces if p["role"] in {"main", "aux"}]
    mains = [p for p in real_pieces if p["role"] == "main"]
    auxes = [p for p in real_pieces if p["role"] == "aux"]
    if enforce_role:
        assert mains and auxes, "variant inputs must contain both main and auxiliary sprites"
    assert real_pieces, "input must contain valid non-anchor pieces"

    canvas = np.full((h, w), bg, dtype=int)
    placed = canvas
    open_ports: dict[tuple[tuple[int, int], ...], dict] = {}
    occupied_reds: set[tuple[int, int]] = set()
    for anchor in anchors:
        ar0, ac0, _, _ = anchor["bbox"]
        placed2 = _try_place(placed, anchor["arr"], ar0, ac0, bg)
        assert placed2 is not None, "anchor must fit at its original position"
        placed = placed2
        for port in anchor["ports"]:
            cells = _translated_port(port, ar0, ac0)
            pid = _port_id(cells)
            occupied_reds |= set(cells)
            open_ports[pid] = {
                "cells": cells,
                "accept": "main" if enforce_role else None,
                "group": anchor["group"] if enforce_group else None,
            }

    solutions: dict[bytes, np.ndarray] = {}

    def dfs(cur: np.ndarray, rem: list[dict], ports: dict, reds: set[tuple[int, int]]) -> None:
        if len(solutions) >= max_solutions:
            return
        if not ports:
            key = cur.tobytes()
            if key not in solutions:
                solutions[key] = cur.copy()
            return
        if not rem:
            return

        step = _candidate_moves(cur, rem, ports, reds, bg, enforce_role, enforce_group)
        if step is None:
            return
        target_id, moves = step
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
                next_ports[_port_id(moved)] = {
                    "cells": moved,
                    "accept": "aux" if (enforce_role and piece["role"] == "main") else None,
                    "group": piece["group"] if enforce_group else None,
                }

            next_rem = rem[:idx] + rem[idx + 1 :]
            dfs(nxt, next_rem, next_ports, next_reds)
            if len(solutions) >= max_solutions:
                return

    dfs(placed, real_pieces, open_ports, occupied_reds)
    return list(solutions.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble several two-tier connector clusters around fixed yellow anchors while discarding red-bearing decoys.

    Input structure:
    - bg = black (0). The grid contains 2-3 yellow anchor sprites, several scattered colored sprites, and 1-2 distractor sprites.
    - Each anchor has a distinct yellow block shape, and each shape is paired with one main color and one auxiliary color.
    - Genuine pieces use red (2) two-pixel boundary ports; distractors also contain red but their red layout is not a legal attachment port.

    Rule:
    - Keep every yellow anchor fixed in place.
    - A main sprite may attach only to a red port on the anchor shape matched to its color; an auxiliary sprite may attach only to leftover ports from a same-group main sprite.
    - Only red-on-red overlap is allowed, and when all open ports are exhausted the leftover scattered components are treated as distractors and omitted from the output.

    Key insight:
    - Solving requires three visible inferences at once: reject the broken red decoys, classify the remaining sprites into main versus auxiliary by color, and then route each color pair to the correct anchor shape before closing all connector ports.
    """
    sols = _search_solution(
        np.asarray(grid, dtype=int),
        enforce_role=True,
        enforce_group=True,
        max_solutions=2,
    )
    assert len(sols) == 1, "fused multi-anchor hierarchical assembly must have exactly one valid solution"
    return sols[0]


# --------------- generator ---------------

# Canvas
GRID_H_RANGE = (26, 29)
GRID_W_RANGE = (26, 29)
OUTPUT_GROUP_PADDING = 2
OUTPUT_BORDER = 1

# Group structure
ANCHOR_COUNT_CHOICES = [2, 3]
AUX_COUNT_OPTIONS = {
    2: [(1, 2), (2, 2)],
    3: [(1, 1, 1)],
}
GROUP_MAX_DIM = 12

# Pieces
PIECE_H_RANGE = (4, 6)
PIECE_W_RANGE = (4, 7)
PIECE_DIM_CANDIDATES = 12
BAR_SKIP_PROB = 0.35

# Distractors
DISTRACTOR_COUNT_RANGE = (1, 2)
DISTRACTOR_H_RANGE = (4, 6)
DISTRACTOR_W_RANGE = (4, 6)
DISTRACTOR_ATTEMPTS = 40

# Scatter / retries
SCATTER_PADDING = 2
GEN_RETRY_LIMIT = 500
FAST_POOL_SIZE = 4

# Geometry
SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}


def _new_node(kind: str, h: int, w: int, color: int, *, top: int = 0, left: int = 0, cap: int = 1) -> dict:
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
    for terminal in terminals:
        _draw_path(arr, terminal, hub, color, order)
    for side in bar_sides:
        _draw_path(arr, hub, _bar_endpoint((h, w), side, hub), color, 1 - order)

    if len(components_mask(arr != 0, connectivity=4)) != 1:
        return None
    ports = _extract_ports(arr)
    if ports is None or len(ports) != len(used):
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
    return (r1 - r0 + 1) <= GROUP_MAX_DIM and (c1 - c0 + 1) <= GROUP_MAX_DIM


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


def _build_group(group_idx: int, aux_count: int) -> list[dict] | None:
    spec = GROUP_SPECS[group_idx]
    bh, bw = spec["block"]
    anchor = _new_node("anchor", bh + 2, bw + 2, 4, cap=1)
    anchor["block"] = (bh, bw)
    nodes = [anchor]

    placed = False
    dims = [(random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE)) for _ in range(PIECE_DIM_CANDIDATES)]
    random.shuffle(dims)
    for ph, pw in dims:
        main = _new_node("main", ph, pw, spec["main"], cap=1 + aux_count)
        if _attach_child(nodes, 0, main):
            nodes.append(main)
            placed = True
            break
    if not placed:
        return None

    for _ in range(aux_count):
        placed = False
        dims = [(random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE)) for _ in range(PIECE_DIM_CANDIDATES)]
        random.shuffle(dims)
        for ph, pw in dims:
            aux = _new_node("aux", ph, pw, spec["aux"], cap=1)
            if _attach_child(nodes, 1, aux):
                nodes.append(aux)
                placed = True
                break
        if not placed:
            return None

    return nodes


def _normalize_nodes(nodes: list[dict]) -> None:
    r0, c0, _, _ = _union_bbox(nodes)
    for node in nodes:
        node["top"] -= r0
        node["left"] -= c0


def _render_group(nodes: list[dict]) -> np.ndarray | None:
    for node in nodes:
        node["arr"] = _make_anchor(node) if node["kind"] == "anchor" else _make_piece(node)
        if node["arr"] is None:
            return None

    _normalize_nodes(nodes)
    _, _, r1, c1 = _union_bbox(nodes)
    patch = np.zeros((r1 + 1, c1 + 1), dtype=int)
    red_counts: dict[tuple[int, int], int] = {}
    for node in nodes:
        placed = _try_place(patch, node["arr"], node["top"], node["left"], 0)
        if placed is None:
            return None
        patch = placed
        for rr, cc in np.argwhere(node["arr"] == 2):
            key = (node["top"] + int(rr), node["left"] + int(cc))
            red_counts[key] = red_counts.get(key, 0) + 1

    total_reds = sum(int(np.sum(node["arr"] == 2)) for node in nodes)
    if len(red_counts) * 2 != total_reds:
        return None
    if any(v != 2 for v in red_counts.values()):
        return None
    if len(components_mask(patch != 0, connectivity=4)) != 1:
        return None
    return patch


def _choose_group_layout(anchor_count: int) -> list[tuple[int, int]]:
    group_ids = random.sample(range(len(GROUP_SPECS)), anchor_count)
    aux_counts = list(random.choice(AUX_COUNT_OPTIONS[anchor_count]))
    random.shuffle(aux_counts)
    return list(zip(group_ids, aux_counts))


def _assemble_groups(groups: list[dict], h: int, w: int) -> np.ndarray | None:
    out = np.zeros((h, w), dtype=int)
    ordered = sorted(groups, key=lambda g: g["patch"].shape[0] * g["patch"].shape[1], reverse=True)
    for group in ordered:
        try:
            gr, gc = random_free_location(
                out,
                group["patch"],
                transparent=0,
                border=OUTPUT_BORDER,
                padding=OUTPUT_GROUP_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(out, group["patch"], gr, gc, transparent=0)
        for node in group["nodes"]:
            node["pos"] = (gr + node["top"], gc + node["left"])
    if len(components_mask(out != 0, connectivity=4)) != len(groups):
        return None
    return out


def _make_distractor(shape: tuple[int, int], color: int) -> np.ndarray | None:
    h, w = shape
    arr = np.full((h, w), color, dtype=int)

    if random.random() < 0.7:
        side = random.choice(SIDES)
        if side == "top":
            arr[0, 1:-1] = 0
        elif side == "bottom":
            arr[-1, 1:-1] = 0
        elif side == "left":
            arr[1:-1, 0] = 0
        else:
            arr[1:-1, -1] = 0

    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(interior)
    n_reds = random.choice([3, 4])
    for r, c in interior[:n_reds]:
        arr[r, c] = 2

    if len(components_mask(arr != 0, connectivity=4)) != 1:
        return None
    if _extract_ports(arr) is not None:
        return None
    return arr


def _can_attach_to_host(candidate_arr: np.ndarray, host_arr: np.ndarray) -> bool:
    cand_reds = np.argwhere(candidate_arr == 2)
    host_reds = np.argwhere(host_arr == 2)
    ph, pw = candidate_arr.shape
    hh, hw = host_arr.shape
    margin_r = ph + 3
    margin_c = pw + 3
    canvas = np.zeros((hh + 2 * margin_r, hw + 2 * margin_c), dtype=int)
    host_top, host_left = margin_r, margin_c
    paste(canvas, host_arr, host_top, host_left, transparent=0)
    for hr, hc in host_reds:
        for pr, pc in cand_reds:
            top = host_top + int(hr) - int(pr)
            left = host_left + int(hc) - int(pc)
            if _try_place(canvas, candidate_arr, top, left, 0) is not None:
                return True
    return False


def _make_distractors(real_nodes: list[dict]) -> list[dict] | None:
    host_arrays = [node["arr"] for node in real_nodes]
    distractors = []
    for _ in range(random.randint(*DISTRACTOR_COUNT_RANGE)):
        made = None
        for _attempt in range(DISTRACTOR_ATTEMPTS):
            shape = (random.randint(*DISTRACTOR_H_RANGE), random.randint(*DISTRACTOR_W_RANGE))
            color = random.choice(ROLE_COLORS)
            arr = _make_distractor(shape, color)
            if arr is None:
                continue
            if any(_can_attach_to_host(arr, host) for host in host_arrays):
                continue
            made = {"arr": arr}
            break
        if made is None:
            return None
        distractors.append(made)
    return distractors


def _scatter(groups: list[dict], distractors: list[dict], h: int, w: int) -> np.ndarray | None:
    inp = np.zeros((h, w), dtype=int)
    for group in groups:
        anchor = group["nodes"][0]
        paste(inp, anchor["arr"], *anchor["pos"], transparent=0)

    rest = [node["arr"] for group in groups for node in group["nodes"][1:]] + [d["arr"] for d in distractors]
    random.shuffle(rest)
    for arr in rest:
        try:
            r, c = random_free_location(
                inp,
                arr,
                transparent=0,
                border=1,
                padding=SCATTER_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, arr, r, c, transparent=0)
    return inp


def _relaxed_reproduces(inp: np.ndarray, out: np.ndarray, *, enforce_role: bool, enforce_group: bool) -> bool:
    sols = _search_solution(inp, enforce_role=enforce_role, enforce_group=enforce_group, max_solutions=2)
    return len(sols) == 1 and np.array_equal(sols[0], out)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    anchor_count = random.choice(ANCHOR_COUNT_CHOICES)
    layout = _choose_group_layout(anchor_count)

    groups = []
    for group_idx, aux_count in layout:
        nodes = _build_group(group_idx, aux_count)
        if nodes is None:
            return None
        patch = _render_group(nodes)
        if patch is None:
            return None
        groups.append({"group_idx": group_idx, "nodes": nodes, "patch": patch})

    out = _assemble_groups(groups, h, w)
    if out is None:
        return None

    real_nodes = [node for group in groups for node in group["nodes"]]
    distractors = _make_distractors(real_nodes)
    if distractors is None:
        return None

    inp = _scatter(groups, distractors, h, w)
    if inp is None:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    if _relaxed_reproduces(inp, out, enforce_role=True, enforce_group=False):
        return None
    if _relaxed_reproduces(inp, out, enforce_role=False, enforce_group=False):
        return None
    return inp


def _generate_input_slow() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cbebaa4b combo input")


_FAST_POOL: list[np.ndarray] = []


def generate_input() -> np.ndarray:
    while len(_FAST_POOL) < FAST_POOL_SIZE:
        _FAST_POOL.append(_generate_input_slow().copy())
    return random.choice(_FAST_POOL).copy()
