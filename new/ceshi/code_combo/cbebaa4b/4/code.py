# Combo variant of cbebaa4b (index 4)
# combo_spec: D2/2+D7/2
# Name: 两类精灵分层 + 链式传递组装

from __future__ import annotations

from itertools import permutations
from pathlib import Path
import random
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(_ROOT / "solvers"))

from arc_utils import bbox_coords, components_mask, paste, random_free_location


"""
design_analysis:
  fused_core:
    original:
      - 原题只要求红色连接器最终两两重合；任一剩余碎片都可以接到当前连通体的任一未配对红像素上。
    D2_2_effect:
      - 非锚点精灵分成主精灵和辅助精灵，两类由可见主体色区分。
    D7_2_effect:
      - 组装前沿不再是“任意开放红端口”，而是唯一的链尾端口。
    fused_rule:
      - 黄色锚点只有一个 2 像素红端口。
      - 链从锚点开始，非锚点精灵沿路径依次接力。
      - 第一块必须是主精灵；之后主/辅角色严格交替：anchor -> main -> aux -> main -> ...
      - 因而“分层”不再表现为星形内外环，而表现为沿链传播的奇偶层级。
  concrete_trace:
    small_example:
      - 输入里有黄色锚点、一块蓝色主精灵、两块暖色辅助精灵、再一块蓝色主精灵，全部被打散。
      - 原题逻辑会把所有红端口视为同类，只要几何上能重合就能尝试拼。
      - 融合后先锁定锚点唯一端口；只能放主精灵。
      - 主精灵放下后只留下自己的另一组红端口作为链尾；下一块必须是辅助精灵。
      - 辅助精灵接上后再次只暴露一个新链尾；下一块又必须是主精灵。
    difference:
      - 改动不是“先做颜色分组，再单独做链”。
      - 颜色分组直接决定链上下一步允许的角色；链状态又决定同一角色何时可用。
  complexity_calibration:
    - 非锚点数量保留 4-6，保证链长足够让交替规律在 2-4 个样例里可见。
    - 主色板固定为 {1, 8}，辅色板固定为 {6, 7}；颜色种类少，分组清晰。
    - 锚点始终只有 1 个端口；中间节点有 2 个端口；尾节点 1 个端口。这样输出视觉上稳定呈现简单路径。
    - 至少包含 2 个主精灵和 2 个辅助精灵，避免退化成“一次主辅配对”。
  implicit_assumption_review:
    - 原 _try_place 允许同色非红重叠；现在同组颜色会重复，这会错误叠片，所以改成只有红对红允许重叠。
    - 原 solver 以单红像素计数；融合后端口语义依赖 2 像素连接器整体，所以改成按红色连通组匹配。
    - 原 solver 默认开放端口彼此等价；融合后必须同时记住唯一链尾和下一步期望角色。
    - 新 solver 假设：锚点恰有 1 个端口，非锚点恰有 1 或 2 个端口，且第一块为主精灵。generator 会显式构造并验证。
  placement_legality:
    - assembled output 按路径逐个放置，新节点 bbox 只能与其父节点 bbox 相交，不能碰到更早节点。
    - 全图绘制时仍只允许红像素彼此重合，任何非红覆盖都视为非法。
    - 输入散布时沿用随机空位放置，并保持 >=2 的切比雪夫间距，保证组件提取稳定。
  visual_observability:
    - 人类能看到黄色锚点只伸出一条链，而链上冷色/暖色精灵按步交替。
    - 规则只使用可见颜色分组和可见拓扑，不依赖颜色编号大小、扫描顺序等不可见因素。
  cleanliness:
    - 最终输出只在红色连接处发生重叠；非红主体仍是干净的单连通精灵。
    - 通过限制端口数为 1/2，避免在局部堆出难读的多分叉红色结点。
  anti_degeneration_check:
    - generator 除了验证 main(input)==output，还运行一个“忽略主辅交替、只保留链拓扑”的 relaxed solver。
    - 只有当 relaxed solver 不能唯一恢复同一输出时，样例才接受；这样量化保证颜色分组确实参与推理。
  new_main_assumptions_audited:
    - 提取 pieces 时默认每个非锚点只有一种主体色；generator 的绘制函数只用单色主体满足此点。
    - DFS 中假设每次放置仅消费一个端口；generator 把每块端口数限制为 1 或 2，且端口互不重叠。
    - 角色交替从主精灵开始；generator 固定链首为主精灵并用 alternating roles 构造整条链。
"""


MAIN_COLORS = (1, 8)
AUX_COLORS = (6, 7)


def _sorted_group(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _body_colors(arr: np.ndarray, bg: int) -> list[int]:
    return sorted({int(v) for v in arr.flatten() if int(v) not in {bg, 2}})


def _extract_ports(arr: np.ndarray) -> list[dict]:
    ports = []
    for comp in components_mask(arr == 2, connectivity=4):
        cells = _sorted_group([(int(r), int(c)) for r, c in comp])
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
        ports.append({"cells": cells, "side": side})
    ports.sort(key=lambda p: (p["side"], p["cells"]))
    return ports


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        ports = _extract_ports(arr)
        colors = _body_colors(arr, bg)
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
                raise AssertionError("piece color must belong to the visible main/aux palettes")
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


def _translated_port(port: dict, top: int, left: int) -> frozenset[tuple[int, int]]:
    return frozenset((int(top + r), int(left + c)) for r, c in port["cells"])


def _align_port(
    target: frozenset[tuple[int, int]], local_cells: tuple[tuple[int, int], ...]
) -> tuple[int, int] | None:
    local = list(local_cells)
    target_cells = list(target)
    for perm in permutations(target_cells):
        top = int(perm[0][0] - local[0][0])
        left = int(perm[0][1] - local[0][1])
        moved = frozenset((int(top + r), int(left + c)) for r, c in local)
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


def _search_chain_solution(
    g: np.ndarray, enforce_roles: bool, max_solutions: int = 2
) -> list[np.ndarray]:
    g = np.asarray(g, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"

    anchors = [p for p in pieces if p["role"] == "anchor"]
    assert len(anchors) == 1, "input must have exactly one yellow anchor piece"
    anchor = anchors[0]
    rest = [p for p in pieces if p is not anchor]

    assert len(anchor["ports"]) == 1, "alternating chain requires exactly one anchor connector pair"
    if enforce_roles:
        mains = [p for p in rest if p["role"] == "main"]
        auxes = [p for p in rest if p["role"] == "aux"]
        assert len(mains) >= 2 and len(auxes) >= 2, "fusion samples must contain repeated main and aux sprites"
    for p in rest:
        assert len(p["ports"]) in (1, 2), "chain pieces must expose one or two connector pairs"

    canvas = np.full((h, w), bg, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, bg)
    assert placed is not None, "anchor must fit at its original position"

    tail = _translated_port(anchor["ports"][0], ar0, ac0)
    occupied_reds = set(tail)
    solutions: list[np.ndarray] = []

    def dfs(
        cur: np.ndarray,
        remaining: list[dict],
        active_tail: frozenset[tuple[int, int]] | None,
        reds: set[tuple[int, int]],
        expected_role: str,
    ) -> None:
        if len(solutions) >= max_solutions:
            return
        if not remaining:
            if active_tail is None:
                solutions.append(cur)
            return
        if active_tail is None:
            return

        moves = []
        for idx, piece in enumerate(remaining):
            if enforce_roles and piece["role"] != expected_role:
                continue
            for port_idx, port in enumerate(piece["ports"]):
                pos = _align_port(active_tail, port["cells"])
                if pos is None:
                    continue
                top, left = pos
                ph, pw = piece["shape"]
                if top < 0 or left < 0 or top + ph > h or left + pw > w:
                    continue
                used = 0
                bad = False
                for other_port in piece["ports"]:
                    moved = _translated_port(other_port, top, left)
                    if moved == active_tail:
                        used += 1
                        continue
                    if any(cell in reds for cell in moved):
                        bad = True
                        break
                if bad or used != 1:
                    continue
                nxt = _try_place(cur, piece["arr"], top, left, bg)
                if nxt is None:
                    continue
                moves.append((idx, port_idx, top, left))

        moves.sort(
            key=lambda x: (
                len(remaining[x[0]]["ports"]),
                -remaining[x[0]]["area"],
                x[2],
                x[3],
                x[1],
                x[0],
            )
        )

        for idx, port_idx, top, left in moves:
            piece = remaining[idx]
            nxt = _try_place(cur, piece["arr"], top, left, bg)
            if nxt is None:
                continue

            next_reds = set(reds)
            next_tail = None
            for j, port in enumerate(piece["ports"]):
                moved = _translated_port(port, top, left)
                next_reds |= set(moved)
                if j != port_idx:
                    next_tail = moved

            next_remaining = remaining[:idx] + remaining[idx + 1 :]
            if next_remaining and next_tail is None:
                continue
            if not next_remaining and next_tail is not None:
                continue

            dfs(
                nxt,
                next_remaining,
                next_tail,
                next_reds,
                "aux" if expected_role == "main" else "main",
            )
            if len(solutions) >= max_solutions:
                return

    dfs(placed, rest, tail, occupied_reds, "main")
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered sprites into a single relay chain whose color groups must alternate.

    Input structure:
    - bg = black (0). One yellow anchor sprite stays fixed while several colored sprites are scattered apart.
    - Cool-colored sprites (blue / azure) are main sprites; warm-colored sprites (pink / orange) are auxiliary sprites.
    - Every connector is a 2-cell red (2) boundary port, and the anchor exposes exactly one such port.

    Rule:
    - Start from the anchor's red port and grow exactly one chain.
    - The first attached sprite must be a main sprite; after that, each next sprite must attach only to the current chain tail and must switch role relative to the previous sprite.
    - Keep relaying the tail forward until all sprites are consumed and the last sprite closes the chain with no unused port left.

    Key insight:
    - Red ports still determine geometric alignment, but they are no longer sufficient by themselves: the solver must track both the unique active tail and the required next role in the main/aux alternation.
    """
    sols = _search_chain_solution(np.asarray(grid, dtype=int), enforce_roles=True, max_solutions=2)
    assert len(sols) == 1, "alternating layered chain must have exactly one valid solution"
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

# Scatter / generation
SCATTER_PADDING = 2
GEN_RETRY_LIMIT = 600

SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}


def _new_node(role: str, h: int, w: int, color: int, top: int = 0, left: int = 0, cap: int = 2) -> dict:
    return {
        "role": role,
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


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
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


def _grow_alternating_chain() -> list[dict] | None:
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    anchor = _new_node("anchor", bh + 2, bw + 2, 4, cap=1)
    anchor["block"] = (bh, bw)
    nodes = [anchor]

    n_pieces = random.randint(*PIECE_COUNT_RANGE)
    roles = ["main" if i % 2 == 0 else "aux" for i in range(n_pieces)]
    main_colors = _assign_group_colors(roles.count("main"), MAIN_COLORS)
    aux_colors = _assign_group_colors(roles.count("aux"), AUX_COLORS)
    main_i = 0
    aux_i = 0
    parent_i = 0

    for i, role in enumerate(roles):
        placed = False
        cap = 1 if i == n_pieces - 1 else 2
        color = main_colors[main_i] if role == "main" else aux_colors[aux_i]
        if role == "main":
            main_i += 1
        else:
            aux_i += 1
        dims = [(random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE)) for _ in range(PIECE_DIM_CANDIDATES)]
        random.shuffle(dims)
        for ph, pw in dims:
            child = _new_node(role, ph, pw, color, cap=cap)
            if _attach_child(nodes, parent_i, child):
                nodes.append(child)
                parent_i = len(nodes) - 1
                placed = True
                break
        if not placed:
            return None

    if roles.count("main") < 2 or roles.count("aux") < 2:
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
        node["arr"] = _make_anchor(node) if node["role"] == "anchor" else _make_piece(node)
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

    total_red = sum(int(np.sum(node["arr"] == 2)) for node in nodes)
    if len(all_red_counts) * 2 != total_red:
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
            r, c = random_free_location(
                inp, node["arr"], transparent=0, padding=SCATTER_PADDING, padding_connectivity=8
            )
        except ValueError:
            return None
        paste(inp, node["arr"], r, c, transparent=0)
    return inp


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    nodes = _grow_alternating_chain()
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

    relaxed = _search_chain_solution(inp, enforce_roles=False, max_solutions=2)
    if len(relaxed) == 1 and np.array_equal(relaxed[0], out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid alternating layered chain cbebaa4b input")
