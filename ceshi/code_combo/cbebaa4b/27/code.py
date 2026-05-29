# Combo variant of cbebaa4b (index 27)
# combo_spec: D3/0+D7/1+D12/1
# Name: 多锚点 + 按色分组批量连接 + 颜色编码层级

# Design analysis for the combo variant
# -------------------------------------
# variant_spec:
#   input:
#     - 黑底 26-30 的网格里固定保留 2 个黄色锚点，左锚点与右锚点彼此独立。
#     - 每个锚点簇都有 3 个散落彩色精灵：2 个第一层精灵直接接锚点，1 个第二层精灵再接到该簇内部。
#     - 左簇颜色家族固定为 蓝(1) -> 绿(3)，右簇固定为 青(8) -> 橙(7)。
#     - 同一颜色的精灵会重复出现成批：左簇有两块蓝色第一层精灵，右簇有两块青色第一层精灵。
#   output:
#     - 两个黄色锚点原地不动，分别长成两个彼此断开的装配簇。
#     - 蓝/青批次先把第一层接到各自锚点上；绿/橙批次再把第二层接到对应簇上。
# behavior_trace:
#   - 原题只有一个锚点，DFS 只维护一条不断扩张的红端口 frontier。
#   - 这里先把两个锚点都放到画布上，再按颜色批次求解；每个批次只能消费“上一层已经存在”的同簇红端口。
#   - 例如左簇的两块蓝色精灵必须都直接覆盖左锚点的红端口，不能先放一块蓝色后再让另一块蓝色接到新蓝块上。
# complexity_calibration:
#   - 若扩到 3 个锚点并同时保留两层颜色家族，颜色家族会过多，单个样例里每个颜色缺乏重复支撑，不利于观察“同色成批”。
#   - 因此这里固定 2 个锚点，但每个第一层颜色都重复 2 次，保证多锚点、按色批处理、层级颜色三件事都能在单个样例里直接看见。
# implicit_assumptions_checked:
#   - 原版 `len(anchors) == 1` 失效；这里改成按从左到右排序的 2 个锚点同时作为初始种子。
#   - 原版允许 `arr == sub` 的非红重叠，在复用颜色后会把同色压同色误判成合法；这里收紧为“只有红压红合法”。
#   - 原版默认任意剩余 piece 都可接到任意 frontier；这里新增假设：piece 颜色能唯一决定锚点归属和层级，generator 会显式保证。
#   - 新增“批次 simultaneity”约束：同色 batch 只能接到 batch 开始前就存在的 frontier，上一个同色 piece 产生的新端口不能被同 batch 继续消费。
# placement_legality:
#   - 仍采用 output-first：先分别构造左右两个局部树，再固定成左右布局的输出，最后仅打散非锚点精灵。
#   - 组内放置要求只与父节点红端口重叠，组间在输出里至少横向留 2 格空隙；输入散布仍要求 Chebyshev padding=2。
# visual_observability:
#   - 两个黄色锚点与两套颜色家族并存，肉眼可见“不是单锚点任务”。
#   - 蓝/青总是贴着锚点，绿/橙总是外扩一层，颜色直接暴露层级。
#   - 同色重复出现两次的第一层精灵，让“同色是一批”不只停留在代码语义，而是能在图像里被看见。
# visual_cleanliness:
#   - 每个 piece 仍保持单主体色 + 红端口，避免在一个小 sprite 里塞太多编码。
#   - 输出只保留两个断开的簇，左右布局稳定，减少颜色家族互相干扰。
# inferability_of_change:
#   - 人类无需知道颜色编号大小；可观察规律是“某些颜色只贴锚点，另一些颜色只贴在它们外面，且颜色家族只去固定一侧锚点”。

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


BG = 0
RED = 2
YELLOW = 4

SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}

GRID_H_RANGE = (26, 30)
GRID_W_RANGE = (26, 30)
OUTPUT_GAP_RANGE = (2, 4)
OUTPUT_BORDER = 1
GROUP_MAX_DIM = 11

PIECE_H_RANGE = (4, 5)
PIECE_W_RANGE = (4, 6)
PIECE_DIM_CANDIDATES = 10
SCATTER_PADDING = 2
BAR_SKIP_PROB = 0.35
GEN_RETRY_LIMIT = 500

FAMILY_SPECS = [
    {"block": (3, 3), "colors": (1, 3)},
    {"block": (4, 5), "colors": (8, 7)},
]
COLOR_TO_FAMILY_LAYER = {
    1: (0, 0),
    3: (0, 1),
    8: (1, 0),
    7: (1, 1),
}
BATCH_SEQUENCE = [1, 8, 3, 7]


def _dominant_piece_color(arr: np.ndarray) -> int | None:
    colors = sorted({int(v) for v in arr.flatten()} - {BG, RED, YELLOW})
    if not colors:
        return None
    assert len(colors) == 1, "non-anchor piece must use exactly one non-red body color"
    return colors[0]


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        reds = np.argwhere(arr == RED)
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "reds": [(int(r), int(c)) for r, c in reds],
                "shape": (int(arr.shape[0]), int(arr.shape[1])),
                "area": int(np.sum(arr != bg)),
                "color": _dominant_piece_color(arr),
                "is_anchor": bool(np.any(arr == YELLOW)),
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

    conflict = mask & (sub != bg) & ~((arr == RED) & (sub == RED))
    if np.any(conflict):
        return None

    out = canvas.copy()
    out_sub = out[top : top + ph, left : left + pw]
    out_sub[mask] = arr[mask]
    out[top : top + ph, left : left + pw] = out_sub
    return out


def _state_key(
    canvas: np.ndarray,
    red_counts: dict[tuple[int, int], int],
    red_owner: dict[tuple[int, int], int],
) -> tuple[bytes, tuple[tuple[tuple[int, int], int], ...], tuple[tuple[tuple[int, int], int], ...]]:
    return (
        canvas.tobytes(),
        tuple(sorted(red_counts.items())),
        tuple(sorted(red_owner.items())),
    )


def _unique_states(states: list[tuple[np.ndarray, dict, dict]]) -> list[tuple[np.ndarray, dict, dict]]:
    uniq: dict[tuple, tuple[np.ndarray, dict, dict]] = {}
    for canvas, rc, owner in states:
        uniq.setdefault(_state_key(canvas, rc, owner), (canvas, rc, owner))
    return list(uniq.values())


def _solve_batch(
    canvas: np.ndarray,
    pieces: list[dict],
    red_counts: dict[tuple[int, int], int],
    red_owner: dict[tuple[int, int], int],
    anchor_idx: int,
    bg: int,
) -> list[tuple[np.ndarray, dict[tuple[int, int], int], dict[tuple[int, int], int]]]:
    if not pieces:
        return [(canvas, red_counts, red_owner)]

    h, w = canvas.shape
    seed_frontier = {
        coord for coord, count in red_counts.items() if count == 1 and red_owner.get(coord) == anchor_idx
    }
    assert seed_frontier, "batch must start from an existing frontier owned by its anchor"

    solutions: list[tuple[np.ndarray, dict, dict]] = []

    def dfs(cur: np.ndarray, remaining: list[dict], rc: dict[tuple[int, int], int], owner: dict[tuple[int, int], int]) -> None:
        if not remaining:
            solutions.append((cur, dict(rc), dict(owner)))
            return

        active_frontier = [coord for coord in seed_frontier if rc.get(coord, 0) == 1]
        if not active_frontier:
            return

        best_moves = None
        for tr, tc in active_frontier:
            moves = []
            for idx, p in enumerate(remaining):
                ph, pw = p["shape"]
                for pr, pc in p["reds"]:
                    top = tr - pr
                    left = tc - pc
                    if top < 0 or left < 0 or top + ph > h or left + pw > w:
                        continue

                    bad = False
                    overlap = 0
                    for rr, cc in p["reds"]:
                        gr, gc = top + rr, left + cc
                        prev = rc.get((gr, gc), 0)
                        if prev >= 2:
                            bad = True
                            break
                        if prev == 1:
                            if owner.get((gr, gc)) != anchor_idx or (gr, gc) not in seed_frontier:
                                bad = True
                                break
                            overlap += 1
                    if bad or overlap == 0:
                        continue
                    moves.append((idx, top, left, overlap))
            if not moves:
                continue
            if best_moves is None or len(moves) < len(best_moves):
                best_moves = moves
                if len(best_moves) == 1:
                    break

        if not best_moves:
            return

        dedup: dict[tuple[int, int, int], int] = {}
        for idx, top, left, overlap in best_moves:
            key = (idx, top, left)
            dedup[key] = max(dedup.get(key, 0), overlap)
        moves2 = [(idx, top, left, overlap) for (idx, top, left), overlap in dedup.items()]
        moves2.sort(key=lambda x: (-x[3], -remaining[x[0]]["area"], x[1], x[2], x[0]))

        for idx, top, left, _overlap in moves2:
            p = remaining[idx]
            nxt = _try_place(cur, p["arr"], top, left, bg)
            if nxt is None:
                continue

            next_rc = dict(rc)
            next_owner = dict(owner)
            for rr, cc in p["reds"]:
                gr, gc = top + rr, left + cc
                key = (gr, gc)
                next_rc[key] = next_rc.get(key, 0) + 1
                next_owner.setdefault(key, anchor_idx)

            dfs(nxt, remaining[:idx] + remaining[idx + 1 :], next_rc, next_owner)

    dfs(canvas, pieces, red_counts, red_owner)
    return _unique_states(solutions)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered sprites onto two fixed yellow anchors by solving independent color batches in layer order.

    Input structure:
    - bg = black (0). The grid contains several scattered colored sprites and exactly two yellow anchor sprites.
    - The left anchor belongs to the blue/green family; the right anchor belongs to the teal/orange family.
    - Each non-anchor sprite is monochrome except for red (2) connector pixels on its boundary.

    Rule:
    - Both yellow anchors stay where they are and seed two disconnected assemblies.
    - Non-anchor sprites are grouped by body color; each color batch can connect only to the anchor family encoded by that color.
    - First-layer colors attach to pre-existing anchor ports, then second-layer colors attach to the new frontier created by the earlier batch of the same family. Red connectors must finish in exact pairs.

    Key insight:
    - Color is doing two jobs at once: it tells you which anchor cluster a sprite belongs to and which assembly layer it occupies.
    - Same-color sprites form a simultaneous batch, so a sprite cannot use red ports created earlier in that same batch; it must attach to the frontier that existed before the batch started.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    pieces = _extract_pieces(g, BG)
    assert pieces, "input must contain at least one non-background component"

    anchors = [p for p in pieces if p["is_anchor"]]
    assert len(anchors) == 2, "combo variant expects exactly two yellow anchor pieces"
    anchors.sort(key=lambda p: (p["bbox"][1], p["bbox"][0]))
    rest = [p for p in pieces if not p["is_anchor"]]
    assert rest, "input must contain non-anchor pieces"

    for p in rest:
        assert p["color"] in COLOR_TO_FAMILY_LAYER, "unexpected piece color in combo variant input"

    canvas = np.full((h, w), BG, dtype=int)
    red_counts: dict[tuple[int, int], int] = {}
    red_owner: dict[tuple[int, int], int] = {}
    for anchor_idx, anchor in enumerate(anchors):
        ar0, ac0, _, _ = anchor["bbox"]
        placed = _try_place(canvas, anchor["arr"], ar0, ac0, BG)
        assert placed is not None, "anchor must fit at its original position"
        canvas = placed
        for rr, cc in anchor["reds"]:
            key = (ar0 + rr, ac0 + cc)
            red_counts[key] = red_counts.get(key, 0) + 1
            red_owner[key] = anchor_idx

    batches: dict[int, list[dict]] = {color: [] for color in BATCH_SEQUENCE}
    for p in rest:
        batches[p["color"]].append(p)

    states: list[tuple[np.ndarray, dict[tuple[int, int], int], dict[tuple[int, int], int]]] = [
        (canvas, red_counts, red_owner)
    ]
    for color in BATCH_SEQUENCE:
        batch = batches[color]
        family_idx, _layer = COLOR_TO_FAMILY_LAYER[color]
        next_states: list[tuple[np.ndarray, dict[tuple[int, int], int], dict[tuple[int, int], int]]] = []
        for cur_canvas, cur_rc, cur_owner in states:
            next_states.extend(
                _solve_batch(cur_canvas, batch, cur_rc, cur_owner, family_idx, BG)
            )
        states = _unique_states(next_states)
        assert states, f"color batch {color} must be placeable"

    total_ports = int(sum(len(p["reds"]) for p in pieces))
    assert total_ports % 2 == 0, "red ports must come in pairs"
    final_canvases: dict[bytes, np.ndarray] = {}
    for cur_canvas, cur_rc, _cur_owner in states:
        if len(cur_rc) * 2 != total_ports:
            continue
        if any(v != 2 for v in cur_rc.values()):
            continue
        if len(components_mask(cur_canvas != BG, connectivity=4)) != 2:
            continue
        final_canvases.setdefault(cur_canvas.tobytes(), cur_canvas)

    assert len(final_canvases) == 1, "combo assembly must have exactly one valid final output"
    return next(iter(final_canvases.values()))


def _new_node(kind: str, h: int, w: int, color: int, top: int = 0, left: int = 0, cap: int = 2) -> dict:
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
    axes = (
        [(r, er, c, True), (c, ec, er, False)]
        if order == 0
        else [(c, ec, r, False), (r, er, ec, True)]
    )
    for a_start, a_end, fixed, is_row in axes:
        step = 1 if a_end >= a_start else -1
        for v in range(a_start, a_end + step, step):
            rr, cc = (v, fixed) if is_row else (fixed, v)
            if arr[rr, cc] != RED:
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
            arr[r, c] = RED
            rr, cc = r + dr, c + dc
            arr[rr, cc] = color
            terminals.append((rr, cc))

    arr[hub] = color
    order = random.randint(0, 1)
    for t in terminals:
        _draw_path(arr, t, hub, color, order)
    for side in bar_sides:
        _draw_path(arr, hub, _bar_endpoint((h, w), side, hub), color, 1 - order)

    if len(components_mask(arr != BG, connectivity=4)) != 1:
        return None
    if int(np.sum(arr == RED)) != 2 * len(used):
        return None
    return arr


def _make_anchor(node: dict) -> np.ndarray:
    bh, bw = node["block"]
    h, w = node["shape"]
    arr = np.zeros((h, w), dtype=int)
    arr[1 : 1 + bh, 1 : 1 + bw] = YELLOW
    for side in node["sides"]:
        for r, c in _port_cells(node, side):
            arr[r, c] = RED
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


def _attach_child(nodes: list[dict], parent_choices: list[int], color: int) -> int | None:
    dims = [
        (random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE))
        for _ in range(PIECE_DIM_CANDIDATES)
    ]
    random.shuffle(dims)
    parents = list(parent_choices)
    random.shuffle(parents)
    for ph, pw in dims:
        for parent_i in parents:
            parent = nodes[parent_i]
            sides = [s for s in SIDES if s not in parent["sides"] and len(parent["sides"]) < parent["cap"]]
            random.shuffle(sides)
            for side in sides:
                child = _new_node("piece", ph, pw, color)
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
                return len(nodes) - 1
    return None


def _grow_group(family_idx: int) -> list[dict] | None:
    spec = FAMILY_SPECS[family_idx]
    bh, bw = spec["block"]
    c1, c2 = spec["colors"]
    anchor = _new_node("anchor", bh + 2, bw + 2, YELLOW, cap=2)
    anchor["block"] = (bh, bw)
    nodes = [anchor]

    child_a = _attach_child(nodes, [0], c1)
    if child_a is None:
        return None
    child_b = _attach_child(nodes, [0], c1)
    if child_b is None:
        return None

    outer_parent = random.choice([child_a, child_b])
    child_c = _attach_child(nodes, [outer_parent], c2)
    if child_c is None:
        other = child_b if outer_parent == child_a else child_a
        child_c = _attach_child(nodes, [other], c2)
        if child_c is None:
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
        placed = _try_place(patch, node["arr"], node["top"], node["left"], BG)
        if placed is None:
            return None
        patch = placed
        for rr, cc in np.argwhere(node["arr"] == RED):
            key = (node["top"] + int(rr), node["left"] + int(cc))
            red_counts[key] = red_counts.get(key, 0) + 1

    total_ports = sum(int(np.sum(node["arr"] == RED)) for node in nodes)
    if len(red_counts) * 2 != total_ports:
        return None
    if any(v != 2 for v in red_counts.values()):
        return None
    if len(components_mask(patch != BG, connectivity=4)) != 1:
        return None
    return patch


def _assemble_groups(groups: list[dict], h: int, w: int) -> np.ndarray | None:
    assert len(groups) == 2
    left_patch = groups[0]["patch"]
    right_patch = groups[1]["patch"]
    lh, lw = left_patch.shape
    rh, rw = right_patch.shape
    gap = random.randint(*OUTPUT_GAP_RANGE)

    max_left = w - OUTPUT_BORDER - lw - gap - rw
    if max_left < OUTPUT_BORDER:
        return None
    c0 = random.randint(OUTPUT_BORDER, max_left)
    c1_min = c0 + lw + gap
    c1_max = w - OUTPUT_BORDER - rw
    if c1_min > c1_max:
        return None
    c1 = random.randint(c1_min, c1_max)

    if lh > h - 2 * OUTPUT_BORDER or rh > h - 2 * OUTPUT_BORDER:
        return None
    r0 = random.randint(OUTPUT_BORDER, h - OUTPUT_BORDER - lh)
    r1 = random.randint(OUTPUT_BORDER, h - OUTPUT_BORDER - rh)

    out = np.zeros((h, w), dtype=int)
    paste(out, left_patch, r0, c0, transparent=BG)
    paste(out, right_patch, r1, c1, transparent=BG)

    for node in groups[0]["nodes"]:
        node["pos"] = (r0 + node["top"], c0 + node["left"])
    for node in groups[1]["nodes"]:
        node["pos"] = (r1 + node["top"], c1 + node["left"])

    if len(components_mask(out != BG, connectivity=4)) != 2:
        return None
    return out


def _scatter(groups: list[dict], h: int, w: int) -> np.ndarray | None:
    inp = np.zeros((h, w), dtype=int)
    for group in groups:
        anchor = group["nodes"][0]
        paste(inp, anchor["arr"], *anchor["pos"], transparent=BG)

    rest = [node for group in groups for node in group["nodes"][1:]]
    random.shuffle(rest)
    for node in rest:
        try:
            r, c = random_free_location(
                inp,
                node["arr"],
                transparent=BG,
                border=1,
                padding=SCATTER_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, node["arr"], r, c, transparent=BG)
    return inp


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    groups = []
    for family_idx in range(2):
        nodes = _grow_group(family_idx)
        if nodes is None:
            return None
        patch = _render_group(nodes)
        if patch is None:
            return None
        groups.append({"nodes": nodes, "patch": patch})

    out = _assemble_groups(groups, h, w)
    if out is None:
        return None
    inp = _scatter(groups, h, w)
    if inp is None:
        return None

    try:
        piece_colors = sorted(
            p["color"] for p in _extract_pieces(inp, BG) if not p["is_anchor"]
        )
    except AssertionError:
        return None
    if piece_colors.count(1) != 2 or piece_colors.count(8) != 2:
        return None
    if piece_colors.count(3) != 1 or piece_colors.count(7) != 1:
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
    raise RuntimeError("failed to generate a valid cbebaa4b combo input")
