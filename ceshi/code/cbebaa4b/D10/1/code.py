# Variant of cbebaa4b
# Dimension: D10 — 连接器像素的数量与分布 (meta)
# Change: 每个精灵有少量红色连接器像素分布在边缘，锚点方块边缘也有红色像素
#   → 精灵有 2-4 个候选红色连接器，分布在不同边缘，但最终只使用其中一个与锚点对齐

# Design analysis for the D10 variant
# ----------------------------------
# spec_variant:
#   grid: 22-26 x 22-26, bg black(0)
#   anchor:
#     core: inset yellow block, same role as original, fixed in input/output
#     targets: exactly one red target cell per non-anchor piece, all on the anchor perimeter
#   pieces:
#     count: 4-6
#     colors: unique colors from {1,3,5,6,7,8,9}
#     connectors_per_piece: 2-4 single red cells on distinct sides
#     active_connector: exactly one connector per piece is used in the final assembly
#     inactive_connectors: visible in both input and output, never overlap anything
#   assembly:
#     topology: star around the anchor; each piece attaches directly to the anchor
#     motion: translation only, no rotation or reflection
#     legality: each piece must overlap the anchor in exactly one red-on-red cell; every other
#       non-black cell must stay conflict-free
#
# a) concretization_and_trace:
#   原题里 piece 的全部红点都要成对配对，所以 solver 用 red_counts 把所有红点都当作“必须被消费”的端口。
#   这里改成：piece 上有多个候选红点，但只有一个红点真的和锚点重合。
#   例子：某个蓝色 piece 在 top/right/left 三条边各有一个红点。输出时只有 left 边红点压到锚点右侧红点，
#   top/right 两个红点依然裸露可见。原逻辑会错误地继续要求那两个红点也去配对；新逻辑只把“恰好一个
#   红点与锚点重合”当作合法落位条件。
#
# b) complexity_calibration:
#   如果每块只有 2 个候选红点，视觉变化太弱；如果每条边都能有多个红点，组合数又会太高。
#   因此把每块控制在 2-4 个红点且每个红点占不同边。这样人眼能稳定看出“有多选一”的行为，
#   solver 的搜索分支也仍然可控。
#
# c) implicit_assumption_audit:
#   原 main() 的关键旧前提是：
#   - total_ports 必须为偶数，且所有红点最终都二二重合
#   - 当前未配对红点集合决定下一步搜索
#   - 非锚点 piece 可以继续提供新的可配对红点，形成树
#   这些前提在变体中都不成立。新 solver 改为：
#   - 不再统计所有红点的配对完成度
#   - 只允许 piece 与 anchor 的某个红目标重合一次
#   - piece 之间不允许通过红点互相连接；未激活红点允许保留裸露
#
# d) placement_legality:
#   生成器先在局部坐标里构造完整输出，再整体平移进画布。对每个 piece 的候选摆放，都逐像素检查：
#   - 与锚点的重叠必须恰好是一格 red/red，且就是指定 target
#   - 与其他 piece 完全不能重叠
#   - 总包围盒不能超过 MAX_ASSEMBLY_DIM
#
# e) visual_observability:
#   输入中每个 piece 明显带有多个红点，输出中每个 piece 只用其中一个接到锚点，其余红点仍然保留。
#   这与原题“红点都要配对完”在 2-3 个样例上肉眼就能区分。
#
# f) visual_cleanliness:
#   连接关系全部围绕中央锚点展开，避免旧树结构里 piece-piece 红点覆盖造成的局部拥挤。
#   同时限制同侧锚点 target 的数量与间距，降低输出堆叠感。
#
# g) color_inference:
#   没有新增颜色映射规则；颜色只用于区分 piece，本变体的核心信息完全由形状和红点位置承载。
#
# h) generator_validation_metric:
#   变体是否真的发生，可量化为 inactive_connectors = sum(piece_reds - 1)。
#   对本生成器，所有样本都应满足 inactive_connectors >= piece_count，且每块 piece_reds in [2,4]。
#
# i) new_solver_assumptions_vs_generator:
#   新 solver 假设“所有可用目标红点都在锚点上，且数量等于非锚点 piece 数量”。
#   生成器通过 star 布局和 anchor-per-piece target 构造保证这一点。

from __future__ import annotations

import itertools
import random
from pathlib import Path
import sys

import numpy as np

for parent in Path(__file__).resolve().parents:
    if (parent / "arc_utils.py").exists():
        if str(parent) not in sys.path:
            sys.path.append(str(parent))
        break

from arc_utils import bbox_coords, components_mask, paste, random_free_location


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        reds = np.argwhere(arr == 2)
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "reds": [(int(r), int(c)) for r, c in reds],
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


def _overlapping_reds(
    canvas: np.ndarray, arr: np.ndarray, top: int, left: int, bg: int
) -> list[tuple[int, int]]:
    h, w = canvas.shape
    ph, pw = arr.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return []
    sub = canvas[top : top + ph, left : left + pw]
    mask = (arr == 2) & (sub == 2)
    pts = np.argwhere(mask)
    return [(int(top + r), int(left + c)) for r, c in pts]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered sprites around a fixed yellow anchor when each sprite offers several red choices but only one is real.

    Input structure:
    - bg is black (0). One component contains the yellow anchor block and red target pixels on its perimeter.
    - Several non-anchor sprites are scattered elsewhere.
    - Every non-anchor sprite has 2-4 red (2) connector pixels on different edges, but most of them are decoys.

    Rule:
    - Keep the anchor in place.
    - Translate each non-anchor sprite so that exactly one of its red pixels lands on exactly one unused red pixel of the anchor.
    - No other overlap is allowed; unused red pixels remain visible and do not need to pair with anything.

    Key insight:
    - Red no longer means “all of these must be matched”. It means “this sprite has several candidate docking points”, and the correct assembly is the unique layout where each sprite uses one candidate to dock to the anchor without creating any other collision.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"

    anchors = [p for p in pieces if np.any(p["arr"] == 4)]
    assert len(anchors) == 1, "input must have exactly one yellow anchor piece"
    anchor = anchors[0]
    rest = [p for p in pieces if p is not anchor]

    canvas = np.full((h, w), bg, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, bg)
    assert placed is not None, "anchor piece must fit on canvas at its original position"

    anchor_targets = {(ar0 + rr, ac0 + cc) for rr, cc in anchor["reds"]}
    assert len(anchor_targets) == len(anchor["reds"]), "anchor targets must be distinct"
    assert len(rest) == len(anchor_targets), "piece count must match anchor target count"

    solutions: list[np.ndarray] = []

    def _moves_for_target(
        cur: np.ndarray, remaining: list[dict], target: tuple[int, int], unused_targets: set[tuple[int, int]]
    ) -> list[tuple[int, int, int, int]]:
        tr, tc = target
        moves = []
        for idx, p in enumerate(remaining):
            ph, pw = p["shape"]
            for pr, pc in p["reds"]:
                top = tr - pr
                left = tc - pc
                if top < 0 or left < 0 or top + ph > h or left + pw > w:
                    continue
                overlaps = _overlapping_reds(cur, p["arr"], top, left, bg)
                overlap_set = set(overlaps)
                if overlap_set != {target}:
                    continue
                # A legal move uses this target and no other anchor/piece red.
                nxt = _try_place(cur, p["arr"], top, left, bg)
                if nxt is None:
                    continue
                # Prefer pieces with fewer connectors first and tighter placements after that.
                moves.append((idx, top, left, len(p["reds"])))
        moves.sort(key=lambda x: (x[3], -remaining[x[0]]["area"], x[1], x[2], x[0]))
        return moves

    def dfs(cur: np.ndarray, remaining: list[dict], unused_targets: set[tuple[int, int]]) -> None:
        if len(solutions) >= 2:
            return
        if not remaining:
            if not unused_targets:
                solutions.append(cur)
            return
        if len(remaining) != len(unused_targets):
            return

        best_target = None
        best_moves = None
        for target in sorted(unused_targets):
            moves = _moves_for_target(cur, remaining, target, unused_targets)
            if not moves:
                return
            if best_moves is None or len(moves) < len(best_moves):
                best_target = target
                best_moves = moves
                if len(best_moves) == 1:
                    break

        assert best_target is not None and best_moves is not None

        seen: set[tuple[int, int, int]] = set()
        for idx, top, left, _reds in best_moves:
            key = (idx, top, left)
            if key in seen:
                continue
            seen.add(key)
            p = remaining[idx]
            nxt = _try_place(cur, p["arr"], top, left, bg)
            if nxt is None:
                continue
            next_remaining = remaining[:idx] + remaining[idx + 1 :]
            next_targets = set(unused_targets)
            next_targets.remove(best_target)
            dfs(nxt, next_remaining, next_targets)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, set(anchor_targets))
    assert len(solutions) == 1, "assembly must have exactly one valid solution"
    return solutions[0]


# --------------- generator (D10 variant) ---------------

GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 24

PIECE_COUNT_RANGE = (4, 6)
PIECE_H_RANGE = (4, 7)
PIECE_W_RANGE = (4, 8)
PIECE_SHAPE_TRIES = 36
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
PIECE_CONNECTOR_RANGE = (2, 4)
BAR_SKIP_PROB = 0.35

ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]

SCATTER_PADDING = 2
GEN_RETRY_LIMIT = 600

SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}


def _new_node(kind, h, w, color, top=0, left=0):
    return {
        "kind": kind,
        "shape": (int(h), int(w)),
        "color": int(color),
        "top": int(top),
        "left": int(left),
        "ports": {},
    }


def _port_cell(node, side):
    h, w = node["shape"]
    k = node["ports"][side]
    if side == "top":
        return (0, k)
    if side == "bottom":
        return (h - 1, k)
    if side == "left":
        return (k, 0)
    return (k, w - 1)


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


def _sample_spaced_positions(span: int, count: int) -> list[int] | None:
    combos = [
        list(combo)
        for combo in itertools.combinations(range(1, span + 1), count)
        if all(b - a >= 2 for a, b in zip(combo, combo[1:]))
    ]
    if not combos:
        return None
    return random.choice(combos)


def _choose_anchor_targets(block: tuple[int, int], count: int) -> list[dict] | None:
    bh, bw = block
    side_span = {"top": bw, "bottom": bw, "left": bh, "right": bh}
    side_cap = {s: min(2, (side_span[s] + 1) // 2) for s in SIDES}
    side_counts = {s: 0 for s in SIDES}

    chosen_sides = []
    for _ in range(count):
        options = [s for s in SIDES if side_counts[s] < side_cap[s]]
        if not options:
            return None
        side = random.choice(options)
        side_counts[side] += 1
        chosen_sides.append(side)

    targets = []
    for side in SIDES:
        k = side_counts[side]
        if k == 0:
            continue
        pos = _sample_spaced_positions(side_span[side], k)
        if pos is None:
            return None
        for v in pos:
            if side == "top":
                coord = (0, v)
            elif side == "bottom":
                coord = (bh + 1, v)
            elif side == "left":
                coord = (v, 0)
            else:
                coord = (v, bw + 1)
            targets.append({"side": side, "coord": coord})
    random.shuffle(targets)
    return targets


def _make_anchor(node):
    bh, bw = node["block"]
    h, w = node["shape"]
    arr = np.zeros((h, w), dtype=int)
    arr[1 : 1 + bh, 1 : 1 + bw] = 4
    for target in node["targets"]:
        r, c = target["coord"]
        arr[r, c] = 2
    return arr


def _make_piece(node):
    h, w = node["shape"]
    color = node["color"]
    connector_sides = list(node["connector_sides"])
    arr = np.zeros((h, w), dtype=int)
    hub = (random.randint(1, h - 2), random.randint(1, w - 2))

    bar_sides = [s for s in SIDES if s not in connector_sides]
    if len(bar_sides) > 1 and random.random() < BAR_SKIP_PROB:
        bar_sides.remove(random.choice(bar_sides))
    for side in bar_sides:
        _paint_bar(arr, side, color, connector_sides)

    terminals = []
    for side in connector_sides:
        r, c = _port_cell(node, side)
        arr[r, c] = 2
        dr, dc = INWARD[side]
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
    reds = int(np.sum(arr == 2))
    if reds != len(connector_sides):
        return None
    return arr


def _overlap_details(a_node, b_node):
    a_box = _bbox_of(a_node)
    b_box = _bbox_of(b_node)
    if not _rects_overlap(a_box, b_box):
        return []

    r0 = max(a_box[0], b_box[0])
    c0 = max(a_box[1], b_box[1])
    r1 = min(a_box[2], b_box[2])
    c1 = min(a_box[3], b_box[3])
    details = []
    for gr in range(r0, r1 + 1):
        for gc in range(c0, c1 + 1):
            av = a_node["arr"][gr - a_node["top"], gc - a_node["left"]]
            bv = b_node["arr"][gr - b_node["top"], gc - b_node["left"]]
            if av != 0 and bv != 0:
                details.append((gr, gc, int(av), int(bv)))
    return details


def _can_add_piece(nodes, child):
    r0, c0, r1, c1 = _union_bbox(nodes + [child])
    if (r1 - r0 + 1) > MAX_ASSEMBLY_DIM or (c1 - c0 + 1) > MAX_ASSEMBLY_DIM:
        return False

    expected = child["target_coord"]
    for other in nodes:
        details = _overlap_details(other, child)
        if not details:
            continue
        if other["kind"] != "anchor":
            return False
        if len(details) != 1:
            return False
        gr, gc, av, bv = details[0]
        if (gr, gc) != expected or av != 2 or bv != 2:
            return False
    return True


def _grow_star():
    n_pieces = random.randint(*PIECE_COUNT_RANGE)
    colors = random.sample(PIECE_COLORS, n_pieces)

    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    anchor = _new_node("anchor", bh + 2, bw + 2, 4)
    anchor["block"] = (bh, bw)
    targets = _choose_anchor_targets(anchor["block"], n_pieces)
    if targets is None:
        return None
    anchor["targets"] = targets
    anchor["arr"] = _make_anchor(anchor)

    nodes = [anchor]
    for color, target in zip(colors, targets):
        placed = False
        active_side = OPPOSITE[target["side"]]
        for _ in range(PIECE_SHAPE_TRIES):
            ph = random.randint(*PIECE_H_RANGE)
            pw = random.randint(*PIECE_W_RANGE)
            count = random.randint(*PIECE_CONNECTOR_RANGE)
            connector_sides = [active_side]
            others = [s for s in SIDES if s != active_side]
            connector_sides.extend(random.sample(others, count - 1))
            node = _new_node("piece", ph, pw, color)
            node["active_side"] = active_side
            node["connector_sides"] = tuple(connector_sides)
            for side in connector_sides:
                span = ph if side in ("left", "right") else pw
                node["ports"][side] = random.randint(1, span - 2)
            node["arr"] = _make_piece(node)
            if node["arr"] is None:
                continue
            pr, pc = _port_cell(node, active_side)
            tr, tc = target["coord"]
            node["top"] = tr - pr
            node["left"] = tc - pc
            node["target_coord"] = (tr, tc)
            node["target_side"] = target["side"]
            if not _can_add_piece(nodes, node):
                continue
            nodes.append(node)
            placed = True
            break
        if not placed:
            return None
    return nodes


def _assemble(nodes, h, w):
    r0, c0, r1, c1 = _union_bbox(nodes)
    if (r1 - r0 + 1) > h or (c1 - c0 + 1) > w:
        return None
    sr = random.randint(-r0, h - 1 - r1)
    sc = random.randint(-c0, w - 1 - c1)

    out = np.zeros((h, w), dtype=int)
    anchor = nodes[0]
    anchor_top = anchor["top"] + sr
    anchor_left = anchor["left"] + sc
    anchor["pos"] = (anchor_top, anchor_left)
    out = _try_place(out, anchor["arr"], anchor_top, anchor_left, 0)
    if out is None:
        return None

    for node in nodes[1:]:
        top = node["top"] + sr
        left = node["left"] + sc
        node["pos"] = (top, left)
        overlaps = _overlapping_reds(out, node["arr"], top, left, 0)
        expected = [(node["target_coord"][0] + sr, node["target_coord"][1] + sc)]
        if overlaps != expected:
            return None
        out = _try_place(out, node["arr"], top, left, 0)
        if out is None:
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
    nodes = _grow_star()
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
    raise RuntimeError("failed to generate a valid cbebaa4b D10 input")
