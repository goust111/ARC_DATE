# Combo variant of cbebaa4b (index 36)
# combo_spec: D1/5+D10/1
# Name: 数量匹配 + 多连接器选一
#
# Design analysis for the combo variant
# -------------------------------------
# spec_variant:
#   grid: 22-26 x 22-26, bg black(0)
#   anchor:
#     fixed: same yellow core role as the original; its position is unchanged
#     slots: 4-6 red edge runs around the anchor perimeter; each run length is 1-3
#   pieces:
#     count: 4-6
#     colors: unique colors from {1,3,5,6,7,8,9}
#     connectors: each piece exposes 2-4 candidate red edge runs on distinct sides
#     active_connector: exactly one candidate run has the same red-count as its assigned anchor slot
#     inactive_connectors: remain visible in both input and output
#   assembly:
#     topology: star centered on the anchor
#     motion: translation only; no rotation or reflection
#     docking: the chosen piece edge is translated flush against the anchor side; there is no red overlap
#     match_rule: a piece may use only the candidate edge whose red-count equals the anchor slot count
#
# a) concretization_and_trace:
#   原题把红点当作必须两两重合的端口，因此 solver 会追踪“哪些红点还没配对完”。
#   这里改成：anchor 某条边上有长度为 2 的红段，某 piece 的四条候选边里可能分别有长度 1/2/3 的红段。
#   输出时这块 piece 只能用长度 2 的那条边贴到该 slot；长度 1 和 3 的边继续裸露。
#   所以执行流程从“找能重合的红点”变成“先数每条候选边的红像素数量，再选择唯一匹配的停靠边”。
#
# b) complexity_calibration:
#   如果每块 piece 只有 2 个候选边，且长度只在 {1,2} 之间变化，视觉区分会太弱。
#   本实现把候选边数控制在 2-4、长度控制在 1-3，并让同一块 piece 的非活跃边都避开目标长度。
#   这样样例里能稳定看出“先数数量，再选边”，同时搜索分支仍然较小。
#
# c) implicit_assumption_audit:
#   原 main() 的旧假设包括：
#   - 所有红点最终都要被消费完
#   - 下一步由当前未配对红点集合决定
#   - piece 可以继续给后续 piece 提供新的红目标
#   融合后这些都不成立。新 solver 改为：
#   - 只把 anchor 上的红段视为目标 slot
#   - 每个 piece 只使用一个候选连接器，其余红段允许保留
#   - 拓扑固定为 star，piece 之间既不重叠也不接触
#
# d) placement_legality:
#   生成器先在局部坐标里构造完整输出，再整体平移到画布。每个 piece 都要满足：
#   - 与 anchor 不重叠，只在指定 slot 上产生恰好 length 个红-红邻接
#   - 与其他 piece 不重叠且不发生 4 邻接触碰
#   - 总包围盒不超过 MAX_ASSEMBLY_DIM
#
# e) visual_observability:
#   输入里每个 piece 都明显带有多条红边段；输出里只有其中一条贴到 anchor 对应边段。
#   而且贴边后是“等长相接”而不是原题那种“红点重合”，肉眼从 2-3 组样例就能区分出规则变化。
#
# f) visual_cleanliness:
#   输出保持围绕 anchor 的星形布局，避免 piece-piece 链式咬合带来的局部拥挤。
#   同时把 anchor 的额外 slot 限制在 top/bottom 两侧，减少角落遮挡和难读接触关系。
#
# g) color_inference:
#   没有颜色映射规则。颜色只用于区分 piece；真正决定变换的是边上的红段数量与停靠方位。
#
# h) generator_validation_metric:
#   可以量化两个核心行为：
#   - inactive_connectors = sum(len(piece.connectors) - 1)，应始终 >= piece_count
#   - unique_count_choice = 每个 piece 内与目标长度相等的候选边数量，必须恒为 1
#   若这两个指标退化，就说明融合行为没有真正发生。
#
# i) new_solver_assumptions_vs_generator:
#   新 solver 假设：
#   - anchor 的所有目标都是边界上的连续红段
#   - 每个非 anchor piece 每边至多一段红连接器，且候选边数在 2-4 之间
#   - 对于每个 piece，只有一个候选边长度等于其目标 slot 长度
#   生成器会逐项构造并验证这些条件。

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


GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 24

PIECE_COUNT_RANGE = (4, 6)
PIECE_H_RANGE = (4, 7)
PIECE_W_RANGE = (4, 8)
PIECE_SHAPE_TRIES = 80
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
CONNECTOR_COUNT_RANGE = (2, 4)
CONNECTOR_LEN_CHOICES = (1, 2, 3)
BAR_SKIP_PROB = 0.35

ANCHOR_BLOCK = (4, 5)
ANCHOR_SIDE_CAP = {"top": 2, "bottom": 2, "left": 1, "right": 1}

SCATTER_PADDING = 2
GEN_RETRY_LIMIT = 900

SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "shape": (int(arr.shape[0]), int(arr.shape[1])),
                "area": int(np.sum(arr != bg)),
            }
        )
    return pieces


def _try_place(canvas: np.ndarray, arr: np.ndarray, top: int, left: int, bg: int) -> np.ndarray | None:
    h, w = canvas.shape
    ph, pw = arr.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return None
    sub = canvas[top : top + ph, left : left + pw]
    mask = arr != bg
    if np.any(mask & (sub != bg)):
        return None
    out = canvas.copy()
    out[top : top + ph, left : left + pw][mask] = arr[mask]
    return out


def _line_runs(line: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    i = 0
    n = int(line.size)
    while i < n:
        if int(line[i]) != 2:
            i += 1
            continue
        j = i
        while j + 1 < n and int(line[j + 1]) == 2:
            j += 1
        runs.append((i, j - i + 1))
        i = j + 1
    return runs


def _edge_runs(arr: np.ndarray, side: str) -> list[dict]:
    if side == "top":
        line = arr[0, :]
    elif side == "bottom":
        line = arr[-1, :]
    elif side == "left":
        line = arr[:, 0]
    else:
        line = arr[:, -1]
    return [{"side": side, "start": int(s), "length": int(k)} for s, k in _line_runs(line)]


def _connector_cells(shape: tuple[int, int], side: str, start: int, length: int) -> list[tuple[int, int]]:
    h, w = shape
    if side == "top":
        return [(0, c) for c in range(start, start + length)]
    if side == "bottom":
        return [(h - 1, c) for c in range(start, start + length)]
    if side == "left":
        return [(r, 0) for r in range(start, start + length)]
    return [(r, w - 1) for r in range(start, start + length)]


def _connector_inward_cells(shape: tuple[int, int], side: str, start: int, length: int) -> list[tuple[int, int]]:
    dr, dc = INWARD[side]
    return [(r + dr, c + dc) for r, c in _connector_cells(shape, side, start, length)]


def _global_cells(node: dict, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(int(node["top"] + r), int(node["left"] + c)) for r, c in cells]


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


def _piece_connectors(arr: np.ndarray) -> list[dict]:
    connectors: list[dict] = []
    for side in SIDES:
        runs = _edge_runs(arr, side)
        assert len(runs) <= 1, "each side may carry at most one connector run"
        connectors.extend(runs)
    return connectors


def _anchor_slots(arr: np.ndarray) -> list[dict]:
    slots: list[dict] = []
    for side in SIDES:
        slots.extend(_edge_runs(arr, side))
    return slots


def _placement_for_slot(anchor_bbox: tuple[int, int, int, int], slot: dict, piece: dict, conn: dict) -> tuple[int, int]:
    ar0, ac0, ar1, ac1 = anchor_bbox
    ph, pw = piece["shape"]
    if slot["side"] == "top":
        top = ar0 - ph
        left = ac0 + slot["start"] - conn["start"]
    elif slot["side"] == "bottom":
        top = ar1 + 1
        left = ac0 + slot["start"] - conn["start"]
    elif slot["side"] == "left":
        top = ar0 + slot["start"] - conn["start"]
        left = ac0 - pw
    else:
        top = ar0 + slot["start"] - conn["start"]
        left = ac1 + 1
    return int(top), int(left)


def _expected_contact_pairs(anchor_bbox: tuple[int, int, int, int], slot: dict, top: int, left: int, conn: dict) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    ar0, ac0, ar1, ac1 = anchor_bbox
    pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for offset in range(slot["length"]):
        if slot["side"] == "top":
            a = (ar0, ac0 + slot["start"] + offset)
            p = (top + conn["side_row"], left + conn["start"] + offset)
        elif slot["side"] == "bottom":
            a = (ar1, ac0 + slot["start"] + offset)
            p = (top + conn["side_row"], left + conn["start"] + offset)
        elif slot["side"] == "left":
            a = (ar0 + slot["start"] + offset, ac0)
            p = (top + conn["start"] + offset, left + conn["side_col"])
        else:
            a = (ar0 + slot["start"] + offset, ac1)
            p = (top + conn["start"] + offset, left + conn["side_col"])
        pairs.add((p, a))
    return pairs


def _contact_pairs(canvas: np.ndarray, arr: np.ndarray, top: int, left: int, bg: int) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    ph, pw = arr.shape
    pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for r in range(ph):
        for c in range(pw):
            if int(arr[r, c]) == bg:
                continue
            gr, gc = top + r, left + c
            for nr, nc in ((gr - 1, gc), (gr + 1, gc), (gr, gc - 1), (gr, gc + 1)):
                if 0 <= nr < canvas.shape[0] and 0 <= nc < canvas.shape[1] and int(canvas[nr, nc]) != bg:
                    pairs.add(((gr, gc), (nr, nc)))
    return pairs


def _is_legal_slot_move(
    canvas: np.ndarray,
    anchor_bbox: tuple[int, int, int, int],
    slot: dict,
    piece: dict,
    conn: dict,
    top: int,
    left: int,
    bg: int,
) -> bool:
    nxt = _try_place(canvas, piece["arr"], top, left, bg)
    if nxt is None:
        return False
    actual_pairs = _contact_pairs(canvas, piece["arr"], top, left, bg)
    expected_pairs = _expected_contact_pairs(anchor_bbox, slot, top, left, conn)
    if actual_pairs != expected_pairs:
        return False
    for (pr, pc), (ar, ac) in actual_pairs:
        if int(piece["arr"][pr - top, pc - left]) != 2 or int(canvas[ar, ac]) != 2:
            return False
    return True


def _annotate_connectors(piece: dict) -> None:
    enriched = []
    h, w = piece["shape"]
    for conn in _piece_connectors(piece["arr"]):
        conn = dict(conn)
        conn["side_row"] = 0 if conn["side"] == "top" else h - 1 if conn["side"] == "bottom" else None
        conn["side_col"] = 0 if conn["side"] == "left" else w - 1 if conn["side"] == "right" else None
        enriched.append(conn)
    piece["connectors"] = enriched


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered sprites by choosing the only candidate edge whose red-count matches an anchor slot.

    Input structure:
    - The background is black (0). One fixed anchor contains a yellow block and several red runs on its outer edges.
    - Each non-anchor sprite is a separate colored component with 2-4 candidate red edge runs on different sides.
    - A candidate connector is the full red run on one side, and most of those runs are decoys.

    Rule:
    - Keep the anchor where it is.
    - For each anchor slot, choose one unused piece whose opposite-side red run has the same number of red pixels.
    - Translate that piece so the matching run docks flush against the slot; pieces never overlap, and unused connectors stay visible.

    Key insight:
    - The puzzle no longer asks you to pair every red pixel. Red now encodes edge counts: first count the red cells on each candidate side, then use the only equal-count side to decide how that sprite attaches to the anchor.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one piece"

    anchors = [p for p in pieces if np.any(p["arr"] == 4)]
    assert len(anchors) == 1, "input must have exactly one anchor"
    anchor = anchors[0]
    rest = [p for p in pieces if p is not anchor]

    for p in rest:
        _annotate_connectors(p)
        assert 2 <= len(p["connectors"]) <= 4, "each piece must expose 2-4 candidate connectors"
        assert len({c["side"] for c in p["connectors"]}) == len(p["connectors"]), "candidate connectors must live on distinct sides"

    slots = _anchor_slots(anchor["arr"])
    assert len(rest) == len(slots), "piece count must match anchor slot count"

    canvas = np.full((h, w), bg, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, bg)
    assert placed is not None, "anchor must fit"

    anchor_bbox = anchor["bbox"]
    solutions: list[np.ndarray] = []

    def moves_for_slot(cur: np.ndarray, remaining: list[dict], slot: dict) -> list[tuple[int, int, int, int, int]]:
        moves = []
        for idx, piece in enumerate(remaining):
            match_count = 0
            for conn in piece["connectors"]:
                if conn["side"] != OPPOSITE[slot["side"]] or conn["length"] != slot["length"]:
                    continue
                match_count += 1
                top, left = _placement_for_slot(anchor_bbox, slot, piece, conn)
                if not _is_legal_slot_move(cur, anchor_bbox, slot, piece, conn, top, left, bg):
                    continue
                moves.append((idx, top, left, len(piece["connectors"]), piece["area"]))
            assert match_count <= 1, "generator guarantees at most one matching connector per piece"
        moves.sort(key=lambda x: (x[3], -x[4], x[1], x[2], x[0]))
        return moves

    def dfs(cur: np.ndarray, remaining: list[dict], unused_slots: list[dict]) -> None:
        if len(solutions) >= 2:
            return
        if not remaining:
            if not unused_slots:
                solutions.append(cur)
            return
        if len(remaining) != len(unused_slots):
            return

        best_slot = None
        best_moves = None
        for slot in unused_slots:
            cur_moves = moves_for_slot(cur, remaining, slot)
            if not cur_moves:
                return
            if best_moves is None or len(cur_moves) < len(best_moves):
                best_slot = slot
                best_moves = cur_moves
                if len(best_moves) == 1:
                    break

        assert best_slot is not None and best_moves is not None

        seen: set[tuple[int, int, int]] = set()
        next_slots = [s for s in unused_slots if s is not best_slot]
        for idx, top, left, _nconn, _area in best_moves:
            key = (idx, top, left)
            if key in seen:
                continue
            seen.add(key)
            piece = remaining[idx]
            nxt = _try_place(cur, piece["arr"], top, left, bg)
            if nxt is None:
                continue
            dfs(nxt, remaining[:idx] + remaining[idx + 1 :], next_slots)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, slots)
    assert len(solutions) == 1, "assembly must have exactly one valid solution"
    return solutions[0]


def _new_node(kind: str, h: int, w: int, color: int, top: int = 0, left: int = 0) -> dict:
    return {
        "kind": kind,
        "shape": (int(h), int(w)),
        "color": int(color),
        "top": int(top),
        "left": int(left),
    }


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


def _sample_side_runs(span: int, count: int) -> list[tuple[int, int]] | None:
    intervals = [(start, length) for length in CONNECTOR_LEN_CHOICES for start in range(span - length + 1)]
    combos = []
    for combo in itertools.combinations(intervals, count):
        ordered = sorted(combo)
        ok = True
        for (s0, l0), (s1, _l1) in zip(ordered, ordered[1:]):
            if s0 + l0 >= s1:
                ok = False
                break
        if ok:
            combos.append(ordered)
    if not combos:
        return None
    return random.choice(combos)


def _choose_anchor_slots(n_pieces: int) -> list[dict] | None:
    counts = {"top": 1, "right": 1, "bottom": 1, "left": 1}
    if n_pieces == 5:
        counts[random.choice(("top", "bottom"))] += 1
    if n_pieces == 6:
        counts["top"] += 1
        counts["bottom"] += 1
    bh, bw = ANCHOR_BLOCK
    side_spans = {"top": bw + 2, "bottom": bw + 2, "left": bh + 2, "right": bh + 2}
    slots = []
    for side in SIDES:
        run_count = counts[side]
        assert run_count <= ANCHOR_SIDE_CAP[side]
        chosen = _sample_side_runs(side_spans[side], run_count)
        if chosen is None:
            return None
        for start, length in chosen:
            slots.append({"side": side, "start": int(start), "length": int(length)})
    random.shuffle(slots)
    return slots


def _make_anchor(node: dict) -> np.ndarray:
    bh, bw = node["block"]
    h, w = node["shape"]
    arr = np.zeros((h, w), dtype=int)
    arr[1 : 1 + bh, 1 : 1 + bw] = 4
    for slot in node["slots"]:
        for r, c in _connector_cells((h, w), slot["side"], slot["start"], slot["length"]):
            arr[r, c] = 2
    return arr


def _sample_run_start(span: int, length: int) -> int:
    interior = [s for s in range(span - length + 1) if s > 0 and s + length < span]
    if interior:
        return random.choice(interior)
    return random.randint(0, span - length)


def _make_piece(node: dict) -> np.ndarray | None:
    h, w = node["shape"]
    color = node["color"]
    connector_sides = set(node["connectors"])
    arr = np.zeros((h, w), dtype=int)
    hub = (random.randint(1, h - 2), random.randint(1, w - 2))

    bar_sides = [s for s in SIDES if s not in connector_sides]
    if len(bar_sides) > 1 and random.random() < BAR_SKIP_PROB:
        bar_sides.remove(random.choice(bar_sides))
    for side in bar_sides:
        _paint_bar(arr, side, color, connector_sides)

    terminals = []
    for side, spec in node["connectors"].items():
        cells = _connector_cells((h, w), side, spec["start"], spec["length"])
        inwards = _connector_inward_cells((h, w), side, spec["start"], spec["length"])
        for r, c in cells:
            arr[r, c] = 2
        for r, c in inwards:
            arr[r, c] = color
        terminals.append(inwards[len(inwards) // 2])

    arr[hub] = color
    order = random.randint(0, 1)
    for t in terminals:
        _draw_path(arr, t, hub, color, order)
    for side in bar_sides:
        _draw_path(arr, hub, _bar_endpoint((h, w), side, hub), color, 1 - order)

    if len(components_mask(arr != 0, connectivity=4)) != 1:
        return None
    expected_reds = sum(spec["length"] for spec in node["connectors"].values())
    if int(np.sum(arr == 2)) != expected_reds:
        return None
    return arr


def _adjacency_pairs(a: dict, b: dict) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    arr = a["arr"]
    for r, c in np.argwhere(arr != 0):
        gr, gc = int(a["top"] + r), int(a["left"] + c)
        for nr, nc in ((gr - 1, gc), (gr + 1, gc), (gr, gc - 1), (gr, gc + 1)):
            br, bc = nr - b["top"], nc - b["left"]
            if 0 <= br < b["shape"][0] and 0 <= bc < b["shape"][1] and int(b["arr"][br, bc]) != 0:
                pairs.add(((gr, gc), (nr, nc)))
    return pairs


def _expected_anchor_pairs(anchor: dict, piece: dict) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    slot = piece["slot"]
    conn = piece["connectors"][piece["active_side"]]
    ar0, ac0, ar1, ac1 = _bbox_of(anchor)
    pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for offset in range(slot["length"]):
        if slot["side"] == "top":
            piece_cell = (piece["top"] + piece["shape"][0] - 1, piece["left"] + conn["start"] + offset)
            anchor_cell = (ar0, ac0 + slot["start"] + offset)
        elif slot["side"] == "bottom":
            piece_cell = (piece["top"], piece["left"] + conn["start"] + offset)
            anchor_cell = (ar1, ac0 + slot["start"] + offset)
        elif slot["side"] == "left":
            piece_cell = (piece["top"] + conn["start"] + offset, piece["left"] + piece["shape"][1] - 1)
            anchor_cell = (ar0 + slot["start"] + offset, ac0)
        else:
            piece_cell = (piece["top"] + conn["start"] + offset, piece["left"])
            anchor_cell = (ar0 + slot["start"] + offset, ac1)
        pairs.add((piece_cell, anchor_cell))
    return pairs


def _can_add_piece(nodes: list[dict], child: dict) -> bool:
    r0, c0, r1, c1 = _union_bbox(nodes + [child])
    if (r1 - r0 + 1) > MAX_ASSEMBLY_DIM or (c1 - c0 + 1) > MAX_ASSEMBLY_DIM:
        return False

    child_box = _bbox_of(child)
    anchor = nodes[0]
    if _rects_overlap(child_box, _bbox_of(anchor)):
        return False

    anchor_pairs = _adjacency_pairs(child, anchor)
    if anchor_pairs != _expected_anchor_pairs(anchor, child):
        return False
    for (pr, pc), (ar, ac) in anchor_pairs:
        if int(child["arr"][pr - child["top"], pc - child["left"]]) != 2:
            return False
        if int(anchor["arr"][ar - anchor["top"], ac - anchor["left"]]) != 2:
            return False

    for other in nodes[1:]:
        if _rects_overlap(child_box, _bbox_of(other)):
            return False
        if _adjacency_pairs(child, other):
            return False
    return True


def _grow_star() -> list[dict] | None:
    n_pieces = random.randint(*PIECE_COUNT_RANGE)
    colors = random.sample(PIECE_COLORS, n_pieces)

    bh, bw = ANCHOR_BLOCK
    anchor = _new_node("anchor", bh + 2, bw + 2, 4)
    anchor["block"] = (bh, bw)
    slots = _choose_anchor_slots(n_pieces)
    if slots is None:
        return None
    anchor["slots"] = slots
    anchor["arr"] = _make_anchor(anchor)
    nodes = [anchor]

    for color, slot in zip(colors, slots):
        active_side = OPPOSITE[slot["side"]]
        placed = False
        for _ in range(PIECE_SHAPE_TRIES):
            ph = random.randint(*PIECE_H_RANGE)
            pw = random.randint(*PIECE_W_RANGE)
            connector_count = random.randint(*CONNECTOR_COUNT_RANGE)
            connector_sides = [active_side]
            others = [s for s in SIDES if s != active_side]
            connector_sides.extend(random.sample(others, connector_count - 1))

            node = _new_node("piece", ph, pw, color)
            node["slot"] = dict(slot)
            node["active_side"] = active_side
            node["connectors"] = {}

            for side in connector_sides:
                span = ph if side in ("left", "right") else pw
                length = slot["length"] if side == active_side else random.choice([k for k in CONNECTOR_LEN_CHOICES if k != slot["length"]])
                if span < length:
                    break
                node["connectors"][side] = {
                    "start": _sample_run_start(span, length),
                    "length": int(length),
                }
            else:
                node["arr"] = _make_piece(node)
                if node["arr"] is None:
                    continue
                active = node["connectors"][active_side]
                if slot["side"] == "top":
                    node["top"] = anchor["top"] - ph
                    node["left"] = anchor["left"] + slot["start"] - active["start"]
                elif slot["side"] == "bottom":
                    node["top"] = anchor["top"] + anchor["shape"][0]
                    node["left"] = anchor["left"] + slot["start"] - active["start"]
                elif slot["side"] == "left":
                    node["top"] = anchor["top"] + slot["start"] - active["start"]
                    node["left"] = anchor["left"] - pw
                else:
                    node["top"] = anchor["top"] + slot["start"] - active["start"]
                    node["left"] = anchor["left"] + anchor["shape"][1]
                if not _can_add_piece(nodes, node):
                    continue
                nodes.append(node)
                placed = True
                break
        if not placed:
            return None
    return nodes


def _assemble(nodes: list[dict], h: int, w: int) -> np.ndarray | None:
    r0, c0, r1, c1 = _union_bbox(nodes)
    if (r1 - r0 + 1) > h or (c1 - c0 + 1) > w:
        return None
    sr = random.randint(-r0, h - 1 - r1)
    sc = random.randint(-c0, w - 1 - c1)

    out = np.zeros((h, w), dtype=int)
    for node in nodes:
        top = node["top"] + sr
        left = node["left"] + sc
        node["pos"] = (top, left)
        out = _try_place(out, node["arr"], top, left, 0)
        if out is None:
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


def _gen_attempt() -> np.ndarray | None:
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
    raise RuntimeError("failed to generate a valid cbebaa4b combo input")
