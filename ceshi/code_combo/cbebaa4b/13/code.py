# Combo variant of cbebaa4b (index 13)
# combo_spec: D1/1+D16/1
# Name: 边缘颜色序列匹配 + 多点同时对齐

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis
# - Concrete behavior:
#   原题靠红点重叠来连接；这里改成“贴边装配”。每个可连接侧面使用去角后的整条内部边带
#   作为接口，例如 5x7 精灵的上侧接口是 `arr[0,1:-1]`，长度为 5。融合后，一次连接必须同时满足：
#   1. 这整条接口颜色序列逐像素完全相同；
#   2. 其中所有红色像素都要在相邻的目标边上同时对位，不能只靠某一个红点决定平移。
#   solver 因此不再围绕单个红坐标回溯，而是围绕“开放侧边序列”回溯。
# - Small traced example:
#   若某个 leaf 的左侧接口是 `[2, 6, 2, 6, 2]`，候选 branch 右侧若是 `[2, 6, 2, 6, 2]`，
#   则该 leaf 可以整体贴上；若候选 branch 右侧是 `[2, 6, 6, 2, 2]`，虽然共享红色且颜色集合相近，
#   但第 3/4 个位置不一致，必须拒绝。再弱一点，只看“是否至少有一个红点对上”，这两条边都可能被误收，
#   正是 D16 要排除的情况。
# - Complexity calibration:
#   为避免退化成“看一眼一个红点就够了”，所有真实接口都强制含 2-3 个红点，且边长至少 3。
#   生成器还同时制造两种可见对照：
#   1. 一对接口共享同一红掩码但非红颜色不同，逼解题者使用 D1 的整序列匹配；
#   2. 另一对接口共享同一非红颜色但红掩码不同，逼解题者使用 D16 的全红点同时对齐。
# - Implicit-assumption audit against the original solver:
#   1. 原 `_try_place` 允许 red/red 重叠；贴边规则下任何重叠都非法，必须改成只允许画到背景。
#   2. 原 `red_counts` 把“未配对红点”当状态；这里状态改成“尚未消费的开放边接口”。
#   3. 原 DFS 用“把 piece 的某个红点移到目标红点”生成候选平移；现在平移由
#      “目标开放边 + 方向相反且长度一致的 piece 边”唯一决定。
#   4. 原 `main()` 默认单个红点足以定义连接；融合后这条隐式前提失效，必须显式检查整侧所有红点。
# - Placement legality:
#   生成器采用 output-first。先构造一棵 7 节点树状矩形布局，保证父子只在一整条边上相邻，
#   其余组件保持间隔，因此 assembled 输出无重叠、无越界、且只有预期接触。
#   再把非锚点散开，锚点保持原位。
# - Visual observability:
#   在每个样例里都能同时看到两类证据：一对接口“同红位形不同颜色”，另一对接口
#   “同颜色不同红位形”。因此既能看出 D1 的整序列颜色匹配，也能看出 D16 的全红点对齐。
# - Cleanliness:
#   所有对象都是去角矩形，主体为单色，连接信息只放在边缘内部条带；输出中只有并排贴边，
#   没有多层覆盖或复杂穿插，视觉上仍然像整洁的拼装板。
# - New solver assumptions and generator guarantees:
#   1. solver 假设每个真实接口至少含 2 个红点；generator 对所有真实连接边都保证这一点。
#   2. solver 假设连接只发生在相反方向、长度一致的整条边之间；generator 只生成这种树状布局。
#   3. solver 假设新增 piece 与装配体恰好接触一侧；generator 用布局结构和散开策略排除了额外接触。
#   4. solver 假设黄色只用于锚点；generator 保证只有锚点主体使用颜色 4。


SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}


def _edge_sequence(arr: np.ndarray, side: str) -> tuple[int, ...]:
    if side == "top":
        return tuple(int(v) for v in arr[0, 1:-1])
    if side == "bottom":
        return tuple(int(v) for v in arr[-1, 1:-1])
    if side == "left":
        return tuple(int(v) for v in arr[1:-1, 0])
    return tuple(int(v) for v in arr[1:-1, -1])


def _red_mask(seq: tuple[int, ...]) -> tuple[bool, ...]:
    return tuple(v == 2 for v in seq)


def _red_count(seq: tuple[int, ...]) -> int:
    return int(sum(v == 2 for v in seq))


def _mask_templates(span: int) -> list[tuple[bool, ...]]:
    assert span >= 3, "connector span must be at least 3"
    if span == 3:
        return [
            (True, False, True),
            (True, True, False),
            (False, True, True),
        ]
    mid = span // 2
    candidates = [
        tuple(i in {0, mid, span - 1} for i in range(span)),
        tuple(i in {0, 1, mid} for i in range(span)),
        tuple(i in {0, mid, span - 2} for i in range(span)),
        tuple(i in {1, mid, span - 1} for i in range(span)),
        tuple(i in {0, span - 2, span - 1} for i in range(span)),
        tuple((i % 2) == 0 for i in range(span)),
    ]
    out: list[tuple[bool, ...]] = []
    for mask in candidates:
        reds = sum(mask)
        if 2 <= reds <= 3 and mask not in out:
            out.append(mask)
    assert len(out) >= 3, "need at least three multi-red templates"
    return out


def _build_sequence(mask: tuple[bool, ...], accent: int) -> tuple[int, ...]:
    return tuple(2 if is_red else int(accent) for is_red in mask)


def _mask_pair_for_single_red_ablation(span: int) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    templates = _mask_templates(span)
    pairs = []
    for i, a in enumerate(templates):
        for b in templates[i + 1 :]:
            if any(x and y for x, y in zip(a, b)):
                pairs.append((a, b))
    assert pairs, "need at least one template pair sharing a red position"
    return random.choice(pairs)


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        edges = {side: _edge_sequence(arr, side) for side in SIDES}
        conn_sides = [side for side in SIDES if _red_count(edges[side]) >= 2]
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "shape": (int(arr.shape[0]), int(arr.shape[1])),
                "area": int(np.sum(arr != bg)),
                "edges": edges,
                "conn_sides": conn_sides,
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
    out_sub = out[top : top + ph, left : left + pw]
    out_sub[mask] = arr[mask]
    out[top : top + ph, left : left + pw] = out_sub
    return out


def _neighbor_sequence(
    canvas: np.ndarray, top: int, left: int, shape: tuple[int, int], side: str, bg: int
) -> tuple[int, ...] | None:
    ph, pw = shape
    if side == "top":
        r = top - 1
        if r < 0:
            return None
        return tuple(int(v) for v in canvas[r, left + 1 : left + pw - 1])
    if side == "bottom":
        r = top + ph
        if r >= canvas.shape[0]:
            return None
        return tuple(int(v) for v in canvas[r, left + 1 : left + pw - 1])
    if side == "left":
        c = left - 1
        if c < 0:
            return None
        return tuple(int(v) for v in canvas[top + 1 : top + ph - 1, c])
    c = left + pw
    if c >= canvas.shape[1]:
        return None
    return tuple(int(v) for v in canvas[top + 1 : top + ph - 1, c])


def _position_from_opening(opening: dict, piece: dict, side: str) -> tuple[int, int] | None:
    target_side = opening["side"]
    th, tw = opening["shape"]
    ph, pw = piece["shape"]
    if side != OPPOSITE[target_side]:
        return None
    if target_side in ("top", "bottom") and pw != tw:
        return None
    if target_side in ("left", "right") and ph != th:
        return None
    if target_side == "top":
        return opening["top"] - ph, opening["left"]
    if target_side == "bottom":
        return opening["top"] + th, opening["left"]
    if target_side == "left":
        return opening["top"], opening["left"] - pw
    return opening["top"], opening["left"] + tw


def _match_sequences(piece_seq: tuple[int, ...], target_seq: tuple[int, ...], mode: str) -> bool:
    if len(piece_seq) != len(target_seq):
        return False
    if mode == "strong":
        return piece_seq == target_seq and _red_count(piece_seq) >= 2
    if mode == "mask_only":
        return _red_mask(piece_seq) == _red_mask(target_seq) and _red_count(piece_seq) >= 2
    if mode == "single_red":
        if not any(a == 2 and b == 2 for a, b in zip(piece_seq, target_seq)):
            return False
        for a, b in zip(piece_seq, target_seq):
            if a != 2 and b != 2 and a != b:
                return False
        return True
    raise ValueError(f"unknown match mode: {mode}")


def _contact_fits(
    canvas: np.ndarray,
    piece: dict,
    top: int,
    left: int,
    matched_side: str,
    bg: int,
    mode: str,
) -> bool:
    if _try_place(canvas, piece["arr"], top, left, bg) is None:
        return False
    touches = 0
    for side in SIDES:
        vals = _neighbor_sequence(canvas, top, left, piece["shape"], side, bg)
        if vals is None:
            if side == matched_side:
                return False
            continue
        if any(v != bg for v in vals):
            if side != matched_side:
                return False
            if not _match_sequences(piece["edges"][side], vals, mode):
                return False
            touches += 1
        elif side == matched_side:
            return False
    return touches == 1


def _solve(grid: np.ndarray, mode: str = "strong") -> list[np.ndarray]:
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

    openings = [
        {
            "top": ar0,
            "left": ac0,
            "shape": anchor["shape"],
            "side": side,
            "seq": anchor["edges"][side],
        }
        for side in anchor["conn_sides"]
    ]

    solutions: list[np.ndarray] = []
    seen: set[bytes] = set()

    def record_solution(cur: np.ndarray) -> None:
        key = cur.tobytes()
        if key not in seen:
            seen.add(key)
            solutions.append(cur.copy())

    def dfs(cur: np.ndarray, remaining: list[dict], cur_openings: list[dict]) -> None:
        if len(solutions) >= 2:
            return
        if not remaining:
            if not cur_openings:
                record_solution(cur)
            return
        if not cur_openings:
            return

        best_idx = None
        best_moves = None
        for oi, opening in enumerate(cur_openings):
            moves = []
            for idx, piece in enumerate(remaining):
                for side in piece["conn_sides"]:
                    if side != OPPOSITE[opening["side"]]:
                        continue
                    if not _match_sequences(piece["edges"][side], opening["seq"], mode):
                        continue
                    pos = _position_from_opening(opening, piece, side)
                    if pos is None:
                        continue
                    top, left = pos
                    if not _contact_fits(cur, piece, top, left, side, bg, mode):
                        continue
                    moves.append((idx, side, top, left))
            if not moves:
                return
            if best_moves is None or len(moves) < len(best_moves):
                best_idx = oi
                best_moves = moves
                if len(best_moves) == 1:
                    break

        assert best_idx is not None and best_moves is not None
        dedup = {(idx, side, top, left): remaining[idx]["area"] for idx, side, top, left in best_moves}
        moves2 = [(idx, side, top, left) for idx, side, top, left in dedup]
        moves2.sort(key=lambda x: (-remaining[x[0]]["area"], x[2], x[3], x[0], x[1]))

        for idx, side, top, left in moves2:
            piece = remaining[idx]
            nxt = _try_place(cur, piece["arr"], top, left, bg)
            if nxt is None:
                continue
            next_openings = cur_openings[:best_idx] + cur_openings[best_idx + 1 :]
            for new_side in piece["conn_sides"]:
                if new_side == side:
                    continue
                next_openings.append(
                    {
                        "top": top,
                        "left": left,
                        "shape": piece["shape"],
                        "side": new_side,
                        "seq": piece["edges"][new_side],
                    }
                )
            next_remaining = remaining[:idx] + remaining[idx + 1 :]
            dfs(nxt, next_remaining, next_openings)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, openings)
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered side-coded sprites around the fixed yellow anchor by exact edge-sequence docking.

    Input structure:
    - bg = black (0). Several rectangular sprites are scattered without overlap.
    - Exactly one sprite contains yellow (4); that anchor stays at its input position.
    - Every usable connector is an entire side-interior strip, and each true strip contains multiple red cells.

    Rule:
    - Translate the non-anchor sprites so they abut the growing assembly without overlap.
    - A docking is valid only when the touching side is the opposite orientation, has the same full color sequence cell-by-cell, and all red cells on that side line up simultaneously.
    - Each placed sprite must touch the current assembly on exactly one connector side; all connector sides are consumed pairwise until no open sides remain.

    Key insight: Red does not mark a single peg anymore. Each connector side is an indivisible multi-cell code strip, so the correct move is determined by matching the whole edge sequence and satisfying every red position at once.
    """
    solutions = _solve(grid, mode="strong")
    assert len(solutions) == 1, "combo edge-sequence assembly must have exactly one valid solution"
    return solutions[0]


# --------------- generator (combo D1/1 + D16/1) ---------------

GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 24

PIECE_H_RANGE = (5, 7)
PIECE_W_RANGE = (5, 8)
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]

ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]
SCATTER_PADDING = 2
GEN_RETRY_LIMIT = 600


def _new_node(kind: str, h: int, w: int, color: int, top: int = 0, left: int = 0) -> dict:
    return {
        "kind": kind,
        "shape": (int(h), int(w)),
        "color": int(color),
        "top": int(top),
        "left": int(left),
        "sides": set(),
        "edge_seq": {},
    }


def _bbox_of(node: dict) -> tuple[int, int, int, int]:
    h, w = node["shape"]
    return (node["top"], node["left"], node["top"] + h - 1, node["left"] + w - 1)


def _union_bbox(nodes: list[dict]) -> tuple[int, int, int, int]:
    boxes = [_bbox_of(n) for n in nodes]
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _render_node(node: dict) -> np.ndarray:
    h, w = node["shape"]
    fill = 4 if node["kind"] == "anchor" else node["color"]
    arr = np.full((h, w), fill, dtype=int)
    arr[0, 0] = 0
    arr[0, -1] = 0
    arr[-1, 0] = 0
    arr[-1, -1] = 0
    for side, seq in node["edge_seq"].items():
        if side == "top":
            arr[0, 1:-1] = seq
        elif side == "bottom":
            arr[-1, 1:-1] = seq
        elif side == "left":
            arr[1:-1, 0] = seq
        else:
            arr[1:-1, -1] = seq
    return arr


def _assign_edge_sequences(nodes: list[dict], edges: list[tuple[int, str, int, str]]) -> None:
    assert len(edges) == 6, "this generator builds a fixed 6-edge tree"

    anchor_span = len(_edge_sequence(np.zeros(nodes[0]["shape"], dtype=int), edges[0][1]))
    anchor_templates = _mask_templates(anchor_span)
    if len(anchor_templates) >= 2:
        anchor_mask_a, anchor_mask_b = random.sample(anchor_templates, 2)
    else:
        anchor_mask_a = anchor_mask_b = anchor_templates[0]

    mid_span = len(_edge_sequence(np.zeros(nodes[edges[2][0]]["shape"], dtype=int), edges[2][1]))
    mid_mask = random.choice(_mask_templates(mid_span))

    outer_span = len(_edge_sequence(np.zeros(nodes[edges[4][0]]["shape"], dtype=int), edges[4][1]))
    outer_mask_a, outer_mask_b = _mask_pair_for_single_red_ablation(outer_span)

    edge_masks = [anchor_mask_a, anchor_mask_b, mid_mask, mid_mask, outer_mask_a, outer_mask_b]
    for edge_idx, (parent_i, parent_side, child_i, child_side) in enumerate(edges):
        seq = _build_sequence(edge_masks[edge_idx], nodes[child_i]["color"])
        nodes[parent_i]["edge_seq"][parent_side] = seq
        nodes[child_i]["edge_seq"][child_side] = seq


def _grow_layout() -> list[dict] | None:
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    anchor = _new_node("anchor", bh + 2, bw + 2, 4)

    palette = random.sample(PIECE_COLORS, 5)
    branch_colors = palette[:2]
    mid_colors = palette[2:4]
    outer_color = palette[4]

    nodes = [anchor]
    edges: list[tuple[int, str, int, str]] = []

    if random.random() < 0.5:
        branch_h = random.randint(*PIECE_H_RANGE)
        mid_w = random.randint(*PIECE_W_RANGE)
        outer_w = random.randint(*PIECE_W_RANGE)
        grow_side = random.choice(["left", "right"])

        top_piece = _new_node("piece", branch_h, anchor["shape"][1], branch_colors[0], top=-branch_h, left=0)
        bottom_piece = _new_node(
            "piece", branch_h, anchor["shape"][1], branch_colors[1], top=anchor["shape"][0], left=0
        )
        if grow_side == "right":
            top_mid = _new_node("piece", branch_h, mid_w, mid_colors[0], top=top_piece["top"], left=anchor["shape"][1])
            bottom_mid = _new_node(
                "piece", branch_h, mid_w, mid_colors[1], top=bottom_piece["top"], left=anchor["shape"][1]
            )
            top_outer = _new_node(
                "piece", branch_h, outer_w, outer_color, top=top_piece["top"], left=anchor["shape"][1] + mid_w
            )
            bottom_outer = _new_node(
                "piece", branch_h, outer_w, outer_color, top=bottom_piece["top"], left=anchor["shape"][1] + mid_w
            )
            anchor["sides"] = {"top", "bottom"}
            top_piece["sides"] = {"bottom", "right"}
            bottom_piece["sides"] = {"top", "right"}
            top_mid["sides"] = {"left", "right"}
            bottom_mid["sides"] = {"left", "right"}
            top_outer["sides"] = {"left"}
            bottom_outer["sides"] = {"left"}
            nodes.extend([top_piece, bottom_piece, top_mid, bottom_mid, top_outer, bottom_outer])
            edges = [
                (0, "top", 1, "bottom"),
                (0, "bottom", 2, "top"),
                (1, "right", 3, "left"),
                (2, "right", 4, "left"),
                (3, "right", 5, "left"),
                (4, "right", 6, "left"),
            ]
        else:
            top_mid = _new_node("piece", branch_h, mid_w, mid_colors[0], top=top_piece["top"], left=-mid_w)
            bottom_mid = _new_node(
                "piece", branch_h, mid_w, mid_colors[1], top=bottom_piece["top"], left=-mid_w
            )
            top_outer = _new_node("piece", branch_h, outer_w, outer_color, top=top_piece["top"], left=-(mid_w + outer_w))
            bottom_outer = _new_node(
                "piece", branch_h, outer_w, outer_color, top=bottom_piece["top"], left=-(mid_w + outer_w)
            )
            anchor["sides"] = {"top", "bottom"}
            top_piece["sides"] = {"bottom", "left"}
            bottom_piece["sides"] = {"top", "left"}
            top_mid["sides"] = {"right", "left"}
            bottom_mid["sides"] = {"right", "left"}
            top_outer["sides"] = {"right"}
            bottom_outer["sides"] = {"right"}
            nodes.extend([top_piece, bottom_piece, top_mid, bottom_mid, top_outer, bottom_outer])
            edges = [
                (0, "top", 1, "bottom"),
                (0, "bottom", 2, "top"),
                (1, "left", 3, "right"),
                (2, "left", 4, "right"),
                (3, "left", 5, "right"),
                (4, "left", 6, "right"),
            ]
    else:
        branch_w = random.randint(*PIECE_W_RANGE)
        mid_h = random.randint(*PIECE_H_RANGE)
        outer_h = random.randint(*PIECE_H_RANGE)
        grow_side = random.choice(["top", "bottom"])

        left_piece = _new_node("piece", anchor["shape"][0], branch_w, branch_colors[0], top=0, left=-branch_w)
        right_piece = _new_node(
            "piece", anchor["shape"][0], branch_w, branch_colors[1], top=0, left=anchor["shape"][1]
        )
        if grow_side == "bottom":
            left_mid = _new_node("piece", mid_h, branch_w, mid_colors[0], top=anchor["shape"][0], left=left_piece["left"])
            right_mid = _new_node(
                "piece", mid_h, branch_w, mid_colors[1], top=anchor["shape"][0], left=right_piece["left"]
            )
            left_outer = _new_node(
                "piece", outer_h, branch_w, outer_color, top=anchor["shape"][0] + mid_h, left=left_piece["left"]
            )
            right_outer = _new_node(
                "piece", outer_h, branch_w, outer_color, top=anchor["shape"][0] + mid_h, left=right_piece["left"]
            )
            anchor["sides"] = {"left", "right"}
            left_piece["sides"] = {"right", "bottom"}
            right_piece["sides"] = {"left", "bottom"}
            left_mid["sides"] = {"top", "bottom"}
            right_mid["sides"] = {"top", "bottom"}
            left_outer["sides"] = {"top"}
            right_outer["sides"] = {"top"}
            nodes.extend([left_piece, right_piece, left_mid, right_mid, left_outer, right_outer])
            edges = [
                (0, "left", 1, "right"),
                (0, "right", 2, "left"),
                (1, "bottom", 3, "top"),
                (2, "bottom", 4, "top"),
                (3, "bottom", 5, "top"),
                (4, "bottom", 6, "top"),
            ]
        else:
            left_mid = _new_node("piece", mid_h, branch_w, mid_colors[0], top=-mid_h, left=left_piece["left"])
            right_mid = _new_node(
                "piece", mid_h, branch_w, mid_colors[1], top=-mid_h, left=right_piece["left"]
            )
            left_outer = _new_node("piece", outer_h, branch_w, outer_color, top=-(mid_h + outer_h), left=left_piece["left"])
            right_outer = _new_node(
                "piece", outer_h, branch_w, outer_color, top=-(mid_h + outer_h), left=right_piece["left"]
            )
            anchor["sides"] = {"left", "right"}
            left_piece["sides"] = {"right", "top"}
            right_piece["sides"] = {"left", "top"}
            left_mid["sides"] = {"bottom", "top"}
            right_mid["sides"] = {"bottom", "top"}
            left_outer["sides"] = {"bottom"}
            right_outer["sides"] = {"bottom"}
            nodes.extend([left_piece, right_piece, left_mid, right_mid, left_outer, right_outer])
            edges = [
                (0, "left", 1, "right"),
                (0, "right", 2, "left"),
                (1, "top", 3, "bottom"),
                (2, "top", 4, "bottom"),
                (3, "top", 5, "bottom"),
                (4, "top", 6, "bottom"),
            ]

    r0, c0, r1, c1 = _union_bbox(nodes)
    if (r1 - r0 + 1) > MAX_ASSEMBLY_DIM or (c1 - c0 + 1) > MAX_ASSEMBLY_DIM:
        return None

    _assign_edge_sequences(nodes, edges)
    return nodes


def _place_node_on_canvas(out: np.ndarray, node: dict, sr: int, sc: int) -> bool:
    arr = node["arr"]
    top, left = node["top"] + sr, node["left"] + sc
    node["pos"] = (top, left)
    h, w = arr.shape
    if top < 0 or left < 0 or top + h > out.shape[0] or left + w > out.shape[1]:
        return False
    sub = out[top : top + h, left : left + w]
    mask = arr != 0
    if np.any(mask & (sub != 0)):
        return False
    paste(out, arr, top, left, transparent=0)
    return True


def _assemble(nodes: list[dict], h: int, w: int) -> np.ndarray | None:
    for node in nodes:
        node["arr"] = _render_node(node)
        if len(components_mask(node["arr"] != 0, connectivity=4)) != 1:
            return None
        for side, seq in node["edge_seq"].items():
            if _red_count(seq) < 2:
                return None

    r0, c0, r1, c1 = _union_bbox(nodes)
    if (r1 - r0 + 1) > h or (c1 - c0 + 1) > w:
        return None
    sr = random.randint(-r0, h - 1 - r1)
    sc = random.randint(-c0, w - 1 - c1)

    out = np.zeros((h, w), dtype=int)
    for node in nodes:
        if not _place_node_on_canvas(out, node, sr, sc):
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
    nodes = _grow_layout()
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
    raise RuntimeError("failed to generate a valid cbebaa4b combo D1/1+D16/1 input")
