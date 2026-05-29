# Variant of cbebaa4b
# Dimension: D1 — 连接器匹配条件 (match_cond)
# Change: 精灵的红色像素与目标（中央黄色方块或已连接精灵）的红色像素对齐，形成连接... → 边缘颜色序列匹配

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis
# - Concrete behavior:
#   输入仍然是一个固定位置的黄色锚点和若干散落精灵，输出仍然把它们拼成一棵连通结构。
#   但连接不再靠“红点重合”，而改成“整条接触边的颜色序列完全一致”。
#   为了让多侧连接时不在角点发生冲突，真正参与匹配的是每条边去掉两个角后的内部边段。
#   例如一个 6x7 精灵的上边可见接触序列长度是 5，对应 `arr[0,1:-1]`；左边则是 `arr[1:-1,0]`。
#   每条可连接边都包含若干红色像素和一种强调色；两条边只有在整段序列逐像素完全相同的时候才能贴合。
# - Small traced example:
#   原逻辑里，若两条边上的红像素位置都在第 1 和第 3 格，就可能被视为可接。
#   变体里，若一条边是 `[2, 8, 2]`，另一条边是 `[2, 3, 2]`，虽然红像素位置相同，但不能连接；
#   只有同样是 `[2, 8, 2]` 的目标边才允许贴边。差异正体现在“非红色位置也要匹配”。
# - Complexity calibration:
#   若连接边只是纯色或长度过短，人类会退化成“看红点就行”。因此 piece 尺寸下限上调到 5，
#   让接触序列至少长度 3；同时只使用少数红色掩码模板，促使同一红点布局在不同颜色下重复出现，
#   从而让“必须看整条颜色序列”这件事在视觉上可观察。
# - Implicit-assumption audit against the original solver:
#   1. 原 `_try_place` 允许红点重叠；这里改成完全禁止重叠，因为连接方式变成相邻贴边。
#   2. 原 `reds`/`red_counts` 假设“连接状态 = 红点坐标配对数”；这里失效，必须改成“开放接触边”状态机。
#   3. 原 DFS 通过“把 piece 的某个红点移到未配对红点坐标”生成候选位移；这里不成立，候选位移由
#      “某个开放边 + 某个相反方向的 matching edge”唯一决定。
#   4. 原冲突检测里同色覆盖是无害的，因为旧题允许红点或相同颜色重合；这里若继续无条件覆盖，会把
#      相邻连接错误地当成重叠，因此只能允许背景上色，不能覆盖已有非背景像素。
# - Placement legality:
#   生成器先在局部坐标里构造树状矩形布局。父子节点只允许在一整条边上相邻；对非父子节点，要求至少
#   保持 1 格空隙，避免意外接触。随后整体随机平移到大网格里，再把非锚点精灵重新散开。
# - Visual observability:
#   每条连接边都含红色，但真正决定是否可接的是“红色+强调色”的完整序列。由于不同连接会重复使用
#   相同红色掩码而配不同强调色，2-3 组样例里就能看出：只看红点位置不足以确定连接。
# - Cleanliness:
#   所有精灵都是去角矩形，中心填充单一主体色，连接信息只放在边缘内部条带，避免输出里出现难以辨认
#   的多层重叠或复杂穿插。
# - New solver assumptions and generator guarantees:
#   1. solver 假设每个可连接边都含红色，generator 只在真实连接边上放红色。
#   2. solver 假设连接发生在相反方向的整条边上，generator 只构造整边贴合。
#   3. solver 假设新增 piece 与已有装配体恰好接触一侧，generator 通过布局检查排除额外接触。
#   4. solver 假设锚点唯一包含黄色块，generator 保证只有锚点使用颜色 4 作为主体。


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


def _match_key(seq: tuple[int, ...], exact_colors: bool) -> tuple[int, ...] | tuple[bool, ...]:
    if exact_colors:
        return seq
    return tuple(v == 2 for v in seq)


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        edges = {side: _edge_sequence(arr, side) for side in SIDES}
        conn_sides = [side for side in SIDES if 2 in edges[side]]
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


def _try_place(
    canvas: np.ndarray, arr: np.ndarray, top: int, left: int, bg: int
) -> np.ndarray | None:
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


def _contact_fits(
    canvas: np.ndarray,
    piece: dict,
    top: int,
    left: int,
    matched_side: str,
    bg: int,
    exact_colors: bool,
) -> bool:
    if _try_place(canvas, piece["arr"], top, left, bg) is None:
        return False
    for side in SIDES:
        vals = _neighbor_sequence(canvas, top, left, piece["shape"], side, bg)
        if side == matched_side:
            if vals is None:
                return False
            if _match_key(vals, exact_colors) != _match_key(piece["edges"][side], exact_colors):
                return False
        else:
            if vals is not None and any(v != bg for v in vals):
                return False
    return True


def _solve(grid: np.ndarray, exact_colors: bool) -> list[np.ndarray]:
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
            target_key = _match_key(opening["seq"], exact_colors)
            for idx, piece in enumerate(remaining):
                for side in piece["conn_sides"]:
                    if side != OPPOSITE[opening["side"]]:
                        continue
                    if _match_key(piece["edges"][side], exact_colors) != target_key:
                        continue
                    pos = _position_from_opening(opening, piece, side)
                    if pos is None:
                        continue
                    top, left = pos
                    if not _contact_fits(cur, piece, top, left, side, bg, exact_colors):
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
        opening = cur_openings[best_idx]

        dedup = {}
        for idx, side, top, left in best_moves:
            key = (idx, side, top, left)
            dedup[key] = remaining[idx]["area"]
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
    Reassemble scattered edge-coded sprites around a fixed yellow anchor by matching full side sequences.

    Input structure:
    - bg = black (0). Several rectangular sprites are scattered without overlap.
    - One anchor sprite contains the only yellow (4) block and stays at its input location.
    - Connectable sides carry red-marked edge strips; each strip is the side interior excluding the two corners.

    Rule:
    - The anchor remains fixed while the other sprites are translated into contact with the growing assembly.
    - A placement is valid only when a sprite touches the assembly on exactly one side and the entire touching edge sequence matches cell-by-cell.
    - No pixels overlap; sprites simply abut along the matched side until all red-marked sides are paired.

    Key insight: The red pixels merely flag candidate connector sides; the actual assembly criterion is exact equality of the whole edge color sequence, not just the red offsets.
    """
    solutions = _solve(grid, exact_colors=True)
    assert len(solutions) == 1, "edge-sequence assembly must have exactly one valid solution"
    return solutions[0]


# --------------- generator (D1 edge_match variant) ---------------

# Canvas
GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 24

# Pieces
PIECE_H_RANGE = (5, 7)
PIECE_W_RANGE = (5, 8)
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
PIECE_CAP = 2

# Anchor
ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]
ANCHOR_CAP_CHOICES = [1, 2]

# Scatter / generation
SCATTER_PADDING = 2
GEN_RETRY_LIMIT = 400


def _new_node(kind, h, w, color, top=0, left=0, cap=PIECE_CAP):
    return {
        "kind": kind,
        "shape": (int(h), int(w)),
        "color": int(color),
        "top": int(top),
        "left": int(left),
        "cap": int(cap),
        "sides": set(),
        "edge_seq": {},
    }


def _bbox_of(node):
    h, w = node["shape"]
    return (node["top"], node["left"], node["top"] + h - 1, node["left"] + w - 1)


def _union_bbox(nodes):
    boxes = [_bbox_of(n) for n in nodes]
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _mask_templates(span: int) -> list[tuple[bool, ...]]:
    if span == 3:
        return [
            (True, False, True),
            (False, True, False),
        ]
    mid = span // 2
    templates = [
        tuple(i in {0, mid, span - 1} for i in range(span)),
        tuple(i in {1, span - 2} for i in range(span)),
        tuple((i % 2) == 0 for i in range(span)),
    ]
    uniq = []
    for tpl in templates:
        if tpl not in uniq and any(tpl) and not all(tpl):
            uniq.append(tpl)
    return uniq


def _build_sequence(span: int, accent: int, mask_id: int) -> tuple[int, ...]:
    mask = _mask_templates(span)[mask_id % len(_mask_templates(span))]
    return tuple(2 if is_red else int(accent) for is_red in mask)


def _assign_edge_sequences(nodes: list[dict], edges: list[tuple[int, str, int, str]]) -> None:
    by_span: dict[int, list[int]] = {}
    for ei, (parent_i, parent_side, _child_i, _child_side) in enumerate(edges):
        span = len(_edge_sequence(np.zeros(nodes[parent_i]["shape"], dtype=int), parent_side))
        by_span.setdefault(span, []).append(ei)

    mask_ids = {}
    for span, eids in by_span.items():
        for j, ei in enumerate(eids):
            if len(eids) >= 2 and j < 2:
                mask_ids[ei] = 0
            else:
                mask_ids[ei] = j % len(_mask_templates(span))

    for ei, (parent_i, parent_side, child_i, child_side) in enumerate(edges):
        span = len(_edge_sequence(np.zeros(nodes[parent_i]["shape"], dtype=int), parent_side))
        accent = nodes[child_i]["color"]
        seq = _build_sequence(span, accent, mask_ids[ei])
        nodes[parent_i]["edge_seq"][parent_side] = seq
        nodes[child_i]["edge_seq"][child_side] = seq


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


def _grow_tree():
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    anchor = _new_node("anchor", bh + 2, bw + 2, 4, cap=random.choice(ANCHOR_CAP_CHOICES))
    anchor["block"] = (bh, bw)
    colors = random.sample(PIECE_COLORS, 4)
    nodes = [anchor]
    edges: list[tuple[int, str, int, str]] = []

    if random.random() < 0.5:
        branch_h = random.randint(*PIECE_H_RANGE)
        leaf_w = random.randint(*PIECE_W_RANGE)
        leaf_side = random.choice(["left", "right"])

        top_piece = _new_node("piece", branch_h, anchor["shape"][1], colors[0], top=-branch_h, left=0)
        bottom_piece = _new_node(
            "piece", branch_h, anchor["shape"][1], colors[1], top=anchor["shape"][0], left=0
        )
        if leaf_side == "right":
            top_leaf = _new_node(
                "piece", branch_h, leaf_w, colors[2], top=top_piece["top"], left=anchor["shape"][1]
            )
            bottom_leaf = _new_node(
                "piece", branch_h, leaf_w, colors[3], top=bottom_piece["top"], left=anchor["shape"][1]
            )
            top_piece["sides"] = {"bottom", "right"}
            bottom_piece["sides"] = {"top", "right"}
            top_leaf["sides"] = {"left"}
            bottom_leaf["sides"] = {"left"}
            anchor["sides"] = {"top", "bottom"}
            nodes.extend([top_piece, bottom_piece, top_leaf, bottom_leaf])
            edges = [
                (0, "top", 1, "bottom"),
                (0, "bottom", 2, "top"),
                (1, "right", 3, "left"),
                (2, "right", 4, "left"),
            ]
        else:
            top_leaf = _new_node("piece", branch_h, leaf_w, colors[2], top=top_piece["top"], left=-leaf_w)
            bottom_leaf = _new_node(
                "piece", branch_h, leaf_w, colors[3], top=bottom_piece["top"], left=-leaf_w
            )
            top_piece["sides"] = {"bottom", "left"}
            bottom_piece["sides"] = {"top", "left"}
            top_leaf["sides"] = {"right"}
            bottom_leaf["sides"] = {"right"}
            anchor["sides"] = {"top", "bottom"}
            nodes.extend([top_piece, bottom_piece, top_leaf, bottom_leaf])
            edges = [
                (0, "top", 1, "bottom"),
                (0, "bottom", 2, "top"),
                (1, "left", 3, "right"),
                (2, "left", 4, "right"),
            ]
    else:
        branch_w = random.randint(*PIECE_W_RANGE)
        leaf_h = random.randint(*PIECE_H_RANGE)
        leaf_side = random.choice(["top", "bottom"])

        left_piece = _new_node("piece", anchor["shape"][0], branch_w, colors[0], top=0, left=-branch_w)
        right_piece = _new_node(
            "piece", anchor["shape"][0], branch_w, colors[1], top=0, left=anchor["shape"][1]
        )
        if leaf_side == "bottom":
            left_leaf = _new_node(
                "piece", leaf_h, branch_w, colors[2], top=anchor["shape"][0], left=left_piece["left"]
            )
            right_leaf = _new_node(
                "piece", leaf_h, branch_w, colors[3], top=anchor["shape"][0], left=right_piece["left"]
            )
            left_piece["sides"] = {"right", "bottom"}
            right_piece["sides"] = {"left", "bottom"}
            left_leaf["sides"] = {"top"}
            right_leaf["sides"] = {"top"}
            anchor["sides"] = {"left", "right"}
            nodes.extend([left_piece, right_piece, left_leaf, right_leaf])
            edges = [
                (0, "left", 1, "right"),
                (0, "right", 2, "left"),
                (1, "bottom", 3, "top"),
                (2, "bottom", 4, "top"),
            ]
        else:
            left_leaf = _new_node("piece", leaf_h, branch_w, colors[2], top=-leaf_h, left=left_piece["left"])
            right_leaf = _new_node(
                "piece", leaf_h, branch_w, colors[3], top=-leaf_h, left=right_piece["left"]
            )
            left_piece["sides"] = {"right", "top"}
            right_piece["sides"] = {"left", "top"}
            left_leaf["sides"] = {"bottom"}
            right_leaf["sides"] = {"bottom"}
            anchor["sides"] = {"left", "right"}
            nodes.extend([left_piece, right_piece, left_leaf, right_leaf])
            edges = [
                (0, "left", 1, "right"),
                (0, "right", 2, "left"),
                (1, "top", 3, "bottom"),
                (2, "top", 4, "bottom"),
            ]

    r0, c0, r1, c1 = _union_bbox(nodes)
    if (r1 - r0 + 1) > MAX_ASSEMBLY_DIM or (c1 - c0 + 1) > MAX_ASSEMBLY_DIM:
        return None

    _assign_edge_sequences(nodes, edges)
    return nodes


def _place_node_on_canvas(out, node, sr, sc):
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


def _assemble(nodes, h, w):
    for node in nodes:
        node["arr"] = _render_node(node)
        if len(components_mask(node["arr"] != 0, connectivity=4)) != 1:
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

    try:
        red_only = _solve(inp, exact_colors=False)
    except Exception:
        red_only = []
    if len(red_only) == 1 and np.array_equal(red_only[0], out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cbebaa4b D1 input")
